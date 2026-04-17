# Hume 代码库架构概述

## 1. 主要技术栈

- **语言:** Python (>=3.10)
- **ML框架:** PyTorch、HuggingFace `transformers` 和 `lerobot`
- **包管理:** `uv` (astral.sh)
- **构建:** setuptools

## 2. 目录结构

| 目录 | 用途 |
|------|------|
| `src/hume/` | 主 Python 包（核心模型、训练、部署） |
| `packages/openpi-client/` | OpenPI 协议客户端包 |
| `config/` | 数据集配置 JSON（LIBERO、Bridge、Fractal） |
| `experiments/` | 实验/评测代码 |
| `scripts/` | 训练脚本 |

## 3. 项目结构

**monorepo 风格**，根目录 `pyproject.toml` 使用 `uv` workspace 模式。

## 4. 核心组件

- **模型** (`src/hume/models/`): `HumePolicy`、`System2`、`PaliGemmaWithExpert`、`ValueQueryHead` 等
- **训练** (`src/hume/training/`): System 2 训练、VQH + System 1 联合训练
- **部署** (`src/hume/serving/`): 策略服务基础设施

## 5. 架构亮点

- **双系统架构**: System 1（快速反应）+ System 2（深思熟虑的规划/重规划）
- **Value Query Head (VQH)**: 评估动作序列得分的可学习模块
- **基于 PaliGemma**: 集成 DinoV2 视觉编码器的视觉语言动作模型
- **LeRobot 数据格式**: 支持机器人学习数据标准化

## 6. 工作流程

### 训练阶段总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: System2 预训练                                          │
│  train_s2.py → System2Policy → System2 (SigLIP + PaliGemma)     │
│  输出: hume_s2 checkpoint                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: VQH + System1 联合训练                                   │
│  train_vqh_s1.py → HumePolicy (加载 S2 checkpoint)               │
│  - System2: 可冻结或微调                                          │
│  - System1: DINO 视觉编码器 + Flow Matching                       │
│  - VQH: CalQL (保守摊销线性Q学习) 评估候选动作                      │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: System2 训练 (`train_s2.py`)

**配置参数:**
- `chunk_size`: 动作序列长度 (默认 50)
- `pretrained_paligemma_path`: 预训练 PaliGemma 模型路径
- `freeze_vision_encoder`: 是否冻结视觉编码器

**训练流程:**
```python
# 1. 创建 System2Policy
policy = make_policy(cfg, ds_meta, policy_cls=System2Policy)

# 2. 可选: 加载预训练 PaliGemma
paligemma = PaliGemmaForConditionalGeneration.from_pretrained(path)
policy.model.paligemma_with_expert.paligemma = paligemma

# 3. 单次前向传播计算 loss
loss, output_dict = policy.forward(batch)

# 4. 损失函数: MSE(u_t, v_t)
# u_t = noise - actions (目标 velocity)
# v_t = policy 输出
accelerator.backward(loss)
```

**数据流:**
```
batch = {
    "observation.images.image": [...],     # 图像
    "observation.state": [...],            # 机器人状态
    "task": ["pick up the cup\n", ...],   # 语言指令
    "action": [...],                      # 动作序列
    "stamp": [...],                       # 时间戳 (0-1)
}
```

### Stage 2: VQH + System1 训练 (`train_vqh_s1.py`)

**配置参数:**
- `pretrained_s2_path`: Stage 1 输出的 System2 checkpoint
- `actor_lr`, `critic_lr`, `temp_lr`: 各优化器学习率
- `s1_chunk_size`, `s2_chunk_size`: System1/2 动作块大小
- `vqh_chunk_size`: VQH 评估的动作块长度
- `freeze_s2`: 是否冻结 System2

**多优化器架构:**
```python
optimizers = {
    "trunk_optimizer":      # FastVisuoExpert + state_proj + action投影层
    "actor_optimizer":       # CalQL policy network
    "critic_optimizer":      # CalQL Q-network (双网络: online + target)
    "temperature_optimizer": # Q-learning temperature
}
```

**训练流程:**
```python
# 1. 加载 HumePolicy (包含 S1, S2, VQH)
policy = make_policy(cfg, ds_meta, policy_cls=HumePolicy)
policy.s2_model.load_state_dict(System2Policy.from_pretrained(s2_path).model.state_dict())

# 2. 前向传播
chunk_loss, temperature_loss, policy_loss, critic_loss, output_dict = policy.forward(batch)

# 3. 总损失
total_loss = chunk_loss + temperature_loss + policy_loss + critic_loss

# 4. 软更新目标网络 (每 target_critic_update_period 步)
soft_update(target_critics, critics, tau=soft_target_critic_update_rate)

# 5. 反向传播
accelerator.backward(total_loss)
```

### 推理阶段 (`HumePolicy.infer`)

```python
def infer(self, observation):
    # 1. 维护状态历史
    self.history_state.append(observation["state"])

    # 2. 每 s2_replan_steps 步重新规划
    if self.infer_step % s2_replan_steps == 0:
        # 生成 s2_candidates_num 个候选动作
        noise_actions = [
            s2_model.sample_actions(images, state, noise_temp=i/s2_candidates_num)
            for i in range(s2_candidates_num)
        ]
        # VQH 选择最优候选
        action_index, q_values = vqh.select_q_actions(images, noise_actions)
        selected_action = noise_actions[action_index]

    # 3. System1 细粒度去噪
    s1_action = s1_model.sample_actions(images, state, selected_action, stamp)

    # 4. 返回单步动作
    return action_queue.popleft()
```

### 数据规格

| 字段 | 形状 | 说明 |
|------|------|------|
| `observation.images` | `B x H x W x C` | 图像 (RGB) |
| `observation.state` | `B x state_dim` | 机器人关节状态 |
| `task` | `List[str]` | 语言指令 |
| `action` | `B x chunk_size x action_dim` | 动作序列 |
| `stamp` | `B` | 时间步 (0-1) |
| `reward.vqh` | `B` | VQH 奖励信号 |
| `mc.vqh` | `B` | 蒙特卡洛返回值 |

### 输出日志指标

| 指标 | 说明 |
|------|------|
| `s1_loss` | System1 Flow Matching 损失 |
| `actor_loss` | VQH policy 损失 |
| `critic_loss` | VQH Q-network 损失 |
| `cql_loss` | Conservative Q-Learning 正则化损失 |
| `temperature` | Q-learning temperature 参数值 |
| `entropy` | 动作分布熵 |
| `online_q` / `target_q` | Online/Target Q 值 |
| `calql_bound_rate` | Q 值约束满足率 |

---

## 7. 硬件平台要求

### GPU 要求

| 训练阶段 | GPU 数量 | 每卡 Batch Size | 总 Batch Size | 精度 |
|---------|----------|----------------|---------------|------|
| **System2 训练** (`train_s2.sh`) | 2 (默认) | 32 | 64 | fp32/bf16 |
| **VQH+S1 训练** (`train_vqh_s1.sh`) | 8 (默认) | 16 | 128 | bf16 |
| **Debug 模式** | 1 | 8 | 8 | bf16 |

### 推理要求

- **CUDA**: 需要 CUDA 支持
- **内存**: 推理时模型需加载到 GPU，建议 16GB+ 显存
- **设备**: `cuda` (不支持 CPU 推理)

### 依赖框架

```
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
accelerate==1.5.2
```

### 分布式训练配置

```bash
# 单机多卡
GPUS=8
GPUS_PER_NODE=8
ACCELERATE_ARGS="--num_machines 1 --num_processes ${GPUS} --multi_gpu --mixed_precision=bf16"

# 多机多卡
GPUS=16
GPUS_PER_NODE=8
NODES=2
ACCELERATE_ARGS="--num_machines ${NODES} --num_processes=${GPUS} --multi_gpu --mixed_precision=bf16"
```

### 存储要求

- **数据集**: LeRobot 格式数据集 (如 LIBERO 系列)
- **Checkpoint**: 每 5000 步保存一次，保留数量可配置 (`checkpoints_total_limit`)
- **日志**: WandB 或本地日志

### 环境变量

```bash
source scripts/env.sh  # 需设置 WANDB_PROJECT, WANDB_ENTITY 等
```

---

## 8. 机械臂参数要求

### 状态维度 (Observation State)

| 数据集 | state_dim | 说明 |
|--------|-----------|------|
| **LIBERO** | 8 | 7 DOF 机械臂 + 1 gripper |
| **Bridge** | 8 | 7 DOF 机械臂 + 1 gripper |
| **Fractal** | 32 | 支持更复杂的状态 |

**状态格式:**
```python
observation.state: np.array(B, state_dim)  # float32
# 例如: [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

### 动作维度 (Action)

| 数据集 | action_dim | 说明 |
|--------|------------|------|
| **LIBERO** | 7 | 7 DOF 机械臂 (无独立 gripper 维度) |
| **Bridge** | 7 | 7 DOF 机械臂 |
| **Fractal** | 32 | 支持更复杂的动作 |

**动作格式:**
```python
action: np.array(B, action_dim)  # float32
# 例如: [delta_joint_0, delta_joint_1, ..., delta_joint_6]
# 或带 gripper: [..., gripper_position]
```

### 图像输入

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 256x256 或 224x224 | resize 后 |
| 通道 | 3 (RGB) | uint8 [0, 255] |
| 相机数量 | 1-2 | 主相机 + 手腕相机 |
| 预处理 | /255 → [-1,1] | SigLIP 期望 |

**LIBERO 示例:**
```python
observation.images.image:      # 主相机
observation.images.wrist_image:  # 手腕相机
```

**Bridge 示例:**
```python
observation.images.image_0:  # 单相机
```

### 配置参数 (config/*.json)

```json
{
    "policy": {
        "n_obs_steps": 1,           // 观测帧数
        "max_state_dim": 32,        // 状态 padding 维度
        "max_action_dim": 32,       // 动作 padding 维度
        "n_action_steps": 50,        // 每次预测的动作步数
        "resize_imgs_with_padding": [224, 224],  // 图像 resize
        "tokenizer_max_length": 48,  // 语言 token 长度
        "proj_width": 1024,          // 投影层维度
        "num_steps": 10              // System2 去噪步数
    }
}
```

### 归一化方式

```python
normalization_mapping = {
    "VISUAL": "IDENTITY",    # 图像不归一化
    "STATE": "MEAN_STD",     # 状态均值标准差归一化
    "ACTION": "MEAN_STD"     # 动作均值标准差归一化
}
```

### 适配不同机械臂

**`adapt_to_pi_aloha`**: 适配 Aloha 机械臂 (双手机器人)
- 关节翻转补偿
- gripper 角度/线性转换

```python
# Aloha 特有的状态/动作处理
_motor_idx = [1, 2, 8, 9]  # 翻转关节
_gripper_idx = [6, 13]      # gripper 转换
```

---

# System1 (FastVisuoMatching) 模型结构

## 整体架构

```
FastVisuoMatching
├── FastVisuoExpertModel      # 视觉编码器 (DINO + Gemma Expert)
├── state_proj               # 状态投影层
├── action_in_proj           # 动作输入投影
├── action_out_proj          # 动作输出投影
└── action_time_mlp_in/out   # 动作-时间融合MLP
```

## 核心组件

| 组件 | 类型 | 说明 |
|------|------|------|
| `fast_visuo_expert` | `FastVisuoExpertModel` | 视觉编码器，基于 DINO + Gemma Expert |
| `state_proj` | `Linear(max_state_dim, s1_proj_width)` | 机器人状态投影 |
| `action_in_proj` | `Linear(max_action_dim, s1_proj_width)` | 动作嵌入投影 |
| `action_out_proj` | `Linear(s1_proj_width, max_action_dim)` | 输出动作预测 |
| `action_time_mlp_*` | MLP (2层) | 融合动作和时间步嵌入 |

## 输入处理

### embed_prefix (仅图像，无语言)

- 图像 → DINO 归一化 (`mean=0.485, std=0.229` 等)
- `fast_visuo_expert.embed_image(img)` → 视觉 token 序列

### embed_suffix (状态+动作+时间)

- `state` → `state_proj` → 状态嵌入
- `stamp` → 正弦位置编码
- `timestep` → 正弦位置编码
- `noisy_actions` → `action_in_proj` → 与时间融合 → MLP

## 与 System2 的关键区别

| 特性 | System1 | System2 |
|------|---------|---------|
| 视觉编码器 | **DINO** | SigLIP + PaliGemma |
| 语言输入 | **无** | 有 |
| 状态历史 | **有** (`s1_his_state_size`) | 无 |
| 去噪范式 | Flow Matching (线性) | EDM (欧拉) |
| 输出 | 直接预测干净动作 | 预测噪声 |

## forward 流程

```
1. x_t = t * noise + (1-t) * actions      # 线性插值
2. u_t = noise - actions                  # 目标 velocity
3. prefix_embs = embed_prefix(images)     # DINO视觉
4. suffix_embs = embed_suffix(state, x_t, time, stamp)
5. inputs_embeds = concat([prefix_embs, suffix_embs])
6. output = fast_visuo_expert(inputs_embeds)
7. v_t = action_out_proj(output[:, -s1_action_steps:])
8. loss = MSE(u_t, v_t)
```

System1 通过 DINO 视觉编码器实现快速反应，配合 Flow Matching 去噪，能在无需语言指令的情况下直接从视觉输入生成动作预测。

---

# System2 模型结构

## 整体架构

```
System2
├── PaliGemmaWithExpertModel     # 视觉-语言编码器 (SigLIP + PaliGemma + Gemma Expert)
├── state_proj                   # 状态投影层
├── action_in_proj               # 动作输入投影
├── action_out_proj              # 动作输出投影
├── action_time_mlp_in/out       # 动作-时间融合MLP
```

## 核心组件

| 组件 | 类型 | 说明 |
|------|------|------|
| `paligemma_with_expert` | `PaliGemmaWithExpertModel` | 视觉-语言编码器，SigLIP 视觉 + Gemma Expert |
| `state_proj` | `Linear(max_state_dim, proj_width)` | 机器人状态投影 |
| `action_in_proj` | `Linear(max_action_dim, proj_width)` | 动作嵌入投影 |
| `action_out_proj` | `Linear(proj_width, max_action_dim)` | 输出动作预测 |
| `action_time_mlp_*` | MLP (2层) | 融合动作嵌入与时间正弦编码 |

## 输入处理

### embed_prefix (图像 + 语言)

```
images → SigLIP 视觉编码 → PaliGemma 图像 token
lang_tokens → Gemma 嵌入层 → 语言 token
图像与语言之间全注意力连接
```

### embed_suffix (状态 + 动作 + 时间)

```
state → state_proj → 状态嵌入
timestep → 正弦位置编码 (sin-cos embedding)
noisy_actions → action_in_proj → 与 time_emb 拼接 → MLP(silu) → 融合嵌入
```

## 注意力 mask 机制

使用自定义的 2D 注意力掩码 (`make_att_2d_masks`):
- 图像 token 之间全连接
- 语言 token 之间全连接
- 图像/语言 → 状态/动作: **无注意力** (`att_masks=1`)
- 动作 token 之间: **因果掩码** (`prefixLM` 风格)

## forward 流程 (训练)

```python
# 1. Flow Matching 插值
time = sample_beta(1.5, 1.0)  # t ∈ (0,1)
x_t = t * noise + (1-t) * actions    # 观测到的带噪动作
u_t = noise - actions                 # 目标 velocity

# 2. 编码
prefix_embs = embed_prefix(images, img_masks, lang_tokens, lang_masks)  # 视觉+语言
suffix_embs = embed_suffix(state, x_t, timestep)  # 状态+动作+时间

# 3. Transformer 前向
inputs_embeds = [prefix_embs, suffix_embs]
output = paligemma_with_expert.forward(inputs_embeds)
v_t = action_out_proj(output[:, -n_action_steps:])

# 4. 损失
loss = MSE(u_t, v_t)  # 预测 velocity
```

## sample_actions 流程 (推理)

```python
# 使用欧拉法进行 ODE 求解 (EDM 范式)
x_t = noise  # 从纯噪声开始
time = 1.0
dt = -1.0 / num_steps

while time >= -dt/2 + (1 - theta2):
    v_t = denoise_step(state, prefix_kv_cache, x_t, time)
    x_t += dt * v_t * noise_temp  # 欧拉更新
    time += dt
return x_t
```

## 与 System1 的关键区别

| 特性 | System1 | System2 |
|------|---------|---------|
| **视觉编码器** | DINO | SigLIP + PaliGemma |
| **语言输入** | ❌ 无 | ✅ 有 |
| **状态历史** | ✅ 有 | ❌ 无 |
| **推理步数** | `s1_num_steps` | `num_steps` |
| **去噪范式** | 线性 Flow Matching | EDM (欧拉 ODE) |

## 协作方式

在 `HumePolicy` 中：
1. **System2** 生成多个候选动作序列 (`s2_candidates_num`)
2. **ValueQueryHead** 评估这些候选动作的 Q 值，选择最优
3. **System1** 接收 System2 的降噪动作 + 状态历史，执行细粒度去噪，输出最终动作
