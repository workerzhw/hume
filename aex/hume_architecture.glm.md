# Hume 软件架构文档

> Hume: Dual-System Vision-Language-Action Model with System-2 Thinking
> 论文: arXiv:2505.21432 | 主页: https://hume-vla.github.io/

---

## 1. 项目概览

Hume 是一个**机器人策略架构**，灵感来源于认知科学中的"双系统理论"（Daniel Kahneman）。它通过一个慢速但精确的 **System 2**（深思熟虑）和一个快速但粗糙的 **System 1**（直觉反应）协同工作，实现高效且精确的机器人动作预测。一个可学习的 **Value Query Head (VQH)** 模块负责在推理时动态选择两个系统的输出。

### 核心创新点

- **System 2** — 基于 PaliGemma（SigLIP 视觉编码器 + Gemma-2B LLM）的"慢思考"路径，通过多步扩散去噪生成高质量动作候选
- **System 1** — 基于 DINOv2-Small + 轻量 Gemma Expert 的"快反应"路径，单次前向传播即可输出动作
- **Value Query Head** — 基于 CalQL（Calibrated Q-Learning）的 RL 组件，学习何时使用 System 2 vs System 1 的输出

---

## 2. 技术栈与构建

| 类别 | 技术 |
|------|------|
| 语言 | Python >= 3.10 |
| ML 框架 | PyTorch 2.6.0, HuggingFace Transformers, HuggingFace LeRobot |
| 基础模型 | PaliGemma (SigLIP + Gemma-2B), DINOv2-Small |
| 分布式训练 | HuggingFace Accelerate (多 GPU / BF16 混合精度) |
| 包管理 | uv (astral.sh), setuptools (src layout) |
| 代码质量 | Ruff (line-length=88, target=py310), beartype + jaxtyping 运行时类型检查 |
| 通信协议 | WebSocket + msgpack (numpy 序列化) |
| 实验追踪 | WandB |

---

## 3. 目录结构

```
hume/
├── src/hume/                    # 核心 Python 包
│   ├── models/                  #   模型定义
│   │   ├── modeling_hume.py     #     HumePolicy, System2, FastVisuoMatching, ValueQueryHead
│   │   ├── configuration_hume.py#     HumeConfig, System2Config
│   │   ├── paligemma_with_expert.py # PaliGemmaWithExpertModel (SigLIP + Gemma + Expert)
│   │   ├── fast_visuo_expert.py #     FastVisuoExpertModel (DINOv2 + Gemma Expert)
│   │   └── value_query.py       #     VQHBackbone, CalQL
│   ├── training/                #   训练逻辑
│   │   ├── train_s2.py          #     Stage 1: System 2 训练
│   │   ├── train_vqh_s1.py      #     Stage 2: VQH + System 1 联合训练
│   │   ├── dataset.py           #     LeRobotDataset (数据加载与增强)
│   │   ├── transforms.py        #     图像变换管线
│   │   └── lerobot_patch.py     #     LeRobot 工厂函数猴子补丁
│   ├── serving/                 #   推理服务
│   │   └── websocket_policy_server.py # WebSocket 策略服务器
│   ├── serve_policy.py          #   服务入口脚本
│   └── array_typing.py          #   类型定义 (InferBatchObs, CalQlBatch, etc.)
├── packages/openpi-client/      # 独立客户端库
│   └── src/openpi_client/       #   WebSocket 客户端 + Runtime 框架
├── config/                      # 数据集配置 JSON
│   ├── libero.json              #   LIBERO 仿真任务
│   ├── bridge.json              #   Bridge 真实机器人数据
│   └── fractal.json             #   Fractal 数据集
├── experiments/libero/          # LIBERO 评测脚本
├── 3rd/LIBERO/                  # LIBERO benchmark (git submodule)
├── scripts/                     # Shell 脚本 (训练/服务/环境)
├── pyproject.toml               # 项目配置 (uv + setuptools)
├── requirements.txt             # 核心依赖
└── uv.lock                      # 依赖锁文件
```

---

## 4. 系统架构

### 4.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HumePolicy                                    │
│  (顶层策略包装器, 继承 PreTrainedPolicy)                              │
│                                                                      │
│  ┌─────────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │     System 2         │  │    System 1       │  │  Value Query   │ │
│  │  (慢思考路径)         │  │  (快反应路径)      │  │    Head        │ │
│  │                      │  │                   │  │  (动作选择)     │ │
│  │  ┌────────────────┐  │  │ ┌───────────────┐│  │                │ │
│  │  │PaliGemmaWith   │  │  │ │FastVisuoExpert││  │ ┌────────────┐ │ │
│  │  │ExpertModel     │  │  │ │Model          ││  │ │ VQHBackbone│ │ │
│  │  │                │  │  │ │               ││  │ │            │ │ │
│  │  │ SigLIP (视觉)  │  │  │ │ DINOv2 (视觉) ││  │ │ PaliGemma  │ │ │
│  │  │ Gemma-2B (LLM)│  │  │ │ Gemma Expert ││  │ │ 编码器(共享)│ │ │
│  │  │ Gemma Expert  │  │  │ │ (13层,512维)  ││  │ └────────────┘ │ │
│  │  │ (动作去噪)    │  │  │ │               ││  │ ┌────────────┐ │ │
│  │  └────────────────┘  │  │ └───────────────┘│  │ │   CalQL    │ │ │
│  │                      │  │                   │  │ │ (RL Critic)│ │ │
│  │  输入: 图像+语言      │  │  输入: 图像(无语言) │  │ └────────────┘ │ │
│  │  输出: 动作候选序列   │  │  输出: 精细化动作   │  │ 输出: Q值评估 │ │
│  └─────────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 推理数据流

```
                         观测输入
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────▼─────┐           ┌──────▼──────┐
        │  System 2  │           │   图像预处理  │
        │ 重新规划    │           │   归一化      │
        └─────┬─────┘           └─────────────┘
              │
     每 s2_replan_steps 步触发
              │
    ┌─────────▼──────────┐
    │ 生成 N 个候选动作    │  (s2_candidates_num 个,
    │ (不同 noise_temp)   │   EDM 欧拉 ODE 去噪)
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │ Value Query Head    │  CalQL 评估每个候选的 Q 值
    │ 选择最优候选         │  → argmax Q(s, a)
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  System 1          │  以 S2 动作 + 状态历史为条件
    │  细粒度去噪          │  Flow Matching 单步/少步去噪
    └─────────┬──────────┘
              │
        ┌─────▼─────┐
        │ 输出动作    │  从 action_queue 逐帧返回
        └───────────┘
```

### 4.3 类继承关系

```
PreTrainedPolicy (lerobot)
├── HumePolicy                    # 完整双系统策略 (hume)
│   ├── s2_model: System2         # System 2 模块
│   ├── s1_model: FastVisuoMatching  # System 1 模块
│   └── value_query_head: ValueQueryHead  # VQH 模块
│
└── System2Policy                 # 独立 System 2 策略 (system2)
    └── model: System2

nn.Module
├── System2                       # System 2 核心网络
│   └── paligemma_with_expert: PaliGemmaWithExpertModel
├── FastVisuoMatching             # System 1 核心网络
│   └── fast_visuo_expert: FastVisuoExpertModel
└── ValueQueryHead                # VQH 核心网络
    ├── paligemma_with_expert     # (与 System2 共享)
    ├── vqh_backbone: VQHBackbone
    └── calql: CalQL
        ├── policy (Actor)
        ├── critics (双 Q 网络 + Target 网络)
        └── temperature (可学习温度参数)
```

---

## 5. 核心模块详解

### 5.1 System 2 — 慢思考路径

**文件:** [modeling_hume.py:1009](../src/hume/models/modeling_hume.py#L1009)
**配置:** [configuration_hume.py](../src/hume/models/configuration_hume.py)

System 2 是一个基于 Flow Matching 的扩散模型，使用 PaliGemma 作为骨干网络：

| 组件 | 说明 |
|------|------|
| `PaliGemmaWithExpertModel` | SigLIP 视觉编码器 + Gemma-2B LLM + Gemma Expert head |
| `state_proj` | `Linear(max_state_dim → proj_width=1024)` |
| `action_in_proj` | `Linear(max_action_dim → proj_width)` |
| `action_out_proj` | `Linear(proj_width → max_action_dim)` |
| `action_time_mlp` | 2 层 MLP 融合动作嵌入与时间步编码 |

**训练流程 (Flow Matching):**
1. 采样时间步 `t ~ Beta(1.5, 1.0)`, 确保 `t ∈ (0.001, 1.0)`
2. 插值: `x_t = t * noise + (1 - t) * actions`
3. 目标: `u_t = noise - actions` (velocity 目标)
4. 编码前缀: 图像 → SigLIP + 语言 → Gemma Embedding
5. 编码后缀: 状态投影 + 噪声动作 + 时间步编码
6. Transformer 前向传播 → 预测 `v_t`
7. 损失: `MSE(u_t, v_t)`

**推理流程 (EDM 欧拉 ODE 去噪):**
```python
x_t = noise  # 从纯噪声开始
for step in range(num_steps):  # 默认 10 步
    v_t = model(state, prefix_cache, x_t, time)
    x_t += dt * v_t * noise_temp
    time += dt
```

**注意力机制:**
- 图像 token 之间: 全连接注意力
- 语言 token 之间: 全连接注意力
- 图像/语言 → 状态/动作: 无注意力
- 动作 token 之间: 因果掩码 (Prefix-LM 风格)

### 5.2 System 1 — 快反应路径

**文件:** [modeling_hume.py:1331](../src/hume/models/modeling_hume.py#L1331)

System 1 是一个轻量化的视觉-动作模型，使用 DINOv2 替代 SigLIP/PaliGemma：

| 组件 | 说明 |
|------|------|
| `FastVisuoExpertModel` | DINOv2-Small 视觉编码器 + 13 层 Gemma Expert (512 维) |
| `state_proj` | `Linear(max_state_dim → s1_proj_width)` |
| `action_in_proj` | `Linear(max_action_dim → s1_proj_width)` |
| `action_out_proj` | `Linear(s1_proj_width → max_action_dim)` |

**与 System 2 的关键区别:**

| 特性 | System 1 | System 2 |
|------|---------|---------|
| 视觉编码器 | DINOv2-Small | SigLIP + PaliGemma |
| 语言输入 | 无 | 有 |
| 状态历史 | 有 (`s1_his_state_size` 帧) | 无 |
| 去噪范式 | 线性 Flow Matching | EDM (欧拉 ODE) |
| 推理速度 | 快 (单步或少步) | 慢 (10 步去噪) |
| 输出语义 | 直接预测干净动作 | 预测噪声 velocity |

### 5.3 Value Query Head (VQH)

**文件:** [modeling_hume.py:1620](../src/hume/models/modeling_hume.py#L1620), [value_query.py](../src/hume/models/value_query.py)

VQH 是一个基于 RL 的决策模块，负责评估和选择 System 2 生成的候选动作：

```
ValueQueryHead
├── paligemma_with_expert    # 与 System2 共享的视觉-语言编码器
├── vqh_backbone             # VQH 专用 backbone (Gemma Expert)
├── calql                    # CalQL 强化学习组件
│   ├── policy               # Actor 网络 (策略)
│   ├── critics              # 双 Q 网络 (online + target)
│   └── temperature          # 可学习温度参数 (熵正则化)
└── query_embedding          # 可学习 query token (用于从 PaliGemma 提取特征)
```

**工作原理:**
1. 将候选动作与图像/语言一起编码
2. 通过 `query_embedding` 从 PaliGemma 提取观测特征
3. `VQHBackbone` 进一步编码观测-动作对
4. `CalQL` 的 critic 网络评估每个候选动作的 Q 值
5. 选择 Q 值最高的候选动作

**CalQL 训练损失:**
- `critic_loss`: Bellman 误差 + CQL (Conservative Q-Learning) 正则化
- `actor_loss`: 最大化 Q 值 + 熵正则化
- `temperature_loss`: 自动调节温度参数以维持目标熵
- `calql_bound_rate`: Q 值约束满足率 (确保 Q 值不超过蒙特卡洛回报)

---

## 6. 训练流程

### 6.1 两阶段训练

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: System 2 预训练                                     │
│  入口: scripts/train_s2.sh → train_s2.py                     │
│  GPU: 2 卡 (默认) | Batch: 64 | 精度: fp32/bf16              │
│  优化器: AdamW + CosineDecayWithWarmup                        │
│  输出: System2 checkpoint                                     │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: VQH + System 1 联合训练                             │
│  入口: scripts/train_vqh_s1.sh → train_vqh_s1.py             │
│  GPU: 8 卡 (默认) | Batch: 128 | 精度: bf16                  │
│  优化器: 4 个独立优化器                                        │
│  输出: 完整 HumePolicy checkpoint                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Stage 2 多优化器架构

Stage 2 使用 4 个独立的优化器分别管理不同组件：

| 优化器 | 管理参数 | 学习率配置 |
|--------|---------|-----------|
| `trunk_optimizer` | FastVisuoExpert + state_proj + action 投影层 | `actor_lr` |
| `actor_optimizer` | CalQL policy network | `actor_lr` |
| `critic_optimizer` | CalQL Q-network (双网络 + target) | `critic_lr` |
| `temperature_optimizer` | CalQL temperature | `temp_lr` |

每 `target_critic_update_period` 步执行一次 target network 软更新:
```python
soft_update(target_critics, critics, tau=soft_target_critic_update_rate)
```

### 6.3 数据格式

LeRobot 格式数据集，每个 batch 包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `observation.images.*` | `B × C × H × W` | RGB 图像 (1-2 相机) |
| `observation.state` | `B × state_dim` | 机器人关节状态 |
| `task` | `List[str]` | 语言指令 |
| `action` | `B × chunk_size × action_dim` | 动作序列 (默认 50 步) |
| `stamp` | `B` | 时间戳 (归一化到 0-1) |
| `reward.vqh` | `B` | VQH 奖励信号 |
| `mc.vqh` | `B` | 蒙特卡洛回报估计 |

### 6.4 训练监控指标

| 指标 | 说明 |
|------|------|
| `s1_loss` | System 1 Flow Matching 损失 |
| `s2_loss` | System 2 Flow Matching 损失 |
| `actor_loss` | CalQL policy 损失 |
| `critic_loss` | CalQL Q-network 损失 |
| `cql_loss` | Conservative Q-Learning 正则化损失 |
| `temperature` | 熵正则化温度参数值 |
| `entropy` | 动作分布熵 |
| `online_q` / `target_q` | Online / Target Q 值 |
| `calql_bound_rate` | Q 值约束满足率 |

---

## 7. 推理与部署

### 7.1 C/S 架构

```
┌──────────────────────┐     WebSocket      ┌──────────────────────┐
│  机器人环境            │ ◄──────────────► │  Policy Server        │
│  (LIBERO / 真实机械臂) │   msgpack 序列化  │  (WebsocketPolicyServer)│
│                      │                   │  加载 HumePolicy       │
│  WebsocketClientPolicy│                   │  GPU 推理              │
└──────────────────────┘                   └──────────────────────┘
```

### 7.2 服务端

**入口:** [serve_policy.py](../src/hume/serve_policy.py)
**实现:** [websocket_policy_server.py](../src/hume/serving/websocket_policy_server.py)

```bash
# 启动服务
python src/hume/serve_policy.py --ckpt_path <path> --port 8000
```

服务流程:
1. `HumePolicy.from_pretrained(ckpt_path)` 加载模型
2. `WebsocketPolicyServer` 启动异步 WebSocket 服务
3. 接收 `load` / `infer` 请求 (msgpack 反序列化)
4. 调用 `policy.infer(observation)` 执行推理
5. 返回动作 (msgpack 序列化)

### 7.3 客户端

**包:** [packages/openpi-client/](packages/openpi-client/)

| 组件 | 说明 |
|------|------|
| `WebsocketClientPolicy` | 同步 WebSocket 客户端，发送观测接收动作 |
| `ActionChunkBroker` | 包装 policy，逐帧返回 action chunk 中的动作 |
| `PolicyAgent` | 实现 `Agent` 接口，将 policy 集成到 runtime loop |
| `Runtime` | 驱动 Agent-Environment 交互循环 |
| `image_tools` | 图像 resize/格式转换工具 |

### 7.4 推理配置参数

```python
infer_config = {
    "s2_replan_steps": 50,      # System 2 重规划间隔
    "s2_candidates_num": 4,     # 候选动作数量
    "s2_num_steps": 10,         # System 2 去噪步数
    "s1_num_steps": 1,          # System 1 去噪步数
    "s1_his_state_size": 5,     # System 1 状态历史长度
    "noise_temp": 1.0,          # 噪声温度
}
```

---

## 8. 配置系统

### 8.1 配置层次

```
pyproject.toml           # 项目级配置 (依赖、构建)
    ↓
config/*.json            # 数据集级配置 (图像增强、训练参数)
    ↓
HumeConfig / System2Config  # 模型级配置 (网络结构、优化器)
    ↓
InferConfig              # 推理级配置 (去噪参数、候选数量)
```

### 8.2 数据集配置 (config/*.json)

每个数据集配置包含:

| 配置组 | 说明 |
|--------|------|
| `policy` | 策略参数 (chunk_size, 图像尺寸, 投影维度, 去噪步数) |
| `policy.image_transforms` | 图像增强 (亮度/对比度/饱和度/色调/锐度/裁剪/旋转) |
| `dataset` | 数据集路径、episode 范围 |
| `training` | 训练超参数 (batch_size, 学习率, 调度器) |

### 8.3 机器人参数

| 数据集 | state_dim | action_dim | 图像尺寸 | 相机数 |
|--------|-----------|------------|---------|-------|
| LIBERO | 8 | 7 | 224×224 | 1-2 |
| Bridge | 8 | 7 | 224×224 | 1 |
| Fractal | 32 | 32 | 224×224 | 1 |

**归一化方式:**
- 图像: IDENTITY (仅 /255 → [-1, 1])
- 状态: MEAN_STD (均值标准差归一化)
- 动作: MEAN_STD (均值标准差归一化)

---

## 9. 关键设计决策

### 9.1 参数共享

`ValueQueryHead` 与 `System2` 共享 `paligemma_with_expert` 实例。这避免了重复加载大模型，但意味着 VQH 训练时梯度可能影响 System 2 的视觉编码器。

### 9.2 多步去噪 vs 单步

System 2 默认使用 10 步 EDM 欧拉 ODE 去噪，System 1 使用 1 步 Flow Matching。这体现了"慢思考 vs 快反应"的设计理念。推理时可通过 `s2_num_steps` 参数控制 Test-Time Scaling (TTS)。

### 9.3 候选采样 + Q 值选择

推理时 System 2 生成多个候选动作（通过不同的 `noise_temp`），VQH 评估 Q 值后选择最优。这是一种 **Best-of-N 采样 + RL Critic 过滤** 的策略。

### 9.4 Action Queue 机制

策略一次预测 `chunk_size` 步动作，存入 `action_queue`，逐帧执行。只有当 queue 为空或达到重规划间隔时才重新调用模型，大幅降低推理频率。

### 9.5 猴子补丁 (lerobot_patch.py)

项目通过猴子补丁替换了 LeRobot 的 `make_policy`、`make_dataset`、`make_optimizer_and_scheduler` 等工厂函数，以支持自定义的多优化器训练逻辑。

---

## 10. 依赖关系图

```
HumePolicy
├── System2
│   └── PaliGemmaWithExpertModel
│       ├── SigLIPModel (transformers)
│       ├── GemmaForCausalLM (transformers)
│       └── GemmaExpert (自定义)
├── FastVisuoMatching
│   └── FastVisuoExpertModel
│       ├── DINOv2 (torchvision / facebookresearch)
│       └── GemmaExpert (自定义, 13层 512维)
└── ValueQueryHead
    ├── PaliGemmaWithExpertModel (与 System2 共享)
    ├── VQHBackbone
    │   └── GemmaExpert (自定义)
    └── CalQL
        ├── DoubleCritic (2 个 Q 网络 + 2 个 Target Q 网络)
        ├── GaussianActor (高斯策略网络)
        └── Temperature (可学习标量)
```

---

## 11. 硬件需求

### 训练

| 阶段 | GPU 数 | 每 GPU Batch | 总 Batch | 精度 |
|------|--------|-------------|---------|------|
| System 2 训练 | 2 | 32 | 64 | fp32/bf16 |
| VQH+S1 训练 | 8 | 16 | 128 | bf16 |
| Debug 模式 | 1 | 8 | 8 | bf16 |

### 推理

- CUDA GPU, 建议 16GB+ 显存
- 支持 BF16 加速
- 不支持 CPU 推理

---

## 12. 入口命令

```bash
# 环境配置
source scripts/env.sh

# Stage 1: 训练 System 2
bash scripts/train_s2.sh

# Stage 2: 训练 VQH + System 1
bash scripts/train_vqh_s1.sh

# 启动推理服务
bash scripts/serve_policy.sh

# LIBERO 评测
bash experiments/libero/scripts/eval_libero.sh
```
