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

### 4.2 推理数据流概览

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

## 5. 推理数据流详解

本节从源码级别逐步描述推理时数据如何在 System 2、VQH、System 1 之间流转。

入口方法: `HumePolicy.infer()` → `HumePolicy.select_action()` → 各子系统

### 5.1 阶段 0: 观测预处理 (`HumePolicy.infer`)

**源码:** [modeling_hume.py:282-369](../src/hume/models/modeling_hume.py#L282-L369)

每收到一帧环境观测，`infer()` 执行以下操作：

```python
def infer(observation):
    # observation = {
    #   "observation.images.image":      np.array (B, H, W, C), uint8 [0,255]
    #   "observation.images.wrist_image": np.array (B, H, W, C), uint8 [0,255]  (可选)
    #   "observation.state":              np.array (B, state_dim), float32
    #   "task":                           List[str]
    # }
```

**Step 0a — 状态历史维护**

```python
if not self.history_state:
    # 首次调用: 用当前状态填满整个历史窗口
    history_state.extend(
        current_state.repeat(s1_his_state_size)   # (s1_his_state_size, state_dim)
    )
else:
    # 后续调用: 追加最新状态, 自动淘汰最旧状态 (deque maxlen)
    history_state.append(current_state)
```

- `history_state` 是一个 `deque(maxlen=s1_his_state_size)`，默认保留最近 5 帧状态
- 拼接后 `observation.state` 形状变为 `(B, s1_his_state_size, state_dim)`

**Step 0b — 图像归一化**

```python
# uint8 [0,255] → float32 [-1,1]
images = torch.tensor(img / 255).permute(0,3,1,2)   # (B, C, H, W)
```

**Step 0c — 触发条件判断**

```python
if not self.action_plan:                              # action_plan 为空 = 上一轮 chunk 执行完毕
    if infer_step % s2_replan_steps == 0:             # 达到 S2 重规划间隔
        outputs = {}                                  # 清空缓存 → 触发完整的 S2+VQH+S1 管线
    # 否则 outputs 保留上一轮的 S2 缓存 → 仅重新执行 S1
```

**Step 0d — 计算 stamp (时间位置)**

```python
stamp = (infer_step % s2_replan_steps) / s2_chunk_size
# stamp 表示当前时刻在 S2 长动作序列中的相对位置, 范围 [0, 1)
```

**Step 0e — 进入 select_action → 管线核心**

如果 `action_plan` 不为空，直接 `popleft()` 返回缓存的下一步动作，不进入模型。

---

### 5.2 阶段 1: System 2 候选动作生成 (`System2.sample_actions`)

**源码:** [modeling_hume.py:1226-1287](../src/hume/models/modeling_hume.py#L1226-L1287)

当 `outputs` 为空 (`"noise_action" not in outputs`) 时，调用 System 2 生成候选。

**调用链:**

```
HumePolicy.select_action()
  ├── normalize_inputs(batch)           # 状态/动作归一化 (MEAN_STD)
  ├── prepare_images(batch)             # 图像 resize + padding → list[Tensor]
  ├── prepare_state(batch)              # 状态 padding → (B, state_dim)
  ├── prepare_language(batch)           # 语言 tokenize → (B, seq_len)
  │
  └── 循环 i = 0..s2_candidates_num-1:
      └── self.s2_model.sample_actions(
              images, img_masks, lang_tokens, lang_masks,
              state[:, -1, :],          # ← 仅取最新一帧状态 (S2 不支持历史)
              time_temp, noise_temp      # ← 每个候选使用不同温度
          )
```

**每个候选的温度参数:**

```python
for i in range(s2_candidates_num):
    time_temp  = (i / s2_candidates_num) * (upper - lower) + lower
    noise_temp = (i / s2_candidates_num) * (upper - lower) + lower
```

- 候选 0: `time_temp=lower, noise_temp=lower` (低扰动, 更确定)
- 候选 N-1: `time_temp=upper, noise_temp=upper` (高扰动, 更随机)
- 这创造了多样性：从保守到激进的不同动作策略

**System2.sample_actions 内部流程:**

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 编码前缀 (图像 + 语言) — 只执行一次, 缓存 KV cache  │
│                                                              │
│   images ──→ SigLIP 视觉编码器 ──→ 视觉 token 序列           │
│              (img * 0.5 + 0.5 → SigLIP → √d 归一化)         │
│                                                              │
│   lang_tokens ──→ Gemma Embedding ──→ 语言 token 序列        │
│                   (√d 归一化)                                │
│                                                              │
│   concat([img_tokens, lang_tokens])                          │
│     │                                                        │
│     ├── 注意力掩码:                                           │
│     │   img_tokens 之间: 全连接 (att_mask=0)                  │
│     │   lang_tokens 之间: 全连接 (att_mask=0)                 │
│     │                                                        │
│     └── PaliGemma forward (prefix_only)                      │
│         → 缓存 KV cache (图像+语言的 key/value)               │
│         → 后续去噪步可复用, 无需重复编码                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: 初始化纯噪声                                        │
│                                                              │
│   x_t = N(0, 1), shape = (B, s2_chunk_size, max_action_dim) │
│   time = time_temp (标量, 从 time_temp 开始)                  │
│   dt = -1.0 / num_steps                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: EDM 欧拉 ODE 去噪循环 (num_steps 次, 默认 10)       │
│                                                              │
│   while time >= -dt/2 + (1 - theta2):                        │
│     │                                                        │
│     ├── 编码后缀:                                             │
│     │   state[:, -1, :] → state_proj → (B, 1, proj_width)   │
│     │   x_t          → action_in_proj                        │
│     │   timestep     → 正弦位置编码 → time_emb               │
│     │   concat([action_emb, time_emb]) → MLP → 动作-时间嵌入  │
│     │   concat([state_emb, action_time_emb])                  │
│     │                                                        │
│     ├── 注意力掩码:                                           │
│     │   state tokens:  att_mask=1 (不回看 prefix)            │
│     │   action tokens: 因果掩码 (第一个=1, 其余=0)            │
│     │   → prefix token 不 attend to suffix token             │
│     │   → action token 之间是因果的 (前面的看不了后面)         │
│     │                                                        │
│     ├── PaliGemma forward (suffix only, 复用 prefix KV cache) │
│     │   → 取最后 n_action_steps 个 token 的输出               │
│     │   → action_out_proj → v_t (预测的 velocity)            │
│     │                                                        │
│     └── 欧拉步更新:                                           │
│         x_t += dt * v_t * noise_temp                         │
│         time += dt                                           │
│                                                              │
│   return x_t  # shape: (B, s2_chunk_size, max_action_dim)   │
└─────────────────────────────────────────────────────────────┘
```

**关键细节:**
- **KV Cache 复用**: 图像+语言编码只执行一次，后续 num_steps 步去噪都复用 prefix KV cache，大幅节省计算
- **注意力隔离**: Prefix (图像+语言) 不 attend to Suffix (状态+动作)，但 Suffix 可以 attend to Prefix
- **noise_temp 控制多样性**: 乘在欧拉步的 `v_t` 上，温度越高，去噪轨迹越偏离"最可能"路径

**候选动作堆叠:**

```python
noise_actions = torch.stack([action_0, action_1, ..., action_N-1], dim=1)
# shape: (B, s2_candidates_num, s2_chunk_size, max_action_dim)
```

---

### 5.3 阶段 2: VQH 候选评估与选择 (`ValueQueryHead.select_q_actions`)

**源码:** [modeling_hume.py:1838-1878](../src/hume/models/modeling_hume.py#L1838-L1878), [value_query.py:1141-1151](../src/hume/models/value_query.py#L1141-L1151)

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 截取 VQH 评估范围                                    │
│                                                              │
│   noise_actions_wo_pad = noise_actions[                      │
│       :, :, :vqh_chunk_size, :original_action_dim            │
│   ]                                                          │
│   # 只取前 vqh_chunk_size 步 (≤ s2_chunk_size)               │
│   # 去掉 padding 维度, 只保留真实 action_dim                  │
│   # shape: (B, s2_candidates_num, vqh_chunk_size, act_dim)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: 编码观测 (图像+语言 → query_embedding)               │
│                                                              │
│   VQH.embed_prefix(images, img_masks, lang_tokens, lang_masks) │
│     │                                                        │
│     ├── 图像 → SigLIP 视觉编码 → 视觉 token                  │
│     │        (与 S2 共享 paligemma_with_expert)               │
│     │                                                        │
│     ├── 语言 → Gemma Embedding → 语言 token                  │
│     │        (语言嵌入被 .detach() — VQH训练时梯度不回传LLM)  │
│     │                                                        │
│     ├── 全连接注意力: img_tokens ↔ lang_tokens               │
│     │                                                        │
│     └── 插入可学习 query_embedding:                           │
│         在序列的有效长度位置插入一个 learnable token            │
│         query_embedding: Parameter(hidden_size, bf16)        │
│         该 token 可以 attend to 所有 img+lang token           │
│                                                              │
│   VQHBackbone.forward(embs, pad_masks, att_masks)            │
│     │                                                        │
│     └── Gemma Expert 处理 → 取 query_embedding 位置处的输出   │
│         → query_embedding_output: (B, hidden_size)           │
│         这就是观测的紧凑表征                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: CalQL Critic 评估 Q 值                               │
│                                                              │
│   # 展平候选动作:                                             │
│   noise_actions = noise_actions.reshape(B, N, -1)            │
│   # (B, N, vqh_chunk_size * action_dim)                      │
│                                                              │
│   CalQL.get_q_values(query_embedding, noise_actions)          │
│     │                                                        │
│     ├── target_critics.forward(obs, actions)                  │
│     │   # 双 Q 网络 ensemble, 每个独立评估                    │
│     │   # Q_i(s, a_j) for i in {1,2}, j in {0..N-1}        │
│     │   # 使用 target_critics (更稳定)                        │
│     │                                                        │
│     └── q_values = min(Q_1, Q_2)  取 ensemble 最小值         │
│         # shape: (B, N) — 每个 batch 中 N 个候选的 Q 值      │
│                                                              │
│   action_index = argmax(q_values, dim=1)                     │
│   # shape: (B,) — 选择 Q 值最高的候选索引                    │
│   # (实际上 B=1 推理时, 就是选最好的那一个)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 4: 选择最优候选                                         │
│                                                              │
│   selected_noise_action = noise_actions[batch_idx, action_index] │
│   # shape: (B, s2_chunk_size, max_action_dim)               │
│                                                              │
│   outputs = {"noise_action": selected_noise_action}           │
│   q_value_cache.append(q_values)     # 记录用于日志/调试      │
│   action_cache.append(unnormalized_actions)                   │
└─────────────────────────────────────────────────────────────┘
```

**关键细节:**
- **共享编码器**: VQH 与 S2 共享 `paligemma_with_expert`，图像只编码一次（但由于 S2 的 KV cache 是独立的，VQH 需要重新编码前缀）
- **语言嵌入 detach**: `lang_emb = paligemma_with_expert.embed_language_tokens(lang_tokens).detach()` — VQH 的梯度不会反向传播到语言模型
- **Query Embedding 机制**: 类似 Transformer Decoder 的 query token，通过注意力从图像+语言上下文中"提取"与决策相关的信息到一个固定维度的向量
- **Target Critic**: 使用 target 网络（EMA 平滑版本）而非 online critic，减少 Q 值估计的方差
- **Double Critic**: 取两个 Q 网络的最小值 `min(Q_1, Q_2)`，这是 TD3/CQL 的标准做法，防止 Q 值过估计

---

### 5.4 阶段 3: System 1 细粒度去噪 (`FastVisuoMatching.sample_actions`)

**源码:** [modeling_hume.py:1543-1617](../src/hume/models/modeling_hume.py#L1543-L1617)

System 1 以 System 2 选出的动作作为"初始引导"，进行细粒度的 Flow Matching 去噪。

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 从 S2 动作中滑窗截取子序列                           │
│                                                              │
│   # stamp 指示当前在 S2 序列中的位置                          │
│   idcs = (stamp * s2_chunk_size).long() + arange(s1_chunk_size)│
│   noise_action_slides = selected_noise_action[batch_idcs, idcs]│
│                                                              │
│   # 例: s2_chunk_size=50, s1_chunk_size=10, stamp=0.3       │
│   #   → idcs = [15, 16, 17, ..., 24]                        │
│   #   → 截取 S2 动作序列的第 15-24 步作为 S1 的引导信号       │
│                                                              │
│   # shape: (B, s1_chunk_size, max_action_dim)                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: S1 前缀编码 (仅图像, 无语言)                         │
│                                                              │
│   FastVisuoMatching.embed_prefix(images, img_masks)           │
│     │                                                        │
│     ├── 图像预处理:                                           │
│     │   img = TF.normalize(img * 0.5 + 0.5,                  │
│     │                       mean=[0.485, 0.456, 0.406],      │
│     │                       std=[0.229, 0.224, 0.225])       │
│     │   # [-1,1] → [0,1] → ImageNet 归一化 (DINOv2 标准)    │
│     │                                                        │
│     ├── DINOv2-Small 视觉编码:                                │
│     │   img → DINOv2 → 视觉 token 序列                       │
│     │   → √d 归一化                                          │
│     │                                                        │
│     └── 注意力掩码: 全连接 (att_mask=0)                      │
│                                                              │
│   注: 无 KV cache 机制, 每次去噪步都重新编码前缀              │
│       (DINOv2 + 小 Expert 推理快, 无需缓存优化)              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: 初始化噪声                                           │
│                                                              │
│   x_t = N(0, 1), shape = (B, s1_action_steps, max_action_dim)│
│   time = theta1  (S1 的时间上界, < 1.0)                      │
│   dt = -theta1 / s1_num_steps                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 4: Flow Matching 去噪循环 (s1_num_steps 次, 默认 1)    │
│                                                              │
│   while time >= -dt/2:                                       │
│     │                                                        │
│     ├── 编码后缀:                                             │
│     │   state → state_proj → 状态嵌入 (B, his, proj_width)   │
│     │   stamp → 正弦位置编码 → stamp_emb (B, 1, proj_width)  │
│     │   x_t   → action_in_proj                               │
│     │   concat([action_emb, time_emb]) → MLP(SiLU) → 融合嵌入│
│     │                                                        │
│     │   最终后缀 = [state_emb, stamp_emb, action_time_emb]   │
│     │                                                        │
│     ├── 注意力掩码:                                           │
│     │   state tokens:  att_mask=1 (不回看 prefix)            │
│     │   stamp token:   att_mask=1                            │
│     │   action tokens: 因果掩码 (第一个=1, 其余=0)            │
│     │                                                        │
│     ├── FastVisuoExpert forward:                              │
│     │   concat([prefix_embs, suffix_embs])                   │
│     │     │                                                  │
│     │     └── DINOv2 特征 + Gemma Expert (13层, 512维)       │
│     │         → 取最后 s1_action_steps 个 token              │
│     │         → action_out_proj → v_t (velocity 预测)        │
│     │                                                        │
│     └── 欧拉步更新:                                           │
│         x_t += dt * v_t    (注意: S1 没有 noise_temp 缩放)  │
│         time += dt                                           │
│                                                              │
│   return x_t  # shape: (B, s1_action_steps, max_action_dim) │
└─────────────────────────────────────────────────────────────┘
```

**关键细节:**
- **S2 动作作为引导 vs 初始噪声**: S1 的 `sample_actions` 接口虽然使用独立的随机噪声 `x_t` 作为起点，但在训练时 S2 动作通过 `noise_action_slides` 提供条件信息（训练的 forward 中直接使用 S2 动作作为目标）。S1 学习的是"在给定 S2 计划的局部片段下，如何精细调整动作"
- **stamp 机制**: stamp 告诉 S1 "我们现在在执行 S2 长序列的哪个位置"，使 S1 能产生与时间位置匹配的精细动作
- **状态历史**: S1 接收 `s1_his_state_size` 帧历史状态，提供更丰富的时序上下文
- **无语言输入**: S1 不接收语言指令，完全依赖视觉和 S2 动作引导
- **默认 1 步去噪**: `s1_num_steps=1` 使得 S1 推理只需一次前向传播，速度极快
- **不同的视觉预处理**: S1 使用 ImageNet 归一化（DINOv2 标准），S2 使用 SigLIP 标准的 [-1,1] 输入

---

### 5.5 阶段 4: 后处理与动作执行 (`HumePolicy.infer` 后半段)

**源码:** [modeling_hume.py:355-369](../src/hume/models/modeling_hume.py#L355-L369)

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 去归一化                                             │
│                                                              │
│   s1_action = outputs["s1_action"]  # (B, s1_chunk_size, act_dim)│
│   actions = unnormalize_outputs({"action": s1_action})        │
│   # MEAN_STD 反归一化 → 还原到原始动作空间                   │
│   # 去掉 padding: actions[:, :, :original_action_dim]        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: 可选后处理                                           │
│                                                              │
│   if post_process_action:                                    │
│       action_chunk[..., -1] = 2 * (1 - action_chunk[..., -1]) - 1│
│       # 对 gripper 维度做特殊变换                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: 填充 action_plan                                    │
│                                                              │
│   # 转置: (s1_chunk_size, B, action_dim)                     │
│   action_chunk = action_chunk.transpose(1, 0, 2)             │
│                                                              │
│   # 只取前 replan_steps 步 (≤ s1_chunk_size)                 │
│   action_plan.extend(action_chunk[:replan_steps])             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 4: 逐帧返回动作                                        │
│                                                              │
│   infer_step += 1                                            │
│   action = action_plan.popleft()                             │
│   return np.asarray(action)  # (B, action_dim)               │
└─────────────────────────────────────────────────────────────┘
```

---

### 5.6 推理时序总结

以下展示一个完整的推理周期 (`s2_replan_steps=50`, `s1_chunk_size=10`, `replan_steps=5`):

```
时间步:   0   1   2   3   4   5   6   7   8   9  10  ...  49  50  51 ...
          │                                                               │
          ├─ S2: 生成 4 个候选 (10步去噪)                                  │
          ├─ VQH: 评估 Q 值, 选择最优候选                                  │
          ├─ S1: 细粒度去噪 → 10 步动作 (stamp=0.0)                       │
          │  plan: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]              │
          │                                                               │
          ├─ 执行 a0                                                      │
          ├─ 执行 a1                                                      │
          ├─ 执行 a2                                                      │
          ├─ 执行 a3                                                      │
          ├─ 执行 a4    ← replan_steps=5 达到                              │
          │                                                               │
          ├─ S1: 细粒度去噪 → 10 步动作 (stamp=0.1)                       │
          │  (S2缓存复用, 跳过 S2+VQH)                                    │
          │  plan: [a5', a6', a7', a8', a9', a10', a11', a12', a13', a14']│
          │                                                               │
          ├─ 执行 a5'                                                     │
          ├─ 执行 a6'                                                     │
          ├─ 执行 a7'                                                     │
          ├─ 执行 a8'                                                     │
          ├─ 执行 a9'    ← replan_steps=5 达到                              │
          │                                                               │
          ├─ S1: 细粒度去噪 → 10 步动作 (stamp=0.2)                       │
          │  ...                                                          │
          │                                                               │
          │  (每 5 步重复 S1, stamp 递增 0.1)                              │
          │                                                               │
          ├─ 步骤 45-49: S1 (stamp=0.9)                                   │
          │                                                               │
          ├─ 步骤 50: infer_step=50, 50%50==0 → outputs={}                 │
          │  ← 完整 S2+VQH+S1 管线重新触发!                               │
          │  ...下一轮循环                                                 │
```

**计算量分布:**
- **步骤 0, 50, 100, ...**: 完整管线 (S2 × N候选 × num_steps步去噪 + VQH + S1 × 1步) — 昂贵
- **步骤 5, 10, 15, ...**: 仅 S1 (DINOv2 + 小Expert × 1步去噪) — 便宜
- **步骤 1-4, 6-9, ...**: 仅 popleft() — 几乎零开销

---

### 5.7 张量形状流转总结

```
观测输入:
  image:            np.uint8 (B, H, W, C) [0, 255]
  state:            np.float32 (B, state_dim)
  task:             List[str]

┌─── normalize_inputs ───┐
  image:            Tensor (B, C, 224, 224) [-1, 1]
  state:            Tensor (B, state_dim) (mean/std normalized)

┌─── prepare_images ───┐
  images:           list[Tensor]  each (B, C, 224, 224)
  img_masks:        list[Tensor]  each (B,) bool

┌─── prepare_state ───┐
  state:            Tensor (B, state_dim) → padded (B, max_state_dim)

┌─── prepare_language ───┐
  lang_tokens:      Tensor (B, tokenizer_max_length) int64
  lang_masks:       Tensor (B, tokenizer_max_length) bool

┌─── System2.sample_actions (×N candidates) ───┐
  输入: images, img_masks, lang_tokens, lang_masks, state[:, -1, :]
  内部:
    SigLIP:          (B, C, 224, 224) → (B, num_img_tokens, hidden_size)
    Lang Embed:      (B, seq_len) → (B, seq_len, hidden_size)
    prefix_embs:     (B, num_img_tokens + seq_len, hidden_size) bf16
    KV cache:        缓存 prefix 的 key/value
    ---
    state_proj:      (B, max_state_dim) → (B, 1, proj_width)
    action_in_proj:  (B, s2_chunk_size, max_action_dim) → (B, s2_chunk_size, proj_width)
    time_emb:        (B,) → sinusoidal → (B, 1, proj_width)
    action+time MLP: concat→Linear→SiLU→Linear → (B, s2_chunk_size, proj_width)
    suffix_embs:     (B, 1 + s2_chunk_size, proj_width)
    ---
    去噪循环 (10步):  suffix → PaliGemma(复用KV) → action_out_proj → v_t
    output:          (B, s2_chunk_size, max_action_dim)

  堆叠:              (B, N, s2_chunk_size, max_action_dim)

┌─── VQH.select_q_actions ───┐
  截取:              (B, N, vqh_chunk_size, original_action_dim)
  VQH embed_prefix:  图像+语言 → SigLIP+Gemma → 插入 query_embedding
  VQHBackbone:       (B, seq_len+1, hidden_size) → query位置输出 → (B, hidden_size)
  展平动作:           (B, N, vqh_chunk_size * action_dim)
  Target Critics:    (B, hidden_size) + (B, N, flat_dim) → Q_i → (B, N)
  取 min:            (B, N)
  argmax:            (B,) → selected index
  选取:              (B, s2_chunk_size, max_action_dim)

┌─── 滑窗截取 ───┐
  stamp → idcs:      (B, s1_chunk_size)
  noise_action_slides: (B, s1_chunk_size, max_action_dim)

┌─── System1.sample_actions ───┐
  输入: images, img_masks, state (含历史), noise_action_slides, stamp
  内部:
    DINOv2:          (B, C, 224, 224) → ImageNet normalize → (B, num_tokens, dino_dim)
    prefix_embs:     (B, num_tokens, dino_dim) bf16
    ---
    state_proj:      (B, s1_his_state_size, max_state_dim) → (B, his, s1_proj_width)
    stamp_emb:       (B,) → sinusoidal → (B, 1, s1_proj_width)
    action_in_proj:  (B, s1_chunk_size, max_action_dim) → (B, s1_chunk_size, s1_proj_width)
    time_emb:        (B,) → sinusoidal → (B, 1, s1_proj_width)
    action+time MLP: → (B, s1_chunk_size, s1_proj_width)
    suffix_embs:     (B, his + 1 + s1_chunk_size, s1_proj_width)
    ---
    去噪循环 (1步):  concat([prefix, suffix]) → FastVisuoExpert → action_out_proj → v_t
    output:          (B, s1_chunk_size, max_action_dim)

┌─── unnormalize_outputs ───┐
  actions:          (B, s1_chunk_size, original_action_dim) (原始尺度)
  截取 replan_steps: (replan_steps, B, original_action_dim)

逐帧返回:
  每步:              (B, original_action_dim) → numpy
```

---

## 6. 核心模块详解

### 6.1 System 2 — 慢思考路径

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

### 6.2 System 1 — 快反应路径

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

### 6.3 Value Query Head (VQH)

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

## 7. 训练流程

### 7.1 两阶段训练

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

### 7.2 Stage 2 多优化器架构

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

### 7.3 数据格式

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

### 7.4 训练监控指标

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

## 8. 推理与部署

### 8.1 C/S 架构

```
┌──────────────────────┐     WebSocket      ┌──────────────────────┐
│  机器人环境            │ ◄──────────────► │  Policy Server        │
│  (LIBERO / 真实机械臂) │   msgpack 序列化  │  (WebsocketPolicyServer)│
│                      │                   │  加载 HumePolicy       │
│  WebsocketClientPolicy│                   │  GPU 推理              │
└──────────────────────┘                   └──────────────────────┘
```

### 8.2 服务端

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

### 8.3 客户端

**包:** [packages/openpi-client/](packages/openpi-client/)

| 组件 | 说明 |
|------|------|
| `WebsocketClientPolicy` | 同步 WebSocket 客户端，发送观测接收动作 |
| `ActionChunkBroker` | 包装 policy，逐帧返回 action chunk 中的动作 |
| `PolicyAgent` | 实现 `Agent` 接口，将 policy 集成到 runtime loop |
| `Runtime` | 驱动 Agent-Environment 交互循环 |
| `image_tools` | 图像 resize/格式转换工具 |

### 8.4 推理配置参数

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

## 9. 配置系统

### 9.1 配置层次

```
pyproject.toml           # 项目级配置 (依赖、构建)
    ↓
config/*.json            # 数据集级配置 (图像增强、训练参数)
    ↓
HumeConfig / System2Config  # 模型级配置 (网络结构、优化器)
    ↓
InferConfig              # 推理级配置 (去噪参数、候选数量)
```

### 9.2 数据集配置 (config/*.json)

每个数据集配置包含:

| 配置组 | 说明 |
|--------|------|
| `policy` | 策略参数 (chunk_size, 图像尺寸, 投影维度, 去噪步数) |
| `policy.image_transforms` | 图像增强 (亮度/对比度/饱和度/色调/锐度/裁剪/旋转) |
| `dataset` | 数据集路径、episode 范围 |
| `training` | 训练超参数 (batch_size, 学习率, 调度器) |

### 9.3 机器人参数

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

## 10. 关键设计决策

### 10.1 参数共享

`ValueQueryHead` 与 `System2` 共享 `paligemma_with_expert` 实例。这避免了重复加载大模型，但意味着 VQH 训练时梯度可能影响 System 2 的视觉编码器。

### 10.2 多步去噪 vs 单步

System 2 默认使用 10 步 EDM 欧拉 ODE 去噪，System 1 使用 1 步 Flow Matching。这体现了"慢思考 vs 快反应"的设计理念。推理时可通过 `s2_num_steps` 参数控制 Test-Time Scaling (TTS)。

### 10.3 候选采样 + Q 值选择

推理时 System 2 生成多个候选动作（通过不同的 `noise_temp`），VQH 评估 Q 值后选择最优。这是一种 **Best-of-N 采样 + RL Critic 过滤** 的策略。

### 10.4 Action Queue 机制

策略一次预测 `chunk_size` 步动作，存入 `action_queue`，逐帧执行。只有当 queue 为空或达到重规划间隔时才重新调用模型，大幅降低推理频率。

### 10.5 猴子补丁 (lerobot_patch.py)

项目通过猴子补丁替换了 LeRobot 的 `make_policy`、`make_dataset`、`make_optimizer_and_scheduler` 等工厂函数，以支持自定义的多优化器训练逻辑。

---

## 11. 依赖关系图

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

## 12. 硬件需求

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

## 13. 入口命令

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
