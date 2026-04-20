# Hume 推理数据流详解

> 本文档从源码级别逐步描述推理时每一层数据的精确 shape 与处理方式。
> 默认配置以 LIBERO 数据集为例: `state_dim=8, action_dim=7, max_state_dim=32, max_action_dim=32`。

---

## 符号约定

| 符号 | 含义 | 默认值 |
|------|------|--------|
| B | Batch size (推理时通常=1) | 1 |
| H_img | 图像高度 (resize后) | 224 |
| W_img | 图像宽度 (resize后) | 224 |
| state_dim | 真实状态维度 | 8 (LIBERO) |
| action_dim | 真实动作维度 | 7 (LIBERO) |
| max_state_dim | 状态 padding 维度 | 32 |
| max_action_dim | 动作 padding 维度 | 32 |
| s2_chunk_size | System 2 输出动作序列长度 | 50 |
| s1_chunk_size | System 1 输出动作序列长度 | 10 |
| vqh_chunk_size | VQH 评估的动作块长度 | 50 |
| s2_candidates_num | S2 候选动作数量 | 4 |
| s2_num_steps | S2 去噪迭代步数 | 10 |
| s1_num_steps | S1 去噪迭代步数 | 1 |
| s1_his_state_size | S1 状态历史帧数 | 5 |
| proj_width | S2 投影维度 | 1024 |
| s1_proj_width | S1 投影维度 | (由 s1_gemma_expert_config.hidden_size 决定, 默认 1024) |
| tokenizer_max_length | 语言 token 最大长度 | 48 |
| hidden_size | PaliGemma/Gemma hidden size | 2048 |
| num_img_tokens | SigLIP 输出 token 数 | 256 |
| head_dim | Gemma attention head dim | 256 |
| num_att_heads | Gemma attention heads | 8 |
| num_kv_heads | Gemma KV heads | 1 |
| dino_hidden_size | DINOv2-Small hidden size | 384 |
| dino_num_img_tokens | DINOv2 输出 token 数 | 256 |
| dino_num_layers | DINOv2 层数 | 12 |
| paligemma_num_layers | PaliGemma Gemma 层数 | 18 |
| s2_expert_num_layers | S2 Gemma Expert 层数 | 18 |
| s1_expert_num_layers | S1 Gemma Expert 层数 | 8 |
| vqh_expert_num_layers | VQH Gemma Expert 层数 | 4 |

---

## 全局时序图

```
┌────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 环境    │     │ HumePolicy   │     │ System2  │     │   VQH    │     │ System1  │
│ (Client)│     │  .infer()    │     │          │     │          │     │          │
└───┬─────┘     └──────┬───────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
    │                  │                   │                 │                 │
    │ observation      │                   │                 │                 │
    │─────────────────>│                   │                 │                 │
    │                  │                   │                 │                 │
    │                  │ (if action_plan   │                 │                 │
    │                  │  empty AND        │                 │                 │
    │                  │  replan_step=0)   │                 │                 │
    │                  │                   │                 │                 │
    │                  │ sample_actions×N  │                 │                 │
    │                  │──────────────────>│                 │                 │
    │                  │                   │                 │                 │
    │                  │  (B,N,50,32) 候选 │                 │                 │
    │                  │<──────────────────│                 │                 │
    │                  │                   │                 │
    │                  │ select_q_actions  │                 │                 │
    │                  │─────────────────────────────────────>│                 │
    │                  │                   │                 │
    │                  │ (B,) action_index │                 │                 │
    │                  │<─────────────────────────────────────│                 │
    │                  │                   │                 │
    │                  │ sample_actions    │                 │                 │
    │                  │ (with S2 action)  │                 │                 │
    │                  │─────────────────────────────────────────────────────────>│
    │                  │                   │                 │                 │
    │                  │ (B,10,32) actions │                 │                 │
    │                  │<─────────────────────────────────────────────────────────│
    │                  │                   │                 │                 │
    │ action           │                   │                 │                 │
    │<─────────────────│                   │                 │                 │
    │                  │                   │                 │                 │
    │  ...(next frame) │                   │                 │                 │
    │ observation      │                   │                 │                 │
    │─────────────────>│                   │                 │                 │
    │                  │                   │                 │                 │
    │                  │ (action_plan 非空: 直接 popleft)     │                 │
    │ action           │                   │                 │                 │
    │<─────────────────│                   │                 │                 │
    │                  ╳                   │                 │                 │
```

---

- 全局时序图 — 环境 ↔ HumePolicy ↔ System2 ↔ VQH ↔ System1 的 UML 交互图
- Phase 0: 观测预处理 — 状态历史维护、numpy→torch 格式转换、stamp 计算
- Phase 1: 通用预处理 — prepare_images/prepare_state/prepare_language 的每步 shape
- Phase 2: System 2 候选生成 — SigLIP 编码 → KV Cache 填充 → 后缀编码 → 10 步 EDM 欧拉去噪的逐层 shape，包括注意力掩码的可视化
- Phase 3: VQH 候选评估 — query embedding 插入机制 → VQHBackbone (4层) → CalQL 双 Critic MLP (2398→256→256→1) → argmax Q 选择
- Phase 4: System 1 细粒度去噪 — DINOv2 (ImageNet归一化) → 5帧状态历史+stamp+动作 → FastVisuoExpert (8层) → 1步 Flow Matching
- Phase 5: 后处理与动作返回

---

## Phase 0: 观测预处理 (`HumePolicy.infer`)

**源码:** `src/hume/models/modeling_hume.py:282-369`

### 0.1 输入观测

```
observation = {
  "observation.images.image":       np.uint8  (B, H_raw, W_raw, 3)   # [0, 255]
  "observation.images.wrist_image": np.uint8  (B, H_raw, W_raw, 3)   # [0, 255], 可选
  "observation.state":              np.float32 (B, 8)                 # 7 DOF + gripper
  "task":                           List[str]  ["pick up the cup\n"]  # 长度 B
}
```

### 0.2 状态历史维护

```
if not history_state:                                    # 首次调用
    history_state.extend(
        state.repeat(s1_his_state_size, axis=1)          # 复制当前状态填充整个窗口
        .transpose(1, 0, 2)
    )
else:                                                    # 后续调用
    history_state.append(state)                          # 追加最新, 淘汰最旧

# 拼接后:
observation["observation.state"] = np.asarray(history_state).transpose(1, 0, 2)
# shape: (B, s1_his_state_size, state_dim) = (1, 5, 8)
```

### 0.3 格式转换: numpy → torch

```
图像:
  np.uint8 (B, H, W, 3) [0,255]
    → torch.tensor(v / 255)        # float32, [0.0, 1.0]
    → .permute(0, 3, 1, 2)         # (B, 3, H, W)
    → .to(device).float()
  结果: Tensor (B, 3, H_raw, W_raw) float32 [0.0, 1.0]

语言:
  List[str] → 原样传递

状态:
  np.float32 (B, 5, 8)              # 含历史
    → torch.tensor(v).to(device).float()
  结果: Tensor (B, s1_his_state_size, state_dim) = (1, 5, 8) float32
```

### 0.4 触发条件判断

```
action_plan 为空? ─── 否 ──→ popleft() 返回缓存动作, 不进入模型
        │
       是
        │
infer_step % s2_replan_steps == 0? ─── 是 ──→ outputs = {} (清空, 触发完整 S2+VQH+S1)
        │
       否
        │
outputs 保留上一轮 S2 缓存 → 仅重跑 S1 (新 stamp)
```

### 0.5 计算 stamp

```
stamp = (infer_step % s2_replan_steps) / s2_chunk_size
# 例: infer_step=0  → stamp=0.0
#     infer_step=5  → stamp=0.1
#     infer_step=45 → stamp=0.9
# shape: Tensor (B,) float32, 值域 [0, ~1)
```

---

## Phase 1: 进入 `select_action` 通用预处理

**源码:** `src/hume/models/modeling_hume.py:371-463`

### 1.1 归一化 (`normalize_inputs`)

```
state:
  (B, s1_his_state_size, state_dim) = (1, 5, 8)
    → MEAN_STD 归一化 (使用训练集统计量)
  (1, 5, 8) float32

图像已在 infer 中归一化到 [0,1], 此处不再归一化 (VISUAL → IDENTITY)
```

### 1.2 `prepare_images`

```
对每个图像 key (如 "observation.images.image"):
  img: (B, 3, H_raw, W_raw) float32 [0,1]
    → resize_with_pad(img, 224, 224, pad_value=0)
      保持长宽比 resize, 不足部分 pad 0
    → img * 2.0 - 1.0
      映射到 [-1, 1] (SigLIP 期望输入范围)
  结果: (B, 3, 224, 224) float32 [-1, 1]

  img_mask: torch.ones(B, dtype=bool)  # 全 True (有效图像)

返回:
  images:   list[Tensor]  长度=相机数(1~2), 每个 (B, 3, 224, 224)
  img_masks: list[Tensor] 长度=相机数,    每个 (B,) bool
```

### 1.3 `prepare_state` (for S2)

```
state:
  (B, state_dim) = (1, 8)
    → pad_vector(state, max_state_dim=32)   # 零填充到 max_state_dim
  (B, max_state_dim) = (1, 32) float32

注: S2 只使用 state[:, -1, :], 即最新一帧
    但 prepare_state 接收的是 batch[OBS_ROBOT]
    在 infer 中已拼成 (B, his, dim), 但传入 S2 时只取 state[:, -1, :]
```

### 1.4 `prepare_language`

```
tasks: List[str] = ["pick up the cup\n"]
  → language_tokenizer(tasks, padding="max_length", max_length=48)

lang_tokens: (B, 48) int64          # token IDs, 右侧 padding=0
lang_masks:  (B, 48) bool           # True=有效 token, False=padding
```

---

## Phase 2: System 2 候选动作生成

**源码:** `src/hume/models/modeling_hume.py:1226-1287`

当 `"noise_action" not in outputs` 时触发。

### 2.1 循环生成 N 个候选

```
noise_actions = []
for i in range(s2_candidates_num):   # N=4
    time_temp  = (i/N) * (upper - lower) + lower
    noise_temp = (i/N) * (upper - lower) + lower

    action_i = self.s2_model.sample_actions(
        images, img_masks, lang_tokens, lang_masks,
        state[:, -1, :],              # (B, max_state_dim) = (1, 32) — 仅最新帧
        time_temp=time_temp,
        noise_temp=noise_temp,
    )
    noise_actions.append(action_i)    # (B, s2_chunk_size, max_action_dim) = (1, 50, 32)

noise_actions = torch.stack(noise_actions, dim=1)
# (B, N, s2_chunk_size, max_action_dim) = (1, 4, 50, 32)
```

### 2.2 System2.sample_actions 内部流程

以下展示**单个候选**的完整处理流程。

#### 2.2.1 前缀编码: 图像 + 语言 (只执行一次, 缓存 KV)

```
System2.embed_prefix(images, img_masks, lang_tokens, lang_masks)
```

```
┌──────────────────────────────────────────────────────────────────────┐
│ 图像编码 (SigLIP)                                                    │
│                                                                      │
│  每个 image:                                                         │
│    img: (B, 3, 224, 224) bf16 [-1,1]                                │
│      → self.paligemma_with_expert.embed_image(img)                   │
│        = self.paligemma.get_image_features(img)                      │
│        → SigLIP Vision Tower (27层, hidden=1152, patch=14)           │
│          224/14 = 16 → 16×16 = 256 patches                          │
│        → multi_modal_projector: Linear(1152 → 2048)                  │
│      img_emb: (B, 256, 2048) bf16                                   │
│                                                                      │
│    → img_emb * √2048 = √hidden_size 归一化                           │
│    img_emb: (B, 256, 2048) bf16                                     │
│                                                                      │
│    img_mask: (B,) → expand → (B, 256)                               │
│    att_masks: [0]*256  (全连接注意力)                                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ 语言编码 (Gemma Embedding)                                           │
│                                                                      │
│  lang_tokens: (B, 48) int64                                         │
│    → self.paligemma_with_expert.embed_language_tokens(lang_tokens)   │
│      = self.paligemma.language_model.model.embed_tokens(lang_tokens) │
│    lang_emb: (B, 48, 2048) bf16                                     │
│                                                                      │
│  → lang_emb * √2048 归一化                                           │
│  lang_emb: (B, 48, 2048) bf16                                       │
│                                                                      │
│  lang_masks: (B, 48) bool                                           │
│  att_masks: [0]*48  (全连接注意力)                                    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ 合并前缀                                                              │
│                                                                      │
│  prefix_embs:     cat([img_emb, lang_emb])                           │
│                  (B, 256+48, 2048) = (1, 304, 2048) bf16            │
│                                                                      │
│  prefix_pad_masks: cat([img_mask_expand, lang_masks])                 │
│                  (B, 304) bool                                       │
│                                                                      │
│  prefix_att_masks: cat([[0]*256, [0]*48])                            │
│                  (304,) bool, 全 0 (全连接)                           │
│                  → expand → (B, 304)                                 │
│                                                                      │
│  → make_att_2d_masks(pad_masks, att_masks)                           │
│    prefix_att_2d_masks: (B, 304, 304) bool                          │
│    全 True (所有 token 互相可见)                                      │
│                                                                      │
│  prefix_position_ids: cumsum(pad_masks) - 1                          │
│                  (B, 304) int64                                      │
└──────────────────────────────────────────────────────────────────────┘
```

如果有 2 个相机 (image + wrist_image):
```
prefix_embs: (B, 256+256+48, 2048) = (1, 560, 2048)
prefix_pad_masks: (B, 560)
```

#### 2.2.2 PaliGemma 前向: 填充 KV Cache

```
PaliGemmaWithExpertModel.forward(
    attention_mask = prefix_att_2d_masks,  # (B, 304, 304)
    position_ids   = prefix_position_ids,  # (B, 304)
    past_key_values = None,
    inputs_embeds  = [prefix_embs, None],  # [前缀, 无后缀]
    use_cache      = True,
    fill_kv_cache  = True,
)
```

内部逐层处理 (共 paligemma_num_layers=18 层 + s2_expert_num_layers=18 层):

```
┌──────────────────────────────────────────────────────────────────────┐
│ 第 L 层 (L = 0..17: PaliGemma Gemma; L = 18..35: Gemma Expert)     │
│                                                                      │
│  hidden_states: (B, 304, 2048) bf16                                 │
│                                                                      │
│  → input_layernorm(hidden_states)                                    │
│    (B, 304, 2048) bf16                                               │
│                                                                      │
│  → .view(B, 304, num_att_heads, head_dim)                            │
│    = .view(B, 304, 8, 256)                                           │
│                                                                      │
│  Q = q_proj(normed): (B, 304, 8, 256) bf16                          │
│  K = k_proj(normed): (B, 304, 1, 256) bf16  ← num_kv_heads=1       │
│  V = v_proj(normed): (B, 304, 1, 256) bf16                          │
│                                                                      │
│  → apply_rope(Q, position_ids)                                       │
│  → apply_rope(K, position_ids)                                       │
│                                                                      │
│  KV Cache (fill_kv_cache=True):                                      │
│    past_key_values[L] = {key_states: K, value_states: V}             │
│    K cache: (B, 304, 1, 256), V cache: (B, 304, 1, 256)             │
│                                                                      │
│  → K.expand(1→8): (B, 304, 8, 256)   ← GQA repeat                  │
│  → V.expand(1→8): (B, 304, 8, 256)                                 │
│                                                                      │
│  → Q.transpose(1,2): (B, 8, 304, 256)                               │
│  → K.transpose(1,2): (B, 8, 304, 256)                               │
│                                                                      │
│  → att_weights = Q @ K^T / √256                                     │
│    (B, 8, 304, 304) float32                                          │
│                                                                      │
│  → masked_att = where(mask, att_weights, -2.38e38)                   │
│  → softmax(masked_att, dim=-1)                                       │
│    (B, 8, 304, 304) float32                                          │
│                                                                      │
│  → att_output = softmax @ V.permute(0,2,1,3)                         │
│    (B, 8, 304, 256)                                                  │
│  → .permute(0,2,1,3).reshape(B, 304, 2048)                          │
│    → bf16                                                            │
│                                                                      │
│  → o_proj(att_output): (B, 304, 2048) bf16                          │
│  → + hidden_states (残差连接1)                                        │
│  → post_attention_layernorm                                          │
│  → mlp (hidden→intermediate→hidden: 2048→16384→2048)                │
│  → + after_first_residual (残差连接2)                                 │
│                                                                      │
│  输出: (B, 304, 2048) bf16                                           │
│  传递给下一层                                                         │
└──────────────────────────────────────────────────────────────────────┘

所有 36 层结束后:
  → final RMSNorm
  prefix_output: (B, 304, 2048) bf16

返回: (prefix_output, past_key_values)
  past_key_values: dict, 36 个 entry, 每个 {key_states, value_states}
```

#### 2.2.3 初始化噪声

```
actions_shape = (B, s2_chunk_size, max_action_dim) = (1, 50, 32)
noise = N(0, 1), shape (1, 50, 32) float32
x_t = noise                                       # 初始纯噪声
time = time_temp (标量, float32)                    # 从 time_temp 开始
dt = -1.0 / num_steps = -0.1                       # 步长
```

#### 2.2.4 去噪循环 (num_steps=10 次)

每次迭代调用 `denoise_step`:

```
┌──────────────────────────────────────────────────────────────────────┐
│ denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)   │
│                                                                      │
│  state: (B, max_state_dim) = (1, 32) float32                        │
│  x_t:   (B, s2_chunk_size, max_action_dim) = (1, 50, 32) float32   │
│  time:  (B,) float32, 标量扩展                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**后缀编码:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ embed_suffix(state, x_t, timestep)                                   │
│                                                                      │
│ ① 状态投影                                                           │
│    state: (B, 32) float32                                            │
│      → state_proj: Linear(32 → 1024)                                 │
│      → bf16                                                          │
│    state_emb: (B, 1, 1024) bf16     ← unsqueeze dim=1               │
│    state_mask: (B, 1) = True                                         │
│    att_masks += [1]                  ← state 不回看 prefix           │
│                                                                      │
│ ② 时间步编码                                                         │
│    timestep: (B,) float32                                            │
│      → create_sinusoidal_pos_embedding(timestep, proj_width=1024,    │
│          min_period=4e-3, max_period=4.0)                            │
│        sin/cos encoding: (B, 1024) float64 → float32                 │
│    time_emb: (B, 1024) float32                                       │
│                                                                      │
│ ③ 动作-时间融合                                                      │
│    x_t: (B, 50, 32) float32                                          │
│      → action_in_proj: Linear(32 → 1024)                             │
│    action_emb: (B, 50, 1024) float32                                 │
│                                                                      │
│    time_emb: (B, 1024) → unsqueeze(1) → expand                       │
│    time_emb: (B, 50, 1024)   ← expand_as(action_emb)                │
│                                                                      │
│    concat([action_emb, time_emb], dim=2):                             │
│    (B, 50, 2048)                                                     │
│      → action_time_mlp_in: Linear(2048 → 1024)                       │
│      → SiLU 激活                                                     │
│      → action_time_mlp_out: Linear(1024 → 1024)                      │
│    action_time_emb: (B, 50, 1024)                                    │
│    action_mask: (B, 50) = True                                       │
│    att_masks += [1] + [0]*49  ← 因果掩码: 第一个动作不回看,           │
│                                 后续动作可见前面的动作                  │
│                                                                      │
│ ④ 合并后缀                                                           │
│    suffix_embs:     cat([state_emb, action_time_emb])                 │
│                    (B, 1+50, 1024) = (1, 51, 1024) bf16             │
│    suffix_pad_masks: cat([state_mask, action_mask])                   │
│                    (B, 51) bool                                      │
│    suffix_att_masks: (B, 51)                                         │
│                    [1, 1, 0, 0, ..., 0]                              │
└──────────────────────────────────────────────────────────────────────┘
```

**注意力掩码构建:**

```
suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
  (B, 51, 51) bool

  可视化 (简化为 5 个 token: 1 state + 4 action):
           state  act0  act1  act2  act3
  state  [  1      0     0     0     0  ]   ← state 只看自己 (att_mask=1 → causal)
  act0   [  1      1     0     0     0  ]   ← act0 看 state + 自己
  act1   [  1      1     1     0     0  ]   ← act1 看 state + act0 + 自己
  act2   [  1      1     1     1     0  ]
  act3   [  1      1     1     1     1  ]

prefix_pad_2d_masks: prefix_pad_masks expand → (B, 51, 304)
  suffix token 对 prefix token 全部可见

full_att_2d_masks: cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
  (B, 51, 304+51) = (B, 51, 355)

  可视化:
           prefix(304)         suffix(51)
           img  lang  |  state  act0  act1 ...
  state  [  1    1   |   1      0     0  ... ]  ← state 看全部 prefix + 自己
  act0   [  1    1   |   1      1     0  ... ]  ← act0 看全部 prefix + state + 自己
  act1   [  1    1   |   1      1     1  ... ]
  ...
```

**复用 KV Cache 的 PaliGemma 前向:**

```
PaliGemmaWithExpertModel.forward(
    attention_mask  = full_att_2d_masks,     # (B, 51, 355)
    position_ids    = prefix_offsets + cumsum(suffix_pad) - 1,  # (B, 51)
    past_key_values = cached,                # 36 层的 prefix KV cache
    inputs_embeds   = [None, suffix_embs],   # [无前缀, 后缀]
    use_cache       = True,
    fill_kv_cache   = False,
)

内部每层:
  Q/K/V 只计算 suffix 部分: (B, 51, 8, 256)
  K = cat([cached_K, new_K], dim=1)  → (B, 304+51, 1, 256) = (B, 355, 1, 256)
  V = cat([cached_V, new_V], dim=1)  → (B, 355, 1, 256)
  → GQA expand → (B, 355, 8, 256)

  attention: Q @ K^T → (B, 8, 51, 355) → softmax → @ V → (B, 51, 2048)

  取 suffix_output (第 2 个输入的输出):
  suffix_out: (B, 51, 2048) bf16

→ 取最后 n_action_steps=50 个 token:
  suffix_out[:, -50:]  → (B, 50, 2048) bf16
  → .to(float32)
  → action_out_proj: Linear(1024 → 32)     # 注意: 2048→? 内部维度映射
  v_t: (B, 50, 32) float32                  # 预测的 velocity
```

**欧拉步更新:**

```
x_t += dt * v_t * noise_temp
time += dt
```

10 步去噪后:
```
return x_t    # (B, 50, 32) float32 — 去噪后的动作序列
```

---

## Phase 3: VQH 候选评估与选择

**源码:** `src/hume/models/modeling_hume.py:1838-1878`

### 3.1 截取与准备

```
noise_actions: (B, N, s2_chunk_size, max_action_dim) = (1, 4, 50, 32)

noise_actions_wo_pad = noise_actions[:, :, :vqh_chunk_size, :action_dim]
  (1, 4, 50, 7)    # 去掉 padding, 只取前 vqh_chunk_size 步 × 真实 action_dim
```

### 3.2 VQH.embed_prefix — 编码观测

```
┌──────────────────────────────────────────────────────────────────────┐
│ VQH.embed_prefix(images, img_masks, lang_tokens, lang_masks)         │
│                                                                      │
│  与 S2 embed_prefix 类似, 但有 2 个关键区别:                          │
│                                                                      │
│  ① 图像编码: 使用共享的 paligemma_with_expert.embed_image(img)       │
│     img → SigLIP → (B, 256, 2048) bf16 → √d 归一化                  │
│                                                                      │
│  ② 语言编码: embed_language_tokens(lang_tokens).detach()             │
│     lang_tokens → Gemma embed → (B, 48, 2048) bf16                  │
│     .detach() — VQH 训练时梯度不回传到语言模型                        │
│     → √d 归一化                                                      │
│                                                                      │
│  ③ 合并: embs = cat([img_emb, lang_emb])                             │
│     (B, 304, 2048) bf16                                              │
│     pad_masks: (B, 304)                                              │
│     att_masks: 全 0 (全连接)                                          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ 插入 Query Embedding                                                 │
│                                                                      │
│  query_embedding: nn.Parameter(2048) bf16 — 可学习的 query token    │
│                                                                      │
│  在每个序列的有效长度位置 (seq_len-1 后) 插入:                        │
│                                                                      │
│  原 embs:     [img0, img1, ..., img255, lang0, ..., lang47]          │
│               ← 304 tokens →                                         │
│                                                                      │
│  新序列:      [img0, ..., img255, lang0, ..., lang47, QUERY]         │
│               ← 305 tokens →                                         │
│                                                                      │
│  new_embs:     (B, 305, 2048) bf16                                   │
│  new_pad_masks: (B, 305) bool                                        │
│  new_att_masks: (B, 305) — QUERY 位置 att_mask=False (可被后续看到)  │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 VQHBackbone 前向

```
┌──────────────────────────────────────────────────────────────────────┐
│ VQHBackbone.forward(attention_mask, position_ids, inputs_embeds)     │
│                                                                      │
│  inputs: (B, 305, 2048) bf16                                        │
│                                                                      │
│  VQHBackbone 内部: Gemma Expert (4 层, hidden=2048, heads=8)        │
│                                                                      │
│  逐层处理 (4 层):                                                    │
│    hidden_states: (B, 305, 2048)                                     │
│      → input_layernorm                                               │
│      → .view(B, 305, 8, 256)                                        │
│      Q/K/V → (B, 305, 8, 256)                                       │
│      → apply_rope                                                    │
│      → eager_attention (full att, 305 tokens)                        │
│      → o_proj + 残差 + layernorm + mlp + 残差                        │
│    输出: (B, 305, 2048)                                              │
│                                                                      │
│  → final RMSNorm                                                     │
│  suffix_out: (B, 305, 2048) bf16                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 提取 Query Embedding 输出

```
batch_indices = arange(B)
query_embedding_idx = pad_masks.sum(-1).long() - 1   # 最后一个有效 token 的位置

query_embedding = suffix_out[batch_indices, query_embedding_idx]
  (B, 2048) bf16 → float32

这就是观测的紧凑表征向量: 编码了图像+语言的所有决策相关信息
```

### 3.5 CalQL Target Critics 评估 Q 值

```
┌──────────────────────────────────────────────────────────────────────┐
│ CalQL.get_q_values(query_embedding, noise_actions)                   │
│                                                                      │
│  query_embedding: (B, 2048) float32                                 │
│  noise_actions:  (B, N, vqh_chunk_size * action_dim)                 │
│                 = (1, 4, 50*7) = (1, 4, 350)                        │
│                                                                      │
│  ┌────────────────────────────────────────────────┐                  │
│  │ 每个 Critic 网络 (共 2 个, ensemble):           │                  │
│  │                                                │                  │
│  │  obs: (B, 2048) → unsqueeze(1) → expand         │                  │
│  │       (B, N, 2048) = (1, 4, 2048)              │                  │
│  │                                                │                  │
│  │  inputs = cat([obs, actions], dim=-1)            │                  │
│  │         (B, N, 2048 + 350) = (1, 4, 2398)      │                  │
│  │                                                │                  │
│  │  MLP: 2398 → 256 → 256 → activate_final         │                  │
│  │  backbone_out: (B, N, 256)                      │                  │
│  │                                                │                  │
│  │  output_layer: Linear(256 → 1)                   │                  │
│  │  Q_i(s, a_j): (B, N) = (1, 4)                  │                  │
│  └────────────────────────────────────────────────┘                  │
│                                                                      │
│  2 个 Critic 堆叠:                                                   │
│  q_values: (2, B, N) = (2, 1, 4) float32                           │
│                                                                      │
│  → min(dim=0): 取两个 Q 网络的最小值                                  │
│  q_values: (B, N) = (1, 4) float32                                  │
│                                                                      │
│  例: q_values = [[0.72, 0.85, 0.61, 0.78]]                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.6 选择最优候选

```
action_index = argmax(q_values, dim=1)
  (B,) int64 = tensor([1])     # 第 2 个候选 Q 值最高

selected_noise_action = noise_actions[batch_idx, action_index]
  (B, s2_chunk_size, max_action_dim) = (1, 50, 32)

outputs = {"noise_action": selected_noise_action}
```

---

## Phase 4: System 1 细粒度去噪

**源码:** `src/hume/models/modeling_hume.py:1543-1617`

### 4.1 从 S2 动作中滑窗截取

```
stamp: (B,) = tensor([0.0])        # 当前在 S2 序列中的位置

idcs = (stamp * s2_chunk_size).long() + arange(s1_chunk_size)
     = (0.0 * 50).long() + [0, 1, ..., 9]
     = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  shape: (B, s1_chunk_size) = (1, 10)

batch_idcs = arange(B).unsqueeze(1) = [[0]]
  shape: (B, 1) = (1, 1)

noise_action_slides = selected_noise_action[batch_idcs, idcs]
  (B, s1_chunk_size, max_action_dim) = (1, 10, 32) float32
```

当 stamp=0.3 时: `idcs = [15, 16, ..., 24]` — 取 S2 序列中对应位置的动作片段。

### 4.2 FastVisuoMatching.sample_actions

```
s1_actions = self.s1_model.sample_actions(
    images,       # list[Tensor], each (B, 3, 224, 224)
    img_masks,    # list[Tensor], each (B,) bool
    state,        # (B, s1_his_state_size, max_state_dim) = (1, 5, 32)  ← 含历史
    noise=noise_action_slides,  # (B, s1_chunk_size, max_action_dim) = (1, 10, 32)
    stamp=stamp,  # (B,) float32
)
```

#### 4.2.1 前缀编码 (DINOv2, 无语言)

```
┌──────────────────────────────────────────────────────────────────────┐
│ FastVisuoMatching.embed_prefix(images, img_masks)                    │
│                                                                      │
│  每个 image:                                                         │
│    img: (B, 3, 224, 224) float32 (来自 infer, 已 /255 到 [0,1])     │
│                                                                      │
│    ① ImageNet 归一化 (DINOv2 标准):                                  │
│       img = TF.normalize(img * 0.5 + 0.5,                           │
│            mean=[0.485, 0.456, 0.406],                               │
│            std=[0.229, 0.224, 0.225])                                │
│       (B, 3, 224, 224) float32                                      │
│       [0,1] → [0.5,1.5] → ImageNet归一化 → ~N(0,1)                 │
│                                                                      │
│    ② DINOv2-Small 编码:                                             │
│       → self.fast_visuo_expert.embed_image(img)                      │
│         = vision_tower(img).last_hidden_state                        │
│         → DINOv2 (12层, hidden=384, patch=14, 224/14=16 → 256 tkn)  │
│       selected_feature: (B, 256, 384) bf16                          │
│                                                                      │
│         → multi_modal_projector: Linear(384 → 1024)                  │
│       img_features: (B, 256, 1024) bf16                             │
│                                                                      │
│         → ÷ √1024 归一化                                             │
│       img_emb: (B, 256, 1024) bf16                                  │
│                                                                      │
│    img_mask: (B,) → expand → (B, 256)                               │
│    att_masks: [0]*256                                                │
│                                                                      │
│  合并: embs: (B, 256, 1024) bf16    (单相机)                        │
│        pad_masks: (B, 256)                                          │
│        att_masks: (B, 256) — 全 0 (全连接)                           │
│                                                                      │
│  注: 无 KV Cache 机制 — 每次去噪步都重新计算前缀编码                  │
│      (DINOv2-Small 很小, 计算开销低)                                  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 初始化噪声

```
actions_shape = (B, s1_chunk_size, max_action_dim) = (1, 10, 32)
noise = noise_action_slides   # (1, 10, 32) ← 直接使用 S2 的动作片段!

# 实际上 sample_actions 中:
#   noise = noise_action_slides (来自 S2, 作为初始 x_t)
#   但去噪从 theta1 开始, 所以:
x_t = noise_action_slides     # (1, 10, 32)
time = theta1                 # 标量 (例如 0.5)
dt = -theta1 / s1_num_steps   # 步长
```

#### 4.2.3 去噪循环 (s1_num_steps=1 次, 默认仅 1 步)

每次迭代调用 `denoise_step`:

```
┌──────────────────────────────────────────────────────────────────────┐
│ denoise_step(state, prefix_embs, prefix_pad_masks,                   │
│              prefix_att_masks, x_t, timestep, stamp)                 │
│                                                                      │
│  state:    (B, s1_his_state_size, max_state_dim) = (1, 5, 32)       │
│  x_t:      (B, s1_chunk_size, max_action_dim) = (1, 10, 32)        │
│  timestep: (B,) float32                                              │
│  stamp:    (B,) float32                                              │
└──────────────────────────────────────────────────────────────────────┘
```

**后缀编码:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ embed_suffix(state, x_t, timestep, stamp)                            │
│                                                                      │
│ ① 状态投影 (含历史):                                                │
│    state: (B, 5, 32) float32                                         │
│      → state_proj: Linear(32 → 1024)                                 │
│    state_emb: (B, 5, 1024) bf16                                     │
│    state_mask: (B, 5) = True                                         │
│    att_masks += [1, 1, 1, 1, 1]  ← 5 个 state token 全部 att_mask=1 │
│                                                                      │
│ ② Stamp 编码:                                                       │
│    stamp: (B,) float32                                               │
│      → sinusoidal_pos_embedding(stamp, s1_proj_width=1024)           │
│    stamp_emb: (B, 1024) → unsqueeze(1) → (B, 1, 1024)              │
│    stamp_mask: (B, 1) = True                                         │
│    att_masks += [1]                                                  │
│                                                                      │
│ ③ 时间步编码:                                                       │
│    timestep: (B,) float32                                            │
│      → sinusoidal_pos_embedding(timestep, 1024)                      │
│    time_emb: (B, 1024)                                               │
│                                                                      │
│ ④ 动作-时间融合:                                                     │
│    x_t: (B, 10, 32)                                                  │
│      → action_in_proj: Linear(32 → 1024)                             │
│    action_emb: (B, 10, 1024)                                         │
│                                                                      │
│    time_emb: (B, 1024) → unsqueeze(1) → expand                       │
│    time_emb: (B, 10, 1024)                                           │
│                                                                      │
│    concat([action_emb, time_emb], dim=2):                             │
│    (B, 10, 2048)                                                     │
│      → action_time_mlp_in: Linear(2048 → 1024)                       │
│      → SiLU                                                          │
│      → action_time_mlp_out: Linear(1024 → 1024)                      │
│    action_time_emb: (B, 10, 1024)                                    │
│    action_mask: (B, 10) = True                                       │
│    att_masks += [1, 0, 0, ..., 0]  ← 因果掩码 (第1个=1, 其余=0)     │
│                                                                      │
│ ⑤ 合并后缀:                                                         │
│    suffix_embs:     cat([state_emb, stamp_emb, action_time_emb])      │
│                    (B, 5+1+10, 1024) = (1, 16, 1024) bf16           │
│    suffix_pad_masks: cat([state_mask, stamp_mask, action_mask])       │
│                    (B, 16) bool                                      │
│    suffix_att_masks: (B, 16)                                         │
│                    [1,1,1,1,1, 1, 1,0,0,0,0,0,0,0,0,0]             │
└──────────────────────────────────────────────────────────────────────┘
```

**合并 prefix + suffix:**

```
pad_masks: cat([prefix_pad, suffix_pad])  → (B, 256+16) = (1, 272)
att_masks: cat([prefix_att, suffix_att])  → (B, 272)

att_2d_masks: (B, 272, 272)

注意力模式:
           prefix(256)       suffix(16)
           all img tokens  |  state(5)  stamp  action(10)
           [0,...,0]        |  [1,...,1]  [1]   [1,0,...,0]
prefix:    全连接           |  不看 suffix
suffix st: 看全部 prefix    |  因果 (只看 state+自己)
suffix ac: 看全部 prefix    |  因果 (看前面所有 suffix)

inputs_embeds: cat([prefix_embs, suffix_embs], dim=1)
  (B, 272, 1024) bf16
```

**FastVisuoExpert 前向:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ FastVisuoExpertModel.forward(attention_mask, position_ids,           │
│                               inputs_embeds)                         │
│                                                                      │
│  inputs: (B, 272, 1024) bf16                                        │
│                                                                      │
│  内部: Gemma Expert (8 层, hidden=1024, heads=8, head_dim=256,      │
│         intermediate=4096, kv_heads=1)                               │
│                                                                      │
│  逐层处理 (8 层):                                                    │
│    hidden_states: (B, 272, 1024)                                     │
│      → input_layernorm                                               │
│      → .view(B, 272, 8, 256)                                        │
│      Q: (B, 272, 8, 256)                                            │
│      K: (B, 272, 1, 256) → expand → (B, 272, 8, 256)               │
│      V: (B, 272, 1, 256) → expand → (B, 272, 8, 256)               │
│      → apply_rope(Q, position_ids)                                   │
│      → apply_rope(K, position_ids)                                   │
│      → eager_attention (272×272)                                     │
│      → o_proj + 残差 + layernorm + mlp + 残差                        │
│    输出: (B, 272, 1024)                                              │
│                                                                      │
│  → final RMSNorm                                                     │
│  输出: (B, 272, 1024) bf16                                          │
└──────────────────────────────────────────────────────────────────────┘
```

**提取动作预测:**

```
suffix_out = output[:, -s1_action_steps:]
  (B, 10, 1024) bf16
  → .to(float32)

v_t = action_out_proj(suffix_out)
  Linear(1024 → 32)
  v_t: (B, 10, 32) float32      # 预测的 velocity
```

**欧拉步更新:**

```
x_t += dt * v_t
time += dt
```

1 步去噪后:
```
return x_t    # (B, 10, 32) float32 — S1 输出的精细动作
```

---

## Phase 5: 后处理与动作执行

**源码:** `src/hume/models/modeling_hume.py:355-369`

### 5.1 去归一化

```
s1_action: (B, s1_chunk_size, max_action_dim) = (1, 10, 32) float32

actions = s1_action[:, :, :original_action_dim]    # 去 padding
  (1, 10, 7) float32

actions = unnormalize_outputs({"action": actions})["action"]
  MEAN_STD 反归一化: action = action * std + mean
  (1, 10, 7) float32 — 还原到原始动作空间尺度
```

### 5.2 可选 Gripper 后处理

```
if post_process_action:
    actions[..., -1] = 2 * (1 - actions[..., -1]) - 1
    # 对 gripper 维度做线性变换
```

### 5.3 填充 action_plan

```
action_chunk = actions.transpose(1, 0, 2)
  (s1_chunk_size, B, action_dim) = (10, 1, 7) float32

action_plan.extend(action_chunk[:replan_steps])
  取前 replan_steps=5 步:
  plan = [a0, a1, a2, a3, a4]   # 每个 (1, 7) float32
```

### 5.4 逐帧返回

```
infer_step += 1
action = action_plan.popleft()
  → np.asarray(action)
  (B, action_dim) = (1, 7) float32 → numpy

返回给客户端
```

---

## 完整 Shape 流转一览

```
输入
  image:       np.uint8 (1, H, W, 3) [0,255]
  state:       np.float32 (1, 8)
  task:        ["pick up the cup\n"]
                  │
Phase 0: 预处理  │
  image:       Tensor (1, 3, 224, 224) [0,1]
  state:       Tensor (1, 5, 8)           # 含历史
  stamp:       Tensor (1,)                # [0, 1)
                  │
Phase 1: prepare  │
  images:      list[(1, 3, 224, 224)]     # [-1, 1]
  img_masks:   list[(1,)]
  state:       Tensor (1, 32)             # padded
  lang_tokens: Tensor (1, 48)             # int64
  lang_masks:  Tensor (1, 48)             # bool
                  │
         ┌────────┴────────┐
         │                  │
Phase 2: System 2     Phase 3: VQH
         │                  │
    SigLIP:                SigLIP (共享):
    (1,224,224)→(1,256,2048)    (1,224,224)→(1,256,2048)
         │                  │
    Gemma Embed:        Gemma Embed (.detach):
    (1,48)→(1,48,2048)        (1,48)→(1,48,2048)
         │                  │
    prefix: (1,304,2048) prefix+QUERY: (1,305,2048)
         │                  │
    KV Cache 填充           VQHBackbone (4层):
    36层 × {K,V: (1,304,1,256)}  (1,305,2048) → (1,305,2048)
         │                  │
    suffix: state_proj          query_embedding:
    (1,32)→Linear→(1,1,1024)    (1,2048)
         │                  │
    action+time MLP:        CalQL Target Critics:
    (1,50,32)→(1,50,1024)   concat → MLP(2398→256→256→1)
         │                  │
    suffix: (1,51,1024)     Q values: (2,1,4) → min → (1,4)
         │                  │
    PaliGemma (复用KV):     argmax → action_index: (1,)
    (1,51,355) → (1,51,2048)
         │                  │
    取后50个 → action_out   选取最优:
    (1,50,2048)→Linear→(1,50,32)  (1,50,32)
         │                  │
    去噪×10步                   │
    (1,50,32) × N              │
         │                      │
    stack: (1,4,50,32)          │
         └──────────┬───────────┘
                    │
                    │ (1,50,32) 选出的 S2 动作
                    │
              滑窗截取 (stamp)
              (1,50,32) → (1,10,32)
                    │
Phase 4: System 1    │
                    │
    DINOv2:          │
    (1,224,224)→ImageNet norm
    →DINOv2→(1,256,384)
    →Linear(384→1024)→(1,256,1024)
                    │
    suffix: state(5帧)+stamp+action
    state_proj: (1,5,32)→(1,5,1024)
    stamp_emb:  (1,1,1024)
    action+time: (1,10,32)→(1,10,1024)
    suffix: (1,16,1024)
                    │
    concat prefix+suffix:
    (1,272,1024)
                    │
    FastVisuoExpert (8层):
    (1,272,1024)→(1,272,1024)
                    │
    取后10个 → action_out
    (1,10,1024)→Linear→(1,10,32)
                    │
    去噪×1步
    (1,10,32)
                    │
Phase 5: 后处理      │
    去 padding: (1,10,7)
    反归一化:  (1,10,7)
    取前5步:   action_plan
    popleft:   (1,7) → numpy
                    │
输出
  action: np.float32 (1, 7)
```

---

## 模型参数量级

| 组件 | 层数 | Hidden | 参数量级 |
|------|------|--------|---------|
| SigLIP Vision Tower | 27 | 1152 | ~400M |
| PaliGemma Gemma | 18 | 2048 | ~2B |
| S2 Gemma Expert | 18 | 1024 | ~500M |
| DINOv2-Small | 12 | 384 | ~22M |
| S1 multi_modal_projector | 1 | 384→1024 | ~0.4M |
| S1 Gemma Expert | 8 | 1024 | ~200M |
| VQH Gemma Expert | 4 | 2048 | ~200M |
| CalQL Critics (×2) | MLP | 256→256→1 | ~0.3M |
| CalQL Actor | MLP | 256→256 | ~0.2M |

**总计:** 约 ~3.3B 参数 (其中 SigLIP + PaliGemma 共享约 2.4B)
