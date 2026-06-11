<!-- arxiv: 2605.19342 -->
<!-- venue: ICML 2026 -->
<!-- tags: VLA, 视觉推理, 强化学习 -->

# SLVR: Semantic-Enriched Latent Visual Reasoning

> **论文信息**
> - 作者：Tianrun Xu, Yue Sun, Qixun Wang, Jingyi Lu, Yuan Wang, Tianren Zhang, Longteng Guo, Fengyun Rao, Jing LYU, Feng Chen†, Jing Liu†
> - 通讯作者：Feng Chen (chenfeng@mail.tsinghua.edu.cn), Jing Liu (jliu@nlpr.ia.ac.cn)
> - 单位：清华大学自动化系 & 电子系、中关村人工智能学院、中国农业大学、北京大学、北京理工大学、中科院自动化所、腾讯微信视觉
> - 发表：ICML 2026
> - arXiv ID：2605.19342
> - 代码：https://github.com/tinnel123666888/slvr
> - 数据集：SLV-Set（[tinnel123/slv-set](https://huggingface.co/datasets/tinnel123/slv-set)）、SV-QA（[tinnel123/sv-qa](https://huggingface.co/datasets/tinnel123/sv-qa)）

---

## 一、核心问题

现有的多模态潜在空间推理（latent visual reasoning）方法，如 LVR（Latent Visual Reasoning），试图用紧凑的潜在表征替代显式的"看图思考"（thinking with images）过程。但这些方法存在两个根本性问题：

1. **语义贫乏**：LVR 仅依赖视觉特征重建作为监督信号（MSE alignment to vision encoder features），学到的 latent 主要编码外观级（appearance-level）信息，缺乏对物体属性、状态、关系等细粒度语义的显式建模。
2. **单一查询视角**：现有方法的 latent 训练由单一任务驱动，没有统一的机制让同一个 rich latent 支持同一区域上的多种推理粒度。当同一区域被从不同语义角度提问时，单查询训练的 latent 无法稳定支持所有回答。

![图1：三种视觉推理范式对比](assets/slvr/figure1_n.png)

*图1：三种视觉推理范式对比。(a) 显式推理或裁剪证据；(b) 仅视觉监督的 latent 推理；(c) SLVR：视觉 + 语义双重监督，跨问题对比对齐。*

---

## 二、核心思路 / 方法

SLVR 提出一个**两阶段学习框架**，核心思想：先让 latent 编码丰富的属性级语义，再通过多查询优化对齐 latent 与多样的推理目标。

### 2.1 Stage 1：结构化 Latent 学习（Structured Latent Learning）

![图2：SLVR 框架概览（Stage 1 & Stage 2）](assets/slvr/figure2.png)

*图2：SLVR 框架概览。(a) Stage 1 结构化 latent 学习；(b) Stage 2 M-GRPO 多查询优化。*

**输入与 Token 构造：**

给定图像 $I$ 和问题 $q$，通过 vision encoder 编码后，用 bounding box 指定的 ROI token 序列被包裹在 `<|vision_start|>` 和 `<|vision_end|>` 之间，形成**区域视觉 latent** $\mathcal{H}_{vis} = \{\mathbf{h}^{lat}_t\}_{t=1}^{T_v}$。

在 `<|vision_end|>` 之后插入特殊 token `<sem>`，其隐藏状态作为**语义 latent** $\mathbf{z}_{sem}$，用于聚合区域级语义信息。

**三个监督信号：**

| 损失 | 目标 | 公式 |
|------|------|------|
| $\mathcal{L}_{vis}$ | 区域视觉 latent 对齐 vision encoder 特征 | $\sum_{t=1}^{T_v} \| \mathbf{h}^{lat}_t - \mathbf{v}^{enc}_t \|_2^2$ |
| $\mathcal{L}_{sem}$ | 语义 latent 对齐属性语义嵌入 | $\text{MSE}(W\mathbf{z}_{sem}, \mathbf{e})$，其中 $\mathbf{e} \in \mathbb{R}^{4096}$ 是 Qwen3 embedding 编码的属性描述 |
| $\mathcal{L}_{ans}$ | 答案监督（标准 LM loss） | 自回归交叉熵 |

总目标：$\mathcal{L}_{stage1} = \mathcal{L}_{vis} + \mathcal{L}_{sem}$

**关键设计（代码印证）：**

```python
# 特殊 token 定义（src/constants.py）
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
SEM_TOKEN = "<|sem|>"
SEM_END_TOKEN = "</sem>"

# 语义投影头（src/model/slvr_heads.py）
class SLVRTextHead(nn.Module):
    # 将 hidden_size → 4096 维语义嵌入空间
    # 结构：LayerNorm → Linear → GELU → Linear
```

### 2.2 Stage 2：多查询分组相对策略优化（M-GRPO）

Stage 2 的核心是 **M-GRPO**（Multi-query Group Relative Policy Optimization），继承自 DeepSeek-R1 的 GRPO 框架，但针对多查询场景做了扩展。

**核心思想：** 对同一区域 $r$ 的两个语义不同的问题 $q_1, q_2$，模型分别产生 latent 对 $(H_{vis}^{(i)}, z_{sem}^{(i)})$，M-GRPO 通过三项奖励联合优化——

| 奖励项 | 作用 | 公式 |
|--------|------|------|
| $\mathcal{R}_{ans}^{(i)}$ | 答案正确性（LLM judge 评估） | $\mathbb{I}(\hat{y}_i = y_i)$ |
| $\mathcal{R}_{cons}$ | **跨查询 latent 一致性**（同时约束视觉和语义 latent） | $-\sum_{i \neq j} \left( \lambda_{sem}\|z_{sem}^{(i)} - z_{sem}^{(j)}\|_2 + \lambda_{vis}\frac{1}{T_v}\sum_t \|h_t^{(i)} - h_t^{(j)}\|_2 \right)$ |
| $\mathcal{R}_{stab}^{(i)}$ | **稳定性正则化**（防止偏离 Stage 1 分布） | $-\max(0, \|z_{sem}^{(i)} - \bar{z}_{sem}\|_2 - \tau_{sem}) - \max(0, \frac{1}{T_v}\sum_t \|h_t^{(i)} - \bar{h}_t\|_2 - \tau_{vis})$ |

**GRPO 更新公式：**

$$\mathcal{J}_{\text{M-GRPO}}(\theta) = \mathbb{E}_{(I,r), (q_1,q_2), o \sim \pi_{\theta_{old}}} \left[ \frac{1}{2}\sum_{i=1}^{2} \frac{1}{G}\sum_{g=1}^{G} \frac{1}{|O_{i,g}|}\sum_t \min(\rho_{i,g,t} \hat{A}_{i,g,t}, \text{clip}(\rho_{i,g,t}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,g,t}) - \beta D_{KL}(\pi_\theta \| \pi_{\theta_{old}}) \right]$$

**代码印证（src/params.py）：**

```python
# M-GRPO 关键超参数
lambda_sem: float = 0.01    # 语义 latent 一致性权重
lambda_vis: float = 0.01    # 视觉 latent 一致性权重
tau_sem: float = 1.0        # 语义 latent 稳定容忍度
tau_vis: float = 1.0        # 视觉 latent 稳定容忍度
lambda_consistency: float = 0.05   # 整体一致性奖励
lambda_stability: float = 0.05     # 整体稳定性奖励
```

**Judge 机制（src/train/mgrpo_reward_funcs.py）：**
- 使用 Qwen3-Max 作为 LLM judge，通过 vLLM 推理服务
- 后台线程定期从 broker 刷新 IP 池，ThreadPoolExecutor 并发调用
- 支持 judge 不可用时自动回退到字符串匹配
- 正确答案性判断：语义等价性比较（支持字母选项 / 文本答案 / 数字匹配）

---

## 三、数据集构建

### 3.1 SLV-Set

基于 Visual-CoT（ViSCoT）数据集构建，包含两个互补组件：

| 组件 | 规模 | 内容 | 用途 |
|------|------|------|------|
| 属性级语义数据集 | ~400K 区域描述 | Qwen3-VL-235B 生成的结构化区域语义档案（实体、颜色、形状、材质、动作、交互、空间关系、文字内容） | Stage 1 语义监督 |
| 多查询数据集 | ~800K QA 对 | 每个区域关联 2-4 个从不同语义角度提问的 QA 对 | Stage 2 M-GRPO 训练 |

![图3：数据集构建示意](assets/slvr/dataset_region_multiq.png)

*图3：数据集构建流程。(a) 区域语义档案构建（属性级标注）；(b) 多查询问题生成。*

**质量验证：** 人工后验检查全部 SLV-Set，发现 29,457 条错误标注（错误率 7.29%）。主要错误类型：颜色错配（19.94%）、动作/姿态错误（14.55%）、空间关系错误（13.28%）、幻觉属性（11.80%）。

每个样本格式：$(q, i, b, \mathcal{A}, \mathbf{e})$ — 问题、图像、边界框、属性集、4096 维语义嵌入。

### 3.2 SV-QA 评估基准

基于 V*、HRBench-4K、HRBench-8K 构建，每个区域从原始问题 $q_1$ 和生成的 $q_2$ 两个语义维度提问。**591 对样本**，经人工审查修正（19 题与原始题过度重叠、8 题含幻觉内容）。

---

## 四、代码架构

### 4.1 项目结构

```
slvr/
├── src/
│   ├── model/
│   │   ├── qwen_slvr_model.py      # QwenWithSLVR 模型定义（含特殊 token embedding）
│   │   ├── qwen_slvr_model_ori.py  # 原始版本
│   │   └── slvr_heads.py           # SLVRHead（视觉）/ SLVRTextHead（语义）/ SLVRHeadGLU
│   ├── train/
│   │   ├── train_sft.py            # Stage 1 SFT 训练入口
│   │   ├── train_mgrpo.py          # Stage 2 M-GRPO 训练入口
│   │   ├── train_grpo.py           # 标准 GRPO 训练入口（消融用）
│   │   ├── mgrpo_reward_funcs.py   # M-GRPO 奖励函数（accuracy + format）
│   │   ├── monkey_patch_forward_slvr.py       # 训练前向 monkey-patch（核心）
│   │   ├── monkey_patch_forward_slvr_rl.py    # RL 专用 monkey-patch
│   │   └── preflight_mgrpo_infer.py           # M-GRPO 推理预检
│   ├── dataset/                    # 数据集加载（SFT / M-GRPO）
│   ├── trainer/
│   │   └── mgrpo_trainer.py       # M-GRPO Trainer（含 consistency/stability 奖励计算）
│   ├── slvr_utils.py              # Bbox → Token 索引映射（QwenVLBboxTokenMapper）
│   ├── params.py                   # 训练参数定义（含 M-GRPO 超参数）
│   └── constants.py                # 特殊 token 定义
├── scripts/
│   ├── finetune_slvr_stage1_7b_viscot.sh    # Stage 1 训练脚本
│   └── finetune_slvr_stage2_7b_mgrpo_viscot.sh  # Stage 2 训练脚本
├── inf_batch_dir_old.py            # 批量推理脚本
├── environment.yaml                # Conda 环境
└── requirements.txt
```

### 4.2 关键技术细节

**Bounding Box → Token 索引映射（`QwenVLBboxTokenMapper`）：**
- 基于 Qwen 2.5 VL 的 vision tower 参数：patch_size=14, spatial_merge_size=2
- 动态计算每张图像的 token grid，支持 xyxy / xywh 两种 bbox 格式
- 自动归一化 bbox 坐标，提供双向转换（bbox → indices / indices → bbox）
- 注意：bbox 标注时维度可能与 Qwen 图像预处理后的实际维度不一致（resize），代码注释提醒使用者**假设 bbox 是归一化的**

**前向传播 Monkey-Patch（`monkey_patch_forward_slvr.py`，~121K，代码库最大文件）：**
- 替换 Qwen2.5-VL 的 `Qwen2_5_VLForConditionalGeneration` 前向传播
- 支持 3 种模式：`inference_mode=False, rl=False`（Stage 1 SFT）、`inference_mode=False, rl=True`（Stage 2 M-GRPO）、`inference_mode=True`（推理）
- 在 hidden states 中提取 `<|vision_start|>` 到 `<|vision_end|>` 之间的视觉 latent，以及 `<sem>` 位置的语义 latent

**M-GRPO Trainer（`mgrpo_trainer.py`）：**
- 继承自 TRL 的 GRPOTrainer
- 每个 sample 包含两个问题（q1, q2）的 completion
- `mgrpo_reward_funcs.py` 中 accuracy_reward 返回 q1 和 q2 答案正确性的加权组合（默认 q1 权重 0.7, q2 权重 0.3）
- format_reward 检查输出格式：`<|vision_start|>[visual latents]<|vision_end|><sem>[semantic latent]</sem><answer>...</answer>`
- 一致性奖励和稳定性奖励在 trainer 内部从 hidden states 计算（使用 L2 距离）

**推理（`inf_batch_dir_old.py`）：**
- 配置类 `Config`：MODEL_PATH, INPUT_DIR, OUTPUT_DIR, STEPS（latent reasoning 步数，默认 8）
- `DECODING_STRATEGY = "latent"` 启用 SLVR 式推理
- 自动检测可用 GPU，OOM 时自动回退

---

## 五、实验结果

### 5.1 标准 VQA 基准

| 模型 | 推理范式 | OKVQA | GQA | VizWiz | ChartQA | TextVQA | AI2D |
|------|---------|-------|-----|--------|---------|---------|------|
| Qwen2.5-VL-7B | 纯文本 | 58.9 | 53.2 | 54.1 | 74.4 | 79.1 | 69.5 |
| LVR | 视觉 latent | 50.6 | **57.4** | 33.1 | 64.4 | 75.1 | **77.3** |
| **SLVR-7B** | 视觉 + 语义 latent | **61.8** | 55.6 | **57.8** | **77.2** | **79.3** | 76.0 |
| *Gain over LVR* | | *+11.2* | *-1.8* | *+24.7* | *+12.8* | *+4.2* | *-1.3* |

**关键发现：** SLVR 在 OKVQA（+11.2）、VizWiz（+24.7）、ChartQA（+12.8）上对 LVR 提升显著，说明语义 latent 在这些需要细粒度语义理解的场景中作用突出。

### 5.2 SV-QA 基准（核心实验）

| 设置 | V* Q1 | V* Q2 | V* Both | HRBench-4K Both | HRBench-8K Both |
|------|-------|-------|---------|------------------|------------------|
| Qwen2.5-VL | 76.4 | 55.5 | 44.0 | 45.8 | 37.5 |
| DeepEyes（显式裁剪） | **83.3** | 79.1 | **70.2** | 60.6 | 49.0 |
| LVR | 81.7 | 77.5 | 65.4 | 57.9 | 46.6 |
| **SLVR-7B** | 82.2 | **80.1** | 69.1 | **61.1** | **50.6** |
| *Gain over LVR* | *+0.5* | *+2.6* | *+3.7* | *+3.2* | *+4.0* |

**关键发现：**
- SLVR 在 Both Correct（两个问题都答对）指标上全面超越 LVR（+3.2 ~ +4.0），证明语义 latent 能同时编码多种语义维度
- 与其他 method 对比：纯文本方法（Qwen2.5-VL）在 Q2 上性能暴跌（Q1 76.4 → Q2 55.5），说明文本推理无法稳定支持语义变化
- DeepEyes 在 V* Q1 上最高（83.3），但这是通过**推理时显式图像裁剪**实现的，计算开销巨大；SLVR 完全通过 latent 推理实现 competitive 的 Q1 性能

### 5.3 VisualPuzzles 推理能力评估

| 模型 | Algorithmic | Analogical | Deductive | Inductive | Spatial | Overall |
|------|-------------|------------|-----------|-----------|---------|---------|
| Qwen2.5-VL | 35.9 | 26.1 | 35.5 | **28.7** | 21.3 | 29.2 |
| LVR | 31.3 | 25.6 | 40.5 | 24.4 | 26.2 | 29.4 |
| **SLVR** | **37.4** | **28.0** | **45.5** | 26.3 | **33.6** | **34.2** |

**关键发现：** SLVR 在 Deductive（+5.0）和 Spatial（+7.4）推理类别上大幅超越 LVR，证明属性级语义监督帮助模型捕获了结构化关系和空间线索，这些是纯视觉监督 latent 所遗漏的。

### 5.4 消融实验

| 设置 | 组件 | V* Both | HRBench-4K Both | HRBench-8K Both |
|------|------|---------|------------------|------------------|
| **文本基线** | | | | |
| SFT (single-Q) | — | 51.8 | 52.8 | 43.5 |
| SFT (multi-Q) | +多查询数据 | 57.1 | 50.1 | 41.1 |
| SFT + GRPO (multi-Q) | +强化学习 | 58.6 | 55.5 | 47.8 |
| **Latent 推理** | | | | |
| LVR | 基线 | 65.4 | 57.9 | 46.6 |
| + Stage 1 | +语义 latent 监督 | 67.5 | 59.6 | 46.9 |
| + GRPO (Single-Q) | +单查询 GRPO | 62.8 | 57.6 | 49.3 |
| + Multi-Q | +多查询 GRPO（无 M-GRPO） | 68.1 | 60.3 | 49.9 |
| **SLVR-7B (Full)** | +M-GRPO 显式一致性约束 | **69.1** | **61.1** | **50.6** |

**核心消融结论：**

1. **文本 vs. Latent**：即使是数据量相同的最强文本基线（SFT + GRPO multi-Q），Both Correct 也全面落后 latent 推理方法（V* 58.6 vs. 69.1），说明文本链式思维在"同一区域多角度提问"场景中难以维持稳定 grounding。
2. **Stage 1 语义监督**：加入语义 latent 后，V* Both 从 65.4 提升到 67.5，验证属性级语义信息有助于 joint correctness。
3. **多查询 vs. 单查询 GRPO**：Multi-Q（多查询数据 + 标准 GRPO）比 Single-Q GRPO 提升了 Both Correct，说明更多样的训练信号本身有益。
4. **M-GRPO 一致性约束是关键**：M-GRPO（显式 latent 一致性约束）在 Multi-Q 基础上进一步提升，证明显式对齐同一区域上的 cross-query latent 对语义一致性至关重要。

---

## 六、关键洞察与讨论

### 6.1 为什么需要语义 latent 而不只是视觉 latent？

LVR 的视觉 latent 通过直接对齐 vision encoder 特征来保留视觉信息，但 vision encoder 的 patch 特征本身不显式编码"这是红色"、"这个人在跑"等属性语义——这些语义分散在深层特征中，缺乏显式的结构化引导。SLVR 引入一个额外的 `<sem>` token 作为**语义聚合点**，直接以结构化属性文本的嵌入作为监督，迫使模型将分散的视觉语义压缩到该 token 的 hidden state 中。

### 6.2 M-GRPO 的 motivation

Stage 1 虽然给单个 latent 编码了语义信息，但没有保证**在不同问题的扰动下该语义能稳定激活**。M-GRPO 的 latent consistency reward 直接对"两个不同问题产生不同 latent"这种 drift 进行惩罚，保证同一区域的核心语义表征不受问题表述方式的影响，而只保留 task-specific 的微小差异。

### 6.3 与 DeepEyes 等"看图思考"方法的关系

DeepEyes 通过推理时反复裁剪区域图像来获取高分辨率视觉证据，准确率高但计算开销大。SLVR 通过 latent 学习一次性地将区域信息压缩为 compact 表征，推理时无需显式图像操作。两者不是互斥的——理想情况下，semantic-enriched latent 可以作为 DeepEyes 等方法的"内化"版本，减少裁剪迭代次数。

### 6.4 局限性

- 语义监督依赖 Qwen3-VL-235B 生成属性标注，存在 7.29% 的错误率（颜色错配、动作/姿态错误等），语义 latent 的上限受制于属性标注质量
- 语义 latent 通过 L2 对齐 Qwen3 embedding 来学习，没有显式验证学到的 latent 是否保留了可解释的语义维度
- 实验在 Qwen2.5-VL-7B 上进行，更大规模模型的 scaling behavior 未知
- M-GRPO 的两问题联合优化目前限制在 q1 + q2 组内，扩展到 3+ 问题可能需要新的 group 构建策略

---

## 七、总结

SLVR 通过两阶段训练为 latent visual reasoning 引入**细粒度属性级语义监督**和**多查询一致性对齐**：

- **Stage 1**：在 LVR 的视觉 latent 重建基础上，新增 `<sem>` 语义 latent 和属性语义对齐损失，使 latent 编码结构化语义
- **Stage 2**：设计 M-GRPO，同时优化同一区域两个语义不同问题的答案正确性、latent 一致性和分布稳定性
- 配套构建 SLV-Set（~400K 属性 + 800K QA 对）和 SV-QA（591 对多角度问题）数据集
- 在 SV-QA 的 Both Correct 指标上全面超越 LVR（+3.2 ~ +4.0），在 VisualPuzzles 的 Deductive 推理上超越 LVR 5 个百分点
