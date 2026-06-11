<!-- arxiv: 2605.22012 -->
<!-- venue: 投稿中 (under review) -->
<!-- tags: 全模态, 多模态理解, 链式思考 -->

# LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning

> **论文信息**
> - 作者：Yifan Dai（SJTU & 快手 Kling）、Zhenhua Wu、Bohan Zeng、Daili Hua、Jialing Liu、Bozhou Li、Yuran Wang、Chengzhuo Tong、Hao Liang、Xiaochen Ma、Junbo Niu、Tianyu Guo、Yang Shi、Yue Ding、Yiyan Ji、Bingyin Mei、Yushuo Guan、Yuanxing Zhang、Pengfei Wan、Fangcheng Fu、Wentao Zhang
> - 机构：上海交大 AI 学院、快手 Kling 团队、北京大学、港科大、中科院自动化所、南京大学、中国人民大学、清华大学
> - arXiv ID：2605.22012
> - 基座模型：Qwen2.5-Omni-7B
> - 代码：未开源

---

## 一、核心问题

当前多模态大模型（MLLM）在音视频联合推理任务上表现不佳。根本原因在于：**主流方法将连续的音频和视觉信号压缩为离散文本 token 进行链式思考（Chain-of-Thought, CoT）**，这一"文本瓶颈"导致两个严重后果：

1. **信息损失（Information Loss）**：高维连续的音视频信号被强制映射到离散词汇空间，时序对齐的细粒度信息（如同步的视听觉线索）在文本化过程中被丢失。
2. **语言绑定现象（Language-Bound Phenomenon）**：模型在推理过程中过度依赖语言先验，对原始音视频输入的注意力逐渐减弱，造成"感官脱离"和幻觉。

> 直接表现：随着推理链加长，纯文本 CoT 对原始音视频信号的注意力持续下降，模型回答更多依赖语言统计规律而非真实的感官证据。

![图1：LatentOmni 与 Explicit Text CoT 基线的对比](assets/latentomni/fig1_new.jpg)

*图1：LatentOmni 与 Explicit Text CoT 基线的定性+定量对比。**左侧**：针对同一个音视频推理问题，基线模型错误地关注了不相关的视觉区域（低注意力），而 LatentOmni 准确锁定了关键的跨模态线索（热力图深色区域），给出了正确答案。**右侧**：在 Daily-Omni 基准的不同任务类型上，LatentOmni 始终保持显著更高的音视频 token 注意力占比（AV Token Attention Ratio），尤其在需要精细对齐的 AV Alignment 任务上优势最为明显，验证了隐空间推理有效防止了"语言绑定"现象。*

---

## 二、核心思路

LatentOmni 的核心洞察是：**连续隐空间（latent space）比离散文本更适合承载和保留音视频感官信息**。与其把所有中间推理都写成文本，不如让模型在需要回顾音视频证据时，直接在连续隐空间中生成隐式推理状态。

具体做法：
- **文本 + 隐空间交替推理**：推理轨迹由文本 token（负责高层逻辑框架）和连续隐状态（负责密集音视频证据的保留和整合）交替组成。
- 当模型需要重新审视音视频信息时，生成特殊 token `<Unified_Latent>` 切换到连续隐空间，生成 K 个隐向量后再用 `</Unified_Latent>` 切回文本生成。
- 通过 **特征级监督** 和 **时序对齐的位置编码** 确保隐状态忠实于原始感官信号。

![图2：LatentOmni 整体架构](assets/latentomni/model_overview_2.jpg)

*图2：LatentOmni 整体框架。**左侧**为推理流程：模型在文本生成和隐空间推理之间交替——当遇到 `<Unified_Latent>` 触发 token 时，从离散词汇空间切换到连续隐空间，自回归生成 K 个隐状态（K_v 个视觉 + K_a 个音频），然后恢复到文本生成。**右侧**为训练目标：联合优化三个损失——文本预测损失（$\mathcal{L}_{\text{text}}$）保持语言能力、隐空间对齐损失（$\mathcal{L}_{\text{latent}}$）将隐状态锚定到感官特征、时序同步损失（$\mathcal{L}_{\text{sync}}$）对齐跨模态时序。*

---

## 三、方法详解

### 3.1 音视频隐空间推理（Audio-Visual Latent Reasoning）

LatentOmni 的推理轨迹是一个**文本与隐状态交替的混合序列**：

$$S = \left[ w_{1:i}, u, z_{1:K}, u', w_{i+1:j}, u, z_{K+1:2K}, u', \dots, a \right]$$

其中：
- $w$ — 标准文本 token（离散）
- $u$ / $u'$ — `<Unified_Latent>` 触发 / 停止 token
- $z_k \in \mathbb{R}^d$ — 连续隐推理状态（transformer 最后一层输出，不经 LM head 映射回词汇表）
- $a$ — 最终答案

每个隐状态 $z_k$ 被直接作为下一个位置的输入 embedding 向前传播，形成**完全连续**的隐推理轨迹。前 $K_v$ 个位置分配给视觉隐状态，后 $K_a$ 个分配给音频隐状态（默认 $K=40$，$K_v=32$，$K_a=8$）。

**设计直觉**：文本提供逻辑骨架（"发生了什么→为什么→答案是什么"），隐空间负责在关键证据节点保留和整合密集的多模态信息——两者各司其职，互不替代。

### 3.2 Omni-Sync 位置编码（OSPE）

问题：视觉和音频隐状态是**依次生成**的（先 $K_v$ 个视觉，再 $K_a$ 个音频），属于同一时刻的视听特征可能在位置上被拉开，导致时序对齐丢失。

解决：OSPE 将 Qwen2.5-Omni 的 time-aligned multimodal RoPE 扩展到统一隐空间：

$$\operatorname{OSPE}(h, t) = h \odot \cos(t \Theta) + \mathcal{R}(h) \odot \sin(t \Theta)$$

- $t$ — 共享的物理时间戳（同一时刻的音视频帧使用相同的 $t$）
- $\Theta$ — 预定义的基频向量
- $\mathcal{R}(\cdot)$ — 在相邻维度上的块对角旋转矩阵

效果：即使视觉和音频隐状态在序列中位置不同，OSPE 通过注入同步的相对位置先验，使得同一时刻的音视频特征在注意力计算中保持对齐。

### 3.3 数据集构建：LatentOmni-Instruct-35K

训练隐空间推理需要一种特殊监督信号：**标注了每一步应该参考哪个音视频片段**的推理轨迹。现有数据集（通常是粗粒度的 QA 对）缺少这种片段级对齐。因此作者设计了三阶段数据合成 pipeline：

![图3：数据构建流水线](assets/latentomni/data.jpg)

*图3：LatentOmni-Instruct-35K 数据构建三阶段流水线。**阶段 1**：从两个时间对齐的高质量音视频 caption 数据集（ASID-1M 和 AVoCaDO）出发，用 Qwen3-235B 将跨模态 caption 转化为符合严格跨模态依赖要求的 AVQA 对，再用 GLM-4.7 对每对 QA 进行三维度评分（难度、逻辑严谨性、模态依赖度，各 1-5 分），总分低于 13 分或类别不平衡超过 3× 的样本被丢弃。**阶段 2**：按时间戳切分原始音视频流，用 Qwen3-30B-Captioner 分别生成视觉和音频的片段级描述，再由 GLM-4.7 过滤幻觉、修复镜头碎片化、重新对齐时序。**阶段 3**：将 AVQA 对与片段级 caption 组合，由 GLM-4.7 生成包含显式片段引用标记的推理轨迹，Gemini-2.5-Flash 审核修正引用错误和冗余分支，最终替换标记为实际音视频片段，得到 35K 样本。*

**数据特点**：
- 每条轨迹要求至少引用 1 个、最多 3 个音视频片段
- 推理过程明确要求从"感官证据"出发（prompt 禁止提及"caption"、"text"等词）
- 包含 10 种推理类型：视听联合感知、动作识别、空间布局理解、时序理解、属性对比、计数量化、情感氛围感知、语义摘要、逻辑关系推理、意图预测

### 3.4 训练目标

LatentOmni 使用三个互补的损失函数联合优化：

**（1）文本预测损失 $\mathcal{L}_{\text{text}}$**

标准自回归交叉熵，仅在离散 token（文本推理步骤 $w$、触发 token $u$、答案 $a$）上计算，保持模型的语言生成能力：

$$\mathcal{L}_{\text{text}} = - \frac{1}{N_{\text{text}}} \sum_{t=1}^{L} \mathbb{I}(s_t \in \mathcal{V}) \log p(s_t \mid S_{<t}, H^v, H^a, H^q)$$

**（2）隐空间对齐损失 $\mathcal{L}_{\text{latent}}$**

核心创新。从标注的音视频片段中提取特征（使用模型自身的视觉/音频编码器），通过无参数的 L2-norm 加权池化压缩为锚点序列 $A = [a_1, \dots, a_K]$，然后用 MSE 将生成的隐状态对齐到锚点：

$$\mathcal{L}_{\text{latent}} = \frac{1}{K} \sum_{k=1}^{K} \left\| z_k - a_k \right\|^2_2$$

> 这是防止"语言绑定"的关键机制：它迫使隐状态 $z_k$ 直接重建原始音视频特征，确保隐空间中的推理忠实于感官证据。

**（3）时序同步损失 $\mathcal{L}_{\text{sync}}$**

对称 InfoNCE 对比损失，对同一时刻 $t$ 的视觉特征 $h_t^v$ 和音频特征 $h_t^a$ 拉近，不同时刻的推开：

$$\mathcal{L}_{\text{sync}} = - \frac{1}{2 |\mathcal{T}|} \sum_{t \in \mathcal{T}} \left( \log \frac{\exp(\operatorname{sim}(h_t^v, h_t^a) / \tau)}{\sum_{t'} \exp(\operatorname{sim}(h_t^v, h_{t'}^a) / \tau)} + \log \frac{\exp(\operatorname{sim}(h_t^a, h_t^v) / \tau)}{\sum_{t'} \exp(\operatorname{sim}(h_t^a, h_{t'}^v) / \tau)} \right)$$

**最终联合目标**：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{text}} + \lambda_1 \mathcal{L}_{\text{latent}} + \lambda_2 \mathcal{L}_{\text{sync}}$$

其中 $\lambda_1 = 0.005$，$\lambda_2 = 1.0$。

**训练配置**：
- 基座模型：Qwen2.5-Omni-7B
- 训练步数：750 steps（2 epochs）
- 学习率：$10^{-5}$，warmup 比例 0.05
- Batch size：1，梯度累积 12 步
- 最大帧数/样本：256
- 隐 token 数：固定 40（32 视觉 + 8 音频）

---

## 四、实验与结果

### 4.1 实验设置

**四个评测基准**（覆盖不同难度的音视频联合推理）：
| 基准 | 侧重点 | 特点 |
|------|--------|------|
| Daily-Omni | 日常场景推理 | 常见事件理解 |
| WorldSense | 物理/时空常识 | 结构化常识推理 |
| OmniVideoBench | 跨模态对齐与问答 | 细粒度音频类型和视频时长分片 |
| LVOmniBench | 长时多感官理解 | 持续长输入推理 |

**对照基线**：
- **Qwen2.5-Omni-7B**：基座模型（零样本推理）
- **Explicit Text CoT**：移除 LatentOmni-Instruct-35K 中所有音视频片段，仅用纯文本推理轨迹微调
- **Vanilla SFT**：直接用 LatentOmni-Instruct-35K 做标准指令微调（不使用隐空间推理）
- 开源多模态模型：VideoLLaMA2-7B、MiniCPM-o-7B、VITA-1.5-7B、HumanOmniV2-7B、Baichuan-Omni-1.5、OmniVinci
- 闭源模型：GPT-4o、Gemini-2.0-Flash、Gemini-2.5-Pro（作为参照）
- 隐推理方法：Monet、LVR（在 VideoMME 视觉子集上对比）

### 4.2 主要结果

| 方法 | Daily-Omni | WorldSense | OmniVideoBench | LVOmniBench |
|------|:----------:|:----------:|:--------------:|:-----------:|
| VideoLLaMA2-7B | 35.2 | 25.4 | 29.2 | 27.0 |
| Qwen2.5-Omni-7B | 62.9 | 45.4 | 29.3 | 32.0 |
| + Explicit Text CoT | 65.6 | 46.6 | 33.2 | 32.1 |
| + Vanilla SFT | 62.0 | 47.5 | 30.5 | 33.2 |
| **LatentOmni** | **67.4** | **48.9** | **35.4** | **35.1** |
| *Gemini-2.5-Pro* | *81.4* | *64.6* | *58.9* | *—* |

**关键发现**：

1. **相对基座模型**：LatentOmni 在四个基准上分别提升 +4.5pp、+3.5pp、+6.1pp、+3.1pp，OmniVideoBench 上提升最显著（+6.1pp），说明隐空间推理特别有利于需要精细跨模态对齐的任务。

2. **相对 Explicit Text CoT**：全面超越纯文本 CoT（+1.8pp ~ +3.0pp），且增益随任务难度增加而增大。这表明隐空间推理的优势不是来自额外的数据或训练，而是来自**在连续空间中保留音视频证据**这一根本设计。

3. **相对 Vanilla SFT**：直接 SFT 反而在 Daily-Omni 上降低了 0.9pp（62.9→62.0），说明缺少隐空间推理机制的普通微调无法利用 interleaved 数据的结构优势；LatentOmni 相比 Vanilla SFT 提升 +5.4pp，差距远超 Text CoT 的 +3.6pp。

4. **OmniVideoBench 细粒度分析**：

| 方法 | Music | Sound | Speech | (0,1]min | (10,30]min | 平均 |
|------|:-----:|:-----:|:------:|:--------:|:----------:|:----:|
| Qwen2.5-Omni-7B | 23.1 | 25.3 | 30.7 | 41.6 | 26.7 | 29.3 |
| + Explicit Text CoT | 30.0 | 32.0 | 33.9 | 39.4 | 30.7 | 33.2 |
| **LatentOmni** | **33.3** | 30.2 | **36.7** | **45.2** | **34.0** | **35.4** |

- 在 Music（+3.3pp）和 Speech（+2.8pp）类别上优势明显，表明隐空间对非视觉信息的保留效果好于文本化
- 在长视频（10-30min）上 LatentOmni 达 34.0%，远超 Text CoT 的 30.7%（+3.3pp），证明隐空间推理对长时序信息保留特别有效

5. **视觉隐推理对比**（VideoMME，纯视觉设定）：

| 方法 | Overall | Short | Medium | Long |
|------|:------:|:-----:|:------:|:----:|
| LVR | 36.7 | 39.2 | 36.6 | 34.3 |
| Monet | 51.6 | 52.9 | 56.0 | 46.0 |
| **LatentOmni** | **60.8** | **70.8** | **60.5** | **50.4** |

即使去掉音频，LatentOmni 的隐推理设计在纯视觉任务上也显著优于专门的视觉隐推理方法，说明其架构通用性。

### 4.3 消融实验

![图4：消融实验](assets/latentomni/ablation_benchmark_bars-2.jpg)

*图4：LatentOmni 隐空间配置消融实验。**左侧两组**：在 Daily-Omni、WorldSense、OmniVideoBench 三个基准上测试隐 token 总数（20/30/40/50）和音视频分配比例（32V+8A vs 其他组合）的影响——40 token（32V+8A）在所有基准上最优，少于 40 限制表征容量，多于 40 引入噪声。**右侧**：测试超参数 $\lambda_1$（$\mathcal{L}_{\text{latent}}$ 权重）和 $\lambda_2$（$\mathcal{L}_{\text{sync}}$ 权重）的敏感度——$\lambda_1$ 过低导致"语言绑定"回退，$\lambda_2$ 过高限制语义灵活性，最终选择 $\lambda_1=0.005, \lambda_2=1.0$ 的均衡配置。*

**组件消融**：

| 配置 | Daily-Omni | WorldSense | OmniVideoBench | LVOmniBench |
|------|:----------:|:----------:|:--------------:|:-----------:|
| LatentOmni（完整） | **67.4** | **48.9** | **35.4** | **35.1** |
| w/o Audio in Latent Space | 65.9 (−1.5) | 47.8 (−1.1) | 33.6 (−1.8) | 31.6 (−3.5) |
| w/o Visual in Latent Space | 63.5 (−3.9) | 47.2 (−1.7) | 33.5 (−1.9) | 32.1 (−3.0) |
| w/o OSPE | 66.0 (−1.4) | 47.8 (−1.1) | 34.9 (−0.5) | 33.1 (−2.0) |
| w/o $\mathcal{L}_{\text{latent}}$ | 61.0 (−6.4) | 45.2 (−3.7) | 31.8 (−3.6) | 30.2 (−4.9) |
| w/o $\mathcal{L}_{\text{sync}}$ | 65.9 (−1.5) | 47.1 (−1.8) | 34.0 (−1.4) | 33.1 (−2.0) |
| Qwen2.5-Omni-7B（基座） | 62.9 | 45.4 | 29.3 | 32.0 |

**关键发现**：
- $\mathcal{L}_{\text{latent}}$ 是最核心的组件，移除后性能暴跌（Daily-Omni 67.4→61.0），印证了特征级监督是防止"语言绑定"的主要机制
- 去掉音频隐状态在 LVOmniBench 上损失最大（−3.5pp），说明长时多感官理解特别依赖音频信息的隐空间保留
- OSPE 在所有基准上都有正向贡献，尤其在长时任务（LVOmniBench）上最明显（−2.0pp），验证了时序对齐对持续推理的重要性
- $\mathcal{L}_{\text{sync}}$ 提供互补增益——单独移除影响中等，但配合 $\mathcal{L}_{\text{latent}}$ 形成协同效果

**隐 token 配置消融**：
- Token 数从 20→40 逐步提升，40 达到峰值，50 反而下降（噪声引入）
- 最优分配比例：32 视觉 + 8 音频——视觉推理需要更大表征带宽，但少量专用音频 token 对跨模态对齐仍然关键

---

## 五、关键洞察与技术亮点

### 5.1 "文本瓶颈"的诊断与解决

论文系统地诊断了 MLLM 音视频推理失败的根因——**不是模型不够大，而是推理媒介不对**。通过注意力可视化（图1）直接证明纯文本 CoT 对音视频信号的注意力持续衰减，这比单纯的准确率下降更有说服力。

### 5.2 特征级隐空间监督（Feature-Level Supervision）

这是 LatentOmni 最精妙的设计。不同于之前工作（如 Coconut）仅通过语言建模目标隐式优化隐状态，LatentOmni 通过 $\mathcal{L}_{\text{latent}}$ 直接要求生成的隐状态**重建原始感官特征**。这相当于在隐空间中设置了一个"锚点"，强制模型在推理过程中不断回望原始证据。

### 5.3 文本与隐空间的混合推理

LatentOmni 不做"纯隐空间推理"（完全抛弃文本），也不做"纯文本 CoT"（完全抛弃连续信息）。通过交替机制，文本负责可解释的逻辑框架，隐空间负责信息密集的证据整合——这是一个务实的折中，既保留了 LLM 的语言推理先验，又克服了文本瓶颈。

### 5.4 实验设计的严谨性

三重对照（基座 vs Text CoT vs Vanilla SFT）的设计非常干净，排除了解释混淆：Vanilla SFT 的倒退（Daily-Omni 上 62.9→62.0）直接证明了"有 interleaved 数据但没有隐空间推理 = 浪费了数据结构"这一核心论点。

### 5.5 数据合成流水线的工程价值

35K 的高质量 interleaved 推理轨迹数据集的构建方法论（三阶段：QA 合成→片段 caption→轨迹合成）为后续工作提供了可复用的范式。特别是"禁止提及 caption/text"的 prompt 设计（将文本描述伪装成感官体验），是一个巧妙的工程技巧。

---

## 六、局限性

1. **模态覆盖范围有限**：当前仅支持视觉+音频+文本，未涉及 3D 空间表示、触觉、动作指令等物理交互信号。将更广泛的异构模态映射到统一隐空间仍是开放挑战。
2. **固定隐 token 数的限制**：当前使用固定 40 个隐 token，可能需要根据不同问题复杂度自适应调整。
3. **训练规模有限**：仅在 35K 数据上微调 2 个 epoch，更大规模数据上的 Scaling 行为未探索。
4. **隐状态不可解释**：连续隐状态无法像文本 CoT 那样直接阅读，调试和可信度评估存在挑战。
5. **与闭源模型的差距**：尽管在开源模型中达到最优，但与 Gemini-2.5-Pro（81.4 vs 67.4 在 Daily-Omni）仍有明显差距。

---

## 七、关键概念速查

| 概念 | 说明 |
|------|------|
| **Language-Bound Phenomenon** | 文本 CoT 导致模型过度依赖语言先验，忽视原始感官信号的倾向 |
| **Unified Latent Space** | 音视频特征和文本 embedding 共享的连续高维空间 $\mathbb{R}^d$ |
| **OSPE** | Omni-Sync Position Embedding，通过共享物理时间戳的 RoPE 对齐跨模态隐状态 |
| **$\mathcal{L}_{\text{latent}}$** | 隐空间对齐损失，用 MSE 将生成隐状态对齐到池化后的音视频特征锚点 |
| **$\mathcal{L}_{\text{sync}}$** | 时序同步损失，对称 InfoNCE 对比损失拉近同时刻的音视频特征 |
| **LatentOmni-Instruct-35K** | 包含音视频片段引用标注的 interleaved 推理轨迹数据集 |
| **混合推理序列 S** | 文本 token 与连续隐状态交替的推理轨迹 |
| **K=40, K_v=32, K_a=8** | 默认隐 token 配置（总数 40，视觉 32，音频 8） |

