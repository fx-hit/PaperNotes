<!-- arxiv: 2602.22766 -->
<!-- venue: ICML 2026 -->
<!-- tags: 视觉推理, MLLM, 因果分析 -->

# Imagination Helps Visual Reasoning, But Not Yet in Latent Space

> **论文信息**
> - 作者：You Li（北京交通大学）、Chi Chen（清华大学，通讯）、Yanghao Li（清华大学，通讯）、Fanhu Zeng（清华大学）、Kaiyu Huang（北京交通大学）、Jinan Xu（北京交通大学）、Maosong Sun（清华大学）
> - 机构：北京交通大学计算机学院、清华大学
> - 投稿方向：ICML 2026（accepted）
> - arXiv ID：2602.22766v2
> - 代码：https://github.com/AI9Stars/CapImagine

---

## 一、核心问题

多模态大模型（MLLM）的视觉推理近年来备受关注。随着任务复杂度提升，研究者开始探索让模型在推理过程中"主动感知"视觉内容，而非仅在文本空间被动编码图像特征。

**Latent Visual Reasoning（LVR，潜在视觉推理）** 试图模仿人类的"想象"过程——模型在隐空间中生成若干 latent token，期望这些 token 编码丰富的视觉语义，从而辅助模型在不显式生成图像的情况下完成视觉推理。然而，这些 latent token 内部到底发生了什么、是否真正承载了视觉想象，至今缺乏系统性的理解。

本文通过**因果中介分析（Causal Mediation Analysis）** 框架，系统考察了 latent token 在视觉推理过程中的实际作用，发现了两个关键的因果断裂，进而提出了一种直接替代方案：**在文本空间进行显式想象**。

---

## 二、核心方法：因果中介分析 + CapImagine

### 2.1 问题形式化

论文将 latent visual reasoning 建模为一条因果链：

$$X \rightarrow Z \rightarrow Y$$

其中：
- $X$：输入（图像 + 问题）
- $Z$：中间 latent token（"想象"的载体）
- $Y$：最终答案

论文对 $X \rightarrow Z$（输入到 latent）和 $Z \rightarrow Y$（latent 到答案）两个环节分别进行了系统性干预和探测分析。

![图1：视觉推理范式对比](assets/capimagine/image1.jpg)

*图1：三种视觉推理范式对比。**(a)** 工具辅助推理——模型通过 zoom-in 或绘图等预定义工具函数感知视觉内容，生成 interleaved 多模态推理轨迹（如框出目标区域、放大关键细节）；**(b)** 潜在空间想象——模型通过隐状态（latent token）在 embedding 空间进行视觉推理，无需显式生成图像；**(c)** 文本空间想象（CapImagine）——将视觉操作转化为文本描述，模型在文本空间中"想象"视觉变化，如描述"高亮区域显示该物体的颜色是红色"。*

![图2：系统性 latent 分析框架](assets/capimagine/image2.jpg)

*图2：系统性 latent 分析框架，揭示 latent token 的内部机制和行为模式。**(a)** Model Inference——展示 latent token 的生成流程：输入图像和问题后，模型通过 `<|latent_start|>` 进入 latent 模式，此时将上一层 hidden state 作为下一层的输入，连续生成多个 latent token，最后以 `<|latent_end|>` 退出 latent 模式，恢复正常文本解码。**(b)** $X \rightarrow Z$ 因果分析——包括 Inter-instance（同一位置、不同实例的 latent token 相似度对比）和 Intra-instance（同一实例内所有 latent token 的相似度变化）两个维度。图中热力图显示 latent token 在跨实例时高度相似（红色），而文本/image token 在跨实例时保持低相似度（蓝色/绿色），呈现鲜明对比。**(c)** $Z \rightarrow Y$ 因果分析——包括 Intervention on $Z$（将 latent token 替换为相同张量、注入高斯噪声、替换为纯噪声、设为接近零小值等干预方式，观察最终答案变化）和 Probing Analysis（用 latent token 作为唯一输入回答衍生问题，测试 latent 编码了多少视觉语义）。实验结果的柱状图显示：无论哪种干预，模型性能几乎不变；而 probing 中 latent-only 输入远低于随机猜测。*

### 2.2 因果分析：$X \rightarrow Z$（Finding 1）

> **Finding 1：Latent token 在不同实例和不同任务间高度相似，并在推理过程中逐步坍缩为几乎相同的状态。**

**实验设置**：在三种代表性 LVR 方法（Monet、LVR、Mirage）上，从 V*、MME、OCRBench-v2、MME-RealWorld-Lite、TableVQA 中采样 100 个测试实例，从两个角度考察 latent token：

- **Inter-instance（跨实例）**：固定位置，比较不同实例的 latent token 余弦相似度
- **Intra-instance（实例内）**：同一实例内，观察所有 latent token 的相似度变化

**关键发现**：

1. **跨实例高度同质**：同一位置、不同实例的 latent token 余弦相似度极高，说明 latent token 几乎没有编码输入图像或问题的差异化信息。不同任务的 latent token 也同样高度相似——连粗粒度的任务级区分都无法做到。

2. **逐步退化现象**：随着推理推进，latent token 之间的相似度越来越高，最终坍缩为几乎一致的表示。相比之下，文本 token、图像 token 和 MLLM 内部表示都携带了丰富的区分性语义。

3. **不同方法退化模式不同**：Monet 退化较慢但最终收敛到高度均匀空间；LVR 快速坍缩但部分 token 保留一定区分性；Mirage（将长视觉 token 压缩为少数 latent）全程几乎无区分性。

**物理直觉**：在标准离散 CoT 中，`lm_head` + `argmax`/采样操作引入了强非线性，将表示从退化锥中拉出。Latent token 完全在连续空间操作，缺乏这种投影机制，因而被困在各向异性的狭窄锥中，导致余弦相似度极高。

### 2.3 因果分析：$Z \rightarrow Y$（Findings 2 & 3）

> **Finding 2：对 latent token $Z$ 施加根本性改变，仅导致答案 $Y$ 的微小变化。**

**干预实验**：

对 Monet（通用场景），将所有 latent token 替换为共享的相同张量：

| 干预方式 | V* | HR-Bench4K | MME-RW-Lite |
|---------|-----|------------|-------------|
| Monet 原版 | 82.7 | 71.1 | 46.9 |
| Monet $do(Z)$（全部同张量） | **83.3** (+0.6) | 70.1 (-1.0) | 46.2 (-0.7) |

在 V* 上性能甚至略有提升！只有 HR-Bench-4K 和 MME-RealWorld-Lite 上出现微小的 1.0% 和 0.7% 退化。

对 Mirage（任务特定），进一步尝试多种干预策略：

| 干预方式 | Mirage-Stage1 | Mirage-Stage2 |
|---------|:---:|:---:|
| 原版 latent | 64.2 | 77.0 |
| 全部替换为同一张量 | 64.0 (-0.2) | 77.2 (+0.2) |
| 注入高斯噪声 | 64.0 (-0.2) | 76.7 (-0.3) |
| 全部替换为高斯噪声 | **64.5 (+0.3)** | 76.2 (-0.8) |
| 全部设为接近零的小值 | 65.0 (+0.8) | 35.5 (-41.5) |

**即使将 latent token 全部替换为随机噪声，性能几乎不变**——这强烈说明模型根本没有关注 latent token，也没有在其内部编码关键信息。Mirage-Stage2 只在"全部设为零"的极端干预下才崩溃（触发了重复生成）。

> **Finding 3：Latent token 仅编码了极有限的视觉语义，无法独立支撑下游推理。**

**探测实验**：从 V* 中采样问题-图像对 $(I_i, q_i)$，收集对应的 latent embedding $\{Z_i\}$，然后构造 30 道不同的多选题——这些题关注同一图像区域但询问不同属性。将 $\{Z_i\}$ 和衍生问题 $\tilde{q}$ 一起喂给模型：

- **Latent-only 输入**（$\{Z_i\}, \tilde{q}$）：远低于随机猜测水平
- **原始图像输入**（$I_i, \tilde{q}$）：Monet 和 Qwen3-VL-32B 都达到 76.67%

说明 latent token 根本没有保留足够的视觉语义——模型不是通过 latent token"看到"图像的。

### 2.4 小结：Latent Token 到底在干什么？

三个 findings 共同指向一个结论：

> **Latent token 更像是 soft prompt 或 placeholder，而非视觉想象或推理的主动载体。模型可能走了一条绕过 latent reasoning 的隐式捷径。**

---

## 三、CapImagine：回到文本空间进行想象

既然 latent token 几乎没用，那如何有效利用 interleaved 多模态推理数据中的视觉信息？答案出奇简单：**把中间图像的操作变成文本描述**。

![图3：CapImagine 方法和数据构建流程](assets/capimagine/image3.jpg)

*图3：CapImagine 方法与数据构建流程。上半部分为原始 interleaved 格式的训练数据（文本推理 + 中间辅助图像）；中间部分对比了两种方法论差异——Latent-space reasoning 尝试将视觉操作压缩到隐空间，而 Text-space imagination 将语义变化转化为文本描述；下半部分展示数据构建流程。*

### 3.1 数据重写（Data Rewriting）

基于 Monet-SFT-125K 数据集（由 Monet 构建的 interleaved 多模态推理数据），将中间辅助图像替换为文本描述：

- **Zoom-in 类数据**（Visual-CoT、Zebra-CoT）：提供原图 + 高亮区域给 Qwen3-VL-4B，让它生成聚焦于高亮区域的准确描述
- **图像操作类数据**（Refocus、CogCoM）：同时呈现原图和操作后的图给 Qwen3-VL-4B，让它描述视觉差异（如标注的数值、高亮的文本实体等）

为避免直接插入导致推理链断裂，还使用 MLLM 进行全局重写，让文本描述自然融入原推理过程。

### 3.2 数据过滤（Data Filtering）

Monet-SFT-125K 中 94.88% 来自 Visual-CoT 数据，但质量堪忧：
1. 原问题的答案常与新生成的视觉观察冲突
2. 大量问题本身模糊到无法回答

使用 MLLM 对整个训练实例进行质量评估，过滤后保留 **17k 高质量实例**（仅占原始的 ~14%）。

### 3.3 训练

基于 Qwen2.5-VL-7B，使用 Monet 代码库做 CoT-SFT 微调。8 张 A800-80G GPU，batch size 1、gradient accumulation 16。在训练中选取 best checkpoint 以应对训练不稳定问题。

---

## 四、实验与结果

### 4.1 主实验：高分辨率感知基准

| 模型 | V* (Overall) | HR-Bench4K | HR-Bench8K | MME-RW-Lite | BLINK (Jigsaw/MV) |
|------|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 67.5 | 59.0 | 55.5 | 52.0 | 55.3/59.4 |
| Qwen2.5VL-7B (基座) | 76.4 | 68.0 | 63.8 | 45.8 | 62.7/42.9 |
| **工具方法** | | | | | |
| PixelReasoner | 80.6 | 72.9 | 66.9 | 49.7 | - |
| DeepEyes | **90.0** | 75.1 | 72.6 | 53.2 | - |
| **想象方法（Latent）** | | | | | |
| LVR | 81.7 | 70.8 | 63.0 | 50.6 | 52.0/46.6 |
| Monet | 83.3 | 71.0 | 68.0 | 46.9 | 50.0/47.4 |
| **想象方法（Text）** | | | | | |
| **CapImagine** | **85.9** | **74.1** | **70.7** | **54.8** | **64.7/49.6** |

**核心观察**：

1. **CapImagine 全面超越 Monet**：V* 提升 2.6%，HR-Bench 平均提升 3.44%，MME-RealWorld-Lite 提升 7.9%。这证明了同一个数据源下，文本空间想象远超潜在空间。

2. **vs 工具方法**：DeepEyes 在部分任务上更强（90.0 vs 85.9 on V*），但它直接操作图像（zoom-in），有天然信息优势。CapImagine 仅靠文本想象就能接近工具方法的水平。

3. **抽象推理泛化**：在 BLINK 的 Jigsaw（64.7 vs 50.0）和 Multi-View（49.6 vs 47.4）上大幅领先 Monet，说明文本想象不仅适用于"放大看细节"，也适用于"空间重构"这类抽象推理。

### 4.2 TableVQA 结果

| 模型 | VWTQ | VWTQ-syn | VTabFact | Overall |
|------|:---:|:---:|:---:|:---:|
| Monet | 55.3 | 60.4 | 78.8 | 64.8 |
| **CapImagine** | **60.9** | **68.0** | **83.2** | **70.7** |

在图表/表格理解任务上，CapImagine 整体提升 5.9%，说明文本空间想象同样适用于非自然图像。

### 4.3 消融实验（严格控制的对比）

| 消融设置 | V* | HR-Bench4K | HR-Bench8K |
|---------|:---:|:---:|:---:|
| **CapImagine（完整版）** | **85.9** | **74.1** | **70.7** |
| + w/o Rewriting（去掉文本想象，用 `<think_image>` 代替） | 82.7 (-3.2) | 74.1 | 69.8 |
| + w/o Filtering（直接用原始 Monet-SFT-125K） | 82.7 (-3.2) | 72.5 | 69.3 |
| Monet + subset（用 CapImagine 相同的 17k 子集训练 Monet） | 79.6 (-6.3) | 70.7 | 67.9 |

三条关键结论：

1. **数据重写至关重要**：去掉文本想象（用 `<think_image>` token 替代），V* 掉 3.1%
2. **数据过滤不可或缺**：不过滤直接用 Monet-SFT-125K，性能继续下降
3. **方法差距是真实的**：在完全相同的数据（17k subset）下训练 Monet，效果远不如 CapImagine。且 SFT 版本的 Monet（去掉 Policy Optimization）与原本持平，进一步质疑了 latent reasoning 中 Policy Optimization 阶段的实际作用

### 4.4 CapImagine 的因果分析

论文对 CapImagine 进行了与 latent token 相同的因果中介分析：

![图4：CapImagine 的因果分析](assets/capimagine/image4.jpg)

*图4：CapImagine 推理过程中隐状态在跨实例（Inter-instance）和实例内（Intra-instance）两个维度的相似度分析。跨实例余弦相似度始终较低，表明 $X \rightarrow Z$ 存在强因果依赖；实例内相邻隐状态之间有显著差异，表明每个想象 token 编码了不同的语义内容。*

**Inter-instance 分析**：跨实例余弦相似度始终较低——说明文本想象的内容确实与输入相关，$X \rightarrow Z$ 具有强因果依赖。

**Intra-instance 分析**：相邻隐状态间有显著差异——每个想象 token 编码了不同语义，而非逐步退化。

**干预 $Z$ 的结果**：使用 Qwen3-32B 故意篡改想象内容使其导向错误答案，再让 CapImagine 基于错误想象继续推理：

| 模型 | V* | HR-Bench4K |
|------|:---:|:---:|
| CapImagine | 85.9 | 74.1 |
| CapImagine $do(Z)$ | **22.5** (-63.4) | **24.0** (-50.1) |

性能断崖式下跌——证明文本想象内容在推理过程中起到了真正的因果作用。这与 latent token 的"改不动"形成鲜明对比。

### 4.5 推理效率

![图5：推理速度对比](assets/capimagine/image5.jpg)

*图5：Monet、CapImagine 和 DeepEyes 在 V* 各子类上的推理速度对比。纵轴为解码时间（秒），横轴为 V* benchmark 的不同类别。CapImagine 虽然使用了较长的文本想象序列，但推理速度与 latent 方法 Monet 基本相当，且耗时仅为工具方法 DeepEyes 的一半左右。*

- CapImagine vs Monet：速度相当（虽然文本更长，但避开了 latent token 的额外计算）
- CapImagine vs DeepEyes：速度快近一倍，同时保持竞争力性能

---

## 五、关键洞察与技术亮点

### 5.1 学术贡献

1. **首次系统性的因果分析**：用因果中介分析框架对 LVR 进行 $X \rightarrow Z$ 和 $Z \rightarrow Y$ 两个方向的系统考察，用数据说话而非直觉猜测。三个 findings 层层递进——latent token 同质化（Finding 1）→ 改了也没影响（Finding 2）→ 根本没有有效语义（Finding 3）。

2. **"皇帝的新衣"式洞察**：LVR 领域一系列方法通过监督 latent token 匹配视觉特征（Mirage → LVR → Monet），在 benchmark 上表现出色，但因果分析揭示这些 latent token 可能是"搭便车"——模型通过其他路径解决问题，latent token 退化成了 soft prompt。这是一个重要的领域反思。

3. **简单方案的胜利**：CapImagine 无需复杂设计（蒸馏、Policy Optimization、latent 监督），仅靠数据重写 + 过滤就从同一数据源获得了显著提升。这种"返璞归真"的实验结果本身就是对 complex latent method 的有力质疑。

### 5.2 技术亮点

1. **从数据角度解构方法**：通过严格的数据平权（相同 17k subset 训练 Monet vs CapImagine），将方法差异从数据差异中隔离出来，消融实验设计严谨。

2. **退化的物理直觉解释**：给出了 latent token 退化现象的合理解释——Transformer 隐状态自然倾向于聚入各向异性锥，而离散解码的 `argmax`/采样提供了"出锥"的非线性力。Latent token 全程连续操作，缺失这种机制。

3. **端到端的因果验证**：不仅分析了 latent token 的"失效"，还对 CapImagine 做了同样的因果中介分析，证明文本想象 $X \rightarrow Z \rightarrow Y$ 的完整因果链——干预 $Z$ 会导致 $Y$ 断崖下跌。

---

## 六、关键概念速查

| 概念 | 解释 |
|------|------|
| **LVR（Latent Visual Reasoning）** | 在 MLLM 隐空间中通过专门的 latent token 进行视觉推理，不显式生成图像 |
| **Causal Mediation Analysis** | 因果中介分析——通过对中间变量 $Z$ 进行干预 $do(Z)$ 并观察结果 $Y$ 的变化，判断 $Z$ 是否真正起到中介作用 |
| **Monet** | 蒸馏式 LVR 方法，将梯度传播限制在 latent token 上，同时保留中间图像和文本线索的语义 |
| **Mirage** | 首个 LVR 方法，通过压缩中间推理图像的视觉特征来监督 latent token |
| **LVR（方法）** | 使用图像特征作为 latent 监督信号的通用 LVR 方法 |
| **CapImagine** | 本文提出的文本空间想象方法，将 interleaved 视觉操作转为文本描述 |
| **Representation Degeneration** | Transformer 隐状态自然聚入各向异性锥的现象，导致表示高度相似 |
| **Inter-instance / Intra-instance** | 跨实例（同一位置不同实例）vs 实例内（同一实例不同位置）的 latent token 分析 |

---

## 七、局限性

1. **数据集构建依赖 MLLM**：数据重写（Qwen3-VL-4B）和过滤（MLLM judge）两个关键步骤都依赖 MLLM，可能引入模型偏差
2. **仅验证了 7B 模型**：所有实验基于 Qwen2.5-VL-7B，不同规模的模型上 latent token 行为是否一致有待验证
3. **文本想象的长度开销**：文本描述天然比 latent token 长，虽然推理速度相当（因为避开了 latent 切换开销），但 prompt 长度和 KV cache 占用可能更大
4. **工具方法的优势**：DeepEyes 等直接操作图像的方法在部分任务上仍然领先，说明真正的视觉操作（zoom-in、draw）的信息量可能超过纯文本描述
5. **因果分析的破坏性干预局限**：将 latent token 全部替换为同一张量或随机噪声是非常强的干预，模型可能通过残差连接"绕路"。更精细的干预设计可能得出更细致的结论

---

*本笔记基于论文 "Imagination Helps Visual Reasoning, But Not Yet in Latent Space"（arXiv: 2602.22766v2, ICML 2026）撰写。*
