<!-- arxiv: 2501.19201 -->
<!-- venue: ICML 2026 -->
<!-- tags: MLLM, CoT, 推理效率, 信息论, 知识蒸馏 -->

# Heima：通过隐藏思考实现高效推理

> **论文**: Efficient Reasoning with Hidden Thinking
> **作者**: Xuan Shen, Yizhou Wang, Yufa Zhou, Xiangxi Shi, Pu Zhao, Yanzhi Wang, Jiuxiang Gu
> **机构**: Zhejiang University / Adobe / Duke / Oregon State / Northeastern
> **代码**: [shawnricecake/Heima](https://github.com/shawnricecake/Heima)

> 本文基于以下本地材料整理：
>
> - 论文 tex 源码：`arXiv-2501.19201v2/main.tex`、`sections/*.tex`
> - 论文图片：`arXiv-2501.19201v2/figures/*.pdf`
> - 官方代码：`Heima/`
> - 本文图片导出目录：`assets/heima/`

---

## 1. 动机与问题

Chain-of-Thought（CoT）推理已成为提升多模态大语言模型（MLLM）复杂问题求解能力的关键技术。然而，CoT 推理需要生成大量中间文本（对复杂问题可达数百 tokens），导致高昂的推理成本。

**核心问题**：CoT 的冗长文本包含大量冗余信息，能否将推理过程压缩到隐空间，以极少的"思考 token"替代冗长的文本 CoT，从而大幅提升推理效率？

---

## 2. 方法概览

Heima（hidden llama 的谐音）提出了一套完整的 CoT 压缩框架：用 MLLM 将多阶段 CoT 推理压缩为极少量的思考 token，再用纯文本 LLM 作为解释器重建推理过程。

![图 1：Heima 整体流程](assets/heima/introduction_whole_pipeline.png)

*图 1：Heima 框架总览，用一个具体 Demo（识别图中汽车品牌）串联全流程。图片分为上下两栏。*

**上栏 (a) Heima 隐空间推理（MLLM）：**
输入为一张黑色 BMW M3 的图片和问题"Which automotive brand does this car belong to, and what visual cues or badges indicate that?"。Heima 不生成冗长的文本 CoT，而是用三个思考 token——`<Thinking_of_Summary>`、`<Thinking_of_Caption>`、`<Thinking_of_Reasoning>`——在隐空间中完成推理，最终输出结论"The image shows a black BMW M3 driving down a road"。每个思考 token 只占 1 个 token 位置，但它的 last hidden state 编码了对应推理阶段所需的全部信息（包括视觉特征）。Heima 的答案基于隐推理直接生成，跳过了传统 CoT 中数百个 token 的文本展开过程。

**下栏 (b) Heima 解释器重建推理（纯 LLM）：**
三个独立的纯文本解释器（基于 Llama-3.1-8B-Instruct）分别接收对应思考 token 的 last hidden state。关键设计：解释器**不看图像**，只接收思考 token 的隐表示 + 问题文本 + 解释性 prompt。Summary 解释器重建出"我将通过检查 logo、配色和设计元素来识别汽车品牌"；Caption 解释器重建出"图像显示一辆 sleek 的黑色现代跑车，侧面有一个十字加圆圈的独特 logo"；Reasoning 解释器重建出"前格栅上的徽章是 BMW，这是 BMW 品牌的常见标志，确认了品牌身份"。

这一 Demo 直接验证了 Heima 的核心主张：思考 token 的隐空间表示不仅保留了语言推理信息，还编码了细粒度的视觉特征（如 BMW logo 的十字+圆圈形状），供纯文本 LLM 在不看原图的情况下重建。*

### 2.1 思考 Token 与渐进式蒸馏

**思考 Token（Thinking Token）**：为每个 CoT 阶段（Summary、Caption、Reasoning）定义一个特殊的 vocabulary token（如 `<Thinking_of_Summary>`），将整个阶段的文本推理压缩到单个 token 的隐状态中。

**渐进式蒸馏（Progressive Distillation）**：不是一次性将所有 CoT 阶段替换为思考 token，而是逐阶段渐进替换：

- **Stage 0**：全部使用文本 CoT（标准 LLaVA-CoT 训练）
- **Stage 1**：仅将 Summary 替换为 `<CoT>_1`，其余保持文本
- **Stage 2**：将 Summary + Caption 替换为 `<CoT>_1`, `<CoT>_2`
- **Stage 3**：全部三个阶段都替换为思考 token
- **Recovering Stage**：最终用纯思考 token 额外微调一步，优化跨阶段 transition

![图 2：渐进式蒸馏过程](assets/heima/method_progressive_encoding-1_column.png)

*图 2：渐进式蒸馏（Progressive Distillation）的四个训练阶段。每行展示一个阶段使用的训练数据格式，从左到右的三个色块分别对应 Summary（蓝色）、Caption（黄色）、Reasoning（绿色）三个阶段，相同颜色 = 同一类推理内容。最后一列为答案（灰色）。

**Stage 0（无思考 token）：**
全部三个阶段都使用完整文本 CoT + 最终答案。这是标准的 LLaVA-CoT SFT 格式，模型学习"看图和问题 → 生成三段文本推理 → 给出答案"。此阶段的训练数据即为原始 LLaVA-CoT-100k，不做任何修改。

**Stage 1：**
Summary 阶段的文本 CoT 被替换为一个 `<Thinking_of_Summary>` 思考 token。Caption 和 Reasoning 仍保持文本 CoT。模型在此阶段学习将"问题复述/摘要"的推理内容压缩到单个 token 中，同时仍依赖 Caption 和 Reasoning 的文本信号来维持答案质量。这是最关键的一步——如果一步到位把所有 CoT 都替换掉，模型会因分布剧烈偏移而崩溃。

**Stage 2：**
Summary 和 Caption 两个阶段都替换为思考 token（`<Thinking_of_Summary>` + `<Thinking_of_Caption>`）。只有 Reasoning 阶段保留文本 CoT。模型在此阶段学习将视觉描述（Caption）也编码到隐空间，训练难度进一步增加。

**Stage 3：**
全部三个阶段都替换为思考 token + 答案。这是最终的 Heima 推理格式。模型仅凭三个思考 token 的隐状态完成全部推理并输出答案，完全不再依赖文本 CoT。

**Recovering Stage（图中未单独展示）：**
在 Stage 3 之后额外执行一步纯思考 token 的蒸馏（学习率从 1e-4 降至 1e-5），优化跨阶段的 transition 和 token 间的交互，确保整个隐推理链的信息流动是连贯的。消融实验表明去掉这一步会使平均准确率从 58.0 降至 56.6。*

训练数据集 $D_H$ 将原始 CoT 文本替换为思考 token：

$$D_H := \bigl\{ (X, \texttt{<CoTs>}, Y) \bigr\}$$

蒸馏目标为标准 next-token prediction：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(X, Y, \texttt{<CoTs>}) \sim D_H} \log P_\theta(\texttt{<CoTs>}, Y \mid X)$$

### 2.2 信息论分析

论文从信息论角度证明了压缩的合理性。定义 $X$ 为输入，$\mathrm{CoTs}$ 为原始文本推理，$\texttt{<CoTs>}$ 为思考 token，$Y$ 为输出答案。

**核心定理**：由于 $\texttt{<CoTs>}=f(X,\mathrm{CoTs})$，由数据处理不等式有：

$$0 < I(Y;\texttt{<CoTs>}\mid X) \le I(Y;\mathrm{CoTs}\mid X)$$

等价于：

$$H(Y\mid X,\mathrm{CoTs}) \le H(Y\mid X,\texttt{<CoTs>}) \le H(Y\mid X)$$

**关键结论**：
- 思考 token 永远不可能比原始 CoT 包含更多关于 $Y$ 的信息
- 但只要 $I(Y;\texttt{<CoTs>}\mid X) > 0$，就保留了非平凡的推理信息
- 信息损失量 $I(Y;\mathrm{CoTs}\mid X,\texttt{<CoTs>})$ 量化了压缩带来的信息差距——解释器就是用来经验性地评估这个差距的大小

### 2.3 解释器（Interpreter）设计

为了量化信息差距，论文设计了基于纯 LLM 的解释器，将思考 token 的隐状态重构回文本 CoT。

![图 3：解释器训练流程](assets/heima/method_adaptive_decoding_train-1_column.png)

*图 3：解释器（Interpreter）训练流程，展示如何将思考 token 的隐状态解码为文本 CoT。图片包含左右两个 panel。

**左栏 (a) Heima 推理——隐状态生成（冻结，不训练）：**
Heima（冻结的 MLLM）基于图像（Vision Input）和问题（Text Query）执行推理。它生成一串 token 序列：先输出三个阶段的思考 token（每个阶段 1 个特殊 token），再输出答案。对于第 s 个思考 token `<CoT>_(s)`，Heima 在其 Transformer 最后一层产生一个 last hidden state $H_{\texttt{<CoT>}_{(s)}} \in \mathbb{R}^{d}$（$d$ 为 hidden dimension，Llama-3.2-11B 中为 4096）。这个向量就是"压缩后的推理"——单个 token 的文本符号本身不携带信息，所有推理内容都编码在这个 4096 维向量中。

**右栏 (b) 解释器训练——隐状态替换 + 文本重建（可训练）：**
解释器是一个基于纯 LLM（Llama-3.1-8B-Instruct）的 LoRA 可训练模型。训练时：① Heima 生成思考 token 并提取 `H_Caught`（即 $H_{\texttt{<CoT>}_{(s)}}$）；② 在解释器的输入序列中，思考 token 的 **embedding 向量被 $H_{\texttt{<CoT>}_{(s)}}$ 替换**——这是核心操作，发生在 word embedding 层之后、Transformer 第一层之前；③ 解释器接收的完整输入为"解释性 prompt（According to the question, can you explain the thinking progress?）+ 纯文本问题 + 被替换的隐状态"；④ 解释器用标准 next-token prediction 自回归地生成重建的文本 CoT。

**为何不直接输入 token 符号？**
如果只输入 `<CoT>_(s)` 这个特殊 token 的文本符号（不带隐状态），解释器看到的只是一个固定的 token ID，没有任何与具体输入图像/问题相关的内容信息。Heima 的推理信息存储在思考 token 的 **hidden state** 中（隐状态随输入不同而变化），而非其 token identity（token 符号在所有样本中是固定的）。用 $H_{\texttt{<CoT>}_{(s)}}$ 替换 embedding 等于把 Heima 内部的"思考内容"直接传递给解释器。

**训练细节：**
解释器用 LoRA（rank=16，alpha=32）微调，学习率 5e-4，batch size=8，1 个 epoch。Heima 全程冻结。三个 CoT 阶段各训练一个独立的解释器（共 3 个），评估在 4300 个测试样本上进行。*

**训练方式**：
1. 使用冻结的 Heima 生成思考 token
2. 提取思考 token 的 last hidden state $H_{\texttt{<CoT>}_{(k)}}$
3. 在解释器中，用 $H_{\texttt{<CoT>}_{(k)}}$ **替换** token embedding，而非直接输入 token 符号
4. 使用解释性 prompt 引导重构：*"According to question: $X_q$, can you explain the thinking progress $\texttt{<CoT>}_{(k)}$?"*

**关键设计**：解释器为纯文本 LLM，不接收图像输入。这意味着思考 token 的隐表示中**必须编码了视觉信息**，解释器才能重构出包含视觉细节的推理过程。

---

## 3. 实验配置

| 组件 | 模型 | 训练框架 |
|------|------|----------|
| Heima Encoder | Llama-3.2-11B-Vision-Instruct | LoRA (rank=16, alpha=32) |
| Heima Decoder | Llama-3.1-8B-Instruct | LoRA |
| 辅助验证 | LLaVA-Next-Vicuna-7B | LoRA |

- **训练数据**：LLaVA-CoT-100k（100k 图像-QA 对，含 Summary/Caption/Reasoning 三阶段 CoT）
- **训练硬件**：8× H100 GPU
- **评估基准**：MMStar, MMBench V1.1, MMVet, MathVista, AI2D, HallusionBench
- **评估框架**：VLMEvalKit

---

## 4. 主要结果

### 4.1 推理效率与精度

| 方法 | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | Avg |
|------|--------|---------|-------|-----------|------|-----------|-----|
| Llama3.2-11B | 48.1 (140) | 58.2 (65) | 50.2 (106) | 50.3 (240) | 68.5 (75) | 37.2 (91) | 52.1 |
| LLaVA-CoT | 54.0 (181) | 70.7 (155) | 49.8 (227) | 50.9 (216) | 77.6 (179) | 63.8 (178) | 61.1 |
| Heima (w/o prog.) | 49.7 (13) | 72.5 (13) | 39.0 (72) | 39.3 (14) | 75.9 (13) | 61.3 (16) | 56.3 |
| Heima (w/o rec.) | 49.8 (13) | 71.6 (13) | 42.8 (80) | 39.8 (14) | 77.3 (13) | 58.5 (18) | 56.6 |
| **Heima** | **49.9 (13)** | **72.8 (13)** | **43.3 (76)** | **43.6 (14)** | **77.5 (13)** | **60.6 (17)** | **58.0** |

*表 1：主实验结果。括号内为平均生成 token 数，Avg 列为 6 个基准的平均准确率。Heima 将 token 数降至原始 LLaVA-CoT 的约 6-8%（如 MMStar 上 12.8 vs 181.0）。"w/o prog." 表示一次性蒸馏所有 CoT 阶段（非渐进式），"w/o rec." 表示去掉最后的 recovering stage。*

**关键发现**：
- Heima 用不到 10% 的 token 保持了 LLaVA-CoT 的绝大部分精度（58.0 vs 61.1）
- 在 MMBench 上甚至**超越**了 LLaVA-CoT（72.8 vs 70.7），说明压缩过程可能去除了噪声
- 渐进式蒸馏提升 1.7 个百分点（58.0 vs 56.3），recovering stage 提升 1.4 个百分点（58.0 vs 56.6）——两者缺一不可
- 相比无 CoT 的 Llama3.2-11B 基线，Heima 在全部 6 个基准上均有大幅提升（58.0 vs 52.1）

### 4.2 跨模型架构泛化

在 LLaVA-Next-Vicuna-7B 上的验证：

| 方法 | Avg Acc |
|------|---------|
| LLaVA-Next-7B (无 CoT) | 44.4 |
| LLaVA-Next-7B (CoT) | 55.0 |
| **Heima (7B)** | **53.8** |

同样将 token 降至约 6%，保持了 CoT 精度的 97.8%，验证了方法的通用性（不限于 Llama 模型家族）。

### 4.3 解释器重建质量

![图 4：解释器重建评估](assets/heima/results_decoder_eval_metrics_gpt4o.png)

*图 4：解释器重建质量的定量评估。包含两个子图，在 4300 个测试样本上评估解释器重建的文本与原始 CoT 文本的相似度。

**子图 (a) 自动评估指标（左，四组柱状图）：**
展示 BLEU-4、METEOR、ROUGE-L、BERTScore 四个指标的结果，每组有三根柱子分别对应 Summary（蓝色）、Caption（黄色）、Reasoning（绿色）。横轴为评估指标，纵轴为分数。结果呈现一致的阶梯状：Summary > Caption > Reasoning。具体数据——BLEU：15.9 / 12.8 / 11.2；METEOR：40.1 / 35.5 / 32.7；ROUGE-L：41.6 / 37.9 / 32.7；BERTScore：73.4 / 71.4 / 66.6。

Summary 阶段重建质量最高（BERTScore 73.4），因为 Summary 本质是对问题的复述——"识别汽车品牌需要检查 logo 和设计元素"——这更多依赖文本理解，不需要太多视觉信息。Caption 居中（71.4），因为它需要从隐状态中提取图像的视觉描述，有一定难度。Reasoning 最低（66.6），因为 Reasoning 涉及多步逻辑推理链（"logo 是十字加圆圈 → 这是 BMW 的特征 → 确认品牌为 BMW"），任何一个环节出错都会影响最终匹配。但 BERTScore 66.6 仍属于"语义高度相关"区间，说明压缩保留了推理的核心逻辑。

**子图 (b) GPT-4o 语义相似度评估（右）：**
使用 GPT-4o 作为评判器，对三个阶段的 4300 个重建-原始文本对进行 1-5 分打分——1 分表示完全无关（不同主题、无内容重叠），5 分表示几乎完全一致（仅有措辞差异）。评分时 GPT-4o 同时参考原始图像和问题作为上下文。横轴为三个阶段，纵轴为平均相似度分数（越高越好）。

三个阶段的 GPT-4o 相似度分数均较高，与左侧自动指标的趋势一致（Summary > Caption > Reasoning）。GPT-4o 的评判比 BLEU/METEOR 等 n-gram 指标更能捕捉语义等价（例如"identify the brand by logo"和"check visual cues for brand recognition"在 BLEU 中匹配度低，但 GPT-4o 能识别其语义一致性）。这证实了解释器的重建不仅在词面层面对齐，在语义层面也与原始 CoT 高度一致。

**综合解读：**
左右两图共同验证了 Theorem 1 的信息论结论——压缩引入的信息差距 $I(Y;\mathrm{CoTs}\mid X,\texttt{<CoTs>})$ 在实践中很小：解释器不需要图像就能从思考 token 的隐状态中重建出语义正确的推理过程，说明思考 token 成功保留了 CoT 中的非平凡互信息。*

| Stage | BLEU | METEOR | ROUGE-L | BERTScore |
|-------|------|--------|---------|-----------|
| Summary | 15.9 | 40.1 | 41.6 | 73.4 |
| Caption | 12.8 | 35.5 | 37.9 | 71.4 |
| Reasoning | 11.2 | 32.7 | 32.7 | 66.6 |

**尤其值得注意的是，解释器是纯文本 LLM，不接收图像输入，却能重建出包含视觉细节的内容——这直接证明了思考 token 的隐表示中成功编码了多模态信息。**

---

## 5. 消融实验

### 5.1 思考 token 数量

每个 CoT 阶段用不同数量的思考 token 进行蒸馏：

| #Token | Avg Acc |
|--------|---------|
| **1** | **58.0** |
| 2 | 56.5 |
| 4 | 56.2 |
| 8 | 56.7 |
| 16 | 56.9 |
| 32 | 57.1 |

**结论**：单个思考 token 效果最优。更多 token 不会提升精度，反而可能引入噪声或过拟合。这说明一个 hidden state 的容量已经足够编码单阶段 CoT 的推理信息。

### 5.2 自适应保留比例

按原始 CoT 长度的固定比例保留思考 token（10%-90%）：

| Ratio | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | Avg |
|-------|--------|---------|-------|-----------|------|-----------|-----|
| 0.1 | 49.1 | 69.7 | 37.2 | 41.3 | 75.9 | 59.1 | 55.4 |
| 0.2 | 49.7 | 71.5 | 39.4 | 41.2 | 75.3 | 60.0 | 56.2 |
| 0.3 | 48.1 | 71.9 | 40.6 | 39.9 | 75.3 | 59.4 | 55.8 |
| 0.4 | 47.9 | 70.3 | 38.6 | 39.2 | 76.3 | 59.7 | 55.3 |
| 0.5 | 47.2 | 70.1 | 40.5 | 39.5 | 75.2 | 57.6 | 55.0 |
| 0.6 | 48.4 | 70.9 | 42.0 | 38.8 | 76.6 | 60.5 | 56.2 |
| 0.7 | 48.7 | 69.8 | 41.1 | 39.0 | 75.4 | 59.7 | 55.6 |
| 0.8 | 49.9 | 69.3 | 40.9 | 37.2 | 75.3 | 59.4 | 55.3 |
| 0.9 | 49.2 | 70.5 | 40.1 | 38.4 | 75.7 | 60.1 | 55.7 |

精度在 55.0-56.2 之间无规律波动，远低于单 token 方案的 58.0，且 token 数超过 70% 时甚至超过基线模型。**结论**：固定比例的自适应方法不适用于 CoT 压缩，核心矛盾不在 token 数量而在信息密度的编码方式。

### 5.3 解释器数量

用单一 LLM 同时解释三个阶段 vs 三个独立解释器的对比：

| 指标 | #解释器 | Summary | Caption | Reasoning |
|------|---------|---------|---------|-----------|
| BLEU | 1 | 9.5 | 6.8 | **11.3** |
| | 3 | **15.9** | **12.8** | 11.2 |
| METEOR | 1 | 32.7 | 25.4 | 32.5 |
| | 3 | **40.1** | **35.5** | **32.7** |
| ROUGE-L | 1 | 36.8 | 29.7 | 31.9 |
| | 3 | **41.6** | **37.9** | **32.7** |
| BERTScore | 1 | 67.8 | 60.6 | **67.3** |
| | 3 | **73.4** | **71.4** | 66.6 |

三个独立解释器在 Summary 和 Caption 阶段优势显著（BERTScore +5.6 / +10.8），但在 Reasoning 阶段单一解释器反而略优（BLEU 11.3 vs 11.2，BERTScore 67.3 vs 66.6）。**结论**：Summary 和 Caption 阶段内容差异大，需要独立解释器；Reasoning 阶段的文本模式相对固定，单一解释器也能胜任。

---

## 6. 代码仓库结构

```
Heima/
├── heima/
│   ├── configs/          # 训练/评估 YAML 配置
│   ├── scripts/           # Shell 运行脚本
│   └── main_python/       # 核心 Python 脚本
│       ├── 1_*-organize_dataset-*.py   # 数据准备
│       ├── 2-*-training-pipeline-*.py   # 渐进式蒸馏训练
│       ├── 4_*-eval-*.py               # 推理与评估
│       └── 5-*-demo-*.py               # Demo 推理
├── torchtune_pkg/        # torchtune 训练框架
└── zero-shot-evaluation/  # VLMEvalKit 评估
```

---

## 7. 总结与思考

### 核心贡献
1. **首个 MLLM CoT 压缩框架**：将冗长的多阶段 CoT 推理压缩为极少量的思考 token，实现了"隐空间推理"
2. **信息论保证**：从理论上证明了只要保留非平凡互信息，压缩后的推理仍有效，且信息差距可以通过解释器经验性地评估
3. **可解释性验证**：通过纯文本 LLM 解释器成功重建多模态推理过程，验证了压缩的有效性和可解释性

### 值得关注的点
- 思考 token 的隐表示编码了视觉信息——单个 4096 维的 hidden state 能"存储"足够的视觉细节供纯文本 LLM 重建图像描述，这在图 1 的 demo 中得到了直观验证
- 渐进式蒸馏的策略很实用，避免了直接将整个 CoT 替换为未知 token 对模型造成的剧烈分布偏移
- 解释器设计巧妙：不输入 token 符号而是输入其 last hidden state，绕过了"单个 token 符号携带信息有限"的问题

### 局限与展望
- 多个独立解释器增加了系统复杂度，统一解释器是未来方向
- 当前仅在 LLaVA-CoT-100k（三阶段 CoT）上验证，更多样的 CoT 结构有待探索
- 论文提到会扩展到更大的模型
