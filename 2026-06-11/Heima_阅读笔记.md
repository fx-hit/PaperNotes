<!-- arxiv: 2501.19201 -->
<!-- venue: ICML 2026 -->
<!-- tags: MLLM, CoT, 推理效率, 信息论, 知识蒸馏 -->

# Heima：通过隐藏思考实现高效推理

> **论文**: Efficient Reasoning with Hidden Thinking  
> **作者**: Xuan Shen, Yizhou Wang, Yufa Zhou, Xiangxi Shi, Pu Zhao, Yanzhi Wang, Jiuxiang Gu  
> **机构**: Zhejiang University / Adobe / Duke / Oregon State / Northeastern  
> **代码**: [shawnricecake/Heima](https://github.com/shawnricecake/Heima)

---

## 1. 动机与问题

Chain-of-Thought（CoT）推理已成为提升多模态大语言模型（MLLM）复杂问题求解能力的关键技术。然而，CoT 推理需要生成大量中间文本（对复杂问题可达数百 tokens），导致高昂的推理成本。

**核心问题**：CoT 的冗长文本包含大量冗余信息，能否将推理过程压缩到隐空间，以极少的"思考 token"替代冗长的文本 CoT，从而大幅提升推理效率？

---

## 2. 方法概览

Heima（hidden llama 的谐音）提出了一套完整的 CoT 压缩框架，包含三个核心组件：

![Heima 整体流程](assets/heima/introduction_whole_pipeline.pdf)

### 2.1 思考 Token 与渐进式蒸馏

**思考 Token（Thinking Token）**：为每个 CoT 阶段（Summary、Caption、Reasoning）定义一个特殊的 vocabulary token（如 `<Thinking_of_Summary>`），将整个阶段的文本推理压缩到单个 token 的隐状态中。

**渐进式蒸馏（Progressive Distillation）**：不是一次性将所有 CoT 阶段替换为思考 token，而是逐阶段渐进替换：

- **Stage 0**：全部使用文本 CoT（标准 LLaVA-CoT 训练）
- **Stage 1**：仅将 Summary 替换为 `<CoT>_1`，其余保持文本
- **Stage 2**：将 Summary + Caption 替换为 `<CoT>_1`, `<CoT>_2`
- **Stage 3**：全部三个阶段都替换为思考 token
- **Recovering Stage**：最终用纯思考 token 额外微调一步，优化跨阶段 transition

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
- 信息损失量 $I(Y;\mathrm{CoTs}\mid X,\texttt{<CoTs>})$ 量化了压缩带来的信息差距

### 2.3 解释器（Interpreter）设计

为了量化信息差距，论文设计了基于纯 LLM 的解释器，将思考 token 的隐状态重构回文本 CoT。

![解释器训练流程](assets/heima/method_adaptive_decoding_train-1_column.pdf)

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

> 括号内为平均生成 token 数。Heima 将 token 数降至原始 LLaVA-CoT 的约 **6-8%**。

**关键发现**：
- Heima 用不到 10% 的 token 保持了 LLaVA-CoT 的绝大部分精度（58.0 vs 61.1）
- 在 MMBench 上甚至**超越**了 LLaVA-CoT（72.8 vs 70.7），说明压缩过程可能去除了噪声
- 渐进式蒸馏提升 1.7 个百分点，recovering stage 提升 1.4 个百分点——两者缺一不可
- 相比无 CoT 的 Llama3.2-11B 基线，Heima 在全部 6 个基准上均有大幅提升

### 4.2 跨模型架构泛化

在 LLaVA-Next-Vicuna-7B 上的验证（Table A2）：

| 方法 | Avg Acc |
|------|---------|
| LLaVA-Next-7B (无 CoT) | 44.4 |
| LLaVA-Next-7B (CoT) | 55.0 |
| **Heima (7B)** | **53.8** |

同样将 token 降至 6%，保持了 CoT 精度的 97.8%，验证了方法的通用性。

### 4.3 解释器重建质量

![解释器评估](assets/heima/results_decoder_eval_metrics_gpt4o.pdf)

| Stage | BLEU | METEOR | ROUGE-L | BERTScore |
|-------|------|--------|---------|-----------|
| Summary | 15.9 | 40.1 | 41.6 | 73.4 |
| Caption | 12.8 | 35.5 | 37.9 | 71.4 |
| Reasoning | 11.2 | 32.7 | 32.7 | 66.6 |

GPT-4o 5-point 相似度评估确认所有三个阶段的推理过程均被有效重建。**尤其值得注意的是，解释器是纯文本 LLM，不接收图像输入，却能重建出包含视觉细节的内容，证明思考 token 中成功编码了多模态信息。**

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

**结论**：单个思考 token 效果最优。更多 token 并不会提升精度，反而可能引入噪声。

### 5.2 自适应保留比例

按原始 CoT 长度的固定比例保留思考 token（10%-90%），结果精度无规律波动，且 token 数超过 70% 时甚至超过基线模型的 token 数，证明自适应方法不适用于 CoT 压缩。

### 5.3 解释器数量

消融实验表明，用单一 LLM 同时解释三个阶段，summary 和 caption 重建质量大幅下降，但 reasoning 阶段尚可。**结论**：三个独立解释器是必要的。

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
2. **信息论保证**：从理论上证明了只要保留非平凡互信息，压缩后的推理仍有效
3. **可解释性验证**：通过纯文本 LLM 解释器成功重建多模态推理过程，验证了压缩的有效性和可解释性

### 值得关注的点
- 思考 token 的隐表示编码了视觉信息这一发现非常有趣——单个 4096 维的 hidden state 竟然能"存储"足够的视觉细节供纯文本 LLM 重建图像描述
- 渐进式蒸馏的策略很实用，避免了直接将整个 CoT 替换为未知 token 对模型造成的剧烈分布偏移
- 解释器设计巧妙：不输入 token 符号而是输入其 last hidden state，绕过了"单个 token 携带信息有限"的问题

### 局限与展望
- 多个独立解释器增加了系统复杂度，统一解释器是未来方向
- 当前仅在 LLaVA-CoT-100k（三阶段 CoT）上验证，更多样的 CoT 结构有待探索
- 论文提到会扩展到更大的模型
