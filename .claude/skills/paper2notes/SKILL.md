---
name: paper2notes
description: Read arxiv TeX paper source (and accompanying code if present), produce illustrated markdown notes with embedded figures, then convert to standalone HTML. Triggers when the user says "读论文做笔记", "paper2notes", "整理论文", "read this paper", or provides a directory containing arxiv TeX source.
---

# Paper → Illustrated Notes + HTML

将 arxiv TeX 论文源码（及配套代码仓库）转化为图文并茂的 markdown 笔记和独立 HTML 网页。

## 触发词

- "读论文做笔记" / "整理论文" / "paper2notes"
- 用户提供一个包含 `.tex` 文件的 arxiv 目录

## 工作流程

### 第一步：扫目录，定位论文和代码

```
ls -la <target_dir>/
```

- 找到 `.tex` 主文件（通常是 `main.tex`、`neurips_*.tex`、`cvpr.tex` 等）
- 找到 `fig/` 或 `figures/` 目录
- 找到 `sec/` 目录（分章节编译的论文）
- 检查是否有代码仓库（同目录下的项目目录，含 `pyproject.toml`、`README.md`、`setup.py` 等）

### 第二步：通读论文源码

按优先级阅读：
1. 主 `.tex` 文件（了解结构、section 组织、标题/作者/投稿方向）
2. `00README.json`（arxiv 元信息）
3. `sec/0_abstract.tex` → `sec/1_intro.tex` → `sec/2_*.tex` → ...（按顺序读所有章节）
4. `appendix.tex` 或 `sec/X_suppl.tex`（附录）
5. 注意 `\caption{}` 中的图标题——这些是插图的直接说明

### 第三步：阅读代码（如果有）

- 先读 `README.md` 了解项目结构
- 找到核心模型文件（通常在 `models/` 或 `src/` 下）
- 理解关键类/函数的论文→代码映射
- 记录到笔记的"代码实现解读"章节
- **必须用 ASCII 字符画**（`┌─┐│└─┘├┤┬┴┼` 等）绘制架构图、数据流图、推理流程图，帮助读者直观理解代码结构

### 第四步：写 Markdown 笔记

笔记结构规范：

```markdown
# 论文标题（英文）

> **论文信息**
> - 作者：...
> - 通讯作者：...
> - 投稿方向：...
> - arXiv ID：...
> - 代码：...（如果有）

---

## 一、核心问题

## 二、核心思路 / 方法

## 三、训练目标（如果有）

## 四、实验与结果

## 五、关键洞察与技术亮点

## 六、代码实现解读（如果有代码）

## 七、局限性（如果有）

---

## 八、关键概念速查
```

**写作要求**：
- 正文用中文叙述，关键术语保留英文
- 数学公式用 `$...$`（行内）和 `$$...$$`（块级）
- 数据用表格展示，对齐规范
- 强调**直觉理解**而非翻译论文，重点突出
- 引用原文 key finding 时用引用块 `>`
- 保留论文中的关键数字和对比数据

### 第五步：处理图片

```bash
python3 .claude/skills/paper2notes/scripts/pdf2jpg.py <fig_dir> --out-dir <output_dir> --quality 92
```

- 将论文的 PDF 图转为 JPG（质量 92%，宽度 1600px）
- PNG 原图直接复制后用 Pillow 转 JPG
- 根据论文中 `\includegraphics{fig/xxx}` 和图标题，选择 4–8 张关键图插入笔记

**插图规则**：
- 每张图标注 `![图X：描述](relative/path/to/fig.jpg)`
- 优先插入：总览图 → 架构图 → 方法细节图 → 实验结果图 → 可视化/案例图
- 图放在相关小节之后、下一小节之前

**多子图处理规则（重要）**：

很多论文使用 `\begin{subfigure}...\end{subfigure}` 将多个独立 PDF 组合成一个 figure（如消融实验的三栏对比图，或泛化分析的 3×3 网格图）。这类情况必须特别注意：

1. **在第二步通读 TeX 源码时**，识别所有 `\begin{subfigure}` / `\end{subfigure}` 环境，记录每个 figure 包含哪些子图文件。
2. **不要只插入一个子图来代表整个 figure**——这会导致读者只看到子图之一，无法理解完整内容。
3. **处理策略**：
   - **2-4 个子图**：使用 HTML `<table>` 并排布局（如 3 列 `<td width="33%">`），匹配 TeX 源码中的排列方式。
   - **5-9 个子图（网格型）**：使用 HTML `<table>` 按 TeX 的行列排布（如 3×3 网格用 `<tr>` 分组行、`<td>` 分组列）。
   - **10+ 个子图**：选择 2-3 个最有信息量的代表性子图，其余用文字描述。
4. **HTML 表格模板**：
   ```html
   <!-- 3列并排 -->
   <table><tr>
   <td width="33%"><img src="path/a.jpg" width="100%"><br><em>(a) 子图标题</em></td>
   <td width="33%"><img src="path/b.jpg" width="100%"><br><em>(b) 子图标题</em></td>
   <td width="33%"><img src="path/c.jpg" width="100%"><br><em>(c) 子图标题</em></td>
   </tr></table>
   
   <!-- 3×3网格 -->
   <table>
   <tr><td width="33%">...</td><td width="33%">...</td><td width="33%">...</td></tr>
   <tr><td width="33%">...</td><td width="33%">...</td><td width="33%">...</td></tr>
   <tr><td width="33%">...</td><td width="33%">...</td><td width="33%">...</td></tr>
   </table>
   ```
5. **注意文件映射**：子图文件通常是 `figures/xxx_a.pdf`、`figures/xxx_b.pdf` 等形式，但也可能是完全不相关的文件名（如 `libero_goal_unseen_2_new.pdf`、`libero_object_unseen_1_new.pdf`）。必须对照 TeX 中的 `\includegraphics` 命令确认哪些文件属于同一个 figure。
6. **编号一致性**：子图编号（a/b/c）应与论文原文一致，方便读者对照论文。

反面示例（禁止）：
```
*只插入 dynamic_sampling.jpg 一张图，但在图注中描述 a/b/c 三个子图。*
→ 读者只能看到 dynamic sampling 的曲线，看不到 clip higher 和 temperature 的图。
```

正面示例（必须达到的标准）：
```
分别插入图3a（dynamic_sampling.jpg）、图3b（clip_higher.jpg）、图3c（temperature_higher.jpg），
每张图配独立图注和解读。
```

**图表解读规则（重要）**：

每张图必须给出充足详细的解释，不能只翻译论文 caption。具体标准：

1. **首先查看图片内容**：用 Read 工具查看已转为 jpg 的图片，不能只看论文 caption 就写图注。
2. **拆解图片结构**：说明图片有几个子图（a/b/c...），每个子图展示的是什么内容。
3. **说明坐标轴和图例**：实验类图表必须解释横轴/纵轴的含义、不同颜色/线型代表什么、数据来源（多少个测试场景等）。
4. **给出关键数据**：标注图中最重要的数字、对比、趋势，并解释为什么这些数字重要。
5. **连接到论文论点**：解释这张图支持了论文的哪个主张，读者为什么应该关注这张图。
6. **逐子图分析**：多子图（如消融实验的 a/b/c 三栏）必须逐个解读，每个子图至少 2-3 句话。

反面示例（禁止）：
```
*图3：消融实验。我们评估了 (a) latent 表征方法、(b) 不同 latent CoT 长度、(c) 自适应 CoT 长度策略。*
```

正面示例（必须达到的标准）：

```
*图3：三组核心消融实验（均在 LIBERO-Spatial 上评估）。*

**子图 (a) Latent 表征方式对比（RL 后 SR）：**
对比四种构造 latent target 的方法——Global Pooling（96.8%）、Convolution（98.4%）、Q-Former（97.2%）、DINOv3（99.8%）。DINOv3 最优，因为它利用了预训练的语义丰富特征，top-k 选择保留了最显著的视觉分量。Global Pooling 最差，简单平均丢失了空间结构。

**子图 (b) Latent CoT 固定长度对比：**
测试 1/2/4/8 四种固定长度 vs Action-Only 基线（95.0%）。关键发现：即使只有 1 个 latent token（96.2%）也优于纯 action-only，验证了"先想再做"设计本身就优于纯 reaction。从 4→8 收益递减（97.8%→98.4%），说明 8 token 已足够。

**子图 (c) Adaptive 候选位置数量 M 对比：**
M=4 最优（99.8%），M=8 反而降到 99.0%——停止位置过多会增加探索方差。所有 adaptive 设置（M≥2）都优于固定长度（M=1 的 98.4%），证明动态推理本身就有正向收益。
```

### 第六步：自查图表与代码解读质量

这是**必须执行**的步骤，写完初版笔记后应立即逐条检查。不满足标准的必须修改后再进入 HTML 转换。

#### 6.1 图表解释充分性检查

对笔记中每一张图，逐条检查：

| 检查项 | 标准 | 不通过时 |
|--------|------|----------|
| **多子图完整性** | 对于 TeX 中使用 `\begin{subfigure}` 的复合图，是否每个子图都有独立的图片嵌入？是否只插入了其中一个子图就描述了全部？ | 对照 TeX 源码补全所有子图图片（2-4 个全部嵌入，5+ 个选代表） |
| **子图拆解** | 多子图（a/b/c）是否逐个解读？每子图至少 2-3 句 | 补充每个子图的分析 |
| **坐标轴/图例** | 实验图表是否说明了横轴、纵轴、颜色/线型含义？ | 从论文正文和 caption 中提取横纵轴信息 |
| **关键数据** | 是否标注了图中最重要的数字和趋势？ | 从论文正文提取具体数值写入图注 |
| **连接论点** | 是否解释了该图支持论文的哪个主张？ | 添加"为什么这张图重要"的解读 |
| **图片内容准确** | 图注描述是否与图片实际内容一致？是否错误描述了不存在的图表类型（如将表格描述为柱状图）？ | 用 Read 工具实际查看图片内容后修正描述 |
| **长度判断** | 图注长度是否明显短于正文中相邻段落的长度？如果只有 1-2 行，大概率不够详细 | 参考正面示例格式扩展 |

**反面示例（禁止，应被自查发现并修改）**：

```
*图3：消融实验。我们评估了 (a) latent 表征方法、(b) 不同 latent CoT 长度、(c) 自适应 CoT 长度策略。*
→ 问题：只翻译 caption，没有解读子图、没有关键数据、没有连接论文论点
```

**必须达标**：每张图的解读应包含子图拆解、关键数据、论文论点连接。参照第五步中的正面示例。

#### 6.2 代码解读章节检查

| 检查项 | 标准 | 不通过时 |
|--------|------|----------|
| **架构字符画** | 模型部分是否有 ASCII art 展示数据流/架构？ | 补充 `┌─┐` 风格字符画 |
| **关键函数映射** | 论文中的公式/算法是否映射到具体代码函数？ | 补充"公式 X → model.py:line Y"映射 |
| **推理流程可视化** | 推理/部署部分是否有流程字符画？ | 补充时间轴或流程图 |

#### 6.3 关键数据完整性检查

扫描笔记全文，确保以下关键数字已出现：
- 论文的主要 benchmark 数字（SR、PS 等）
- 对比方法的关键数字
- 消融实验的关键数字
- 模型参数量、训练数据量、推理速度等

#### 6.4 自查后处理

- 若发现图表解释不足：**立即修改**对应图注，补充详细解读
- 若发现缺少字符画：**立即补充** ASCII art 到代码解读章节
- 修改完成后**重新转 HTML**，确保最终版本包含所有改进

### 第七步：转为 HTML

调用 md2html 技能：

```
python3 .claude/skills/md2html/scripts/convert.py <path/to/笔记.md>
```

### 第八步：验证

- 确认 HTML 文件生成成功
- 确认图片路径在 HTML 中可访问（相对路径正确）
- 确认主要章节都在目录中
- **确认自查中发现的问题已全部修复**（对照第六步检查清单）

## 依赖

- `python3`（Pillow 库，用于图片格式转换）
- macOS 无需额外依赖（使用内置 `qlmanage`）
- Linux：需要 `poppler-utils` 或 `PyMuPDF`
- `markdown` + `pygments`（md2html 依赖）

## 注意事项

- 图片默认生成为 JPG 而非 PNG（体积小 3–7 倍）
- 论文"投稿中"时标注为"投稿（under review）"，不写"已发表"
- 笔记 md 文件放在论文目录的**上级目录**（与 arxiv 目录平级），便于管理
