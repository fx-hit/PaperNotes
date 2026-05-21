# AGENTS.md

PaperNotes — 论文阅读笔记仓库。按日期组织，每篇论文一个子目录，包含 arxiv TeX 源码、官方代码（如提供）、以及生成的图文笔记和独立 HTML 页面。

## 目录结构

```
PaperNotes/
├── YYYY-MM-DD/                  # 按论文阅读日期分组
│   ├── arXiv-XXXX.XXXXXvX/      # arxiv TeX 源码（不提交到 git）
│   ├── paper_code_repo/         # 论文官方代码（不提交到 git）
│   ├── assets/                  # 笔记插图资源（提交）
│   │   ├── paper-name-1/        # 每篇论文独立子目录，存放其插图
│   │   └── paper-name-2/
│   ├── paper_notes.md           # 图文笔记（提交）
│   └── paper_notes.html         # 独立 HTML 页面（提交）
├── .gitignore
├── AGENTS.md
├── generate_index.sh            # 生成 index.html 和 tags.html
├── index.html                   # GitHub Pages 默认首页：按时间倒序浏览（生成物）
├── tags.html                    # 标签浏览页：按标签分组浏览（生成物）
└── .claude/                     # Claude Code 配置
```

> **assets/ 组织规则**：每篇论文在 `assets/` 下拥有独立子目录（如 `assets/dreamzero/`），图片按论文归类存放，避免扁平的散文件混杂。HTML 和 MD 中的图片引用路径统一为 `assets/<paper-name>/xxx.jpg`。

## 命名规范

### 笔记文件

统一使用 `<EnglishShortName>_阅读笔记.md` / `.html` 格式：

```
2026-05-21/DreamZero_阅读笔记.md
2026-05-21/DreamZero_阅读笔记.html
```

- `EnglishShortName` 取论文简称，用连字符 `-` 或下划线 `_` 连接，不含中文和特殊符号
- 同一篇论文有多个笔记时加副标题：`LaST-R1_分享笔记.md`、`LaST-R1_论文与代码笔记.md`

### assets 子目录

与笔记文件名对应：

| 笔记文件 | assets 子目录 |
|---------|-------------|
| `DreamZero_阅读笔记.md` | `assets/dreamzero/` |
| `Fast-WAM_阅读笔记.md` | `assets/fastwam/` |
| `VGGT-Omega_阅读笔记.md` | `assets/vggt-omega/` |

规则：取 `EnglishShortName`，全小写，连字符保持不变。

## 工作流

1. **读论文做笔记** — 对包含 arxiv TeX 源码的目录使用 `paper2notes` skill，生成带有插图的 markdown 笔记。
2. **导出 HTML** — 使用 `md2html` skill 将 markdown 笔记转换为独立 HTML 页面，包含 MathJax、Mermaid 图表、代码高亮和侧边栏目录。
3. **添加元数据** — 在 HTML 页面的 `<title>` 后添加 HTML 注释格式的元数据，以便首页自动提取：
   ```html
   <!-- arxiv: XXXX.XXXXX -->
   <!-- venue: 会议/期刊 年份 -->
   <!-- tags: 标签1, 标签2, 标签3 -->
   ```
   - **venue 格式**：已发表写 `会议 年份`（如 `CVPR 2026 Oral`、`ICML 2026 Spotlight`），投稿中写 `会议 年份（投稿中）`（如 `NeurIPS 2026（投稿中）`），技术报告写 `机构/公司 技术报告`（如 `NVIDIA Tech Report 2026`、`蚂蚁灵波技术报告`）。每个 HTML 必须有 venue，禁止缺失。
   - **tags 规范**：标签用于站点级浏览，不是关键词堆砌。每篇论文建议 2-5 个宽粒度、可复用标签，优先复用既有规范标签，如 `机器人`、`VLA`、`WAM`、`世界模型`、`推理`、`视频生成`、`3D重建`、`策略学习`、`泛化`、`强化学习`、`多模态`、`自动驾驶`。
   - 避免同义标签重复：统一写 `WAM`，不要写 `World Action Model`、`World-Action Model`、`世界动作模型`；统一写 `机器人`，不要写 `Robotics`、`Robot Manipulation`、`机器人操作`、`机器人操控`；统一写 `推理`，不要拆成 `隐式推理`、`因果推理`、`流式推理`；统一写 `视频生成`，不要拆成 `视频扩散`、`视频扩散模型`、`Video Prediction`。
   - 除非预计后续会有多篇论文复用，否则不要新增论文专属的一次性方法标签（如某个 benchmark 名、模块名、损失名）。这类细节留在正文，不进入 `tags`。
4. **更新站点入口** — 运行 `./generate_index.sh --site` 自动重新生成 `index.html` 和 `tags.html`。
   - GitHub Pages 默认打开根目录 `index.html`，该页按日期倒序浏览论文；页面顶部提供"按时间 / 按标签"导航，可切换到 `tags.html`。
   - `tags.html` 按 `<!-- tags: -->` 元数据自动分组，标签导航按论文数量降序排列；当前只展示至少关联 2 篇论文的标签，避免单篇标签污染浏览入口。不要手工维护标签计数、分组或论文列表。
   - `generate_index.sh` 仅扫描日期目录**顶层**（`20??-??-??/*.html`），因此笔记 HTML 必须放在 `YYYY-MM-DD/` 目录下，不能放在子目录中。
   - 两个入口页都从 HTML 注释中提取元数据（`<!-- arxiv: -->`、`<!-- venue: -->`、`<!-- tags: -->`）。每篇笔记必须提供 `venue` 和 `tags`，否则首页/标签页信息会缺失。
   - 首页的"更新于"日期通过 `git log -1 --format="%cs"` 取笔记 HTML 的最后提交日期；`git log` 只能查到已提交文件。因此新增或修改笔记后，先提交笔记文件，再运行 `./generate_index.sh --site`，然后把生成的 `index.html` 和 `tags.html` amend 到同一个 commit，或单独再提交一次入口页更新。
   - `generate_index.sh` 兼容旧用法：`./generate_index.sh` 或 `./generate_index.sh --index` 输出按时间首页到 stdout，`./generate_index.sh --tags` 输出标签页到 stdout；日常维护统一使用 `./generate_index.sh --site`。
   - 首页和标签页不依赖任何外部构建工具或静态站点生成器，每次新增或修改笔记后重新运行脚本并 commit 生成的 `index.html`、`tags.html` 即可。
5. **提交** — 按 commit 规范提交笔记文件。

## Commit 规范

所有 commit message 末尾必须包含共同作者 trailer，署名取决于提交者身份：

### 由 AI 代理提交

```
Co-Authored-By: Claude Code <模型名> <noreply@anthropic.com>
Co-Authored-By: Codex <模型名> <noreply@openai.com>
```

示例：
```
Add paper notes for LaST-R1

Co-Authored-By: Claude Code Opus 4.7 <noreply@anthropic.com>
```

### 由人类提交（fx-hit）

无需添加 trailer。

## Git 忽略规则

`.gitignore` 自动排除以下内容：
- arxiv TeX 源码目录（`**/arXiv-*/`）
- 论文官方代码目录（`**/*_code/` 及已知项目目录）
- PDF 论文
- conda 环境（`**/.conda/`）
- 编译产物、二进制文件、macOS 系统文件
- 日期子目录下的 `.claude/`（根目录 `.claude/` 保留）

### 新增代码目录

每次添加新的代码仓库时，在 `.gitignore` 的"Known code directories"下追加一行：

```
*/<repo-name>/
```

`*/` 前缀确保只匹配日期目录层级的代码目录，不会误伤 `assets/` 下同名的子目录。例如：

```
*/dreamzero/
*/FastWAM/
*/giga-world-policy/
```

此步骤容易遗漏，添加新论文代码后务必检查 `git status` 是否出现了不该提交的代码文件。
