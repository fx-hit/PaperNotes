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
└── .claude/                     # Claude Code 配置
```

> **assets/ 组织规则**：每篇论文在 `assets/` 下拥有独立子目录（如 `assets/dreamzero/`），图片按论文归类存放，避免扁平的散文件混杂。HTML 和 MD 中的图片引用路径统一为 `assets/<paper-name>/xxx.jpg`。

## 工作流

1. **读论文做笔记** — 对包含 arxiv TeX 源码的目录使用 `paper2notes` skill，生成带有插图的 markdown 笔记。
2. **导出 HTML** — 使用 `md2html` skill 将 markdown 笔记转换为独立 HTML 页面，包含 MathJax、Mermaid 图表、代码高亮和侧边栏目录。
3. **添加元数据** — 在 HTML 页面的 `<title>` 后添加 HTML 注释格式的元数据，以便首页自动提取：
   ```html
   <!-- arxiv: XXXX.XXXXX -->
   <!-- venue: 会议/期刊 年份 -->
   <!-- tags: 标签1, 标签2, 标签3 -->
   ```
4. **更新首页** — 运行 `./generate_index.sh` 重新生成 `index.html`。
   - `generate_index.sh` 仅扫描日期目录**顶层**（`find YYYY-MM-DD/ -maxdepth 1 -name "*.html"`），因此笔记 HTML 必须放在 `YYYY-MM-DD/` 目录下，不能放在子目录中。
   - 首页从 HTML 注释中提取元数据（`<!-- arxiv: -->`、`<!-- venue: -->`、`<!-- tags: -->`），"更新于" 日期则通过 `git log -1 --format="%cs"` 取该文件的最后提交日期。因此即使只修改了图片路径等非内容变更，commit 时该文件的时间戳也会刷新。
   - 首页不依赖任何外部构建工具或静态站点生成器，每次新增或修改笔记后重新运行脚本并 commit 生成的 `index.html` 即可。
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

如需添加新的代码目录排除规则，编辑 `.gitignore` 并在"Known code directories"下添加。
