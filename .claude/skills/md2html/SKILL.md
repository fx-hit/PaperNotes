---
name: md2html
description: Convert markdown files to standalone HTML with MathJax, Mermaid diagrams, syntax highlighting, and sidebar table of contents. This skill should be used when the user asks to "convert markdown to html", "转成html", "导出网页", "make an html version", or wants a rendered HTML version of markdown notes/documents. It handles LaTeX math, mermaid flowcharts, code blocks with syntax highlighting, and auto-generates a left sidebar TOC with scroll tracking.
---

# Markdown to HTML Converter

## 用法

用户说「转成html」或「/md2html」时，调用此技能：

```
python3 .claude/skills/md2html/scripts/convert.py <file1.md> [file2.md ...]
```

- 不传参数则转换当前目录所有 `.md` 文件
- 输出 HTML 文件与源 `.md` 同目录

## 转换内容

- **LaTeX 数学公式**：`$...$` 行内、`$$...$$` 和 `````math ````` 块级公式
- **Mermaid 流程图**：`````mermaid ````` 代码块
- **代码高亮**：Pygments 暗色主题
- **表格**、**引用**、**列表**等标准 markdown
- **侧边栏目录**：自动提取 h1-h4 生成，滚动时高亮当前章节
- **响应式布局**：窄屏自动折叠目录

## 依赖

- `python3`（需 `markdown` 和 `pygments` 库）
- 浏览器需要能访问 CDN 加载 MathJax / Mermaid（首次打开需要联网）
