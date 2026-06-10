# PaperNotes

论文阅读笔记，在线浏览：[fx-hit.github.io/PaperNotes](https://fx-hit.github.io/PaperNotes)

## 目录结构

```
YYYY-MM-DD/           # 按日期组织
├── *.md              # 图文笔记
├── *.html            # 独立 HTML 页面
└── assets/           # 插图
```

## 使用

```bash
# 转换单篇笔记、更新入口页并审计
./scripts/build_note.sh 2026-05-21/DreamZero_阅读笔记.md

# 仅重新生成首页和标签页
./generate_index.sh --site

# 全仓检查
python3 scripts/audit_notes.py --check-site
```

每篇笔记通过 [paper2notes](.claude/skills/paper2notes/SKILL.md) 生成 markdown，再通过 [md2html](.claude/skills/md2html/SKILL.md) 转为 HTML。站点元数据写在 Markdown 顶部，并由转换器写入 HTML：

```html
<!-- arxiv: 2602.15922 -->
<!-- venue: NVIDIA Tech Report 2026 -->
<!-- tags: WAM, 世界模型, 视频生成, 泛化 -->
```

## 常用提示词示例

```
1. /paper2notes 阅读 2026-05-25/diwa 代码和 2026-05-25/arXiv-2508.03645v1 论文，
并参考 AGENTS.md 规范撰写笔记。

2. 图注不够详细，要按照 paper2notes skill的要求撰写

3. 按 AGENTS.md 规范运行 scripts/audit_notes.py --check-site，检查文件组织结构，并做 commit 和 push
```
