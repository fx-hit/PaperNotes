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
# 生成首页
./generate_index.sh > index.html
```

每篇笔记通过 [paper2notes](.claude/skills/paper2notes/SKILL.md) 生成 markdown，再通过 [md2html](.claude/skills/md2html/SKILL.md) 转为 HTML。
