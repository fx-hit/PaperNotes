#!/bin/bash
# Generate PaperNotes GitHub Pages entry files from dated HTML notes.
#
# Usage:
#   ./generate_index.sh          # print chronological index HTML to stdout
#   ./generate_index.sh --index  # print chronological index HTML to stdout
#   ./generate_index.sh --tags   # print tag-browsing HTML to stdout
#   ./generate_index.sh --site   # write index.html and tags.html

set -euo pipefail

cd "$(dirname "$0")"

mode="${1:---index}"

case "$mode" in
    --index|--tags|--site)
        ;;
    *)
        echo "Usage: $0 [--index|--tags|--site]" >&2
        exit 2
        ;;
esac

python3 - "$mode" <<'PY'
import html
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


MODE = sys.argv[1]
ROOT = Path.cwd()
TAG_PAGE_MIN_COUNT = 1

TAG_ALIASES = {
    "World Action Model": "WAM",
    "World-Action Model": "WAM",
    "世界动作模型": "WAM",
    "3D感知": "3D重建",
    "4D重建": "3D重建",
    "前馈模型": "3D重建",
    "视觉几何": "3D重建",
    "场景重建": "3D重建",
    "深度估计": "3D重建",
    "视频扩散": "视频生成",
    "视频扩散模型": "视频生成",
    "Video Prediction": "视频生成",
    "Flow Matching": "视频生成",
    "零样本泛化": "泛化",
    "跨具身迁移": "泛化",
    "Generalization": "泛化",
    "Knowledge Distillation": "知识蒸馏",
    "Parameter-Efficient Transfer": "知识蒸馏",
}


@dataclass(frozen=True)
class Note:
    path: str
    date: str
    title: str
    arxiv: str
    venue: str
    tags: tuple[str, ...]
    words_zh: str
    words_en: str
    reading_time: str
    updated: str


def extract_comment(text: str, key: str) -> str:
    match = re.search(rf"<!--\s*{re.escape(key)}:\s*(.*?)\s*-->", text, re.S)
    return match.group(1).strip() if match else ""


def extract_title(text: str, fallback: str) -> str:
    match = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
    if not match:
        return fallback
    return re.sub(r"\s+", " ", match.group(1)).strip() or fallback


def git_updated(path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", path],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def normalize_tags(tags: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        canonical = TAG_ALIASES.get(tag, tag)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return tuple(normalized)


def load_notes() -> list[Note]:
    notes: list[Note] = []
    errors: list[str] = []
    for path in sorted(ROOT.glob("20??-??-??/*.html"), key=lambda p: (p.parent.name, p.name)):
        rel_path = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        tags = normalize_tags([tag.strip() for tag in extract_comment(text, "tags").split(",") if tag.strip()])
        arxiv = extract_comment(text, "arxiv")
        venue = extract_comment(text, "venue")
        if not arxiv:
            errors.append(f"{rel_path}: missing <!-- arxiv: ... -->")
        if not venue:
            errors.append(f"{rel_path}: missing <!-- venue: ... -->")
        if not tags:
            errors.append(f"{rel_path}: missing <!-- tags: ... -->")
        notes.append(
            Note(
                path=rel_path,
                date=path.parent.name,
                title=extract_title(text, path.stem),
                arxiv=arxiv,
                venue=venue,
                tags=tags,
                words_zh=extract_comment(text, "words-zh"),
                words_en=extract_comment(text, "words-en"),
                reading_time=extract_comment(text, "reading-time"),
                updated=git_updated(rel_path),
            )
        )
    if errors:
        print("Metadata errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    return notes


def chronological_notes(notes: list[Note]) -> list[Note]:
    ordered: list[Note] = []
    for date in sorted({note.date for note in notes}, reverse=True):
        ordered.extend(sorted((note for note in notes if note.date == date), key=lambda note: note.path.casefold()))
    return ordered


def e(value: str) -> str:
    return html.escape(value, quote=True)


def tag_anchor(tag: str, used: set[str]) -> str:
    anchor = re.sub(r"\s+", "-", tag.strip())
    anchor = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", anchor).strip("-")
    if not anchor:
        anchor = "tag"
    original = anchor
    suffix = 2
    while anchor in used:
        anchor = f"{original}-{suffix}"
        suffix += 1
    used.add(anchor)
    return anchor


BASE_STYLE = """  :root {
    --bg: #fafafa;
    --card-bg: #ffffff;
    --text: #1a1a1a;
    --text-secondary: #6b7280;
    --border: #e5e7eb;
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --tag-bg: #eff6ff;
    --tag-text: #1e40af;
    --tag-active-bg: #dbeafe;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }
  .container { max-width: 840px; margin: 0 auto; padding: 48px 24px 80px; }

  header {
    text-align: center;
    padding: 60px 0 48px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 48px;
  }
  header h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
  }
  header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
  }
  header .site-intro {
    max-width: 680px;
    margin: 14px auto 0;
    color: #4b5563;
    font-size: 0.95rem;
  }
  .nav-bar {
    text-align: center;
    margin-top: 24px;
  }
  .nav-bar a {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin: 0 4px;
    transition: background 0.15s, color 0.15s;
  }
  .nav-bar a:hover {
    background: var(--tag-bg);
    color: var(--accent);
  }
  .nav-bar a.active {
    background: var(--accent);
    color: #fff;
  }

  .paper-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
  }
  .paper-card:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-color: #c7d2fe;
  }
  .paper-card a {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent);
    text-decoration: none;
  }
  .paper-card a:hover { color: var(--accent-hover); }
  .paper-meta {
    margin-top: 6px;
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  .paper-meta a {
    font-size: 0.85rem;
    font-weight: 400;
  }
  .paper-venue {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 6px;
    font-weight: 600;
  }

  footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.8rem;
    padding: 32px 0;
    border-top: 1px solid var(--border);
    margin-top: 48px;
  }
  footer a { color: var(--accent); text-decoration: none; }"""


INDEX_EXTRA_STYLE = """
  .date-group { margin-bottom: 40px; }
  .date-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    padding-left: 4px;
  }
  .paper-card {
    padding: 20px 24px;
    margin-bottom: 10px;
  }
  .paper-tag {
    display: inline-block;
    background: var(--tag-bg);
    color: var(--tag-text);
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 6px;
    font-weight: 500;
  }
  .paper-updated {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-left: 8px;
  }

  .pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 6px;
    margin: 36px 0;
    flex-wrap: wrap;
  }
  .pagination a,
  .pagination .current,
  .pagination .disabled,
  .pagination span.page-info {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 36px;
    padding: 0 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    text-decoration: none;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    background: var(--card-bg);
    transition: all 0.15s;
  }
  .pagination a:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .pagination .current {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .pagination .disabled {
    opacity: 0.35;
    pointer-events: none;
  }
  .pagination span.page-info {
    border: none;
    background: none;
    font-size: 0.8rem;
    color: #9ca3af;
    margin-right: 8px;
  }"""


TAGS_EXTRA_STYLE = """
  .tag-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 36px;
    padding: 16px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .tag-nav a {
    display: inline-block;
    font-size: 0.8rem;
    padding: 3px 10px;
    border-radius: 4px;
    text-decoration: none;
    color: var(--tag-text);
    background: var(--tag-bg);
    font-weight: 500;
    transition: background 0.15s;
    white-space: nowrap;
  }
  .tag-nav a:hover { background: var(--tag-active-bg); }
  .tag-nav a .count {
    color: var(--text-secondary);
    font-weight: 400;
    margin-left: 2px;
    font-size: 0.75rem;
  }
  .tag-group { margin-bottom: 10px; }
  .tag-group[open] { margin-bottom: 28px; }
  .tag-group:target { scroll-margin-top: 24px; }
  .tag-label {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    padding: 8px 12px;
    background: var(--tag-bg);
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    list-style: none;
    user-select: none;
  }
  .tag-label::-webkit-details-marker { display: none; }
  .tag-label::marker { content: ""; }
  .tag-heading {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .tag-label .tag-count {
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-secondary);
    background: var(--card-bg);
    padding: 2px 8px;
    border-radius: 10px;
  }
  .tag-toggle {
    margin-left: auto;
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--accent);
  }
  .tag-toggle::after { content: "展开"; }
  .tag-group[open] .tag-toggle::after { content: "收起"; }
  .tag-panel {
    padding-top: 12px;
  }
  .tag-panel-actions {
    text-align: right;
    margin-bottom: 8px;
  }
  .back-top {
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--accent);
    text-decoration: none;
  }
  .back-top:hover { text-decoration: underline; }
  .paper-card {
    padding: 16px 24px;
    margin-bottom: 8px;
  }
  .paper-card a { font-size: 1rem; }
  .paper-date {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-left: 8px;
  }"""


def document_head(title: str, extra_style: str) -> list[str]:
    return [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"<title>{e(title)}</title>",
        "<style>",
        BASE_STYLE,
        extra_style,
        "</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        "",
    ]


def header(subtitle: str, active: str, intro: str = "") -> list[str]:
    index_active = ' class="active"' if active == "index" else ""
    tags_active = ' class="active"' if active == "tags" else ""
    lines = [
        "<header>",
        "  <h1>PaperNotes</h1>",
        f"  <p>{e(subtitle)}</p>",
    ]
    if intro:
        lines.append(f'  <p class="site-intro">{e(intro)}</p>')
    lines += [
        '  <div class="nav-bar">',
        f'    <a href="index.html"{index_active}>按时间</a>',
        f'    <a href="tags.html"{tags_active}>按标签</a>',
        "  </div>",
        "</header>",
        "",
    ]
    return lines


def footer() -> list[str]:
    return [
        "",
        "<footer>",
        '  Powered by <a href="https://github.com/fx-hit/PaperNotes">GitHub Pages</a>',
        "  &nbsp;·&nbsp;",
        "  Generated with Claude Code and Codex",
        "</footer>",
        "",
        "</div>",
        "</body>",
        "</html>",
    ]


def tag_page_script() -> list[str]:
    return [
        "<script>",
        "  function openTagGroupFromHash() {",
        "    const id = decodeURIComponent(window.location.hash.slice(1));",
        "    if (!id || id === 'top') return;",
        "    const target = document.getElementById(id);",
        "    if (target && target.tagName.toLowerCase() === 'details') {",
        "      target.open = true;",
        "      target.scrollIntoView({ block: 'start' });",
        "    }",
        "  }",
        "  document.querySelectorAll('.tag-nav a').forEach((link) => {",
        "    link.addEventListener('click', () => {",
        "      const id = decodeURIComponent(link.hash.slice(1));",
        "      const target = document.getElementById(id);",
        "      if (target && target.tagName.toLowerCase() === 'details') {",
        "        target.open = true;",
        "      }",
        "    });",
        "  });",
        "  window.addEventListener('hashchange', openTagGroupFromHash);",
        "  openTagGroupFromHash();",
        "</script>",
    ]


def note_meta(note: Note, include_reading: bool, include_updated: bool) -> str:
    meta = ""
    if note.arxiv and note.arxiv.upper() not in {"N/A", "NA", "NONE"}:
        meta += f"arXiv: {e(note.arxiv)}"
    if note.venue:
        meta += f'<span class="paper-venue">{e(note.venue)}</span>'
    if include_reading and note.reading_time:
        reading_parts: list[str] = []
        if note.words_zh.isdigit() and int(note.words_zh) > 0:
            reading_parts.append(f"{e(note.words_zh)} 字")
        if note.words_en.isdigit() and int(note.words_en) > 0:
            reading_parts.append(f"English {e(note.words_en)} 词")
        reading_text = " / ".join(reading_parts)
        if reading_text:
            meta += f" · {reading_text} · 约 {e(note.reading_time)} 分钟"
        else:
            meta += f" · 约 {e(note.reading_time)} 分钟"
    if include_updated and note.updated:
        meta += f'<span class="paper-updated">更新于 {e(note.updated)}</span>'
    return meta


def index_page_script() -> list[str]:
    return [
        "<script>",
        "(function() {",
        "  var PAGE_SIZE = 10;",
        "  var cards = document.querySelectorAll('.paper-card');",
        "  var total = cards.length;",
        "  if (total === 0) return;",
        "  var totalPages = Math.ceil(total / PAGE_SIZE);",
        "  if (totalPages <= 1) {",
        "    var ph = document.querySelectorAll('.pagination');",
        "    for (var i = 0; i < ph.length; i++) ph[i].style.display = 'none';",
        "    return;",
        "  }",
        "",
        "  function getPage() {",
        "    var m = window.location.hash.match(/^#(\\d+)$/);",
        "    if (m) {",
        "      var p = parseInt(m[1], 10);",
        "      if (p >= 1 && p <= totalPages) return p;",
        "    }",
        "    if (window.location.hash) {",
        "      history.replaceState(null, '', window.location.pathname);",
        "    }",
        "    return 1;",
        "  }",
        "",
        "  var currentPage = getPage();",
        "",
        "  function showPage(page) {",
        "    var start = (page - 1) * PAGE_SIZE;",
        "    var end = Math.min(page * PAGE_SIZE, total);",
        "    for (var i = 0; i < cards.length; i++) {",
        "      cards[i].style.display = (i >= start && i < end) ? '' : 'none';",
        "    }",
        "    var groups = document.querySelectorAll('.date-group');",
        "    for (var g = 0; g < groups.length; g++) {",
        "      var gc = groups[g].querySelectorAll('.paper-card');",
        "      var allHidden = true;",
        "      for (var c = 0; c < gc.length; c++) {",
        "        if (gc[c].style.display !== 'none') { allHidden = false; break; }",
        "      }",
        "      groups[g].style.display = allHidden ? 'none' : '';",
        "    }",
        "    var info = '第 ' + page + ' 页 / 共 ' + totalPages + ' 页（' + (start + 1) + '-' + end + ' / ' + total + ' 篇）';",
        "    var html = '<span class=\"page-info\">' + info + '</span>';",
        "    html += (page > 1) ? '<a href=\"#' + (page - 1) + '\">&larr; 上一页</a>' : '<span class=\"disabled\">&larr; 上一页</span>';",
        "    for (var p = 1; p <= totalPages; p++) {",
        "      html += (p === page) ? '<span class=\"current\">' + p + '</span>' : '<a href=\"#' + p + '\">' + p + '</a>';",
        "    }",
        "    html += (page < totalPages) ? '<a href=\"#' + (page + 1) + '\">下一页 &rarr;</a>' : '<span class=\"disabled\">下一页 &rarr;</span>';",
        "    document.getElementById('pagination-bottom').innerHTML = html;",
        "    currentPage = page;",
        "  }",
        "",
        "  showPage(currentPage);",
        "",
        "  window.addEventListener('hashchange', function() {",
        "    var p = getPage();",
        "    if (p !== currentPage) showPage(p);",
        "  });",
        "})();",
        "</script>",
    ]


def render_index(notes: list[Note]) -> str:
    ordered = chronological_notes(notes)
    lines = document_head("PaperNotes — 论文阅读笔记", INDEX_EXTRA_STYLE)
    lines += header(
        "一个由人类审阅、Claude Code 和 Codex 协作更新的论文阅读笔记站。",
        "index",
        "Agent 负责生成初稿和维护页面，我负责检查理解是否正确、指出问题并推动修改。这里不是自动摘要合集，而是人类判断与 AI 执行力共同产出的阅读记录。",
    )

    current_date = None
    for note in ordered:
        if note.date != current_date:
            if current_date is not None:
                lines.append("</div>")
            current_date = note.date
            lines.extend(
                [
                    "",
                    '<div class="date-group">',
                    f'  <div class="date-label">{e(note.date)}</div>',
                ]
            )
        lines.extend(
            [
                '  <div class="paper-card">',
                f'    <a href="{e(note.path)}" target="_blank" rel="noopener">{e(note.title)}</a>',
                f'    <div class="paper-meta">{note_meta(note, include_reading=True, include_updated=True)}</div>',
            ]
        )
        if note.tags:
            tags = "".join(f'<span class="paper-tag">{e(tag)}</span>' for tag in note.tags)
            lines.append(f"    <div>{tags}</div>")
        lines.append("  </div>")

    if current_date is not None:
        lines.append("</div>")

    lines.append('<div class="pagination" id="pagination-bottom"></div>')
    lines += index_page_script()
    lines += footer()
    return "\n".join(lines) + "\n"


def render_tags(notes: list[Note]) -> str:
    by_tag: dict[str, list[Note]] = defaultdict(list)
    for note in chronological_notes(notes):
        for tag in note.tags:
            by_tag[tag].append(note)

    counts = Counter({tag: len(tag_notes) for tag, tag_notes in by_tag.items()})
    tags = sorted((tag for tag, count in counts.items() if count >= TAG_PAGE_MIN_COUNT), key=lambda tag: (-counts[tag], tag.casefold()))
    used_anchors: set[str] = set()
    anchors = {tag: tag_anchor(tag, used_anchors) for tag in tags}

    lines = document_head("按标签浏览 — PaperNotes", TAGS_EXTRA_STYLE)
    lines += header(
        "一个由人类审阅、Claude Code 和 Codex 协作更新的论文阅读笔记站。",
        "tags",
        "Agent 负责生成初稿和维护页面，我负责检查理解是否正确、指出问题并推动修改。这里不是自动摘要合集，而是人类判断与 AI 执行力共同产出的阅读记录。",
    )

    lines.append('<div class="tag-nav" id="top">')
    for tag in tags:
        lines.append(f'  <a href="#{e(anchors[tag])}"><span class="count">{counts[tag]}</span> {e(tag)}</a>')
    lines.append("</div>")

    for tag in tags:
        icon = "🤖" if tag == "机器人" else "🏷️"
        lines.extend(
            [
                "",
                f'<!-- ====== {e(tag)} ({counts[tag]}) ====== -->',
                f'<details class="tag-group" id="{e(anchors[tag])}">',
                '  <summary class="tag-label">',
                f'    <span class="tag-heading">{icon} {e(tag)} <span class="tag-count">{counts[tag]} 篇</span></span>',
                '    <span class="tag-toggle" aria-hidden="true"></span>',
                "  </summary>",
                '  <div class="tag-panel">',
                '    <div class="tag-panel-actions"><a class="back-top" href="#top">↑ 回到标签列表</a></div>',
            ]
        )
        for note in by_tag[tag]:
            lines.extend(
                [
                    '    <div class="paper-card">',
                    f'      <a href="{e(note.path)}" target="_blank" rel="noopener">{e(note.title)}</a>',
                    f'      <div class="paper-meta">{note_meta(note, include_reading=False, include_updated=False)}<span class="paper-date">{e(note.date)}</span></div>',
                    "    </div>",
                ]
            )
        lines.extend(["  </div>", "</details>"])

    lines += tag_page_script()
    lines += footer()
    return "\n".join(lines) + "\n"


notes = load_notes()

if MODE == "--index":
    sys.stdout.write(render_index(notes))
elif MODE == "--tags":
    sys.stdout.write(render_tags(notes))
elif MODE == "--site":
    (ROOT / "index.html").write_text(render_index(notes), encoding="utf-8")
    (ROOT / "tags.html").write_text(render_tags(notes), encoding="utf-8")
    print(f"Wrote index.html + tags.html, {len(notes)} notes")
PY
