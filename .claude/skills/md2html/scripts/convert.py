#!/usr/bin/env python3
"""Convert markdown files to standalone HTML with MathJax, Mermaid, and sidebar TOC.

Usage:
    python3 convert.py file1.md [file2.md ...]
    python3 convert.py                    # converts all .md in current dir

Output: <filename>.html alongside each input file.
"""

import re
import sys
import os
import markdown
from markdown.extensions import codehilite, fenced_code, tables, toc


# ── nested list indent fix ─────────────────────────────────────────────────
# Python-Markdown 要求嵌套列表用 4 空格缩进，但很多 md 文件只有 2 空格。
# 此函数将 2 空格嵌套列表缩进统一翻倍为 4 空格，避免所有子项被拍平到顶层。

def fix_nested_list_indent(text):
    """将偶数缩进的列表项缩进翻倍 (2→4, 4→8...)，使嵌套列表正确渲染。"""
    lines = text.split('\n')
    result = []
    for line in lines:
        m = re.match(r'^(\s{2,})([-*]\s)', line)
        if m:
            indent = len(m.group(1))
            # 只处理 2 的倍数但不是 4 的倍数等混合情况：只要是 ≥2 的缩进统一翻倍
            new_indent = indent * 2
            line = ' ' * new_indent + line.lstrip(' ')
        result.append(line)
    return '\n'.join(result)


# ── blank line before list fix ─────────────────────────────────────────────
# Python-Markdown 要求在列表前有空行，否则会将 list item 当作段落文本。
# 此函数在每个顶层列表项（非缩进、非紧跟前一个列表项）前插入空行。

def fix_list_blank_lines(text):
    """在列表项前确保有空行，避免列表被当作文本渲染。"""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        stripped = line.lstrip(' ')
        is_list_item = bool(re.match(r'^[-*]\s', stripped))
        # 检查是否是编号列表项 (1. 2. 等)
        is_ordered = bool(re.match(r'^\d+\.\s', stripped))
        is_list = is_list_item or is_ordered

        # 检查上一行（跳过空白行）
        prev_nonempty = None
        for j in range(i - 1, -1, -1):
            if lines[j].strip():
                prev_nonempty = lines[j]
                break

        if is_list and prev_nonempty is not None:
            prev_stripped = prev_nonempty.lstrip(' ')
            prev_is_list = bool(re.match(r'^[-*]\s', prev_stripped)) or bool(re.match(r'^\d+\.\s', prev_stripped))
            # 上一行不是列表项、不是空行、不是表格行、不是代码块
            if not prev_is_list:
                prev_is_special = (
                    prev_stripped.startswith('|') or
                    prev_stripped.startswith('```') or
                    prev_stripped.startswith('>') or
                    prev_stripped.startswith('#')
                )
                if not prev_is_special:
                    result.append('')
        result.append(line)
    return '\n'.join(result)


def protect_special_blocks(text):
    placeholders = []
    counter = [0]

    def add(original, kind):
        counter[0] += 1
        key = f"SPLACEHOLDER{counter[0]:06d}P"
        placeholders.append((key, original, kind))
        return key

    text = re.sub(
        r'```mermaid\s*\n(.*?)```',
        lambda m: '\n\n' + add(m.group(1).strip(), 'mermaid') + '\n\n',
        text, flags=re.DOTALL,
    )
    text = re.sub(
        r'```math\s*\n(.*?)```',
        lambda m: '\n\n' + add(m.group(1).strip(), 'display_math') + '\n\n',
        text, flags=re.DOTALL,
    )
    text = re.sub(
        r'\$\$\s*(.+?)\s*\$\$',
        lambda m: add(m.group(1).strip(), 'display_math'),
        text, flags=re.DOTALL,
    )

    def replace_inline_math(m):
        content = m.group(1)
        if not content.strip():
            return m.group(0)
        content = content.strip('`')
        content = re.sub(r'`', '', content)
        return add(content, 'inline_math')

    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', replace_inline_math, text)

    return text, {key: (orig, kind) for key, orig, kind in placeholders}


def restore_special_blocks(html_text, placeholders):
    for key, (content, kind) in placeholders.items():
        safe = content.replace('<', '&lt;')
        if kind == 'mermaid':
            r = f'<pre class="mermaid">{safe}</pre>'
        elif kind == 'display_math':
            r = f'<div class="math display">$${safe}$$</div>'
        else:
            r = f'<span class="math inline">${safe}$</span>'
        html_text = html_text.replace(key, r)
    return html_text


# ── word count & reading time stats ──────────────────────────────────────────

def compute_stats(html_body):
    """Count Chinese characters, English words, and estimate reading time.

    Strips script/math/code blocks so only visible prose contributes.
    Chinese: ~350 chars/min, English: ~200 words/min.
    """
    text = html_body

    # Remove script blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    # Remove mermaid diagrams
    text = re.sub(r'<pre class="mermaid">.*?</pre>', '', text, flags=re.DOTALL)
    # Remove math display
    text = re.sub(r'<div class="math display">.*?</div>', '', text, flags=re.DOTALL)
    # Remove math inline
    text = re.sub(r'<span class="math inline">.*?</span>', '', text, flags=re.DOTALL)
    # Remove codehilite blocks
    text = re.sub(r'<div class="codehilite">.*?</div>', '', text, flags=re.DOTALL)
    # Remove generic pre blocks
    text = re.sub(r'<pre[^>]*>.*?</pre>', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'<code[^>]*>.*?</code>', '', text, flags=re.DOTALL)

    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    text = re.sub(r'&#\d+;', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Count Chinese chars (CJK Unified Ideographs + Extension A)
    chinese_chars = len(re.findall(r'[一-鿿㐀-䶿]', text))

    # Count English words (letter sequences >= 2 chars)
    english_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', text))

    # Reading time: Chinese 350 chars/min, English 200 words/min, floor 1 minute
    minutes = max(1, round(chinese_chars / 350 + english_words / 200))

    return {'zh': chinese_chars, 'en': english_words, 'minutes': minutes}


def format_stats_html(stats):
    """Visible stats bar for the article page, placed after h1."""
    parts = []
    if stats['zh'] > 0:
        parts.append(f'{stats["zh"]:,} 字')
    if stats['en'] > 0:
        parts.append(f'English {stats["en"]:,} 词')
    if not parts:
        return ''
    parts.append(f'约 {stats["minutes"]} 分钟阅读')
    return '<div class="article-stats">' + ' · '.join(parts) + '</div>\n'


def extract_metadata_comments(text):
    """Extract site metadata comments (arxiv, venue, tags) from markdown."""
    metadata = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('<!--') and any(k in stripped for k in ('arxiv', 'venue', 'tags')):
            metadata.append(stripped)
    return metadata


def format_stats_comment(stats, metadata_comments=None):
    """HTML comments consumed by generate_index.sh."""
    lines = []
    if metadata_comments:
        lines.extend(metadata_comments)
    if stats['zh'] > 0 or stats['en'] > 0:
        lines.append(f'<!-- words-zh: {stats["zh"]} -->')
        lines.append(f'<!-- words-en: {stats["en"]} -->')
        lines.append(f'<!-- reading-time: {stats["minutes"]} -->')
    return '\n'.join(lines)


def insert_stats_into_body(body, stats_html):
    """Insert stats_html right after the first h1 in body."""
    if not stats_html:
        return body
    m = re.search(r'(<h1[^>]*>.*?</h1>)', body, re.DOTALL)
    if m:
        return body[:m.end()] + '\n' + stats_html + body[m.end():]
    return stats_html + body


# ── heading / TOC ──────────────────────────────────────────────────────────

def extract_headings(html_body):
    pattern = re.compile(r'<h([1-4])\s+id="([^"]+)"[^>]*>(.*?)</h\1>', re.DOTALL)
    headings = []
    for m in pattern.finditer(html_body):
        text = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        headings.append((int(m.group(1)), m.group(2), text))
    return headings


def build_toc_html(headings):
    if not headings:
        return '<p class="toc-empty">No sections</p>'

    lines = ['<ul class="toc-list">']
    stack = [1]

    for level, hid, text in headings:
        if level > stack[-1]:
            lines.append('<ul>')
            stack.append(level)
        while level < stack[-1]:
            lines.append('</ul></li>')
            stack.pop()
        if level == stack[-1] and len(lines) > 1:
            lines.append('</li>')
        lines.append(f'<li class="toc-h{level}"><a href="#{hid}">{text}</a>')

    while len(stack) > 1:
        lines.append('</ul></li>')
        stack.pop()
    lines.append('</li></ul>')
    return '\n'.join(lines)


# ── page shell ─────────────────────────────────────────────────────────────

CSS = r"""
  @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #ffffff;
    --text: #1a1a2e;
    --text-secondary: #555;
    --border: #e1e4e8;
    --link: #0969da;
    --table-stripe: #f6f8fa;
    --heading: #0d1117;
    --blockquote-border: #0969da;
    --blockquote-bg: #f0f6ff;
    --inline-code: #d73a49;
    --inline-code-bg: #f6f8fa;
    --sidebar-width: 260px;
    --toc-active: #0969da;
    --content-max-width: 900px;
    --content-padding: 48px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: "Inter", "Noto Sans SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 16px;
    line-height: 1.75;
    color: var(--text);
    background: var(--bg);
  }

  .sidebar {
    position: fixed;
    top: 0; left: 0; bottom: 0;
    width: var(--sidebar-width);
    overflow-y: auto;
    padding: 28px 20px 40px;
    background: #fafbfc;
    border-right: 1px solid var(--border);
    z-index: 10;
    scrollbar-width: thin;
  }
  .sidebar-title {
    font-size: 13px; font-weight: 700; color: var(--text-secondary);
    margin-bottom: 16px; letter-spacing: 0.06em; text-transform: uppercase;
  }
  .toc-list { list-style: none; padding: 0; margin: 0; }
  .toc-list ul { list-style: none; padding-left: 14px; }
  .toc-list li { margin: 1px 0; }
  .toc-list a {
    display: block; padding: 4px 12px; font-size: 13.5px; line-height: 1.45;
    color: var(--text-secondary); text-decoration: none; border-radius: 4px;
    border-left: 2px solid transparent; transition: all 0.15s;
  }
  .toc-list a:hover { color: var(--link); background: #e8f0fe; }
  .toc-list a.active {
    color: var(--toc-active); font-weight: 600;
    border-left-color: var(--toc-active); background: #e8f0fe;
  }
  .toc-h1 a { font-weight: 600; color: var(--heading); font-size: 14px; }
  .toc-h3 a { padding-left: 24px; font-size: 13px; }
  .toc-h4 a { padding-left: 36px; font-size: 12.5px; }
  .toc-empty { color: var(--text-secondary); font-size: 13px; font-style: italic; }

  .main-wrapper {
    margin-left: var(--sidebar-width);
    min-height: 100vh;
    display: flex; justify-content: center;
  }
  .content-inner {
    width: 100%;
    max-width: var(--content-max-width);
    padding: 40px var(--content-padding) 80px;
  }

  h1, h2, h3, h4, h5, h6 {
    color: var(--heading); font-weight: 700; line-height: 1.3;
    margin-top: 2em; margin-bottom: 0.6em; scroll-margin-top: 24px;
  }
  h1 { font-size: 2em; border-bottom: 2px solid var(--border); padding-bottom: 0.3em; margin-top: 0; }
  h2 { font-size: 1.5em; border-bottom: 1px solid var(--border); padding-bottom: 0.25em; }
  h3 { font-size: 1.25em; }
  h4 { font-size: 1.1em; }
  p { margin-bottom: 1em; }
  a { color: var(--link); text-decoration: none; }
  a:hover { text-decoration: underline; }
  strong { font-weight: 600; }

  blockquote {
    border-left: 4px solid var(--blockquote-border);
    background: var(--blockquote-bg);
    padding: 12px 20px; margin: 1em 0;
    border-radius: 0 6px 6px 0; color: var(--text-secondary);
  }
  blockquote p:last-child { margin-bottom: 0; }

  code {
    font-family: "JetBrains Mono", "SF Mono", "Fira Code", monospace;
    font-size: 0.875em; background: var(--inline-code-bg);
    padding: 2px 6px; border-radius: 4px; color: var(--inline-code);
  }
  pre {
    background: #1e1e2e; color: #cdd6f4;
    border-radius: 8px; padding: 16px 20px; overflow-x: auto;
    margin: 1em 0; font-size: 14px; line-height: 1.55;
  }
  pre code { background: none; padding: 0; border-radius: 0; color: inherit; font-size: inherit; }

  .codehilite { background: #1e1e2e; border-radius: 8px; padding: 16px 20px; overflow-x: auto; margin: 1em 0; }
  .codehilite pre { background: none; padding: 0; margin: 0; border-radius: 0; }
  .codehilite .hll { background-color: #2a2a3e; }
  .codehilite .c  { color: #6c7086; font-style: italic; }
  .codehilite .k  { color: #cba6f7; }
  .codehilite .o  { color: #89b4fa; }
  .codehilite .s  { color: #a6e3a1; }
  .codehilite .n  { color: #cdd6f4; }
  .codehilite .p  { color: #bac2de; }
  .codehilite .mi { color: #fab387; }
  .codehilite .nf { color: #89b4fa; }
  .codehilite .nb { color: #f38ba8; }
  .codehilite .nc { color: #f9e2af; }
  .codehilite .bp { color: #f38ba8; }

  table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 0.925em; }
  th, td { border: 1px solid var(--border); padding: 10px 14px; text-align: left; }
  th { background: var(--table-stripe); font-weight: 600; color: var(--heading); }
  tr:nth-child(even) td { background: var(--table-stripe); }

  .math.display { display: block; overflow-x: auto; padding: 12px 0; margin: 1em 0; }
  ul, ol { padding-left: 1.5em; margin-bottom: 1em; }
  li { margin-bottom: 0.3em; }
  hr { border: none; border-top: 1px solid var(--border); margin: 2em 0; }
  img { max-width: 100%; height: auto; border-radius: 6px; margin: 0; }
  p:has(img) { text-align: center; margin-top: 0.6em; margin-bottom: 0; }
  p:has(img) + p { text-align: center; font-size: 0.9em; color: var(--text-secondary); margin-top: 0.15em; margin-bottom: 0.6em; }
  .mermaid { text-align: center; margin: 1.5em 0; background: #fafbfc; border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .article-stats {
    font-size: 0.85rem; color: var(--text-secondary);
    padding: 10px 18px; margin-bottom: 28px;
    background: #f8f9fa; border: 1px solid var(--border); border-radius: 8px;
  }

  @media (max-width: 1024px) {
    .sidebar { position: static; width: 100%; height: auto; max-height: 35vh; border-right: none; border-bottom: 1px solid var(--border); }
    .main-wrapper { margin-left: 0; display: block; }
    .content-inner { max-width: none; padding: 24px 20px 60px; }
  }
"""

JS = r"""
<script>
MathJax = {
  tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true, tags: 'none' },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
  mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
});
</script>
"""

TOC_SCROLL_JS = r"""
<script>
(function() {
  const sidebar = document.getElementById('sidebar');
  const links = sidebar.querySelectorAll('a');
  const headings = document.querySelectorAll('h1[id], h2[id], h3[id], h4[id]');
  if (headings.length === 0) return;

  const linkMap = new Map();
  links.forEach(function(a) {
    const href = a.getAttribute('href');
    if (href && href.startsWith('#')) linkMap.set(href.slice(1), a);
  });

  function onScroll() {
    let currentId = null;
    const scrollTop = window.scrollY + 80;
    headings.forEach(function(h) { if (h.offsetTop <= scrollTop) currentId = h.id; });
    links.forEach(function(a) { a.classList.remove('active'); });
    if (currentId && linkMap.has(currentId)) {
      const el = linkMap.get(currentId);
      el.classList.add('active');
      const sr = sidebar.getBoundingClientRect();
      const er = el.getBoundingClientRect();
      if (er.bottom > sr.bottom || er.top < sr.top) {
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
</script>
"""


def build_page(body, toc_html, title, stats_comment=""):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{stats_comment}
<style>{CSS}</style>
{JS}
</head>
<body>

<nav class="sidebar" id="sidebar">
  <div class="sidebar-title">目录</div>
  {toc_html}
</nav>

<main class="main-wrapper">
  <div class="content-inner" id="main-content">
{body}
  </div>
</main>

{TOC_SCROLL_JS}
</body>
</html>"""


# ── main conversion ────────────────────────────────────────────────────────

def convert_file(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract metadata before any processing
    metadata_comments = extract_metadata_comments(text)

    # Find first # heading for title (skip HTML comments)
    title = os.path.basename(md_path)
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('# ') and not s.startswith('## '):
            title = s.lstrip('#').strip()
            break

    text = fix_nested_list_indent(text)
    text = fix_list_blank_lines(text)
    text, placeholders = protect_special_blocks(text)

    md = markdown.Markdown(extensions=['fenced_code', 'tables', 'codehilite', 'toc'])
    body = md.convert(text)
    body = restore_special_blocks(body, placeholders)

    # Compute stats and inject into body
    stats = compute_stats(body)
    stats_html = format_stats_html(stats)
    stats_comment = format_stats_comment(stats, metadata_comments)
    body = insert_stats_into_body(body, stats_html)

    # Inject heading IDs
    heading_pattern = re.compile(r'<(h[1-4])([^>]*)>(.*?)</\1>', re.DOTALL)
    used_ids = set()
    counter = [0]

    def add_id(m):
        tag, attrs, content = m.group(1), m.group(2), m.group(3)
        if 'id=' in attrs:
            return m.group(0)
        text_content = re.sub(r'<[^>]+>', '', content).strip()
        slug = re.sub(r'[^\w一-鿿]+', '-', text_content.lower()).strip('-')
        if not slug or slug in used_ids:
            counter[0] += 1
            slug = f'section-{counter[0]}'
        used_ids.add(slug)
        return f'<{tag} id="{slug}"{attrs}>{content}</{tag}>'

    body = heading_pattern.sub(add_id, body)

    toc_html = build_toc_html(extract_headings(body))
    full_html = build_page(body, toc_html, title, stats_comment=stats_comment)

    html_path = os.path.splitext(md_path)[0] + '.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    return html_path


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted([f for f in os.listdir('.') if f.endswith('.md')])

    if not files:
        print("No markdown files found.")
        sys.exit(1)

    for f in files:
        if not os.path.isfile(f):
            print(f"  SKIP (not a file): {f}")
            continue
        print(f"Converting: {f}")
        out = convert_file(f)
        print(f"  -> {out}")

    print(f"\nDone. {len(files)} file(s) converted.")


if __name__ == '__main__':
    main()
