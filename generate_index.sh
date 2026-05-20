#!/bin/bash
# Generate index.html by scanning dated directories for HTML note files.
# Run from repo root: ./generate_index.sh

set -euo pipefail

cd "$(dirname "$0")"

cat <<'HEAD'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaperNotes — 论文阅读笔记</title>
<style>
  :root {
    --bg: #fafafa;
    --card-bg: #ffffff;
    --text: #1a1a1a;
    --text-secondary: #6b7280;
    --border: #e5e7eb;
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --tag-bg: #eff6ff;
    --tag-text: #1e40af;
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
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 10px;
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
  .paper-updated {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-left: 8px;
  }

  footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.8rem;
    padding: 32px 0;
    border-top: 1px solid var(--border);
    margin-top: 48px;
  }
  footer a { color: var(--accent); text-decoration: none; }
</style>
</head>
<body>
<div class="container">

<header>
  <h1>PaperNotes</h1>
  <p>论文阅读笔记 · 图文详解 · 公式与图表</p>
</header>

HEAD

# Scan date dirs in descending order, find HTML notes, extract titles.
for date_dir in $(ls -d [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] 2>/dev/null | sort -r); do
    # Find HTML files (skip arxiv dirs, code dirs, conda envs)
    html_files=$(find "$date_dir" -maxdepth 1 -name "*.html" ! -name "index.html" 2>/dev/null | sort)

    if [ -z "$html_files" ]; then
        continue
    fi

    echo ""
    echo "<div class=\"date-group\">"
    echo "  <div class=\"date-label\">$date_dir</div>"

    while IFS= read -r f; do
        title=$(grep -o '<title>[^<]*</title>' "$f" 2>/dev/null | head -1 | sed 's|<title>||;s|</title>||')
        if [ -z "$title" ]; then
            title=$(basename "$f" .html)
        fi
        # Escape HTML entities in title
        title=$(echo "$title" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

        # Extract arxiv id, venue, and tags from HTML comments
        arxiv=$(grep -o '<!-- arxiv: [^ ]* -->' "$f" 2>/dev/null | head -1 | sed 's/<!-- arxiv: //;s/ -->//' || true)
        venue=$(grep -o '<!-- venue: [^ ]*.* -->' "$f" 2>/dev/null | head -1 | sed 's/<!-- venue: //;s/ -->//' || true)
        tags=$(grep -o '<!-- tags: [^ ]*.* -->' "$f" 2>/dev/null | head -1 | sed 's/<!-- tags: //;s/ -->//' || true)

        # Get last update time from git log
        updated=$(git log -1 --format="%cs" -- "$f" 2>/dev/null)

        echo "  <div class=\"paper-card\">"
        echo "    <a href=\"$f\">$title</a>"
        echo -n "    <div class=\"paper-meta\">"
        if [ -n "$arxiv" ]; then
            echo -n "arXiv: $arxiv"
        fi
        if [ -n "$venue" ]; then
            echo -n "<span class=\"paper-venue\">$venue</span>"
        fi
        if [ -n "$updated" ]; then
            echo -n "<span class=\"paper-updated\">更新于 $updated</span>"
        fi
        echo "</div>"
        if [ -n "$tags" ]; then
            echo ""
            echo -n "    <div>"
            IFS=',' read -ra TAG_ARRAY <<< "$tags"
            for tag in "${TAG_ARRAY[@]}"; do
                tag=$(echo "$tag" | sed 's/^ *//;s/ *$//')
                echo -n "<span class=\"paper-tag\">$tag</span>"
            done
            echo "</div>"
        fi
        echo "  </div>"
    done <<< "$html_files"

    echo "</div>"
done

cat <<'FOOT'

<footer>
  Powered by <a href="https://github.com/fx-hit/PaperNotes">GitHub Pages</a>
  &nbsp;·&nbsp;
  Generated with Claude Code
</footer>

</div>
</body>
</html>
FOOT
