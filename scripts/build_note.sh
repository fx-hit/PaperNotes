#!/bin/bash
# Convert one or more note Markdown files, rebuild site entries, and audit output.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 20YY-MM-DD/Name_阅读笔记.md [more.md ...]" >&2
    exit 2
fi

python3 .claude/skills/md2html/scripts/convert.py "$@"
./generate_index.sh --site
python3 scripts/audit_notes.py --check-site
