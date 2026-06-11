#!/usr/bin/env python3
"""Audit PaperNotes source notes, generated HTML, assets, and site entries."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
META_RE = re.compile(r"<!--\s*(arxiv|venue|tags)\s*:\s*(.*?)\s*-->", re.S)
MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMG_RE = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"']", re.I)
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
CANONICAL_TAGS = {
    "VLA",
    "WAM",
    "世界模型",
    "视频生成",
    "3D重建",
    "泛化",
    "强化学习",
    "自动驾驶",
    "扩散模型",
    "知识蒸馏",
    "表征学习",
    "语言模型",
    "导航",
    "图像编辑",
    "最优传输",
    "全模态",
    "多模态理解",
    "链式思考",
    "音视频推理",
    "视觉推理",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def metadata(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in META_RE.findall(text):
        values[key] = value.strip()
    return values


def split_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def local_image_refs(path: Path, text: str) -> list[str]:
    refs = []
    for match in MD_IMG_RE.finditer(text):
        refs.append(match.group(1).strip().split()[0].strip("\"'"))
    for match in HTML_IMG_RE.finditer(text):
        refs.append(match.group(1).strip())
    return [
        ref
        for ref in refs
        if ref and not ref.startswith(("http://", "https://", "data:", "#"))
    ]


def audit_notes(errors: list[str], warnings: list[str]) -> None:
    md_by_stem = {path.with_suffix(""): path for path in ROOT.glob("20??-??-??/*.md")}
    html_by_stem = {path.with_suffix(""): path for path in ROOT.glob("20??-??-??/*.html")}

    for stem, path in sorted(md_by_stem.items()):
        html = html_by_stem.get(stem)
        if html is None:
            errors.append(f"{rel(path)} has no matching HTML file")
            continue

        md_meta = metadata(read(path))
        html_meta = metadata(read(html))
        for key in ("arxiv", "venue", "tags"):
            if not md_meta.get(key):
                errors.append(f"{rel(path)} missing <!-- {key}: ... -->")
            if not html_meta.get(key):
                errors.append(f"{rel(html)} missing <!-- {key}: ... -->")
            if md_meta.get(key) and html_meta.get(key) and md_meta[key] != html_meta[key]:
                errors.append(f"{rel(path)} and {rel(html)} disagree on {key}")

        tags = split_tags(md_meta.get("tags", ""))
        if len(tags) > 5:
            warnings.append(f"{rel(path)} has {len(tags)} tags; AGENTS recommends 2-5")
        for tag in tags:
            if tag in TAG_ALIASES:
                warnings.append(f"{rel(path)} uses alias tag {tag!r}; prefer {TAG_ALIASES[tag]!r}")
            elif tag not in CANONICAL_TAGS:
                warnings.append(f"{rel(path)} uses non-standard tag {tag!r}; add it intentionally if reusable")

        for source in (path, html):
            for ref in local_image_refs(source, read(source)):
                target = (source.parent / ref).resolve()
                try:
                    target.relative_to(ROOT)
                except ValueError:
                    errors.append(f"{rel(source)} image escapes repo: {ref}")
                    continue
                if not target.exists():
                    errors.append(f"{rel(source)} missing image: {ref}")
                if not ref.startswith("assets/"):
                    warnings.append(f"{rel(source)} image is outside assets/: {ref}")

    for stem, path in sorted(html_by_stem.items()):
        if stem not in md_by_stem:
            errors.append(f"{rel(path)} has no matching Markdown file")


def audit_assets(warnings: list[str]) -> None:
    for path in ROOT.glob("20??-??-??/assets/**/*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > 3 * 1024 * 1024:
            warnings.append(f"{rel(path)} is {size / 1024 / 1024:.1f} MB; consider recompressing")


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def audit_site(errors: list[str]) -> None:
    for mode, output in (("--index", "index.html"), ("--tags", "tags.html")):
        generated = run_checked(["./generate_index.sh", mode])
        if generated.returncode != 0:
            errors.append(f"./generate_index.sh {mode} failed:\n{generated.stderr.strip()}")
            continue
        current = (ROOT / output).read_text(encoding="utf-8")
        if generated.stdout != current:
            errors.append(f"{output} is stale; run ./generate_index.sh --site")


def audit_git(warnings: list[str]) -> None:
    result = run_checked(["git", "ls-files", "--others", "--exclude-standard"])
    if result.returncode != 0:
        warnings.append(f"git untracked scan failed: {result.stderr.strip()}")
        return

    for line in result.stdout.splitlines():
        path = Path(line)
        if not path.parts:
            continue
        if DATE_RE.fullmatch(path.parts[0]) and len(path.parts) > 1:
            warnings.append(f"untracked date-dir file is not ignored: {line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-site", action="store_true", help="also verify index.html and tags.html are reproducible")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    audit_notes(errors, warnings)
    audit_assets(warnings)
    audit_git(warnings)
    if args.check_site:
        audit_site(errors)

    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    print(f"Audit complete: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
