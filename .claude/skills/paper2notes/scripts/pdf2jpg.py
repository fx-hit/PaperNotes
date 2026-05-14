#!/usr/bin/env python3
"""Convert PDF and PNG figures to high-quality JPG for web embedding.

Usage:
    python3 pdf2jpg.py <fig_dir> [--out-dir <dir>] [--quality 92] [--width 1600]

- Converts all PDF files via the best available backend (qlmanage on macOS,
  PyMuPDF via pip, or pdf2image+poppler).
- PNG files are converted to JPG using Pillow (RGBA/P → RGB first).
- Output JPG files are placed in --out-dir (default: <fig_dir>/jpg/).
"""

import argparse
import os
import subprocess
import sys
import tempfile


def _pdf_to_png_macos(pdf_path, png_path, width):
    """macOS: use qlmanage to render a thumbnail, then resize."""
    tmp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", str(width), "-o", tmp_dir, pdf_path],
            check=True, capture_output=True, timeout=30,
        )
        # qlmanage names output: <basename>.pdf.png
        src = os.path.join(tmp_dir, os.path.basename(pdf_path) + ".png")
        if not os.path.exists(src):
            # fallback: maybe no .pdf suffix
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            candidates = [f for f in os.listdir(tmp_dir) if f.startswith(base)]
            if candidates:
                src = os.path.join(tmp_dir, candidates[0])
        if os.path.exists(src):
            os.rename(src, png_path)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    finally:
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)
    return False


def _pdf_to_png_pymupdf(pdf_path, png_path, width):
    """Cross-platform: use PyMuPDF (fitz)."""
    try:
        import fitz
    except ImportError:
        return False
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = width / page.rect.width
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(png_path)
    doc.close()
    return True


def _pdf_to_png_pdf2image(pdf_path, png_path, width):
    """Cross-platform: use pdf2image + poppler."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return False
    images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
    if images:
        # resize to target width maintaining aspect ratio
        img = images[0]
        ratio = width / img.width
        img = img.resize((width, int(img.height * ratio)))
        img.save(png_path, "PNG")
        return True
    return False


def pdf_to_png(pdf_path, width=1600):
    """Convert first page of PDF to PNG; returns path to temp PNG or None."""
    import tempfile
    fd, png_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    for fn in (_pdf_to_png_macos, _pdf_to_png_pymupdf, _pdf_to_png_pdf2image):
        if fn(pdf_path, png_path, width):
            return png_path
        # clean up failed attempt
        if os.path.exists(png_path):
            os.remove(png_path)

    print(f"  [WARN] No PDF→PNG backend available for: {pdf_path}", file=sys.stderr)
    print(f"         Install one: pip install pymupdf  OR  brew install poppler && pip install pdf2image", file=sys.stderr)
    return None


def png_to_jpg(png_path, jpg_path, quality=92):
    """Convert PNG to JPG with Pillow."""
    from PIL import Image
    img = Image.open(png_path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode == "CMYK":
        img = img.convert("RGB")
    img.save(jpg_path, "JPEG", quality=quality)
    return img.size


def main():
    parser = argparse.ArgumentParser(description="Convert paper figures to JPG")
    parser.add_argument("fig_dir", help="Directory containing figure files (PDF/PNG)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <fig_dir>/jpg/)")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (1-100)")
    parser.add_argument("--width", type=int, default=1600, help="Target width in pixels")
    args = parser.parse_args()

    fig_dir = os.path.abspath(args.fig_dir)
    out_dir = args.out_dir or os.path.join(fig_dir, "jpg")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(fig_dir):
        print(f"Error: {fig_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(os.listdir(fig_dir))
    pdfs = [f for f in files if f.lower().endswith(".pdf")]
    pngs = [f for f in files if f.lower().endswith(".png")]
    other = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]

    print(f"Found: {len(pdfs)} PDF(s), {len(pngs)} PNG(s), {len(other)} JPG(s)")

    for pdf in pdfs:
        pdf_path = os.path.join(fig_dir, pdf)
        jpg_name = os.path.splitext(pdf)[0] + ".jpg"
        jpg_path = os.path.join(out_dir, jpg_name)

        print(f"  PDF → JPG: {pdf}", end=" ", flush=True)
        png_tmp = pdf_to_png(pdf_path, width=args.width)
        if png_tmp is None:
            print("SKIP (no backend)")
            continue
        try:
            size = png_to_jpg(png_tmp, jpg_path, args.quality)
            kb = os.path.getsize(jpg_path) // 1024
            print(f"→ {jpg_name} ({size[0]}x{size[1]}, {kb}KB)")
        finally:
            os.remove(png_tmp)

    from PIL import Image
    for png in pngs:
        png_path = os.path.join(fig_dir, png)
        jpg_name = os.path.splitext(png)[0] + ".jpg"
        jpg_path = os.path.join(out_dir, jpg_name)

        print(f"  PNG → JPG: {png}", end=" ", flush=True)
        img = Image.open(png_path)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(jpg_path, "JPEG", quality=args.quality)
        kb = os.path.getsize(jpg_path) // 1024
        print(f"→ {jpg_name} ({img.size[0]}x{img.size[1]}, {kb}KB)")

    for jpg in other:
        jpg_path = os.path.join(fig_dir, jpg)
        if out_dir != fig_dir:
            import shutil
            shutil.copy2(jpg_path, os.path.join(out_dir, jpg))
            print(f"  COPY: {jpg}")

    print(f"\nDone. {len(os.listdir(out_dir))} file(s) in {out_dir}")


if __name__ == "__main__":
    main()
