#!/usr/bin/env python3
"""Re-encode the photographs inside the figure PDFs as JPEG so the paper fits the 6 MB limit.

    python tools/compress_figures.py out/*.pdf [--quality 90] [--min-pixels 40000] [--dry-run]

matplotlib's PDF backend and the SVG->PDF route through Chrome both store every raster as a
lossless Flate stream. For the pool composites (Fig. 7) and the photos in Figs. 1 and 3 that is
3-4x larger than a JPEG of the same pixels, and those three figures were ~6.7 MB of a 7.5 MB
paper. This walks every image XObject, and for each 8-bit RGB Flate image above --min-pixels
swaps the stream for a DCT (JPEG) stream at --quality with no chroma subsampling. Pixel
dimensions are untouched; soft masks, indexed/gray images (colorbars) and vector content are
left alone. Runs in place; idempotent (already-DCT images are skipped).
"""
import argparse, io, pathlib, sys

import pikepdf
from pikepdf import Name, PdfImage


def is_candidate(obj, min_pixels):
    if obj.get("/Subtype") != Name.Image or "/ImageMask" in obj:
        return False
    if int(obj.get("/BitsPerComponent", 0)) != 8:
        return False
    w, h = int(obj.Width), int(obj.Height)
    if w * h < min_pixels:
        return False
    filt = obj.get("/Filter")
    filters = [filt] if isinstance(filt, Name) else list(filt or [])
    if any(f in (Name.DCTDecode, Name.JPXDecode, Name.JBIG2Decode, Name.CCITTFaxDecode) for f in filters):
        return False
    cs = obj.get("/ColorSpace")
    if isinstance(cs, pikepdf.Array):          # [/ICCBased stream] or [/Indexed ...]
        if cs[0] != Name.ICCBased or int(cs[1].get("/N", 0)) != 3:
            return False
    elif cs != Name.DeviceRGB:
        return False
    return True


def compress_pdf(path, quality, min_pixels, dry_run):
    before = path.stat().st_size
    pdf = pikepdf.open(path, allow_overwriting_input=True)
    seen, n_done, saved = set(), 0, 0
    for page in pdf.pages:
        xobjs = page.Resources.get("/XObject", {}) if "/Resources" in page else {}
        for _, obj in xobjs.items():
            if obj.objgen in seen or not is_candidate(obj, min_pixels):
                continue
            seen.add(obj.objgen)
            pil = PdfImage(obj).as_pil_image().convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
            jpeg = buf.getvalue()
            old = len(obj.read_raw_bytes())
            if len(jpeg) >= old:
                continue
            saved += old - len(jpeg)
            n_done += 1
            print(f"    {pil.width}x{pil.height}  {old/1024:7.0f}K -> {len(jpeg)/1024:6.0f}K")
            if not dry_run:
                obj.write(jpeg, filter=Name.DCTDecode)
                obj.ColorSpace = Name.DeviceRGB
                if "/DecodeParms" in obj:
                    del obj.DecodeParms
    if n_done and not dry_run:
        pdf.save(path, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    after = path.stat().st_size
    tag = "unchanged" if not n_done else f"{before/1048576:.2f} -> {after/1048576:.2f} MB"
    print(f"  {path.name}: {n_done} image(s) re-encoded, {tag}")
    return before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdfs", nargs="+", type=pathlib.Path)
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    ap.add_argument("--min-pixels", type=int, default=40000,
                    help="skip images smaller than this many pixels (default 40000)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    tot_b = tot_a = 0
    for p in a.pdfs:
        if p.suffix.lower() != ".pdf":
            continue
        b, af = compress_pdf(p, a.quality, a.min_pixels, a.dry_run)
        tot_b += b; tot_a += af
    print(f"total: {tot_b/1048576:.2f} -> {tot_a/1048576:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
