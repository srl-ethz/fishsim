#!/usr/bin/env python3
"""Fig. 3 (the hardware fish, single column): the CAD cutaway with labelled components, and a
photograph of the built fish as an inset.

  (A) the CAD cutaway render of the fish, tail left and head right, with an arrow and a label
      on each component the text names: the AprilTag mast, the compliant PLA tail, the
      Raspberry Pi 5, the antagonistic tendon pair, the tendon pulley, the Dynamixel motor and
      the Adafruit IMU.
  (B) the real fish (a photograph, background removed and lighting corrected by
      tools/cut_real_fish.py, head right like the render), in a framed inset at the bottom right.
      The inset is sized from the cutout's aspect ratio at run time; the photo is rotated by
      the cutout tool to sit roughly level with the render.

The labels and arrows are drawn here, as real SVG text, so they follow the paper's type
standard (paper_style: Times, 9 pt labels, 10 pt panel letters after the include scale) instead
of being baked into a hand-made PDF. The arrow tips were read off the earlier Inkscape drawing
(sofiRL icra_report/Images/draft/experimental_setup.svg) as fractions of the render, so each
arrow lands where it did before; the label positions are laid out here.

Inputs (figures/static/):
  hw_cad_cutaway.png               the CAD cutaway render, 2864 x 1562, white background, head
                                   LEFT (it is mirrored on the page); extracted from the Inkscape
                                   drawing above
  real_fish_cut.png                the photo cutout (RGBA, tight crop, head right), made from
                                   real_fish.jpeg by tools/cut_real_fish.py
Output:  out/fig3_experimental_setup.pdf (+ .png); the editable SVG goes to .gen/svg/
Run via `make figures`, or from .gen/figures:  python fig3_experimental_setup.py [--no-pdf]
The PDF export needs Inkscape or Chrome/Edge on the PATH (see _svg_curriculum.export_pdf).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _svg_pipeline as tp                  # noqa: E402  primitives, icons, data URIs
import _svg_curriculum as cf                # noqa: E402  export_pdf / export_png, column width

from _paths import STATIC as _STATIC, OUT as _OUT
STATIC = _STATIC
OUT = _OUT
SVG_DIR = _OUT / "svg"
DEFAULT_OUT = SVG_DIR / "fig3_experimental_setup.svg"

# design units: the column width in mm; the page is K x this and the paper scales it back
W = cf.W
K = cf.PRINT_SCALE
FS = tp.FS_MAIN                             # 7 pt design -> 9 pt printed: every label
FS_PANEL = 10.0 / K * tp.PT                 # -> 10 pt printed, bold: the panel letters
INK = "#000000"                             # arrows, the inset frame and all text: plain black
text, line, rect, image, group, f = tp.text, tp.line, tp.rect, tp.image, tp.group, tp.f

# ---- layout (design mm) ------------------------------------------------------------------------
# SHRINK scales the artwork (render + inset) inside the fixed column width, so the figure gets
# shorter while every label stays at the paper's 9 pt. 0.9 took the paper from 9 pages back to 8.
SHRINK = 0.9
CAD_W = 76.0 * SHRINK                       # the cutaway render, mirrored so the head is on the right
CAD_X, CAD_Y = (W - CAD_W) / 2, 6.5         # centred in the column
CAD_ASPECT = 2864 / 1562
CAD_H = CAD_W / CAD_ASPECT
INSET_W, INSET_PAD = 28.0 * SHRINK, 1.2     # panel B box and the air around the photo inside it
INSET_BG = "#FFFFFF"
INSET_X = W - 1.5 - INSET_W
INSET_Y = CAD_Y + 1.02 * CAD_H              # a little under the hull's lowest point at the right; the box
                                            #   starts right of the motor's stub, which reaches the render's foot


def inset_h(photo_aspect: float) -> float:
    return (INSET_W - 2 * INSET_PAD) / photo_aspect + 2 * INSET_PAD

# ---- the labels -------------------------------------------------------------------------------
# (label, text anchor, text position, arrow start, arrow tip), positions as fractions of the
# displayed (mirrored) render: x to the right, y down. Tips are from the Inkscape drawing.
LABELS = [
    ("AprilTag",                 "end",    (0.560, -0.015), [((0.575, -0.005), (0.655, 0.075))]),
    ("Compliant PLA Tail",       "start",  (0.000, 0.060), [((0.345, 0.075), (0.360, 0.270))]),
    ("Raspberry Pi 5",           "start",  (0.780, 0.245), [((0.815, 0.262), (0.762, 0.395))]),
    ("Antagonistic Tendon Pair", "start",  (-0.125, 0.870), [((0.355, 0.846), (0.445, 0.607)),
                                                             ((0.355, 0.846), (0.500, 0.578))]),
    ("Tendon Pulley",            "middle", (0.220, 1.030), [((0.365, 1.000), (0.583, 0.660))]),
    ("Dynamixel Motor",          "middle", (0.270, 1.150), [((0.445, 1.120), (0.600, 0.860))]),
    ("Adafruit IMU",             "middle", (0.385, 1.270), [((0.530, 1.240), (0.754, 0.766))]),
]


def P(fx, fy):
    """Render fraction -> design mm."""
    return CAD_X + fx * CAD_W, CAD_Y + fy * CAD_H


def build_svg(cad_uri: str, photo_uri: str, photo_aspect: float) -> str:
    INSET_H = inset_h(photo_aspect)
    H = INSET_Y + INSET_H + 1.5
    parts = []
    # A: the cutaway
    parts.append(image(CAD_X, CAD_Y, CAD_W, CAD_H, cad_uri))
    parts.append(text(1.2, 4.2, "A", fs=FS_PANEL, weight="bold", fill=INK))
    for label, anchor, (tx, ty), arrows in LABELS:
        x, y = P(tx, ty)
        parts.append(text(x, y, label, fs=FS, anchor=anchor, fill=INK))
        for (ax0, ay0), (ax1, ay1) in arrows:
            x0, y0 = P(ax0, ay0)
            x1, y1 = P(ax1, ay1)
            parts.append(line(x0, y0, x1, y1, stroke=INK, sw=0.3, marker=True))
    # B: the photo, framed
    parts.append(rect(INSET_X, INSET_Y, INSET_W, INSET_H, INSET_BG, rx=1.4, stroke=INK, sw=0.3))
    parts.append(image(INSET_X + INSET_PAD, INSET_Y + INSET_PAD, INSET_W - 2 * INSET_PAD,
                       INSET_H - 2 * INSET_PAD, photo_uri))
    parts.append(text(INSET_X + 1.4, INSET_Y + 4.0, "B", fs=FS_PANEL, weight="bold", fill=INK))

    defs = ('<defs>'
            '<marker id="arrow" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4.5" '
            'markerHeight="4.5" orient="auto-start-reverse">'
            f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{INK}"/></marker>'
            '</defs>')
    body = "\n  ".join(parts)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by figures/fig3_experimental_setup.py (swim2real-icra2026) - edit freely in Inkscape. -->
<svg
   width="{f(W * K)}mm" height="{f(H * K)}mm" viewBox="0 0 {f(W)} {f(H)}"
   version="1.1" id="experimental_setup"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   font-family="{tp.FONT}" fill="{INK}">
  <sodipodi:namedview id="namedview" pagecolor="#ffffff" bordercolor="#000000"
     borderopacity="0.25" inkscape:document-units="mm" inkscape:pagecheckerboard="0"/>
  {defs}
  <g id="layer1" inkscape:groupmode="layer" inkscape:label="Figure">
  {body}
  </g>
</svg>
'''


def load_cad(path: Path, width_px: int = 2400) -> Image.Image:
    """The cutaway render, mirrored so the head is on the right, downsampled for the PDF."""
    im = Image.open(path).convert("RGB")
    im = ImageOps.mirror(im)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS)


def load_photo(path: Path) -> Image.Image:
    """The cutout at its full resolution (it is only ~1100 px wide), alpha kept."""
    return Image.open(path).convert("RGBA")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cad", default=str(STATIC / "hw_cad_cutaway.png"))
    p.add_argument("--photo", default=str(STATIC / "real_fish_cut.png"))
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)

    for path in (args.cad, args.photo):
        if not Path(path).is_file():
            sys.exit(f"Missing input {path}")
    photo = load_photo(Path(args.photo))
    svg = build_svg(tp.png_data_uri(load_cad(Path(args.cad))), tp.png_data_uri(photo),
                    photo.width / photo.height)
    H = INSET_Y + inset_h(photo.width / photo.height) + 1.5
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Saved {out}  ({W:.1f} x {H:.1f} mm design, page x{K:.3f}, {out.stat().st_size / 1024:.0f} kB)")
    if not args.no_pdf:
        cf.export_pdf(out, OUT / (out.stem + ".pdf"), W * K, H * K)
    if not args.no_png:
        cf.export_png(out, OUT / (out.stem + ".png"), W * K, H * K, px_per_mm=24.0 / K)   # ~610 dpi: the photo inset stays sharp


if __name__ == "__main__":
    main()
