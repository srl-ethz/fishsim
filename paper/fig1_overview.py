"""Page-1 single-column overview figure: the three contributions in one picture, no technical detail.

Layout (one IEEE column = 86.8 mm wide): two tinted side panels, read like text (a Z):

        SIMULATION (cool tint)         |        HARDWARE (warm tint)
   [MuJoCo ellipsoid twin]   <--(1) identified from--   [hardware CAD render]
        (1) Staged system identification: tail stiffness . fluid . motor
   [sim overhead still]      --(3) deployed zero-shot-->   [pool strobe, target reached]
        (2) Learn in simulation only        (3) Deploy zero-shot, unchanged
            CPG . DR . curriculum               reaching . disturbance . tracking

Real fish and real result share the right panel, twin and training episode the left one. The
steps still run top, bottom-left, bottom-right; the only leftward arrow is the identification.
Text is 7 pt (titles, headers, step names) and 6 pt (details, chips, callouts). Every label is real SVG text (7 pt, the paper's single text size);
rasters are embedded as base64 PNG so the SVG is self-contained and editable in Inkscape.

Inputs (figures/static/, plus the paper repo's strobes):
  real_sofi.png                         hardware CAD render (AprilTag mast, tail right)
  40_hull_hinges_fluid.png              ellipsoid/hinge twin render (same orientation)
  deploy_strobe_montage.png             six deployment strobes (fallback for the pool photo)
  sim_collage_panel*_t*.png             overhead MuJoCo stills (render_sim_collage.py in fishCREATE)
  upstream/sofi_control/rl_deploy/analysis_20260520/report_strobes/*.png   full-res strobes
Output (out/):  fig1_overview.pdf/.png  -- the paper's Figure 1 (the boxed variant, the default);
                the editable SVG goes to .gen/svg/. The other framings are selected with --variant
                and keep their name as a suffix: fig1_overview_{compact,rich,text,minimal}
                (--variant all builds every one of them).

Run via `make figures`, or from .gen/figures after `python tools/configure.py`:
  python fig1_overview.py                    # the paper's figure
  python fig1_overview.py --variant compact  # an earlier framing, out/fig1_overview_compact.*
The PDF export needs Inkscape or Chrome/Edge on the PATH (see _svg_curriculum.export_pdf).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _svg_pipeline as tp                       # noqa: E402  (SVG primitives, eq markup, icons)
import _svg_curriculum as cf                     # noqa: E402  (export helpers, page geometry)

from _paths import STATIC as _STATIC, OUT as _OUT, ANALYSIS as _ANALYSIS
STATIC = _STATIC                                   # renders, stills, montage
OUT = _OUT                                         # generated figures, gitignored
SVG_DIR = _OUT / "svg"                             # editable SVG sources, disposable
DEFAULT_OUT = SVG_DIR / "fig1_overview.svg"       # PDF and PNG twins go to OUT/<stem>

W = cf.W                          # 86.8 mm = \columnwidth, placed unscaled
# type sizes at the printed size (the page is the column width, included unscaled):
# 9 pt (FS), 8 pt (FS2), 10 pt (FS_STEP), paper_style's tiers. K stays 1.
K = cf.PRINT_SCALE
FS = cf.FS                        # 9 pt: titles, headers, step names
FS2 = 8.0 / K / 72.27 * 25.4      # 8 pt: detail lines, chips, callouts (the floor)
LINE = FS * 1.15
TINT_SIM, TINT_REAL = "#EEF3FA", "#F7F2EB"       # side panels: simulation (cool) / hardware (warm)
INK, INK_SOFT = tp.INK, tp.INK_SOFT
ACC = ["#0072B2", "#D55E00", "#009E73"]          # Okabe-Ito: identify / learn / deploy
TINT = ["#EEF3FA", "#FFF4E6", "#EEF7EC"]
TIP = 0.7                                         # arrowhead overshoot (see marker refX)
FS_TITLE = 11.0 / K / 72.27 * 25.4                # 11 pt: the two panel titles only (rich variant)
FS_STEP = 10.0 / K / 72.27 * 25.4                 # 10 pt: the three numbered step titles (PANEL_PT)
COL_SIM_TITLE, COL_HW_TITLE = "#0072B2", "#C0562A" # cool / warm, matching the panel tints
DEPLOY_ASPECT = 2.3                               # rich variant: w/h of the full-width deployment photo
_STROBE_DIR = _ANALYSIS / "report_strobes"
HIRES_STROBES = {                                  # 1920 x 1080 originals of the montage panels (paper repo)
    "a": _STROBE_DIR / "hope_didnt_work_2_highres_traceover_strobe.png",
    "b": _STROBE_DIR / "insane_run_8_middle_bg_highres_traceover_strobe.png",
    "c": _STROBE_DIR / "insane_run_2_highres_traceover_strobe.png",
    "d": _STROBE_DIR / "finetune_run_7_highres_traceover_strobe.png",
    "e": _STROBE_DIR / "finetune_run_8_highres_traceover_strobe.png",
    "f": _STROBE_DIR / "figure_eight_6_clean_highres_traceover_strobe.png",
}
# The deployment collage (fig_deploy_collage_paper.py, box B) crops the same figure-eight
# composite to its tracked paths plus a margin, shifted 47 px down so the pool's top edge is
# out and the fish at the bottom right is whole; it prints "crop figure_eight_6 x 150..1871
# y 47..1049" when it builds. The same window here, so the two figures show the same frame.
FIG7_CROP_F = (150, 47, 1871, 1049)               # (left, top, right, bottom) in 1920 x 1080 px

# ---------------------------------------------------------------------------------------
# rasters
# ---------------------------------------------------------------------------------------

def load_render(path: Path, width_px: int = 1100) -> Image.Image:
    """CAD / MuJoCo render on white -> white faded to alpha, trimmed to content."""
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).copy()
    a[..., 3] = tp.white_to_alpha(a, gain=4.0)
    im = Image.fromarray(a)
    bbox = im.getbbox()
    im = im.crop(bbox)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS)


def montage_panel(path: Path, which: str) -> Image.Image:
    """Crop one 892x501 panel out of draft/deploy_strobe_montage.png (bounds measured)."""
    im = Image.open(path).convert("RGB")
    rows = [(49, 550), (642, 1143)]
    cols = [(4, 896), (904, 1796), (1805, 2697)]
    k = "abcdef".index(which)
    r, c = divmod(k, 3)
    return im.crop((cols[c][0], rows[r][0], cols[c][1], rows[r][1]))


def crop_aspect(im: Image.Image, aspect: float, y_center_frac: float = 0.5,
                x_center_frac: float = 0.5) -> Image.Image:
    w, h = im.size
    if w / h > aspect:            # too wide: cut width
        nw, nh = int(round(h * aspect)), h
    else:
        nw, nh = w, int(round(w / aspect))
    x0 = int(round(x_center_frac * w - nw / 2))
    y0 = int(round(y_center_frac * h - nh / 2))
    x0 = max(0, min(x0, w - nw))
    y0 = max(0, min(y0, h - nh))
    return im.crop((x0, y0, x0 + nw, y0 + nh))


def soften(im: Image.Image) -> Image.Image:
    return im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))


# ---------------------------------------------------------------------------------------
# small pictograms
# ---------------------------------------------------------------------------------------

def num_badge(cx, cy, n, color, r=1.85):
    return (f'<circle cx="{tp.f(cx)}" cy="{tp.f(cy)}" r="{tp.f(r)}" fill="{color}"/>'
            + tp.text(cx, cy + 0.9, str(n), fs=FS, anchor="middle", weight="bold", fill="#FFFFFF"))


CHIP = FS2 * 1.38                                 # baseline spacing of the icon+text lines


def chip(x, y, name, label, color, ico=2.7):
    """Icon + one line of 6 pt text, left-aligned; y is the text baseline."""
    s = (tp.icon(name, x, y - 0.75 - ico / 2, ico, color=color, sw=1.9)
         + tp.text(x + ico + 1.0, y, label, fs=FS2, fill=INK))
    return s


def waveform_glyph(x, y, w, h, color):
    """A capped-frequency smooth stroke: two cycles of a 2-harmonic wave."""
    t = np.linspace(0, 1, 160)
    v = np.cos(2 * np.pi * 2 * t) + 0.25 * np.sin(2 * np.pi * 4 * t)
    v = v / 1.25
    d = "M " + " L ".join(f"{x + tt * w:.2f},{y + h / 2 - vv * h / 2:.2f}" for tt, vv in zip(t, v))
    return tp.path(d, stroke=color, sw=0.35)


def dice_glyph(cx, cy, s, color):
    """Randomization pictogram: a die face."""
    x, y = cx - s / 2, cy - s / 2
    out = [tp.rect(x, y, s, s, "none", rx=0.6, stroke=color, sw=0.3)]
    for fx, fy in [(0.27, 0.27), (0.73, 0.27), (0.5, 0.5), (0.27, 0.73), (0.73, 0.73)]:
        out.append(f'<circle cx="{tp.f(x + fx * s)}" cy="{tp.f(y + fy * s)}" r="{tp.f(s * 0.085)}" fill="{color}"/>')
    return "".join(out)


def callout(x_text, y_text, label, x_dot, y_dot, anchor="start"):
    """White label with dark halo + leader to a dot, for use on top of a photo."""
    halo = tp.text(x_text, y_text, label, fs=FS2, anchor=anchor, fill="#FFFFFF",
                   extra='stroke="#1A2230" stroke-width="0.55" stroke-linejoin="round" paint-order="stroke"')
    lx = x_text if anchor == "start" else x_text
    return (halo
            + tp.line(lx, y_text - 0.9, x_dot, y_dot, stroke="#FFFFFF", sw=0.28, marker=False)
            + f'<circle cx="{tp.f(x_dot)}" cy="{tp.f(y_dot)}" r="0.55" fill="#FFFFFF" stroke="#1A2230" stroke-width="0.2"/>')


# Labels drawn on the pool photo: (label, text x, text y, dot x, dot y, anchor), all as
# fractions of the cropped panel (panel d, --photo-aspect 1.6, x-centre 0.55).
POOL_CALLOUTS = [
    ("AprilTag target", 0.96, 0.36, 0.918, 0.50, "end"),
    ("start", 0.30, 0.19, 0.282, 0.05, "middle"),
    ("reached", 0.42, 0.80, 0.42, 0.63, "middle"),
]


# ---------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------

def step_block(x, y, n, color, title, chips):
    """Numbered step: badge + 7 pt bold title on one line, then 6 pt icon+text lines.
    Returns (svg, height below y)."""
    out = [num_badge(x + 1.85, y - 0.85, n, color, r=1.85),
           tp.text(x + 4.6, y, title, fs=FS, weight="bold", fill=color)]
    yy = y
    for name, label in chips:
        yy += CHIP
        out.append(chip(x + 0.4, yy, name, label, color))
    return "\n    ".join(out), yy - y


TEXT_BLACK = "#000000"


def blacken_text(svg: str) -> str:
    """Set every <text> that uses the ink or soft-ink grey to plain black. Strokes, icons and
    the white numerals in the step badges keep their colours."""
    import re
    return re.sub(r'(<text\b[^>]*?fill=")(?:%s|%s)(")' % (INK, INK_SOFT),
                  lambda m: m.group(1) + TEXT_BLACK + m.group(2), svg)


SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by figures/fig1_overview.py - edit freely in Inkscape. -->
<svg
   width="{PW}mm" height="{PH}mm" viewBox="0 0 {W} {H}"
   version="1.1" id="overview_fig1"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   font-family="{FONT}">
  <sodipodi:namedview id="namedview" pagecolor="#ffffff" bordercolor="#000000"
     borderopacity="0.25" inkscape:document-units="mm" inkscape:pagecheckerboard="0"/>
  {defs}
  <g id="layer1" inkscape:groupmode="layer" inkscape:label="Figure">
  {body}
  </g>
</svg>
"""


def build_svg(hw_uri, hw_aspect, twin_uri, twin_aspect, pool_uri, sim_uri, photo_aspect,
              hw_ratio: float, twin_ratio: float) -> tuple[str, float]:
    """Two tinted side panels, Simulation (left) and Hardware (right), read as a Z:
      row 1  digital twin  <-(1) identified from-  physical fish
      row 2  training episode  -(3) deployed zero-shot->  pool run
    Step texts: (1) centred under row 1, (2) under the sim still, (3) under the pool photo.
    The only leftward arrow is the identification, a correspondence rather than a flow, so
    the eye still ends on the hardware result at bottom right."""
    M = 0.4
    panel_w = 42.0
    gap = W - 2 * M - 2 * panel_w
    xP = [M, M + panel_w + gap]                    # panel left edges: [sim, real]
    pad = 3.0
    img_w = panel_w - 2 * pad
    xS, xH = xP[0] + pad, xP[1] + pad              # image left edges
    cS, cH = xP[0] + panel_w / 2, xP[1] + panel_w / 2

    def arrow(y, color, mid, to_left=False):
        x0, x1 = xS + img_w + 0.5, xH - 0.5
        if to_left:
            x0, x1 = x1, x0 + TIP
        else:
            x1 -= TIP
        return (tp.line(x0, y, x1, y, stroke=color, sw=0.5, marker=False)
                .replace("/>", f' marker-end="url(#arrow{mid})"/>'))

    # ---- headers -----------------------------------------------------------------------------
    y_hdr = M + 1.2 + FS * 0.9
    # ---- row 1: renders ----------------------------------------------------------------------
    y1 = y_hdr + 1.8
    tw_w, hw_w = img_w * twin_ratio, img_w * hw_ratio
    tw_h, hw_h = tw_w / twin_aspect, hw_w / hw_aspect
    row1_h = max(tw_h, hw_h)
    tw_x, tw_y = cS - tw_w / 2, y1 + (row1_h - tw_h) / 2
    hw_x, hw_y = cH - hw_w / 2, y1 + (row1_h - hw_h) / 2
    y1_end = y1 + row1_h
    # ---- step 1 (centred across both panels) -----------------------------------------------
    y_s1 = y1_end + 1.0 + FS * 0.9
    s1_title = "Staged system identification"
    s1_detail = "tail stiffness  ·  15 fluid coefficients  ·  motor gain  ·  inertia  ·  vision delay"
    y_s1b = y_s1 + FS2 * 1.35
    title_w = len(s1_title) * FS * 0.47
    # ---- row 2: photos -----------------------------------------------------------------------
    y2 = y_s1b + 3.0
    ph_h = img_w / photo_aspect
    y2_end = y2 + ph_h
    # ---- steps 2, 3 -------------------------------------------------------------------------
    y_s23 = y2_end + 1.2 + FS * 0.9
    blk2, h2 = step_block(xP[0] + 1.6, y_s23, 2, ACC[1], "Learn in simulation only",
                          [("wave-sine", "Band-limited CPG actions"),
                           ("dice-5", "Domain randomization"),
                           ("adjustments", "Noise + target curriculum")])
    blk3, h3 = step_block(xP[1] + 1.6, y_s23, 3, ACC[2], "Deploy zero-shot, unchanged",
                          [("target-arrow", "Target reaching"),
                           ("wind", "Disturbance rejection"),
                           ("arrows-shuffle", "Moving-target tracking")])
    H = y_s23 + max(h2, h3) + 1.6 + M

    g = [
        tp.group(
            "Side panels",
            tp.rect(xP[0], M, panel_w, H - 2 * M, TINT_SIM, rx=1.6, stroke="none"),
            tp.rect(xP[1], M, panel_w, H - 2 * M, TINT_REAL, rx=1.6, stroke="none"),
            tp.text(cS, y_hdr, "Simulation", fs=FS, anchor="middle", weight="bold"),
            tp.text(cH, y_hdr, "Hardware", fs=FS, anchor="middle", weight="bold"),
        ),
        tp.group(
            "Row 1 identification",
            tp.image(tw_x, tw_y, tw_w, tw_h, twin_uri),
            tp.image(hw_x, hw_y, hw_w, hw_h, hw_uri),
            arrow(y1 + row1_h * 0.55, ACC[0], 0, to_left=True),
        ),
        tp.group(
            "Step 1 text",
            num_badge(W / 2 - title_w / 2 - 1.2, y_s1 - 0.85, 1, ACC[0]),
            tp.text(W / 2 + 1.7, y_s1, s1_title, fs=FS, anchor="middle", weight="bold", fill=ACC[0]),
            tp.text(W / 2, y_s1b, s1_detail, fs=FS2, anchor="middle", fill=INK_SOFT),
        ),
        tp.group(
            "Row 2 transfer",
            tp.image(xS, y2, img_w, ph_h, sim_uri),
            tp.rect(xS, y2, img_w, ph_h, "none", rx=0.4, stroke=INK_SOFT, sw=0.18),
            tp.image(xH, y2, img_w, ph_h, pool_uri),
            tp.rect(xH, y2, img_w, ph_h, "none", rx=0.4, stroke=INK_SOFT, sw=0.18),
            arrow(y2 + ph_h * 0.5, ACC[2], 2),
        ),
        tp.group("Step 2 text", blk2),
        tp.group("Step 3 text", blk3),
    ]

    def marker(mid, color):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4" '
                'markerHeight="4" orient="auto-start-reverse">'
                f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{color}"/></marker>')
    defs = "<defs>" + "".join(marker(f"arrow{i}", c) for i, c in enumerate(ACC)) + "</defs>"
    svg = blacken_text(SVG_TEMPLATE.format(W=tp.f(W), H=tp.f(H), PW=tp.f(W * K), PH=tp.f(H * K), FONT=tp.FONT, defs=defs, body="\n  ".join(g)))
    return svg, H


# ---------------------------------------------------------------------------------------
# minimal (title-page) variant: same Z layout, icons instead of text
# ---------------------------------------------------------------------------------------

def icon_row(x, y, n, color, word, icons, ico=3.8, gap=1.5):
    """Badge + one bold word + a row of icons, left-aligned at x with baseline y.
    Returns (svg, right edge)."""
    out = [num_badge(x + 1.85, y - 0.85, n, color, r=1.85)]
    cx = x + 4.6
    if word:
        out.append(tp.text(cx, y, word, fs=FS, weight="bold", fill=color))
        cx += len(word) * FS * 0.5 + 1.6
    for name in icons:
        out.append(tp.icon(name, cx, y - 0.85 - ico / 2, ico, color=color, sw=1.8))
        cx += ico + gap
    return "\n    ".join(out), cx - gap


def build_svg_minimal(hw_uri, hw_aspect, twin_uri, twin_aspect, pool_uri, sim_uri, photo_aspect,
                      hw_ratio: float, twin_ratio: float, words=True) -> tuple[str, float]:
    """Title-page variant. Two tinted panels (Simulation | Hardware), the four images, three
    plain arrows and three badge+icon rows: (1) identify: stiffness, fluid, motor; (2) learn:
    CPG, randomization, curriculum; (3) deploy: reach, disturbance, tracking. `words=False`
    drops the three verbs and leaves only the two panel headers as text."""
    M = 0.4
    panel_w = 42.0
    gap = W - 2 * M - 2 * panel_w
    xP = [M, M + panel_w + gap]
    pad = 3.0
    img_w = panel_w - 2 * pad
    xS, xH = xP[0] + pad, xP[1] + pad
    cS, cH = xP[0] + panel_w / 2, xP[1] + panel_w / 2

    def arrow(y, color, mid, to_left=False):
        x0, x1 = xS + img_w + 0.5, xH - 0.5
        if to_left:
            x0, x1 = x1, x0 + TIP
        else:
            x1 -= TIP
        return (tp.line(x0, y, x1, y, stroke=color, sw=0.5, marker=False)
                .replace("/>", f' marker-end="url(#arrow{mid})"/>'))

    y_hdr = M + 1.2 + FS * 0.9
    y1 = y_hdr + 1.8
    tw_w, hw_w = img_w * twin_ratio, img_w * hw_ratio
    tw_h, hw_h = tw_w / twin_aspect, hw_w / hw_aspect
    row1_h = max(tw_h, hw_h)
    tw_x, tw_y = cS - tw_w / 2, y1 + (row1_h - tw_h) / 2
    hw_x, hw_y = cH - hw_w / 2, y1 + (row1_h - hw_h) / 2
    y1_end = y1 + row1_h

    ROWH = 5.2                                     # icon-row band height
    y_r1 = y1_end + 1.0 + 3.0                      # baseline of the (1) row (icons 3.8 mm tall)
    row1, r1_right = icon_row(0, y_r1, 1, ACC[0], "identify" if words else "",
                              ["scale", "ripple", "engine"])
    # centre the (1) row across the figure
    dx1 = W / 2 - r1_right / 2
    row1 = f'<g transform="translate({tp.f(dx1)},0)">{row1}</g>'

    y2 = y_r1 + 3.2
    ph_h = img_w / photo_aspect
    y2_end = y2 + ph_h
    y_r23 = y2_end + 1.0 + 3.0
    row2, _ = icon_row(xP[0] + 1.6, y_r23, 2, ACC[1], "learn" if words else "",
                       ["wave-sine", "dice-5", "adjustments"])
    row3, _ = icon_row(xP[1] + 1.6, y_r23, 3, ACC[2], "deploy" if words else "",
                       ["target-arrow", "wind", "arrows-shuffle"])
    H = y_r23 + 1.6 + M

    g = [
        tp.group(
            "Side panels",
            tp.rect(xP[0], M, panel_w, H - 2 * M, TINT_SIM, rx=1.6, stroke="none"),
            tp.rect(xP[1], M, panel_w, H - 2 * M, TINT_REAL, rx=1.6, stroke="none"),
            tp.text(cS, y_hdr, "Simulation", fs=FS, anchor="middle", weight="bold"),
            tp.text(cH, y_hdr, "Hardware", fs=FS, anchor="middle", weight="bold"),
        ),
        tp.group(
            "Row 1",
            tp.image(tw_x, tw_y, tw_w, tw_h, twin_uri),
            tp.image(hw_x, hw_y, hw_w, hw_h, hw_uri),
            arrow(y1 + row1_h * 0.55, ACC[0], 0, to_left=True),
        ),
        tp.group("Step 1 icons", row1),
        tp.group(
            "Row 2",
            tp.image(xS, y2, img_w, ph_h, sim_uri),
            tp.rect(xS, y2, img_w, ph_h, "none", rx=0.4, stroke=INK_SOFT, sw=0.18),
            tp.image(xH, y2, img_w, ph_h, pool_uri),
            tp.rect(xH, y2, img_w, ph_h, "none", rx=0.4, stroke=INK_SOFT, sw=0.18),
            arrow(y2 + ph_h * 0.5, ACC[2], 2),
        ),
        tp.group("Step 2 icons", row2),
        tp.group("Step 3 icons", row3),
    ]

    def marker(mid, color):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4" '
                'markerHeight="4" orient="auto-start-reverse">'
                f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{color}"/></marker>')
    defs = "<defs>" + "".join(marker(f"arrow{i}", c) for i, c in enumerate(ACC)) + "</defs>"
    svg = blacken_text(SVG_TEMPLATE.format(W=tp.f(W), H=tp.f(H), PW=tp.f(W * K), PH=tp.f(H * K), FONT=tp.FONT, defs=defs, body="\n  ".join(g)))
    return svg, H



# ---------------------------------------------------------------------------------------
# rich (title-page) variant: pipeline + trajectory generator + complex-manoeuvre result
# ---------------------------------------------------------------------------------------

def mlp_glyph(x, y, w, h, layers=(4, 5, 5, 3), color=INK, *, sp_max: float = 1.6,
              r: float = 0.45):
    """Small fully connected network, no labels.

    Node spacing is h/max(layers) but capped at `sp_max`, so in the figure the glyph stays a
    compact pictogram however tall a box it is handed. Both the cap and the node radius are
    arguments because the talk video draws this same glyph many times larger, where the printed
    cap would leave the network a tight little knot floating in an otherwise empty panel; the
    strokes scale with the radius so the whole thing enlarges evenly. The defaults are the
    printed figure's values, so its output is unchanged.
    """
    xs = np.linspace(x, x + w, len(layers))
    sp = min(h / max(layers), sp_max)
    cols = [[(cx, y + h / 2 + (i - (n - 1) / 2) * sp) for i in range(n)] for n, cx in zip(layers, xs)]
    k = r / 0.45
    out = []
    for a, b in zip(cols[:-1], cols[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                out.append(f'<line x1="{tp.f(x1)}" y1="{tp.f(y1)}" x2="{tp.f(x2)}" y2="{tp.f(y2)}" '
                           f'stroke="#9AA5B5" stroke-width="{0.08 * k:g}" stroke-opacity="0.7"/>')
    for i, col in enumerate(cols):
        fill = tp.COL_IN if i == 0 else tp.COL_OUT if i == len(cols) - 1 else tp.COL_HID
        for (cx, cy) in col:
            out.append(f'<circle cx="{tp.f(cx)}" cy="{tp.f(cy)}" r="{r:g}" fill="{fill}" '
                       f'stroke="{color}" stroke-width="{0.1 * k:g}"/>')
    return "".join(out)


def tg_waveform(x, y, w, h, color_sig=tp.COL_SIGNAL, color_env=tp.COL_FILT):
    """Two-harmonic stroke with a filtered amplitude step: the trajectory generator's output."""
    t = np.linspace(0, 2.6, 400)
    k = np.where(t < 1.0, 0.45, 0.9 - 0.45 * np.exp(-(t - 1.0) / 0.1))
    phi = 2 * np.pi * 1.15 * t
    sig = k * (np.cos(phi) + 0.25 * np.sin(2 * phi))
    lo, hi = -1.25, 1.25
    X = lambda tt: x + tt / t[-1] * w
    Y = lambda v: y + h - (v - lo) / (hi - lo) * h
    poly = lambda vals: "M " + " L ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(t, vals))
    return "".join([
        tp.line(x, Y(0), x + w, Y(0), stroke=tp.COL_DES, sw=0.12, dash="0.5 0.4", marker=False),
        tp.path(poly(k), stroke=color_env, sw=0.22),
        tp.path(poly(-k), stroke=color_env, sw=0.22),
        tp.path(poly(sig), stroke=color_sig, sw=0.32),
    ])


COL_REAL = "#2B3A55"        # measured / hardware
COL_TWIN = "#D98B4A"        # simulated / identified
COL_CMD = "#8A94A3"         # commanded


def mini_axes(x, y, w, h):
    """L-shaped axes for a schematic mini plot."""
    return tp.path(f"M {tp.f(x)},{tp.f(y)} V {tp.f(y + h)} H {tp.f(x + w)}", stroke=INK_SOFT, sw=0.18)


def mini_tail(x, y, w, h):
    """Re H against omega^2 in air: a straight line whose intercept is k_eff (schematic of the
    identification in Sec. IV-A; numbers loosely follow the measured fit, k_eff = 0.98)."""
    om2 = np.array([40, 160, 355, 630], float)
    lo, hi = -1.0, 1.5
    def X(v): return x + v / 700.0 * w
    def Y(v): return y + h - (v - lo) / (hi - lo) * h
    k, slope = 0.98, -0.00264
    out = [mini_axes(x, y, w, h),
           tp.line(X(0), Y(k), X(700), Y(k + slope * 700), stroke=COL_TWIN, sw=0.3, marker=False)]
    rng = np.random.default_rng(3)
    for o in om2:
        for dv in rng.normal(0, 0.16, 4):
            out.append(f'<circle cx="{tp.f(X(o))}" cy="{tp.f(Y(k + slope * o + dv))}" r="0.35" '
                       f'fill="{COL_REAL}" fill-opacity="0.75"/>')
    out.append(f'<circle cx="{tp.f(X(0))}" cy="{tp.f(Y(k))}" r="0.5" fill="{COL_TWIN}"/>')
    out.append(tp.text(x + w + 0.4, y + h + 0.6, "ω²", fs=FS2, style="italic", fill=INK_SOFT))
    out.append(tp.eq(x - 0.5, y + h / 2 + 0.7, "\\r{Re }H", fs=FS2, anchor="end", fill=INK_SOFT))
    return "\n    ".join(out)


def mini_fluid(x, y, w, h):
    """Tail-marker lateral positions, measured (solid) against the twin (dashed), for three
    markers along the tail -- what the fluid fit minimises (Sec. IV-B)."""
    t = np.linspace(0, 3.0, 240)
    def X(v): return x + v / 3.0 * w
    def Y(v): return y + h / 2 - v * (h / 2) / 1.15
    out = [mini_axes(x, y, w, h)]
    for amp, ph, damp in [(0.35, 0.0, 1.0), (0.65, 0.35, 1.05), (1.0, 0.7, 0.93)]:
        real = amp * np.sin(2 * np.pi * 1.1 * t + ph)
        twin = amp * damp * np.sin(2 * np.pi * 1.1 * t + ph - 0.18)
        out.append(tp.path("M " + " L ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(t, real)),
                           stroke=COL_REAL, sw=0.26))
        out.append(tp.path("M " + " L ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(t, twin)),
                           stroke=COL_TWIN, sw=0.26, dash="0.6 0.4"))
    out.append(tp.eq(x + w + 0.4, y + h + 0.6, "t", fs=FS2, fill=INK_SOFT))
    out.append(tp.text(x - 0.5, y + h / 2 + 0.7, "y", fs=FS2, anchor="end", style="italic", fill=INK_SOFT))
    return "\n    ".join(out)


def mini_motor(x, y, w, h):
    """Commanded (grey, dashed) against measured (dark) and identified-model (orange) motor angle:
    the servo's gain, rotor inertia and transport delay (Sec. IV-C)."""
    t = np.linspace(0, 3.0, 600)
    cmd = 0.45 + 0.55 * np.clip(np.sin(2 * np.pi * 0.8 * t), -0.2, 1.0) ** 3 + 0.12 * np.sin(4 * np.pi * 0.8 * t)
    def lag(sig, tau, delay):
        dt = t[1] - t[0]
        k = int(round(delay / dt))
        src = np.concatenate([np.full(k, sig[0]), sig[:-k]]) if k else sig
        out = np.empty_like(src); v = src[0]
        for i, u in enumerate(src):
            v += (u - v) * dt / tau; out[i] = v
        return out
    meas = lag(cmd, 0.09, 0.03)
    ident = lag(cmd, 0.085, 0.028)
    lo, hi = -0.3, 1.5
    def X(v): return x + v / 3.0 * w
    def Y(v): return y + h - (v - lo) / (hi - lo) * h
    def P(sig): return "M " + " L ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(t, sig))
    out = [mini_axes(x, y, w, h),
           tp.path(P(meas), stroke=COL_REAL, sw=0.34),
           tp.path(P(ident), stroke=COL_TWIN, sw=0.24),
           tp.eq(x + w + 0.4, y + h + 0.6, "t", fs=FS2, fill=INK_SOFT),
           tp.eq(x - 0.5, y + h / 2 + 0.7, "θ", fs=FS2, anchor="end", fill=INK_SOFT)]
    return "\n    ".join(out)


def tail_icon(x, y, size, color=INK, sw=1.9):
    """Bent fish tail with a forked caudal fin, drawn in the 24 x 24 Tabler icon frame so it
    sits like the other icons (top-left at x, y; `size` mm wide)."""
    s = size / 24.0
    body = ('<path d="M 2,9 C 7,7 10,15 15,12"/>'          # upper edge of the tail, bending
            '<path d="M 2,15 C 7,17 10,9 15,12"/>'          # lower edge (tapers to the peduncle)
            '<path d="M 15,12 L 22,5 L 19.5,12 L 22,19 Z"/>')   # forked caudal fin
    return (f'<g transform="translate({tp.f(x)},{tp.f(y)}) scale({s:.4f})" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'inkscape:label="icon:tail">{body}</g>')


def sysid_cards(x0, y, card_w, gap, card_h, fill="#FFFFFF"):
    """Two identification cards side by side: tail stiffness and fluid (the motor servo card was
    dropped in review; the identification is described in the text). Each is a
    small icon + 6 pt title, a schematic mini plot (mini_tail / mini_fluid / mini_motor, kept
    local so this figure does not depend on the pipeline module) and the symbols it fixes. Reads left to right in the paper's staged
    order, which is why the arrow above (hardware -> twin) does not need its own labels."""
    # plot fills the card; a one-line legend under it says what the two traces are
    plot_w, plot_h = card_w - 9.8, card_h - 6.0
    content = [  # icon, title, plot, legend entries (label, colour, dashed)
        ("tail", "Tail dynamics", mini_tail_springs,
         [("measured", COL_REAL, False), ("twin", COL_TWIN, False)]),
        ("ripple", "Fluid model", mini_fluid,
         [("measured", COL_REAL, False), ("twin", COL_TWIN, True)]),
    ]
    out = []
    for i, (ic, title, plot_fn, legend) in enumerate(content):
        x = x0 + i * (card_w + gap)
        px, py, ph = x + 5.6, y + 4.6, plot_h
        if ic == "ripple":                           # the fluid traces fill the card like the tail curve does
            py, ph = y + 4.2, card_h - 5.0           # (was centred in a 5.6 mm plot: air above and below)
        out += [tp.rect(x, y, card_w, card_h, fill, rx=0.8, stroke="none"),      # no card outline
                (tail_icon(x + 1.1, y + 0.8, 3.6, color=ACC[0]) if ic == "tail"
                 else tp.icon(ic, x + 1.1, y + 0.8, 3.6, color=ACC[0], sw=1.9)),
                tp.text(x + 5.4, y + 3.4, title, fs=FS, weight="bold"),
                plot_fn(px, py, plot_w, ph)]
        pass                                             # no legend row: the caption names the two colours
    return "\n    ".join(out)


def card_legend(cx, ly, legend):
    """Centred row of [line sample + 6 pt label], as used under every identification card."""
    out = []
    items = [(lab, col, dash, 3.0 + 0.8 + len(lab) * FS2 * 0.42) for lab, col, dash in legend]
    lx = cx - (sum(wd for *_, wd in items) + 2.0 * (len(items) - 1)) / 2
    for lab, col, dash, wd in items:
        out.append(tp.line(lx, ly - 0.75, lx + 3.0, ly - 0.75, stroke=col, sw=0.3,
                           dash="0.6 0.4" if dash else "", marker=False))
        out.append(tp.text(lx + 3.8, ly, lab, fs=FS2, fill=INK_SOFT))
        lx += wd + 2.0
    return out


def photo_legend(x, y, items):
    """One-line legend drawn over a photo: colour swatch + white haloed 6 pt label per item."""
    out = []
    for lab, col in items:
        out.append(tp.line(x, y - 0.75, x + 3.0, y - 0.75, stroke=col, sw=0.5, marker=False))
        out.append(tp.text(x + 3.8, y, lab, fs=FS2, fill="#FFFFFF",
                           extra='stroke="#1A2230" stroke-width="0.5" stroke-linejoin="round" paint-order="stroke"'))
        x += 3.8 + len(lab) * FS2 * 0.44 + 2.4
    return out


def speed_match_plot(x, y, w, h, csv_path):
    """Twin vs hardware cruising speed against commanded tail-beat frequency (the open-loop
    first-swim runs replayed into the twin; data from fig_speed_vs_freq_paper.py's CSV).
    Frame: L-axes, 6 pt tick and axis labels, legend in the top-left."""
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    f = np.array([float(r["f"]) for r in rows]); amp = np.array([float(r["amp_mm"]) for r in rows])
    vr = np.array([float(r["v_real"]) for r in rows]); vt = np.array([float(r["v_twin"]) for r in rows])
    xj = f + (amp - amp.mean()) * 0.04                       # spread coincident frequencies a little
    ax_x, ax_y, ax_w, ax_h = x + 7.0, y + 2.4, w - 8.2, h - 7.6
    def X(v): return ax_x + (v - 0.5) / 4.0 * ax_w
    def Y(v): return ax_y + ax_h - v / 0.55 * ax_h
    out = [tp.path(f"M {tp.f(ax_x)},{tp.f(ax_y)} V {tp.f(ax_y + ax_h)} H {tp.f(ax_x + ax_w)}",
                   stroke=INK_SOFT, sw=0.2)]
    for fx in (1, 2, 3, 4):
        out.append(tp.line(X(fx), ax_y + ax_h, X(fx), ax_y + ax_h + 0.6, stroke=INK_SOFT, sw=0.2, marker=False))
        out.append(tp.text(X(fx), ax_y + ax_h + 2.6, f"{fx}", fs=FS2, anchor="middle", fill=INK_SOFT))
    for vy in (0.2, 0.4):
        out.append(tp.line(ax_x - 0.6, Y(vy), ax_x, Y(vy), stroke=INK_SOFT, sw=0.2, marker=False))
        out.append(tp.text(ax_x - 1.0, Y(vy) + 0.75, f"{vy:g}", fs=FS2, anchor="end", fill=INK_SOFT))
    out.append(tp.text(ax_x + ax_w / 2, y + h - 0.4, "tail-beat frequency [Hz]", fs=FS2, anchor="middle", fill=INK_SOFT))
    out.append(tp.text(x + 1.9, ax_y + ax_h / 2, "speed [m/s]", fs=FS2, anchor="middle", fill=INK_SOFT,
                       extra=f'transform="rotate(-90 {tp.f(x + 1.9)} {tp.f(ax_y + ax_h / 2)})"'))
    for a, b, c in zip(xj, vr, vt):
        out.append(tp.line(X(a), Y(b), X(a), Y(c), stroke="#BFC5CE", sw=0.25, marker=False))
    for a, b in zip(xj, vr):
        out.append(f'<circle cx="{tp.f(X(a))}" cy="{tp.f(Y(b))}" r="0.75" fill="{COL_REAL}"/>')
    for a, c in zip(xj, vt):
        out.append(tp.rect(X(a) - 0.65, Y(c) - 0.65, 1.3, 1.3, COL_TWIN, rx=0.1, stroke="none"))
    # legend (top-left inside the axes)
    lx, ly = ax_x + 1.2, ax_y + 1.9
    out.append(f'<circle cx="{tp.f(lx)}" cy="{tp.f(ly - 0.75)}" r="0.75" fill="{COL_REAL}"/>')
    out.append(tp.text(lx + 1.6, ly, "hardware", fs=FS2, fill=INK_SOFT))
    out.append(tp.rect(lx - 0.65, ly + 2.2 - 0.75 - 0.65, 1.3, 1.3, COL_TWIN, rx=0.1, stroke="none"))
    out.append(tp.text(lx + 1.6, ly + 2.2, "twin", fs=FS2, fill=INK_SOFT))
    return "\n    ".join(out)


def mini_tail_springs(x, y, w, h):
    """Tail stiffness pictogram: the measured tail bent under a tendon load is one continuous
    curve (dark); the twin approximates it as five rigid links joined by torsional hinges
    (orange straight segments, hinge dots). What is identified is the hinge stiffness and
    damping that make the chain match the curve."""
    L, A = w * 0.86, w * 0.86 * 0.34            # projected length and tip rise
    x0, y0 = x + 0.6, y + h * 0.92
    s = np.linspace(0, 1, 120)
    curve = np.column_stack([x0 + s * L, y0 - A * s ** 2])
    out = [tp.line(x0, y0, x0 + L, y0, stroke=INK_SOFT, sw=0.18, dash="0.6 0.4", marker=False),
           tp.path("M " + " L ".join(f"{a:.2f},{b:.2f}" for a, b in curve), stroke=COL_REAL, sw=0.7)]
    # five rigid links: straight segments between six points on the curve
    n = 5
    sk = np.linspace(0, 1, n + 1)
    nodes = np.column_stack([x0 + sk * L, y0 - A * sk ** 2])
    out.append(tp.path("M " + " L ".join(f"{a:.2f},{b:.2f}" for a, b in nodes), stroke=COL_TWIN, sw=0.45)
               .replace('stroke-linejoin="round"', 'stroke-linejoin="miter"'))
    for (px, py) in nodes[:-1]:                # hinges (the sixth point is the free tip)
        out.append(f'<circle cx="{tp.f(px)}" cy="{tp.f(py)}" r="0.55" fill="#FFFFFF" stroke="{INK}" stroke-width="0.22"/>')
    # torque at the root (the tendon pull)
    out.append(tp.path(f"M {x0 + 2.4:.2f},{y0 + 0.1:.2f} A 2.0 2.0 0 0 0 {x0 + 0.5:.2f},{y0 - 1.9:.2f}",
                       stroke=INK, sw=0.25, marker=True))
    return "\n    ".join(out)


def sim2real_labels(png: Path, full_size, crop_size):
    """'twin' / 'hardware' next to the two fish at the final instant of a fig_sim2real_strobe
    render, read from its JSON sidecar. Positions come back as fractions of the (centre-)cropped
    image. The hardware label sits above its fish, the twin label below, so they never collide."""
    import json
    meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    last = meta["frames"][-1]
    W0, H0 = full_size; Wc, Hc = crop_size
    ox, oy = (W0 - Wc) / 2, (H0 - Hc) / 2
    def frac(px):
        return (px[0] - ox) / Wc, (px[1] - oy) / Hc
    tx, ty = frac(last["twin_px"]); rx, ry = frac(last["real_px"])
    return [("hardware", rx, ry - 0.11, "#2B3A55", "middle"),
            ("twin", tx, ty + 0.10, "#D98B4A", "middle")]


def build_svg_rich(hw_uri, hw_aspect, twin_uri, twin_aspect, pool_uri, sim_uri, photo_aspect,
                   hw_ratio: float, twin_ratio: float, s2r_labels=(), compact=False,
                   pool_aspect: float = DEPLOY_ASPECT) -> tuple[str, float]:
    """Title-page variant with three rows in the Simulation | Hardware panels:
      1  digital twin  <-(1) identify-  physical fish          (renders)
      2  (2) learn: SAC policy -> u* -> trajectory generator  -(3) zero-shot->  same policy on the
         fish at 50 Hz, closed by AprilTag / IMU / encoder feedback   (mini diagrams)
      3  training episode  |  deployed figure-eight pursuit    (overhead stills)"""
    M = 0.4
    panel_w = 42.0
    gap = W - 2 * M - 2 * panel_w
    xP = [M, M + panel_w + gap]
    pad = 3.0
    img_w = panel_w - 2 * pad
    xS, xH = xP[0] + pad, xP[1] + pad
    cS, cH = xP[0] + panel_w / 2, xP[1] + panel_w / 2

    def arrow(x0, x1, y, color, mid, sw=0.5):
        x1 = x1 - TIP if x1 > x0 else x1 + TIP
        return (tp.line(x0, y, x1, y, stroke=color, sw=sw, marker=False)
                .replace("/>", f' marker-end="url(#arrow{mid})"/>'))

    # ---- row 1: renders + (1) --------------------------------------------------------------
    y_hdr = M + 1.2 + FS_TITLE * 0.9
    y1 = y_hdr + 1.8
    tw_w, hw_w = img_w * twin_ratio, img_w * hw_ratio
    tw_h, hw_h = tw_w / twin_aspect, hw_w / hw_aspect
    row1_h = max(tw_h, hw_h)
    tw_x, tw_y = cS - tw_w / 2, y1 + (row1_h - tw_h) / 2
    hw_x, hw_y = cH - hw_w / 2, y1 + (row1_h - hw_h) / 2
    y1_end = y1 + row1_h
    if compact:                                      # no titles, no renders: start with step (1)
        y1_end = M + 0.4
    SLIM = 1.0                                       # (was 0.88 in compact: too light against the photo)
    # (1) identify: badge line, then a strip of three identification cards spanning both panels
    # (schematic mini plots borrowed from the pipeline figure's system-identification strip)
    y_s1 = y1_end + 0.8 + FS * 0.9
    x_s1 = 1.9
    strip_y, strip_h = y_s1 + 1.6, 11.6 * SLIM
    card_w, card_gap = 37.0, 2.0                     # two cards, centred in the column
    strip = sysid_cards((W - (2 * card_w + card_gap)) / 2, strip_y, card_w, card_gap, strip_h)

    # ---- row 2: one control loop across both panels ------------------------------------------
    #   Simulation: [Policy] -u*-> [Trajectory Generator]  --theta-->  Hardware: the fish (CAD render)
    #   and its sensors (camera, IMU, encoder) feed back along a rounded return path into the Policy
    y2 = strip_y + strip_h + 2.0
    y_lab = y2 + FS * 0.9 + 0.4                     # (2) / (3) badge line
    box_h = 8.0 * SLIM
    y_c = y_lab + 2.4 + box_h / 2                    # chain centre line
    cap_y = y_c + box_h / 2 + 2.4                    # captions under the boxes
    loop_y = cap_y + 5.6                             # return path, with air under the captions
    ico = 3.2
    row2_h = (loop_y + ico / 2 + 1.6) - y2
    x_ret = xP[0] + 2.6                              # the return path climbs here, into the Policy
    P = dict(x=xP[0] + 6.6, y=y_c - box_h / 2, w=10.5, h=box_h)
    T = dict(x=xP[0] + 22.0, y=y_c - box_h / 2, w=19.0, h=box_h)
    fish_w = 15.0 * SLIM; fish_h = fish_w / hw_aspect
    fish_x = xP[1] + panel_w - 2.0 - fish_w          # hardware fish: the CAD render, small, far right
    fish_cx = fish_x + fish_w / 2
    left = [
        num_badge(xP[0] + 3.65, y_lab - 0.95, 2, ACC[1], r=2.05),
        tp.text(xP[0] + 6.6, y_lab, "Policy Learning", fs=FS_STEP, weight="bold", fill=INK),
        tp.rect(**P, fill="#FFFFFF", rx=0.8, stroke="none"),                 # no box outline
        mlp_glyph(P["x"] + 1.3, P["y"] + 1.3, P["w"] - 2.6, P["h"] - 2.6),
        tp.text(P["x"] + P["w"] / 2, cap_y, "Policy", fs=FS2, anchor="middle"),
        arrow(P["x"] + P["w"] + 0.3, T["x"] - 0.3, y_c, INK, 3, sw=0.5),
        tp.eq((P["x"] + P["w"] + T["x"]) / 2, y_c - 1.2, "\\b{u}*", fs=FS2, anchor="middle"),
        tp.rect(**T, fill="#FFFFFF", rx=0.8, stroke="none"),                 # no box outline
        tg_waveform(T["x"] + 1.2, T["y"] + 0.9, T["w"] - 2.4, T["h"] - 1.8),
        tp.text(T["x"] + T["w"] / 2, cap_y, "Trajectory Generator", fs=FS2, anchor="middle"),
    ]
    right = [
        num_badge(xP[1] + 1.4, y_lab - 0.95, 3, ACC[2], r=2.05),
        tp.text(xP[1] + 4.3, y_lab, "Zero-Shot Deployment", fs=FS_STEP, weight="bold", fill=INK),
        tp.image(fish_x, y_c - fish_h / 2, fish_w, fish_h, hw_uri),
        tp.text(fish_x + fish_w, cap_y, "Onboard at 50 Hz", fs=FS2, anchor="end"),
    ]
    x_th0, x_th1 = T["x"] + T["w"] + 0.5, fish_x - 0.4
    r = 2.4                                          # corner radius of the return path
    ret = (f"M {tp.f(fish_cx)},{tp.f(cap_y + 0.8)} V {tp.f(loop_y - r)} "
           f"Q {tp.f(fish_cx)},{tp.f(loop_y)} {tp.f(fish_cx - r)},{tp.f(loop_y)} "
           f"H {tp.f(x_ret + r)} Q {tp.f(x_ret)},{tp.f(loop_y)} {tp.f(x_ret)},{tp.f(loop_y - r)} "
           f"V {tp.f(y_c + r)} Q {tp.f(x_ret)},{tp.f(y_c)} {tp.f(x_ret + r)},{tp.f(y_c)} "
           f"H {tp.f(P['x'] - TIP - 0.1)}")
    ic_x0 = (x_th0 + x_th1) / 2 - (3 * ico + 2 * 2.2) / 2   # sensor icons on the return path, centred under theta
    loop = [
        arrow(x_th0, x_th1, y_c, INK, 3, sw=0.5),
        tp.text((x_th0 + x_th1) / 2, y_c - 1.3, "θ", fs=FS2, anchor="middle", style="italic"),
        tp.path(ret, stroke=INK, sw=0.5, marker=True),
        tp.rect(ic_x0 - 0.8, loop_y - ico / 2 - 0.6, 3 * ico + 2 * 2.2 + 1.6, ico + 1.2,
                "#FFFFFF" if compact else TINT_REAL, rx=0.6,
                stroke="none"),
        tp.icon("camera", ic_x0, loop_y - ico / 2, ico, color=INK),
        tp.icon("rotate-3d", ic_x0 + ico + 2.2, loop_y - ico / 2, ico, color=INK),
        tp.icon("engine", ic_x0 + 2 * (ico + 2.2), loop_y - ico / 2, ico, color=INK),
        tp.text(ic_x0 + (3 * ico + 2 * 2.2) / 2, loop_y - ico / 2 - 1.3, "Observations", fs=FS2,
                anchor="middle", fill=INK_SOFT),
    ]

    # ---- row 3: results ----------------------------------------------------------------------
    panel_end = y2 + row2_h + 1.0                   # the tinted side panels stop here
    y3 = panel_end + (0.2 if compact else 1.4)       # no panel edge to clear in the compact variant
    full_w = W - 2 * M
    ph_h = full_w / pool_aspect                     # one wide deployment photo across both columns
    H = y3 + ph_h + M

    g = [
        tp.group(
            "Side panels",
            *([] if compact else [                  # the compact variant has no tinted panels
                tp.rect(xP[0], M, panel_w, panel_end - M, TINT_SIM, rx=1.6, stroke="none"),
                tp.rect(xP[1], M, panel_w, panel_end - M, TINT_REAL, rx=1.6, stroke="none")]),
            # panel titles: the only coloured text besides the badges; larger than body text
            *([] if compact else [
                tp.text(cS, y_hdr, "Simulation", fs=FS_TITLE, anchor="middle", weight="bold", fill=COL_SIM_TITLE),
                tp.text(cH, y_hdr, "Hardware", fs=FS_TITLE, anchor="middle", weight="bold", fill=COL_HW_TITLE)]),
        ),
        tp.group(
            "Row 1 identification",
            *([] if compact else [tp.image(tw_x, tw_y, tw_w, tw_h, twin_uri),
                                  tp.image(hw_x, hw_y, hw_w, hw_h, hw_uri)]),
            num_badge(x_s1 + 2.05, y_s1 - 0.95, 1, ACC[0], r=2.05),
            tp.text(x_s1 + 5.0, y_s1, "System Identification", fs=FS_STEP, weight="bold", fill=INK),
            strip,
        ),
        tp.group("Row 2 learn", *left),
        tp.group("Row 2 deploy", *right),
        tp.group("Row 2 loop", *loop),
        tp.group(
            "Row 3 results",
            tp.image(M, y3, full_w, ph_h, pool_uri),                       # no frame around the photo
        ),
    ]

    def marker(mid, color):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4" '
                'markerHeight="4" orient="auto-start-reverse">'
                f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{color}"/></marker>')
    defs = ("<defs>" + "".join(marker(f"arrow{i}", c) for i, c in enumerate(ACC + [INK]))
            + marker("arrow", INK) + "</defs>")
    svg = blacken_text(SVG_TEMPLATE.format(W=tp.f(W), H=tp.f(H), PW=tp.f(W * K), PH=tp.f(H * K), FONT=tp.FONT, defs=defs, body="\n  ".join(g)))
    return svg, H


# ---------------------------------------------------------------------------------------
# boxed (title-page) variant: three tinted step boxes, no badges, the photo inside the loop
# ---------------------------------------------------------------------------------------

def build_svg_boxed(hw_uri, hw_aspect, twin_uri, twin_aspect, pool_uri, sim_uri, photo_aspect,
                    hw_ratio: float, twin_ratio: float, s2r_labels=(),
                    pool_aspect: float = DEPLOY_ASPECT, white_cards: bool = True,
                    fish_w: float = 17.5, *, layers: bool = False,
                    geom_out: dict | None = None) -> tuple[str, float]:
    """Same content as the compact variant, re-framed:
        [ System Identification: tail dynamics | fluid model ]                (full width)
        [ Policy Learning: policy -> TG ]  --theta-->  [ Zero-Shot Deployment: fish ]
        the observation loop from the fish back into the policy is drawn large, and the
        deployment photo (rounded corners) sits inside it.
    Each step is a tinted rounded box with a centred title in its accent colour; the
    numbered badges are gone (the boxes do their job). The step colours are the compact
    variant's, so the two figures stay interchangeable.

    layers=True splits the five drawing groups into the nine the ICRA video reveals one at a
    time (tools/make_fig1_animation.py); the picture is the same, only the grouping differs.
    geom_out, if given, is filled with the mm rectangles that tool draws its live insets into."""
    M = 0.4
    pad = 2.0                                        # inner padding of every box
    RX = 1.6                                         # corner radius of boxes and photo
    full_w = W - 2 * M
    title_dy = pad + FS_STEP * 0.72                  # box top -> title baseline

    def title(cx, y_base, s, color):
        # titles in ink (black after blacken_text); the box tint alone carries the step colour
        return tp.text(cx, y_base, s, fs=FS_STEP, anchor="middle", weight="bold", fill=INK)

    def arrow(x0, x1, y, color, mid, sw=0.5):
        x1 = x1 - TIP if x1 > x0 else x1 + TIP
        return (tp.line(x0, y, x1, y, stroke=color, sw=sw, marker=False)
                .replace("/>", f' marker-end="url(#arrow{mid})"/>'))

    # ---- box 1: system identification (full width) -----------------------------------------
    y1 = M
    t1 = y1 + title_dy
    card_w, card_gap, card_h = 37.0, 2.0, 11.6
    strip_y = t1 + 2.0
    strip = sysid_cards((W - (2 * card_w + card_gap)) / 2, strip_y, card_w, card_gap, card_h,
                        fill="#FFFFFF" if white_cards else "none")
    box1_h = (strip_y + card_h + pad) - y1

    # ---- boxes 2 and 3: the control chain, split at the theta arrow -------------------------
    y2 = y1 + box1_h + 1.6
    gap23 = 2.4
    box2_w = (full_w - gap23) / 2 + 2.5              # a little wider: the TG caption needs the room
    box3_x = M + box2_w + gap23
    box3_w = W - M - box3_x
    t2 = y2 + title_dy
    box_h = 8.0                                      # policy / TG sub-boxes
    y_c = t2 + 2.6 + box_h / 2                       # chain centre line
    cap_y = y_c + box_h / 2 + 3.3                    # captions, centred in the band under the sub-boxes
    box23_h = (cap_y + 1.7) - y2
    x_ret = M + pad + 0.8                            # left leg of the loop, climbing into the Policy
    P = dict(x=M + pad + 4.4, y=y_c - box_h / 2, w=10.5, h=box_h)
    T = dict(x=P["x"] + P["w"] + 4.8, y=y_c - box_h / 2, w=18.0, h=box_h)
    fish_h = fish_w / hw_aspect
    fish_cx = box3_x + box3_w / 2                    # hardware fish, centred in box 3
    fish_x = fish_cx - fish_w / 2
    x_leg = W - x_ret                                # right leg of the loop, mirroring the left one
    x_th0, x_th1 = T["x"] + T["w"] + 0.5, fish_x - 0.4
    # split where the video's beats split: everything up to and including the theta arrow is
    # "policy -> TG -> theta", the fish and its caption are the hardware. Concatenated below,
    # so the drawing order is what it always was.
    tg_glyph = tg_waveform(T["x"] + 1.2, T["y"] + 0.9, T["w"] - 2.4, T["h"] - 1.8)
    chain_rl = [
        tp.rect(**P, fill="#FFFFFF", rx=0.8, stroke="none"),
        mlp_glyph(P["x"] + 1.3, P["y"] + 1.3, P["w"] - 2.6, P["h"] - 2.6),
        tp.text(P["x"] + P["w"] / 2, cap_y, "Policy", fs=FS2, anchor="middle"),
        arrow(P["x"] + P["w"] + 0.3, T["x"] - 0.3, y_c, INK, 3, sw=0.5),
        tp.eq((P["x"] + P["w"] + T["x"]) / 2, y_c - 1.2, "\\b{u}*", fs=FS2, anchor="middle"),
        tp.rect(**T, fill="#FFFFFF", rx=0.8, stroke="none"),
        tg_glyph,
        tp.text(T["x"] + T["w"] / 2, cap_y, "Trajectory generator", fs=FS2, anchor="middle"),
        arrow(x_th0, x_th1, y_c, INK, 3, sw=0.5),
        # theta sits over the part of the arrow inside box 3, centred between its edge and the fish
        tp.text((box3_x + fish_x) / 2, y_c - 1.3, "θ", fs=FS2, anchor="middle", style="italic"),
    ]
    chain_hw = [
        tp.image(fish_x, y_c - fish_h / 2, fish_w, fish_h, hw_uri),
        tp.text(fish_cx, cap_y, "Onboard (50 Hz)", fs=FS2, anchor="middle"),
    ]
    chain = chain_rl + chain_hw

    # ---- the photo inside the loop ----------------------------------------------------------
    # the return path leaves the fish's right side at the chain height, mirroring the entry
    # into the Policy on the left, then runs down the right edge
    ph_y = y2 + box23_h + 2.0
    ph_x = x_ret + 2.2
    ph_w = (x_leg - 2.2) - ph_x
    ph_h = ph_w / pool_aspect
    loop_y = ph_y + ph_h + 4.2                       # bottom leg, with air under the photo
    ico = 3.2
    H = loop_y + ico / 2 + 1.2 + M
    r = 2.4                                          # corner radius of the return path
    ret = (f"M {tp.f(fish_x + fish_w + 0.4)},{tp.f(y_c)} H {tp.f(x_leg - r)} "
           f"Q {tp.f(x_leg)},{tp.f(y_c)} {tp.f(x_leg)},{tp.f(y_c + r)} "
           f"V {tp.f(loop_y - r)} "
           f"Q {tp.f(x_leg)},{tp.f(loop_y)} {tp.f(x_leg - r)},{tp.f(loop_y)} "
           f"H {tp.f(x_ret + r)} Q {tp.f(x_ret)},{tp.f(loop_y)} {tp.f(x_ret)},{tp.f(loop_y - r)} "
           f"V {tp.f(y_c + r)} Q {tp.f(x_ret)},{tp.f(y_c)} {tp.f(x_ret + r)},{tp.f(y_c)} "
           f"H {tp.f(P['x'] - TIP - 0.1)}")
    # sensor icons + label on the bottom leg, centred, on a white backing
    ico_gap = 2.2
    OBS_LABEL = "Hardware observations"
    lab_w = len(OBS_LABEL) * FS2 * 0.46
    grp_w = 3 * ico + 2 * ico_gap + 2.0 + lab_w
    gx = W / 2 - grp_w / 2
    loop = [
        tp.path(ret, stroke=INK, sw=0.5, marker=True),
        tp.rect(gx - 1.0, loop_y - ico / 2 - 0.6, grp_w + 2.0, ico + 1.2, "#FFFFFF", rx=0.6, stroke="none"),
        tp.icon("camera", gx, loop_y - ico / 2, ico, color=INK),
        tp.icon("rotate-3d", gx + ico + ico_gap, loop_y - ico / 2, ico, color=INK),
        tp.icon("engine", gx + 2 * (ico + ico_gap), loop_y - ico / 2, ico, color=INK),
        tp.text(gx + 3 * ico + 2 * ico_gap + 2.0, loop_y + 0.95, OBS_LABEL, fs=FS2, fill=INK_SOFT),
    ]

    box1 = tp.rect(M, y1, full_w, box1_h, TINT[0], rx=RX, stroke="none")
    box2 = tp.rect(M, y2, box2_w, box23_h, TINT[1], rx=RX, stroke="none")
    box3 = tp.rect(box3_x, y2, box3_w, box23_h, TINT[2], rx=RX, stroke="none")
    ttl1 = title(W / 2, t1, "System Identification", ACC[0])
    ttl2 = title(M + box2_w / 2, t2, "Reinforcement Learning", ACC[1])
    ttl3 = title(box3_x + box3_w / 2, t2, "Zero-Shot Deployment", ACC[2])
    photo = tp.image(ph_x, ph_y, ph_w, ph_h, pool_uri, extra='clip-path="url(#photo_clip)"')

    if layers:
        # one group per reveal, in drawing order. The TG glyph is its own group so the video can
        # hide it and play a live oscillator in the same rectangle.
        g = [
            tp.group("L box1", box1, ttl1),
            tp.group("L ident", strip),
            tp.group("L box2", box2, ttl2),
            tp.group("L rl", *[c for c in chain_rl if c is not tg_glyph]),
            tp.group("L tg", tg_glyph),
            tp.group("L box3", box3, ttl3),
            tp.group("L hw", *chain_hw),
            tp.group("L loop", *loop),
            tp.group("L result", photo),
        ]
    else:
        g = [
            tp.group("Boxes", box1, box2, box3, ttl1, ttl2, ttl3),
            tp.group("Identification", strip),
            tp.group("Control chain", *chain),
            tp.group("Loop", *loop),
            tp.group("Result", photo),
        ]
    if geom_out is not None:
        geom_out.update(W=W, H=H, tg=(T["x"], T["y"], T["w"], T["h"]),
                        photo=(ph_x, ph_y, ph_w, ph_h),
                        fish=(fish_x, y_c - fish_h / 2, fish_w, fish_h),
                        box1=(M, y1, full_w, box1_h), box2=(M, y2, box2_w, box23_h),
                        box3=(box3_x, y2, box3_w, box23_h))

    def marker(mid, color):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4" '
                'markerHeight="4" orient="auto-start-reverse">'
                f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{color}"/></marker>')
    clip = (f'<clipPath id="photo_clip"><rect x="{tp.f(ph_x)}" y="{tp.f(ph_y)}" width="{tp.f(ph_w)}" '
            f'height="{tp.f(ph_h)}" rx="{tp.f(RX)}"/></clipPath>')
    defs = ("<defs>" + "".join(marker(f"arrow{i}", c) for i, c in enumerate(ACC + [INK]))
            + marker("arrow", INK) + clip + "</defs>")
    svg = blacken_text(SVG_TEMPLATE.format(W=tp.f(W), H=tp.f(H), PW=tp.f(W * K), PH=tp.f(H * K), FONT=tp.FONT, defs=defs, body="\n  ".join(g)))
    return svg, H


def boxed_inputs(photo_aspect: float = 1.6, rich_panel: str = "f", render_filter=None) -> dict:
    """Every raster build_svg_boxed draws, assembled the way the paper's figure wants them.

    Split out of main() so tools/make_fig1_animation.py builds the same picture rather than a
    lookalike. render_filter, if given, is applied to the CAD render before it is encoded; the
    video uses it to lift the render's blacks so the tail does not vanish on a dark background.
    """
    hw = load_render(STATIC / "real_sofi.png")
    twin = load_render(STATIC / "40_hull_hinges_fluid.png")
    if render_filter is not None:
        hw, twin = render_filter(hw), render_filter(twin)
    sim_path = sorted(STATIC.glob("sim_collage_panel2_t*.png"))
    if not sim_path:
        sys.exit("need draft/sim_collage_panel2_t*.png (render_sim_collage.py --save-panels)")
    sim = Image.open(sim_path[-1]).convert("RGB").crop((600, 120, 1560, 720))
    sim = crop_aspect(sim, photo_aspect)
    # full-resolution strobe from the paper repo when available, else the montage crop
    hires = HIRES_STROBES.get(rich_panel)
    src = (Image.open(hires).convert("RGB") if hires and Path(hires).is_file()
           else montage_panel(STATIC / "deploy_strobe_montage.png", rich_panel))
    if rich_panel == "f" and hires and Path(hires).is_file() and src.size == (1920, 1080):
        pool8 = src.crop(FIG7_CROP_F)         # the collage's window, see FIG7_CROP_F
    else:
        pool8 = crop_aspect(src, DEPLOY_ASPECT)
    return dict(hw_uri=tp.png_data_uri(hw), hw_aspect=hw.width / hw.height,
                twin_uri=tp.png_data_uri(twin), twin_aspect=twin.width / twin.height,
                pool_uri=tp.png_data_uri(soften(pool8)), sim_uri=tp.png_data_uri(sim),
                photo_aspect=photo_aspect, pool_aspect=pool8.width / pool8.height,
                hw_ratio=0.96, twin_ratio=0.85)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--panel", default="d", choices=list("abcdef"),
                   help="which deployment strobe of the montage to show (default d: out-of-distribution U-turn)")
    p.add_argument("--photo-aspect", type=float, default=1.6)
    p.add_argument("--variant", choices=["boxed", "compact", "rich", "text", "minimal", "all"], default="boxed",
                   help="boxed = the paper's figure, three tinted step boxes with the photo inside the "
                        "observation loop (default, bare output name); compact = the same content "
                        "without the boxes; rich = adds the controller row and the figure-eight "
                        "result; text = step texts under the images; minimal = badges + icons only; "
                        "all = every variant. Non-default variants get their name as a suffix")
    p.add_argument("--plain-cards", action="store_true",
                   help="boxed variant: identification cards without their white backing")
    p.add_argument("--fish-w", type=float, default=17.5, help="boxed variant: CAD render width in mm")
    p.add_argument("--sim2real", default=str(STATIC / "sim2real_strobe.png"),
                   help="rich variant, bottom-left: twin-vs-hardware strobe from fig_sim2real_strobe.py")
    p.add_argument("--rich-panel", default="f", choices=list("abcdef"),
                   help="deployment strobe for the rich variant (default f: figure-eight pursuit)")
    p.add_argument("--no-words", action="store_true",
                   help="minimal variant without the three verbs (identify / learn / deploy)")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)

    hw = load_render(STATIC / "real_sofi.png")
    twin = load_render(STATIC / "40_hull_hinges_fluid.png")
    pool = crop_aspect(montage_panel(STATIC / "deploy_strobe_montage.png", args.panel),
                       args.photo_aspect, x_center_frac=0.55)
    sim_path = sorted(STATIC.glob("sim_collage_panel2_t*.png"))
    if not sim_path:
        sys.exit("need draft/sim_collage_panel2_t*.png (render_sim_collage.py --save-panels)")
    sim = Image.open(sim_path[-1]).convert("RGB")
    # zoom on the fish: 960 x 600 px window of the 1600 x 800 still (the path start is cut off)
    sim = sim.crop((600, 120, 1560, 720))
    sim = crop_aspect(sim, args.photo_aspect)

    # the twin render is drawn a little narrower than the CAD render so the two bodies read at
    # a comparable size (the CAD render includes the AprilTag mast and the full fin sweep)
    imgs = dict(hw_uri=tp.png_data_uri(hw), hw_aspect=hw.width / hw.height,
                twin_uri=tp.png_data_uri(twin), twin_aspect=twin.width / twin.height,
                pool_uri=tp.png_data_uri(soften(pool)), sim_uri=tp.png_data_uri(sim),
                photo_aspect=args.photo_aspect, hw_ratio=1.0, twin_ratio=0.88)
    variants = ["boxed", "compact", "rich", "text", "minimal"] if args.variant == "all" else [args.variant]
    for v in variants:
        # boxed is the paper's figure and takes the bare stem; the others carry their name
        out = Path(args.out) if v == "boxed" else Path(args.out).with_name(Path(args.out).stem + f"_{v}.svg")
        if v == "text":
            svg, H = build_svg(**imgs)
        elif v == "minimal":
            svg, H = build_svg_minimal(**imgs, words=not args.no_words)
        elif v == "boxed":
            svg, H = build_svg_boxed(**boxed_inputs(args.photo_aspect, args.rich_panel),
                                     white_cards=not args.plain_cards, fish_w=args.fish_w)
        else:
            # full-resolution strobe from the paper repo when available, else the montage crop
            hires = HIRES_STROBES.get(args.rich_panel)
            src = (Image.open(hires).convert("RGB") if hires and Path(hires).is_file()
                   else montage_panel(STATIC / "deploy_strobe_montage.png", args.rich_panel))
            if args.rich_panel == "f" and hires and Path(hires).is_file() and src.size == (1920, 1080):
                pool8 = src.crop(FIG7_CROP_F)         # the collage's window, see FIG7_CROP_F
            else:
                pool8 = crop_aspect(src, DEPLOY_ASPECT)
            pool_aspect = pool8.width / pool8.height
            s2r_im = (Image.open(args.sim2real).convert("RGB") if Path(args.sim2real).is_file()
                      else sim)                     # not shipped in this repo: fall back to the training still
            s2r = crop_aspect(s2r_im, 36.0 / (36.0 / args.photo_aspect - 3.6), y_center_frac=0.5)
            svg, H = build_svg_rich(**{**imgs, "pool_uri": tp.png_data_uri(soften(pool8)),
                                       "compact": v == "compact",
                                       "pool_aspect": pool_aspect,
                                       "sim_uri": tp.png_data_uri(s2r),
                                       "s2r_labels": (sim2real_labels(Path(args.sim2real), s2r_im.size, s2r.size)
                                                      if Path(args.sim2real).with_suffix(".json").is_file() else ()),
                                       "hw_ratio": 0.96, "twin_ratio": 0.85})
        out.parent.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        out.write_text(svg, encoding="utf-8")
        print(f"[{v}] Saved {out}  ({W:.1f} x {H:.1f} mm, {out.stat().st_size / 1024:.0f} kB)")
        if not args.no_pdf:
            cf.export_pdf(out, OUT / (out.stem + ".pdf"), W * K, H * K)
        if not args.no_png:
            cf.export_png(out, OUT / (out.stem + ".png"), W * K, H * K, px_per_mm=10.0 / K)


if __name__ == "__main__":
    main()
