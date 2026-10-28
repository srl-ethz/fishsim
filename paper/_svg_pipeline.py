"""Generate the two-column "training / SAC pipeline with TG" figure as an editable SVG.

Layout (left to right), following the hand sketch in icra_report/Figure plans.jpeg:

  [MuJoCo simulation environment]  <-theta_m-  [Trajectory generator]  <-a_t-  [SAC: actor]
     ellipsoid fish (existing fig)                (blocks + u + waveform)            |  a
     1 overhead strobe still (ghosts)    --s_t (24-D)------------------------------> |
     domain randomization note          --d-> [Reward r_t] --r_t------------------> [SAC: critic]

Every box, arrow, label and equation is a plain SVG element (real text, not outlines) so
the file can be edited in Inkscape. Raster content (the fish render and the MuJoCo strobe
still) is embedded as base64 PNG so the SVG is self-contained. Equations are written in
a tiny markup: `_{..}` subscript, `^{..}` superscript, `\\r{..}` upright, `\\b{..}` bold;
everything else is italic (maths convention).

Inputs (all under icra_report/Images/):
  40_hull_hinges_fluid.png                      the ellipsoid/hinge fish render
  draft/sim_collage_strobe.png (+ .json)        from: python icra_report/Images/scripts/render_sim_collage.py
                                                      --times 0 2 2.9 4 --target-deg 20 --no-start-ring --strobe
                                                (the .json sidecar gives the ghost positions for the time labels)
Output:
  icra_report/Images/draft/training_pipeline.svg   (+ .pdf via Inkscape or headless Chrome, + .png preview)

Notation follows the Trajectory Generator section of the report: TG parameters u = (ω, θc, a1, b1, a2, b2),
policy action a_t sets u* (the commanded parameters), low-pass filtered toward u.

Usage (any Python with numpy + Pillow):
  python icra_report/Images/scripts/make_training_pipeline_fig.py [--out PATH]
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

# Helper module for fig1_overview.py / fig2_architecture.py (SVG primitives, equation markup,
# icons, waveform). Its own CLI builds the older stand-alone training-pipeline figure.
from _paths import STATIC as IMG_DIR, OUT as _OUT
DEFAULT_OUT = _OUT / "training_pipeline.svg"

# ── page (mm; IEEE conf \textwidth is 177.8 mm) ───────────────────────────────
W, H = 177.8, 62.0
PT = 25.4 / 72                    # 1 pt in mm
# The diagrams are laid out in millimetres at the final column / text width and included at
# that width, so a font size here IS the printed size. (An earlier version drew them at 7 pt on
# a page 9/7 larger and let LaTeX scale it back, which of course printed 7 pt: a page scaled by
# 1/k shrinks its type by 1/k too. PRINT_SCALE is kept at 1 so the page equals the design.)
PRINT_SCALE = 1.0                 # page / design
FS_MAIN = 9 * PT                  # titles, sub-block headers, signal labels (paper_style.LABEL_PT)
FS_SMALL = 8 * PT                 # everything else: equations, notes, legends, strip labels (TICK_PT, the floor)
SUBSCRIPT = 0.78                  # sub/superscripts relative to their base: 9 pt -> 7 pt, as LaTeX sets them in the plots
ICON_DIR = Path(__file__).resolve().parent / "icons"   # Tabler Icons (MIT), see icons/LICENSE.md

FONT = "'Times New Roman', Times, 'Nimbus Roman', 'Liberation Serif', serif"     # one family, as the plots (paper_style: Times via LaTeX)
INK = "#2B3440"          # text + strokes
INK_SOFT = "#5B6573"     # secondary text
EDGE = 0.25              # box stroke width (mm)
ARROW_W = 0.3

# box fills (very light tints so the figure prints cleanly)
FILL_SIM = "#EEF3FA"     # from ETH-l-blue (205,214,230) used by the deployment diagram
FILL_TG = "#FFF4E6"
FILL_REW = "#EEF7EC"
FILL_SAC = "#F3EFF9"
FILL_SUB = "#FFFFFF"

COL_SIGNAL = "#2B3A55"   # theta_m waveform
COL_FILT = "#D98B4A"     # filtered TG parameter
COL_DES = "#8A94A3"      # commanded (desired) parameter
COL_IN, COL_HID, COL_OUT = "#6B8FBF", "#C9CFD8", "#D98B4A"


# ── equation mini-markup → tspans (unit-tested) ───────────────────────────────

def parse_eq(s: str) -> list[tuple[str, str]]:
    """Split markup into (kind, text) runs; kind ∈ {it, up, bold, sub, sup}."""
    out: list[tuple[str, str]] = []
    buf = ""
    i = 0

    def flush():
        nonlocal buf
        if buf:
            out.append(("it", buf))
            buf = ""

    while i < len(s):
        c = s[i]
        if c in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            flush()
            j = s.index("}", i + 2)
            out.append(("sub" if c == "_" else "sup", s[i + 2:j]))
            i = j + 1
        elif c == "\\" and i + 2 < len(s) and s[i + 2] == "{":
            flush()
            kind = {"r": "up", "b": "bold"}[s[i + 1]]
            j = s.index("}", i + 3)
            out.append((kind, s[i + 3:j]))
            i = j + 1
        else:
            buf += c
            i += 1
    flush()
    return out


def eq_tspans(markup: str, fs: float) -> str:
    """Render markup as <tspan>s. Sub/superscripts use dy shifts (portable to Inkscape)."""
    parts = []
    pending_dy = 0.0
    for kind, txt in parse_eq(markup):
        attrs = []
        dy = pending_dy
        pending_dy = 0.0
        if kind == "sub":
            attrs.append(f'font-size="{fs * SUBSCRIPT:.3f}"')
            dy += 0.30 * fs
            pending_dy = -0.30 * fs
        elif kind == "sup":
            attrs.append(f'font-size="{fs * SUBSCRIPT:.3f}"')
            dy += -0.45 * fs
            pending_dy = 0.45 * fs
        elif kind == "up":
            attrs.append('font-style="normal"')
        elif kind == "bold":
            attrs.append('font-style="normal" font-weight="bold"')
        else:
            attrs.append('font-style="italic"')
        if abs(dy) > 1e-9:
            attrs.append(f'dy="{dy:.3f}"')
        parts.append(f"<tspan {' '.join(attrs)}>{escape(txt)}</tspan>")
    return "".join(parts)


# ── raster helpers (unit-tested where pure) ──────────────────────────────────

def white_to_alpha(rgb: np.ndarray, gain: float = 6.0) -> np.ndarray:
    """Alpha channel that fades near-white pixels out (so a render on white can sit on a
    tinted background). Grey/coloured pixels stay opaque."""
    mean = rgb[..., :3].astype(float).mean(-1)
    return np.clip((255.0 - mean) * gain, 0, 255).astype(np.uint8)


def crop_box(shape_hw: tuple[int, int], x0: int, x1: int, aspect: float,
             y_center: float) -> tuple[int, int, int, int]:
    """Crop (left, top, right, bottom) of width x1-x0 and the given w/h aspect, vertically
    centred on y_center and clamped to the image."""
    h_img = shape_hw[0]
    w = x1 - x0
    h = int(round(w / aspect))
    top = int(round(y_center - h / 2))
    top = max(0, min(top, h_img - h))
    return x0, top, x1, top + h


def png_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_fish(path: Path, width_px: int = 1000) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).copy()
    a[..., 3] = white_to_alpha(a)
    im = Image.fromarray(a)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS)


def load_panel(path: Path, aspect: float, width_px: int = 1400
               ) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop the overhead still to `aspect` (a band centred on the camera's centre row,
    where the start->target line lies) and downsample. Returns (image, crop box)."""
    im = Image.open(path).convert("RGB")
    box = crop_box((im.height, im.width), 40, im.width - 40, aspect, y_center=im.height // 2)
    im = im.crop(box)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS), box


def uv_to_panel(u: float, v: float, box: tuple[int, int, int, int],
                img_w: int, img_h: int) -> tuple[float, float]:
    """Map normalised full-frame coordinates (u, v in [0, 1]) to normalised coordinates of
    the crop `box` = (left, top, right, bottom)."""
    l, t, r, b = box
    return (u * img_w - l) / (r - l), (v * img_h - t) / (b - t)


# ── SVG primitives ────────────────────────────────────────────────────────────

def f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def text(x, y, s, fs=FS_SMALL, anchor="start", weight="normal", style="normal",
         fill=INK, extra=""):
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{f(fs)}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" {extra}>'
            f"{escape(s)}</text>")


def eq(x, y, markup, fs=FS_SMALL, anchor="start", fill=INK):
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{f(fs)}" text-anchor="{anchor}" '
            f'fill="{fill}">{eq_tspans(markup, fs)}</text>')


def rect(x, y, w, h, fill, rx=1.5, stroke=INK, sw=EDGE, extra=""):
    """Rounded rectangle. Pass stroke="none" for a fill-only band."""
    return (f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{f(rx)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" {extra}/>')


def line(x1, y1, x2, y2, stroke=INK, sw=ARROW_W, dash="", marker=True):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if marker else ""
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" '
            f'stroke-width="{f(sw)}" stroke-linecap="butt"{d}{m}/>')


def path(d, stroke=INK, sw=ARROW_W, dash="", marker=False, fill="none"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if marker else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}{m}/>')


def image(x, y, w, h, uri, extra=""):
    return (f'<image x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" '
            f'preserveAspectRatio="none" xlink:href="{uri}" {extra}/>')


def icon_paths(svg_text: str) -> list[str]:
    """Path 'd' strings of a Tabler outline icon, minus its invisible 24x24 bounding box."""
    import re
    ds = re.findall(r'<path[^>]*\bd="([^"]+)"', svg_text)
    return [d for d in ds if d.replace(" ", "") not in ("M00h24v24H0z", "M0,0h24v24H0z")]


def icon(name: str, x: float, y: float, size: float, color: str = INK, sw: float = 1.8) -> str:
    """Place a 24x24 Tabler outline icon with its top-left corner at (x, y), `size` mm wide."""
    ds = icon_paths((ICON_DIR / f"{name}.svg").read_text(encoding="utf-8"))
    s = size / 24.0
    body = "".join(f'<path d="{d}"/>' for d in ds)
    return (f'<g transform="translate({f(x)},{f(y)}) scale({s:.4f})" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'inkscape:label="icon:{name}">{body}</g>')


def group(label: str, *children: str) -> str:
    gid = label.lower().replace(" ", "_").replace("/", "_")
    body = "\n    ".join(c for c in children if c)
    return f'<g id="{gid}" inkscape:label="{label}">\n    {body}\n  </g>'


def fish_top(cx, cy, length, heading_deg=0.0, color=INK):
    """Tiny top-view fish (ellipse body + triangular tail) centred at (cx, cy), `length` mm long,
    heading `heading_deg` clockwise from up (90 = heading right)."""
    b, w = length * 0.62, length * 0.30
    d = (f"M {cx:.3f},{cy - b / 2:.3f} "
         f"C {cx + w / 2:.3f},{cy - b / 2:.3f} {cx + w / 2:.3f},{cy + b / 2:.3f} {cx:.3f},{cy + b / 2:.3f} "
         f"C {cx - w / 2:.3f},{cy + b / 2:.3f} {cx - w / 2:.3f},{cy - b / 2:.3f} {cx:.3f},{cy - b / 2:.3f} Z "
         f"M {cx:.3f},{cy + b / 2 - 0.1:.3f} L {cx - w * 0.75:.3f},{cy + length / 2:.3f} "
         f"L {cx + w * 0.75:.3f},{cy + length / 2:.3f} Z")
    return (f'<g transform="rotate({f(heading_deg)} {f(cx)} {f(cy)})">'
            f'<path d="{d}" fill="{color}" stroke="{color}" stroke-width="0.15" stroke-linejoin="round"/></g>')


def network(x, y, w, h, layers, in_label="s", out_label="a"):  # noqa: E501
    """Small fully-connected MLP glyph with `layers` nodes per column."""
    xs = np.linspace(x, x + w, len(layers))
    spacing = min(h / (max(layers)), 2.0)
    cols = []
    for n, cx in zip(layers, xs):
        ys = y + h / 2 + (np.arange(n) - (n - 1) / 2) * spacing
        cols.append([(cx, cy) for cy in ys])
    out = []
    for a, b in zip(cols[:-1], cols[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                out.append(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" '
                           f'stroke="#9AA5B5" stroke-width="0.08" stroke-opacity="0.7"/>')
    for i, col in enumerate(cols):
        fill = COL_IN if i == 0 else COL_OUT if i == len(cols) - 1 else COL_HID
        for (cx, cy) in col:
            out.append(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="0.55" fill="{fill}" '
                       f'stroke="{INK}" stroke-width="0.12"/>')
    out.append(eq(x - 1.2, y + h / 2 + 0.7, in_label, fs=FS_SMALL, anchor="end"))
    out.append(eq(x + w + 1.2, y + h / 2 + 0.7, out_label, fs=FS_SMALL, anchor="start"))
    return "\n    ".join(out)


# ── TG waveform ──────────────────────────────────────────────────────────────

def cpg_waveform(px0, py0, pw, ph):
    """theta_m(t) for a step in the commanded amplitude at t=1 s, filtered with tau=0.1 s,
    plus the filtered and commanded envelopes. Returns SVG elements."""
    t = np.linspace(0, 2.6, 600)
    f_hz, tau, theta_c = 1.15, 0.1, 0.12
    k_des = np.where(t < 1.0, 0.45, 0.9)
    k = np.where(t < 1.0, 0.45, 0.9 - 0.45 * np.exp(-(t - 1.0) / tau))
    phi = 2 * np.pi * f_hz * t
    theta = theta_c + k * (np.cos(phi) + 0.25 * np.sin(2 * phi))

    lo, hi = -1.5, 1.6
    def X(tt): return px0 + (tt / t[-1]) * pw
    def Y(v): return py0 + ph - (v - lo) / (hi - lo) * ph
    def poly(vals):
        return "M " + " L ".join(f"{X(tt):.2f},{Y(v):.2f}" for tt, v in zip(t, vals))

    out = [
        # baseline theta_c and time axis
        line(px0, Y(theta_c), px0 + pw, Y(theta_c), stroke=COL_DES, sw=0.15, dash="0.6 0.5",
             marker=False),
        line(px0, py0 + ph, px0 + pw + 1.5, py0 + ph, stroke=INK, sw=0.18),
        eq(px0 + pw + 2.2, py0 + ph + 0.7, "t", fs=FS_SMALL),
        eq(px0 - 0.6, Y(theta_c) + 0.7, "θ_{c}", fs=FS_SMALL, anchor="end", fill=INK_SOFT),
        # envelopes
        path(poly(theta_c + k_des), stroke=COL_DES, sw=0.18, dash="0.8 0.5"),
        path(poly(theta_c - k_des), stroke=COL_DES, sw=0.18, dash="0.8 0.5"),
        path(poly(theta_c + k), stroke=COL_FILT, sw=0.22),
        path(poly(theta_c - k), stroke=COL_FILT, sw=0.22),
        # the signal
        path(poly(theta), stroke=COL_SIGNAL, sw=0.32),
        # step marker
        line(X(1.0), py0 + ph, X(1.0), py0 + 0.4, stroke=INK_SOFT, sw=0.12, dash="0.4 0.4",
             marker=False),
        # labels
        eq(px0 + pw + 1.2, Y(theta[-1]) + 2.4, "θ_{m}", fs=FS_SMALL, fill=COL_SIGNAL),
        # legend: filtered (orange) and commanded (grey dashed) parameter envelopes
        line(X(0.02), Y(hi) + 1.1, X(0.02) + 3.0, Y(hi) + 1.1, stroke=COL_FILT, sw=0.28,
             marker=False),
        eq(X(0.02) + 3.8, Y(hi) + 1.8, "\\b{u}", fs=FS_SMALL, fill=INK_SOFT),
        line(X(0.02) + 7.4, Y(hi) + 1.1, X(0.02) + 10.4, Y(hi) + 1.1, stroke=COL_DES, sw=0.22,
             dash="0.8 0.5", marker=False),
        eq(X(0.02) + 11.2, Y(hi) + 1.8, "\\b{u}*", fs=FS_SMALL, fill=INK_SOFT),
    ]
    return "\n    ".join(out)


def harmonic_parts(x0, y, w, h, gap):
    """Four mini plots over two strokes of phase:  theta_c  +  (a1, b1)  +  (a2, b2)  =  theta_m.
    Each plot has a zero baseline; labels (name, symbols) sit underneath."""
    phi = np.linspace(0, 4 * np.pi, 400)
    th_c = 0.55
    h1 = 1.0 * np.cos(phi) + 0.4 * np.sin(phi)
    h2 = 0.32 * np.cos(2 * phi) + 0.2 * np.sin(2 * phi)
    series = [(th_c + h1 + h2, COL_SIGNAL, "motor angle", "θ_{m}(φ)"),
              (np.full_like(phi, th_c), INK_SOFT, "offset", "θ_{c}"),
              (h1, COL_FILT, "1st harmonic", "a_{1}, b_{1}"),
              (h2, COL_IN, "2nd harmonic", "a_{2}, b_{2}")]
    lo, hi = -1.8, 1.8
    out = []
    for i, (vals, col, name, sym) in enumerate(series):
        px = x0 + i * (w + gap)
        def X(v, px=px): return px + v / phi[-1] * w
        def Y(v): return y + h / 2 - v / (hi - lo) * h
        out.append(line(px, Y(0), px + w, Y(0), stroke=INK_SOFT, sw=0.15, marker=False))
        out.append(path("M " + " L ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(phi, vals)),
                        stroke=col, sw=0.32))
        out.append(text(px + w / 2, y + h + 2.6, name, fs=FS_SMALL, anchor="middle", fill=INK_SOFT))
        out.append(eq(px + w / 2, y + h + 5.1, sym, fs=FS_SMALL, anchor="middle"))
        if i < 3:
            out.append(text(px + w + gap / 2, y + h / 2 + 0.9, "=" if i == 0 else "+", fs=FS_MAIN,
                            anchor="middle"))
    return "\n    ".join(out)


# ── figure ────────────────────────────────────────────────────────────────────

def build_svg(fish_uri: str, still_uri: str, still_aspect: float) -> tuple[str, float]:
    """Returns (svg, page height in mm)."""
    # ---- column geometry -----------------------------------------------------
    Hm = H
    SIM = dict(x=1.0, y=1.0, w=60.0, h=Hm - 2.0)
    TG = dict(x=70.0, y=1.0, w=60.0, h=30.0)     # top aligned with the SIM and SAC boxes
    REW = dict(x=70.0, y=Hm - 12.5, w=60.0, h=11.5)
    SAC = dict(x=142.0, y=1.0, w=W - 143.0, h=Hm - 2.0)
    sub_w = SAC["w"] - 4 - 4.5                    # leave a lane on the right for the action bus
    # sub-boxes spaced evenly under the title; a free lane above the actor carries the action
    ACT = dict(x=SAC["x"] + 2, y=11.0, w=sub_w, h=14.5)
    BUF = dict(x=SAC["x"] + 2, y=Hm - 16.0, w=sub_w, h=12.0)
    CRI = dict(x=SAC["x"] + 2, y=(ACT["y"] + ACT["h"] + BUF["y"]) / 2 - 7.25, w=sub_w, h=14.5)
    y_actor = ACT["y"] + ACT["h"] / 2             # actor centreline: state in, action out
    y_buf = BUF["y"] + BUF["h"] / 2               # buffer centreline: action in from the right

    sim_r = SIM["x"] + SIM["w"]
    cpg_r = TG["x"] + TG["w"]
    rew_r = REW["x"] + REW["w"]
    # one signal height for the whole top row: SAC actor -> TG chain -> simulator (theta_m)
    ch_y, ch_h, ch_w = TG["y"] + 5.2, 6.2, 19.5
    y_act = ch_y + ch_h / 2

    # ---- simulation block ----------------------------------------------------
    fish_w = 40.0
    fish_h = fish_w * 760 / 1347
    fish_x = SIM["x"] + (SIM["w"] - fish_w) / 2
    fish_y = 4.6
    fish_b = fish_y + fish_h
    # Under the render: the 5 x 3 matrix of quasi-steady fluid coefficients (blunt, slender and
    # angular drag, Kutta and Magnus lift; one column per ellipsoid zone), with an underbrace
    # on each zone of the render and an elbow leader down to its matrix column. Same layout and
    # notation (c_{b,H} ...) as make_architecture_fig.py / Table "fluid" in the paper.
    zones = [("Head", 0.08, 0.53), ("Tail", 0.54, 0.76), ("Fin", 0.77, 0.94)]
    y_br = fish_b + 0.7                           # underbrace baseline
    y_hdr = y_br + 2.7                            # zone names sit right under the brackets
    ROW = 2.05
    rows = ["Blunt drag", "Slender drag", "Angular drag", "Kutta lift", "Magnus lift"]
    subs = ["b", "s", "a", "K", "M"]
    col_x = [fish_x + (f0 + f1) / 2 * fish_w for _, f0, f1 in zones]   # columns under their zones
    y_rows = [y_hdr + ROW * (i + 1) for i in range(len(rows))]
    coeffs = []
    for (name, f0, f1), cx in zip(zones, col_x):
        x0, x1 = fish_x + f0 * fish_w, fish_x + f1 * fish_w
        xm = (x0 + x1) / 2
        # underbrace under the zone; its name and coefficient column sit right below it
        coeffs.append(path(f"M {f(x0)},{f(y_br - 1.0)} V {f(y_br)} H {f(x1)} V {f(y_br - 1.0)}",
                           stroke=INK_SOFT, sw=0.22))
        coeffs.append(text(cx, y_hdr, name, fs=FS_SMALL, anchor="middle", weight="bold"))
        for yr, sb in zip(y_rows, subs):
            coeffs.append(eq(cx, yr, f"c_{{{sb},{name[0]}}}", fs=FS_SMALL, anchor="middle"))
    for yr, name in zip(y_rows, rows):
        coeffs.append(text(col_x[0] - 4.6, yr, name, fs=FS_SMALL, anchor="end", fill=INK_SOFT))
    # matrix brackets, and the label with some air under the matrix
    bx0, bx1 = col_x[0] - 3.2, col_x[-1] + 3.2
    by0, by1 = y_hdr + 0.9, y_rows[-1] + 0.7
    coeffs.append(path(f"M {f(bx0 + 0.9)},{f(by0)} H {f(bx0)} V {f(by1)} H {f(bx0 + 0.9)}", sw=0.25))
    coeffs.append(path(f"M {f(bx1 - 0.9)},{f(by0)} H {f(bx1)} V {f(by1)} H {f(bx1 - 0.9)}", sw=0.25))
    coeffs.append(text((bx0 + bx1) / 2, by1 + 3.2, "Fluid coefficients (15)", fs=FS_SMALL,
                       anchor="middle", weight="bold"))
    by1 += 3.6
    # one large overhead still (strobe: final pose + ghosts of earlier poses)
    pan_w = SIM["w"] - 4.0
    pan_h = pan_w / still_aspect
    px, pan_y = SIM["x"] + 2.0, by1 + 1.8
    panels = [
        image(px, pan_y, pan_w, pan_h, still_uri),
        rect(px, pan_y, pan_w, pan_h, "none", rx=0.3, stroke=INK_SOFT, sw=0.15),
    ]
    # legend inside the still (bottom-right corner): red trail = swum path, green dot = target
    lg_x, lg_y = px + pan_w - 24.0, pan_y + pan_h - 1.8
    sim = group(
        "Simulation",
        rect(**SIM, fill=FILL_SIM),
        text(SIM["x"] + SIM["w"] / 2, 4.0, "MuJoCo Simulation Environment", fs=FS_MAIN,
             anchor="middle", weight="bold"),
        image(fish_x, fish_y, fish_w, fish_h, fish_uri),
        *coeffs,
        *panels,
        line(lg_x, lg_y, lg_x + 3.5, lg_y, stroke="#D93434", sw=0.4, marker=False),
        text(lg_x + 4.3, lg_y + 0.7, "path", fs=FS_SMALL, fill=INK),
        f'<circle cx="{f(lg_x + 13.5)}" cy="{f(lg_y)}" r="0.7" fill="#19BF19" '
        f'stroke="{INK}" stroke-width="0.12"/>',
        text(lg_x + 14.8, lg_y + 0.7, "target", fs=FS_SMALL, fill=INK),
    )

    # ---- TG block -----------------------------------------------------------
    # chain on top (right to left, at the height of the incoming action and outgoing theta_m):
    #   u* -> [low-pass filter] -> u -> [harmonic oscillator] -> theta_m
    # the parameter vector below it, and how the oscillator composes theta_m from its parts:
    #   offset theta_c  +  1st harmonic (a1, b1)  +  2nd harmonic (a2, b2)  =  theta_m(phi)
    OSC = dict(x=TG["x"] + 5.5, y=ch_y, w=ch_w, h=ch_h)
    LPF = dict(x=cpg_r - 5.5 - ch_w, y=ch_y, w=ch_w, h=ch_h)
    hp_w, hp_h, hp_gap = 10.5, 6.6, 3.0
    hp_x0 = TG["x"] + (TG["w"] - (4 * hp_w + 3 * hp_gap)) / 2
    hp_y = TG["y"] + 16.6
    parts = harmonic_parts(hp_x0, hp_y, hp_w, hp_h, hp_gap)
    cpg = group(
        "TG",
        rect(**TG, fill=FILL_TG),
        text(TG["x"] + TG["w"] / 2, TG["y"] + 3.2, "Trajectory Generator", fs=FS_MAIN,
             anchor="middle", weight="bold"),
        # low-pass filter block
        rect(**LPF, fill=FILL_SUB, rx=0.8),
        text(LPF["x"] + ch_w / 2, y_act - 0.6, "Low-pass filter", fs=FS_SMALL, anchor="middle",
             weight="bold"),
        eq(LPF["x"] + ch_w / 2, y_act + 2.4, "\\b{u}* → \\b{u}", fs=FS_SMALL, anchor="middle",
           fill=INK_SOFT),
        # oscillator block
        rect(**OSC, fill=FILL_SUB, rx=0.8),
        text(OSC["x"] + ch_w / 2, y_act - 0.6, "Harmonic oscillator", fs=FS_SMALL, anchor="middle",
             weight="bold"),
        eq(OSC["x"] + ch_w / 2, y_act + 2.4, "dφ/dt = ω", fs=FS_SMALL, anchor="middle", fill=INK_SOFT),
        # filter -> oscillator (the action arrow into the filter and theta_m out of the
        # oscillator are drawn in the Arrows group so they run straight across the box edges)
        line(LPF["x"], y_act, OSC["x"] + ch_w + 0.4, y_act),
        # the TG parameter vector the policy commands
        eq(TG["x"] + TG["w"] / 2, TG["y"] + 14.6,
           "\\b{u} = (ω, θ_{c}, a_{1}, b_{1}, a_{2}, b_{2})", fs=FS_SMALL, anchor="middle"),
        parts,
    )

    # ---- reward block --------------------------------------------------------
    # left: pictogram -- fish approaching the target, distance d, 5 cm arrival radius
    g_x0, g_w = REW["x"] + 1.5, 14.0
    g_cy = REW["y"] + REW["h"] / 2
    tgt = (g_x0 + g_w - 3.0, g_cy)
    r_arr = 2.3                                   # 5 cm arrival radius (not to scale)
    fish_c = (g_x0 + 2.8, g_cy)
    txt_x0 = g_x0 + g_w + 1.5
    txt_cx = (txt_x0 + rew_r - 1.0) / 2                             # centre of the text column
    rew = group(
        "Reward",
        rect(**REW, fill=FILL_REW),
        # pictogram
        f'<circle cx="{f(tgt[0])}" cy="{f(tgt[1])}" r="{f(r_arr)}" fill="#009E73" fill-opacity="0.15" '
        f'stroke="#009E73" stroke-width="0.2" stroke-dasharray="0.5 0.4"/>',
        f'<circle cx="{f(tgt[0])}" cy="{f(tgt[1])}" r="0.7" fill="#009E73" stroke="{INK}" '
        f'stroke-width="0.12"/>',
        text(tgt[0], tgt[1] + r_arr + 2.2, "5 cm", fs=FS_SMALL, anchor="middle", fill="#009E73"),
        fish_top(fish_c[0], fish_c[1], 3.4, heading_deg=90),
        line(fish_c[0] + 2.1, g_cy, tgt[0] - r_arr - 0.5, g_cy, sw=0.22, dash="0.5 0.4",
             marker=True),
        eq((fish_c[0] + 2.1 + tgt[0] - r_arr) / 2, g_cy - 1.0, "d", fs=FS_SMALL, anchor="middle"),
        # text column: no full equation, just what the reward rewards
        text(txt_cx, REW["y"] + 4.4, "Reward", fs=FS_MAIN, anchor="middle", weight="bold"),
        eq(txt_cx, REW["y"] + 8.4, "r_{t} ∝ −d\\r{,   bonus when }d < \\r{5 cm}",
           fs=FS_SMALL, anchor="middle"),
    )

    # ---- SAC block -----------------------------------------------------------
    net_w, net_h = 14.0, 6.2
    net_x = ACT["x"] + 4.5
    act_r = ACT["x"] + ACT["w"]
    buf_r = BUF["x"] + BUF["w"]
    sac_r = SAC["x"] + SAC["w"]
    bus_x = sac_r - 2.3                            # action bus: actor -> TG and -> buffer
    net_dy = ACT["h"] / 2 - net_h / 2             # network glyph centred in its box
    sac = group(
        "SAC",
        rect(**SAC, fill=FILL_SAC),
        text(SAC["x"] + SAC["w"] / 2, 4.6, "Soft Actor-Critic (SAC)", fs=FS_MAIN, anchor="middle",
             weight="bold"),
        # actor
        rect(**ACT, fill=FILL_SUB, rx=1.0),
        eq(ACT["x"] + 1.8, ACT["y"] + 3.0, "\\b{Actor  }π(\\b{a} | \\b{s})", fs=FS_MAIN),
        network(net_x, ACT["y"] + net_dy, net_w, net_h, [5, 6, 6, 3], "\\b{s}", "\\b{a}"),
        # critic
        rect(**CRI, fill=FILL_SUB, rx=1.0),
        eq(CRI["x"] + 1.8, CRI["y"] + 3.0, "\\b{Critic  }Q(\\b{s}, \\b{a})", fs=FS_MAIN),
        network(net_x, CRI["y"] + net_dy, net_w, net_h, [6, 6, 6, 1], "\\b{s},\\b{a}", "Q"),
        # replay buffer
        rect(**BUF, fill=FILL_SUB, rx=1.0),
        icon("database", BUF["x"] + 1.4, BUF["y"] + 1.8, 4.8, color=INK_SOFT),
        text(BUF["x"] + 7.6, BUF["y"] + 3.5, "Replay Buffer", fs=FS_MAIN, weight="bold"),
        eq(BUF["x"] + 7.6, BUF["y"] + 6.6, "(\\b{s}_{t}, \\b{a}_{t}, r_{t}, \\b{s}_{t+1})", fs=FS_SMALL),
        text(BUF["x"] + 1.5, BUF["y"] + 10.2, "kept across curriculum", fs=FS_SMALL, fill=INK_SOFT),
        # actor -> action bus (right lane) -> down into the buffer
        path(f"M {f(act_r)},{f(y_actor)} H {f(bus_x)} V {f(y_buf)} H {f(buf_r + 0.4)}",
             marker=True),
        # ... and up, over the actor along the free lane, out of SAC and into the TG's filter
        path(f"M {f(bus_x)},{f(y_actor)} V {f(y_act)} H {f(LPF['x'] + ch_w + 0.4)}", marker=True),
        eq(bus_x - 0.8, CRI["y"] - 1.1, "\\b{a}_{t}", fs=FS_SMALL, anchor="end"),
        # buffer -> critic (minibatch), critic -> actor (policy gradient)
        line(BUF["x"] + BUF["w"] * 0.3, BUF["y"], BUF["x"] + BUF["w"] * 0.3, CRI["y"] + CRI["h"] + 0.4,
             sw=0.28),
        line(CRI["x"] + CRI["w"] * 0.3, CRI["y"], CRI["x"] + CRI["w"] * 0.3, ACT["y"] + ACT["h"] + 0.4,
             sw=0.28, dash="0.7 0.5"),
    )

    # ---- observation strip (icons) under the state arrow ------------------------
    y_state = 36.0
    fan_x = SAC["x"] - 3.2                # state fan-out point
    y_rew = REW["y"] + REW["h"] / 2
    obs = [  # (icon, width share, line 1, line 2) -- names only, no dimensions
        ("engine",    13.4, "Motor angle",  "& velocity"),
        ("rotate-3d", 12.4, "IMU",          "yaw rate"),
        ("wave-sine", 11.6, "TG phase",     "cos φ, sin φ"),
        ("camera",    16.6, "Camera",       "target pose"),
        ("history",   13.6, "Past actions", ""),
    ]
    strip_x0, strip_x1 = sim_r + 1.2, fan_x + 0.6
    scale = (strip_x1 - strip_x0) / sum(w for _, w, _, _ in obs)
    ico, y0 = 3.6, y_state + 1.8
    row_h = ico                                   # height of the icon row
    chips, cx = [], strip_x0
    for name, w, l1, l2 in obs:
        w *= scale
        chips.append(icon(name, cx + w / 2 - ico / 2, y0 + (row_h - ico) / 2, ico, color=INK))
        chips.append(text(cx + w / 2, y0 + row_h + 2.4, l1, fs=FS_SMALL, anchor="middle"))
        chips.append(text(cx + w / 2, y0 + row_h + 4.9, l2, fs=FS_SMALL, anchor="middle", fill=INK_SOFT))
        cx += w
    # soft band (no outline) that gathers the state label, its arrow and the observation
    # strip into one visual unit between the simulator and SAC
    band = dict(x=sim_r + 1.0, y=y_state - 3.9, w=fan_x + 1.6 - (sim_r + 1.0),
                h=(y0 + row_h + 4.9 + 1.1) - (y_state - 3.9))

    # ---- arrows between blocks -------------------------------------------------
    arrows = group(
        "Arrows",
        rect(**band, fill="#F3F4F6", rx=1.5, stroke="none", sw=0),
        # TG oscillator -> sim: motor setpoint theta_m, straight out of the chain
        line(OSC["x"], y_act, sim_r + 0.4, y_act),
        eq((sim_r + TG["x"]) / 2, y_act - 1.3, "θ_{m}", fs=FS_MAIN, anchor="middle"),
        text((sim_r + TG["x"]) / 2, y_act + 2.8, "50 Hz", fs=FS_SMALL, anchor="middle",
             fill=INK_SOFT),
        # SAC actor -> TG filter: action (the line itself is drawn from the action bus in the SAC group)
        eq((cpg_r + SAC["x"]) / 2, y_act - 1.3, "\\b{u}* = \\b{a}_{t}", fs=FS_MAIN, anchor="middle"),
        # sim -> SAC: state (fans out to the actor, and into the replay buffer)
        line(sim_r, y_state, fan_x, y_state, marker=False),
        path(f"M {f(fan_x)},{f(y_state)} V {f(y_actor)} H {f(ACT['x'] - 0.4)}", marker=True),
        path(f"M {f(fan_x)},{f(y_state)} V {f(BUF['y'] + 3.0)} H {f(BUF['x'] - 0.4)}", marker=True),
        eq((sim_r + fan_x) / 2, y_state - 1.2, "\\b{State  s}_{t}", fs=FS_MAIN, anchor="middle"),
        *chips,
        # sim -> reward -> replay buffer, one straight line through the reward box
        line(sim_r, y_rew, REW["x"] - 0.4, y_rew),
        eq((sim_r + REW["x"]) / 2, y_rew - 1.2, "d", fs=FS_MAIN, anchor="middle"),
        line(rew_r, y_rew, BUF["x"] - 0.4, y_rew),
        eq(rew_r + 4.0, y_rew - 1.2, "r_{t}", fs=FS_MAIN, anchor="middle"),
    )

    defs = (
        '<defs>'
        # refX sits well behind the tip so the (butt-ended) line is hidden inside the head:
        # with refX at the tip, the line's square end pokes out either side of the point.
        '<marker id="arrow" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4.5" '
        'markerHeight="4.5" orient="auto-start-reverse">'
        f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{INK}"/></marker>'
        '</defs>'
    )

    H_page = Hm
    main = "\n  ".join([sim, cpg, rew, sac, arrows])
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by icra_report/Images/scripts/make_training_pipeline_fig.py - edit freely in Inkscape. -->
<svg
   width="{f(W)}mm" height="{f(H_page)}mm" viewBox="0 0 {f(W)} {f(H_page)}"
   version="1.1" id="training_pipeline"
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
  {main}
  </g>
</svg>
'''
    return svg, H_page


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--fish", default=str(IMG_DIR / "40_hull_hinges_fluid.png"))
    p.add_argument("--still", default=str(IMG_DIR / "draft" / "sim_collage_strobe.png"),
                   help="overhead strobe still; a .json sidecar of the same name gives the poses")
    p.add_argument("--still-aspect", type=float, default=4.2,
                   help="w/h crop aspect for the still (raw render is 2:1)")
    p.add_argument("--no-pdf", action="store_true", help="skip the PDF export next to the SVG")
    p.add_argument("--png-width", type=int, default=2100,
                   help="pixel width of the PNG preview written next to the SVG (2100 = 300 dpi)")
    args = p.parse_args(argv)

    still = Path(args.still)
    if not still.is_file():
        sys.exit("Need the overhead still first: "
                 "python icra_report/Images/scripts/render_sim_collage.py --times 0 2 2.9 4 "
                 "--target-deg 20 --no-start-ring --strobe")

    fish_uri = png_data_uri(load_fish(Path(args.fish)))
    still_img, _box = load_panel(still, args.still_aspect)
    still_uri = png_data_uri(still_img)

    svg, height = build_svg(fish_uri, still_uri, args.still_aspect)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Saved {out}  ({out.stat().st_size / 1024:.0f} kB, {W:g} x {height:g} mm)")
    if not args.no_pdf:
        export_pdf(out, out.with_suffix(".pdf"), height)
    export_png(out, out.with_suffix(".png"), args.png_width, height)


# ── PDF export ────────────────────────────────────────────────────────────────

_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]
_INKSCAPE_CANDIDATES = [
    r"C:\Program Files\Inkscape\bin\inkscape.exe",
    r"C:\Program Files\Inkscape\bin\inkscape.com",
    "/usr/bin/inkscape",
    "/mnt/c/Program Files/Inkscape/bin/inkscape.exe",
]


def _find(names: list[str], candidates: list[str]) -> str | None:
    import shutil
    for n in names:
        if shutil.which(n):
            return shutil.which(n)
    for c in candidates:
        if Path(c).is_file():
            return c
    return None


# When the browser or Inkscape is a Windows binary launched from WSL (/mnt/c/...), it needs
# \\wsl.localhost\... paths for everything it reads or writes.
def _is_windows_exe(exe) -> bool:
    return bool(exe) and str(exe).startswith("/mnt/")


def _bpath(p, exe) -> str:
    p = Path(p).resolve()
    if _is_windows_exe(exe):
        import subprocess as _sp
        return _sp.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


def _uri(p, exe) -> str:
    return "file:///" + _bpath(p, exe).replace("\\", "/") if _is_windows_exe(exe) else Path(p).resolve().as_uri()


def export_pdf(svg: Path, pdf: Path, height: float = H) -> None:
    """Write `pdf` from `svg`. Prefers Inkscape (true vector export); falls back to a
    headless Chrome/Edge print with the page sized exactly to the figure."""
    import subprocess
    import tempfile

    ink = _find(["inkscape"], _INKSCAPE_CANDIDATES)
    if ink:
        subprocess.run([ink, _bpath(svg, ink), "--export-type=pdf", f"--export-filename={_bpath(pdf, ink)}"],
                       check=True, timeout=300)
        print(f"Saved {pdf}  (Inkscape)")
        return

    chrome = _find(["chrome", "google-chrome", "chromium", "msedge"], _CHROME_CANDIDATES)
    if not chrome:
        print("WARNING: no Inkscape or Chrome found; PDF not written. "
              "Open the SVG in Inkscape and File > Save a Copy as PDF.")
        return
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "print.html"
        wrapper.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            f"@page{{size:{W}mm {height}mm;margin:0}} html,body{{margin:0;padding:0}}"
            f"img{{display:block;width:{W}mm;height:{height}mm}}"
            f"</style></head><body><img src='{_uri(svg, chrome)}'></body></html>",
            encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={_bpath(pdf, chrome)}", _uri(wrapper, chrome)],
                       check=True, timeout=120, capture_output=True)
    if not pdf.is_file():
        print("WARNING: Chrome did not produce a PDF.")
        return
    # sanity-check the page box (mm -> pt: x 72/25.4)
    import re
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
    size = f", page {float(m[1]) * 25.4 / 72:.1f} x {float(m[2]) * 25.4 / 72:.1f} mm" if m else ""
    print(f"Saved {pdf}  (Chrome print{size})")


def export_png(svg: Path, png: Path, width_px: int, height: float = H) -> None:
    """Rasterise the SVG to `png` (`width_px` wide) with Inkscape, else headless Chrome."""
    import subprocess
    import tempfile

    ink = _find(["inkscape"], _INKSCAPE_CANDIDATES)
    if ink:
        subprocess.run([ink, _bpath(svg, ink), "--export-type=png", f"--export-width={width_px}",
                        "--export-background=#ffffff", f"--export-filename={_bpath(png, ink)}"],
                       check=True, timeout=300)
        print(f"Saved {png}  (Inkscape)")
        return
    chrome = _find(["chrome", "google-chrome", "chromium", "msedge"], _CHROME_CANDIDATES)
    if not chrome:
        print("WARNING: no Inkscape or Chrome found; PNG not written.")
        return
    height_px = int(round(width_px * height / W))
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "shot.html"
        wrapper.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            "html,body{margin:0;padding:0;background:#fff;overflow:hidden}"
            f"img{{display:block;width:{width_px}px;height:{height_px}px}}"
            f"</style></head><body><img src='{_uri(svg, chrome)}'></body></html>",
            encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={width_px},{height_px}",
                        f"--screenshot={_bpath(png, chrome)}", _uri(wrapper, chrome)],
                       check=True, timeout=120, capture_output=True)
    if png.is_file():
        print(f"Saved {png}  (Chrome, {width_px}x{height_px} px)")
    else:
        print("WARNING: Chrome did not produce a PNG.")


if __name__ == "__main__":
    main()
