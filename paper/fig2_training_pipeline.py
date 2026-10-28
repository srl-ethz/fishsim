#!/usr/bin/env python3
"""Fig. 2 (training pipeline, full text width): simulator, trajectory generator, reward and
SAC in one row, drawn as editable SVG. The question it answers: what does the policy see,
what does it command, and how does the TG sit between the two.

Layout (left to right):

  [MuJoCo simulation environment]  <-theta_m-  [Trajectory generator]  <-u*=a_t-  [SAC actor]
     ellipsoid/hinge twin render                 smoothing -> harmonic oscillator    critic
     15 fluid coefficients                       u and its harmonic parts            replay buffer
     one overhead strobe still           --s_t (icon strip) ------------------------> actor, buffer
                                         --d-> [Reward r_t] --------------------------> buffer

Style: tinted blocks with NO outlines (the tint alone bounds a block; sub-blocks are white on
the tint), one ink colour for all text (no grey tier), 7 pt titles and signal labels, 6 pt
annotations, Times via SVG font-family. Every
label is a real <text> element, so the SVG stays Inkscape-editable. Rasters are embedded as
base64 PNG so the file is self-contained.

Inputs (figures/static/):
  40_hull_hinges_fluid.png     the ellipsoid/hinge fish render
  sim_collage_strobe.png       overhead MuJoCo strobe still of the twin reaching a target on a
                               white floor with a light-grey 0.25 m grid, no swum-path trail,
                               three opaque poses (t = 0, halfway to the target at 2.36 s,
                               arrival at 3.82 s), hull meshes opaque and matte grey, the swum
                               path of the hull centre as a red trail, raw (the tail-beat wobble
                               is real), with every fish composited over it
                               (render_sim_collage.py in the sofiRL/fishCREATE repo:
                               --times 0 2.35 4 --target-deg 20 --no-start-ring --strobe
                               --floor grid --ghost-alpha 1.0 --opaque --matte
                               --trail-rgba 0.85 0.45 0.45 1.0 --trail-radius 0.004
                               --trail-anchor hull --fish-over-trail, checkpoint
                               outputs/e06g760j/SAC_curriculum_stage8_lateral.zip); a static
                               raster like the Fig. 1 stills, with its poses in the .json
                               sidecar. Nothing here regenerates it; the grid is subdivided and darkened
                               at load time (refine_grid).
Output:  out/fig2_training_pipeline.pdf (+ .png); the editable SVG goes to .gen/svg/
Run via `make figures`, or from .gen/figures:  python fig2_training_pipeline.py [--no-pdf]
The PDF export needs Inkscape or Chrome/Edge on the PATH (see _svg_curriculum.export_pdf).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _svg_pipeline as tp                  # noqa: E402  primitives, equation markup, icons, glyphs
import _svg_curriculum as cf                # noqa: E402  export_pdf / export_png

from _paths import STATIC as _STATIC, OUT as _OUT
STATIC = _STATIC
OUT = _OUT
SVG_DIR = _OUT / "svg"                      # editable SVG source, disposable
DEFAULT_OUT = SVG_DIR / "fig2_training_pipeline.svg"   # PDF and PNG twins go to OUT/<stem>

W, H = 177.8, 70.0                          # \textwidth x page height, mm; placed unscaled
FS_MAIN, FS_SMALL = tp.FS_MAIN, tp.FS_SMALL # 7 pt titles / signal labels, 6 pt annotations
INK, INK_SOFT = tp.INK, tp.INK_SOFT
text, eq, rect, line, path, image, icon, group = (tp.text, tp.eq, tp.rect, tp.line, tp.path,
                                                   tp.image, tp.icon, tp.group)
f = tp.f
# Times lacks U+2192 (->) and U+2016 (double bar); without a TrueType fallback Chrome sets
# them from Cambria Math and embeds it as Type 3, which submission systems reject.
FONT = tp.FONT.replace("serif", "'Segoe UI Symbol', 'DejaVu Serif', serif")

# ── block tints (no outlines: the tint is the block) ───────────────────────────
# One muted tint per block, keyed to the Okabe-Ito accents the rest of the paper uses
# (blue = simulator/identified model, amber = trajectory generator, green = reward/arrival,
# purple = learner). Saturated enough to bound a block without a stroke, light enough
# that 6 pt ink on top stays legible in print.
FILL_SIM = "#E4EDF8"          # blue
FILL_TG = "#FBEEDD"           # amber
FILL_REW = "#E3F2EA"          # green
FILL_SAC = "#ECE7F5"          # purple
FILL_SUB = "#FFFFFF"          # sub-blocks, white on the tint
FILL_BAND = "#EEF0F3"         # state band between simulator and SAC
NO_STROKE = dict(stroke="none", sw=0)


def harmonic_parts(x0, y, w, h, gap):
    """Four mini plots over two strokes of phase: theta_m = theta_c + (a1, b1) + (a2, b2).
    Same as _svg_pipeline.harmonic_parts but every label is set in INK (no grey text)."""
    phi = np.linspace(0, 4 * np.pi, 400)
    th_c = 0.55
    h1 = 1.0 * np.cos(phi) + 0.4 * np.sin(phi)
    h2 = 0.32 * np.cos(2 * phi) + 0.2 * np.sin(2 * phi)
    series = [(th_c + h1 + h2, tp.COL_SIGNAL, "Motor angle", "θ_{m}(φ)"),
              (np.full_like(phi, th_c), INK_SOFT, "Offset", "θ_{c}"),
              (h1, tp.COL_FILT, "1st harmonic", "a_{1}, b_{1}"),
              (h2, tp.COL_IN, "2nd harmonic", "a_{2}, b_{2}")]
    lo, hi = -1.8, 1.8
    out = []
    for i, (vals, col, name, sym) in enumerate(series):
        px = x0 + i * (w + gap)
        def X(v, px=px): return px + v / phi[-1] * w
        def Y(v): return y + h / 2 - v / (hi - lo) * h
        out.append(line(px, Y(0), px + w, Y(0), stroke=INK_SOFT, sw=0.15, marker=False))
        out.append(path("M " + " L ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(phi, vals)),
                        stroke=col, sw=0.32))
        out.append(text(px + w / 2, y + h + 3.2, name, fs=FS_SMALL, anchor="middle"))
        out.append(eq(px + w / 2, y + h + 6.4, sym, fs=FS_SMALL, anchor="middle"))
        if i < 3:
            out.append(text(px + w + gap / 2, y + h / 2 + 0.9, "=" if i == 0 else "+", fs=FS_MAIN,
                            anchor="middle"))
    return "\n    ".join(out)


def refine_grid(im: "Image.Image", factor: int = 2, grey: int | None = None) -> "Image.Image":
    """Subdivide the light-grey floor grid of the overhead render `factor` times, and
    optionally repaint it darker.

    The render is orthographic, so the grid is a set of straight, evenly spaced rows and
    columns (3 px wide, grey 212 on white). They are found as rows/columns whose median
    dips below white; new lines of the same width go at the subdivision points, painted only
    over white pixels so the fish, the target and the existing lines are left exactly as
    rendered. With `grey`, the existing lines' pixels (those at the rendered grey, so the fish
    is untouched) and the new lines are set to that level instead of the rendered 212."""
    a = np.asarray(im.convert("RGB")).copy()
    g = a.mean(-1)
    white = (a.min(-1) >= 250)

    def centres(profile):
        idx = np.where(profile < 253)[0]
        if len(idx) == 0:
            return [], 1
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        cs = [float(gr.mean()) for gr in groups]
        width = int(round(np.median([len(gr) for gr in groups])))
        return cs, width

    cols, cw = centres(np.median(g, axis=0))
    rows, rw = centres(np.median(g, axis=1))
    rendered = int(round(min(np.median(g[:, int(round(cols[1]))]) if len(cols) > 1 else 212,
                             np.median(g[int(round(rows[1])), :]) if len(rows) > 1 else 212)))
    colour = rendered if grey is None else int(grey)
    if grey is not None:                              # repaint the rendered lines darker
        on_line = (np.abs(g - rendered) <= 8) & (np.ptp(a, axis=-1) <= 6)   # neutral grey only
        for c in cols:
            x0 = max(0, int(round(c - cw / 2))); x1 = min(a.shape[1], x0 + cw)
            m = on_line[:, x0:x1]; a[:, x0:x1][m] = colour
        for r in rows:
            y0 = max(0, int(round(r - rw / 2))); y1 = min(a.shape[0], y0 + rw)
            m = on_line[y0:y1, :]; a[y0:y1, :][m] = colour
    if factor <= 1:
        return Image.fromarray(a)

    def new_positions(cs, n):
        if len(cs) < 2:
            return []
        step = float(np.median(np.diff(cs)))
        out = []
        c = cs[0] - step                              # also fill the margins outside the outermost lines
        while c < n + step:
            for k in range(1, factor):
                out.append(c + k * step / factor)
            c += step
        return [v for v in out if -1 <= v <= n]

    for c in new_positions(cols, a.shape[1]):
        x0 = int(round(c - cw / 2)); x0 = max(0, x0); x1 = min(a.shape[1], x0 + cw)
        m = white[:, x0:x1]
        a[:, x0:x1][m] = colour
    for r in new_positions(rows, a.shape[0]):
        y0 = int(round(r - rw / 2)); y0 = max(0, y0); y1 = min(a.shape[0], y0 + rw)
        m = white[y0:y1, :]
        a[y0:y1, :][m] = colour
    return Image.fromarray(a)


def load_panel_fine(path: Path, aspect: float, factor: int, grey: int | None = None,
                    width_px: int = 1400):
    """tp.load_panel with the floor grid subdivided (and darkened) first (see refine_grid)."""
    im = refine_grid(Image.open(path).convert("RGB"), factor, grey)
    box = tp.crop_box((im.height, im.width), 0, im.width - 40, aspect, y_center=im.height // 2)
    im = im.crop(box)                                 # no left cut: the first fish's tail reaches the edge
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS), box


def block(x, y, w, h, fill, rx=1.8):
    """A tinted block with no outline."""
    return rect(x, y, w, h, fill, rx=rx, **NO_STROKE)


def flap_fish(cx, cy, sweep_deg=40.0):
    """Top-view fish heading right, drawn in absolute mm, with its tail shown mid-stroke: the fish
    with a straight tail in ink, two soft ghost tails swept +-sweep about the tail root, and a
    two-headed arc just beyond the ghost tips. Reads as "a large tail stroke", which is what the
    action penalty charges for. About 6 mm wide, 3.5 mm tall around (cx, cy) = body centre."""
    import math
    rx, ry = 1.45, 0.8                              # body half-axes
    lt, ht = 2.0, 0.75                              # tail length and half-height at the tip
    root = (cx - rx + 0.15, cy)
    body = (f'<ellipse cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" fill="{INK}" '
            f'stroke="none"/>')
    tail = f"M 0,0 L {-lt:.3f},{-ht:.3f} L {-lt:.3f},{ht:.3f} Z"
    solid = (f'<path d="{tail}" transform="translate({f(root[0])},{f(root[1])})" fill="{INK}" '
             f'stroke="{INK}" stroke-width="0.15" stroke-linejoin="round"/>')
    ghosts = "".join(
        f'<path d="{tail}" transform="translate({f(root[0])},{f(root[1])}) rotate({sg * sweep_deg:.1f})" '
        f'fill="{INK_SOFT}" fill-opacity="0.4" stroke="none"/>' for sg in (-1, 1))
    r_arc = lt + 0.55                               # just beyond the tail tips
    a = math.radians(sweep_deg + 6.0)               # the arc reaches a little past the ghosts
    pts = [(root[0] - r_arc * math.cos(t), root[1] + r_arc * math.sin(t))
           for t in [(-a + 2 * a * i / 24) for i in range(25)]]
    arc = (f'<path d="M ' + " L ".join(f"{x:.3f},{y:.3f}" for x, y in pts) + '" fill="none" '
           f'stroke="{INK}" stroke-width="0.22" stroke-linecap="butt" '
           f'marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
    return f'<g inkscape:label="flapping fish">{ghosts}{body}{solid}{arc}</g>'


def build_svg(fish_uri: str, still_uri: str, still_aspect: float) -> tuple[str, float]:
    """Returns (svg, page height in mm)."""
    # ---- column geometry -----------------------------------------------------
    Hm = H
    # three blocks with air between them: 10 mm SIM->TG, 18.5 mm TG->SAC (the state band
    # and the reward arrow live in the wider gap)
    SIM = dict(x=1.0, y=1.0, w=51.0, h=Hm - 2.0)  # as wide as its title plus margins
    TG = dict(x=59.15, y=1.0, w=76.0, h=33.0)    # top aligned with the SIM and SAC boxes
    REW = dict(x=59.15, y=Hm - 14.5, w=76.0, h=13.5)   # title row over the pictogram row
    SAC = dict(x=W - 1.0 - 34.5, y=1.0, w=34.5, h=Hm - 2.0)
    sub_w = SAC["w"] - 4 - 3.6                    # leave a lane on the right for the action bus
    sub_h = 13.5                                  # actor / critic boxes
    ACT = dict(x=SAC["x"] + 2, y=12.5, w=sub_w, h=sub_h)
    BUF = dict(x=SAC["x"] + 2, y=Hm - 14.5, w=sub_w, h=12.0)
    CRI = dict(x=SAC["x"] + 2, y=(ACT["y"] + ACT["h"] + BUF["y"]) / 2 - sub_h / 2, w=sub_w, h=sub_h)
    y_actor = ACT["y"] + ACT["h"] / 2             # actor centreline: state in, action out
    y_buf = BUF["y"] + BUF["h"] / 2               # buffer centreline: action in from the right

    sim_r = SIM["x"] + SIM["w"]
    cpg_r = TG["x"] + TG["w"]
    rew_r = REW["x"] + REW["w"]
    # one signal height for the whole top row: SAC actor -> TG chain -> simulator (theta_m)
    TITLE_DY = 4.0                                # every block title: baseline this far below its box top
    ch_y, ch_h, ch_w = TG["y"] + 6.0, 7.2, 18.5   # the TG title is centred between box top and these boxes
    y_act = ch_y + ch_h / 2

    # ---- simulation block ----------------------------------------------------
    fish_w = 40.0
    fish_h = fish_w * 760 / 1347
    fish_x = SIM["x"] + (SIM["w"] - fish_w) / 2   # centred in the block
    fish_y = 7.0                                  # a little air under the title
    fish_b = fish_y + fish_h
    # Under the render: the 5 x 3 matrix of quasi-steady fluid coefficients (blunt, slender and
    # angular drag, Kutta and Magnus lift; one column per ellipsoid zone), with an underbrace
    # on each zone of the render. Same notation (c_{b,H} ...) as the fluid table in the paper.
    zones = [("Head", 0.08, 0.53), ("Tail", 0.54, 0.76), ("Fin", 0.77, 0.94)]
    y_br = fish_b + 0.7                           # underbrace baseline
    y_hdr = y_br + 3.3                            # zone names sit right under the brackets
    ROW = 3.0                                     # 8 pt rows
    rows = ["Blunt drag", "Slender drag", "Angular drag", "Kutta lift", "Magnus lift"]
    subs = ["b", "s", "a", "K", "M"]
    col_x = [fish_x + (f0 + f1) / 2 * fish_w for _, f0, f1 in zones]
    col_x[0] = col_x[1] - (col_x[2] - col_x[1])   # Head column at the Tail-Fin spacing, still under its bracket
    y_rows = [y_hdr + ROW * (i + 1) for i in range(len(rows))]
    coeffs = []
    for (name, f0, f1), cx in zip(zones, col_x):
        x0, x1 = fish_x + f0 * fish_w, fish_x + f1 * fish_w
        coeffs.append(path(f"M {f(x0)},{f(y_br - 1.0)} V {f(y_br)} H {f(x1)} V {f(y_br - 1.0)}",
                           stroke=INK_SOFT, sw=0.22))
        coeffs.append(text(cx, y_hdr, name, fs=FS_SMALL, anchor="middle", weight="bold"))
        for yr, sb in zip(y_rows, subs):
            coeffs.append(eq(cx, yr, f"c_{{{sb},{name[0]}}}", fs=FS_SMALL, anchor="middle"))
    for yr, name in zip(y_rows, rows):
        coeffs.append(text(col_x[0] - 4.8, yr, name, fs=FS_SMALL, anchor="end"))
    bx0, bx1 = col_x[0] - 3.2, col_x[-1] + 3.2
    by0, by1 = y_hdr + 0.9, y_rows[-1] + 0.7
    coeffs.append(path(f"M {f(bx0 + 0.9)},{f(by0)} H {f(bx0)} V {f(by1)} H {f(bx0 + 0.9)}", sw=0.25))
    coeffs.append(path(f"M {f(bx1 - 0.9)},{f(by0)} H {f(bx1)} V {f(by1)} H {f(bx1 - 0.9)}", sw=0.25))
    # one large overhead still (strobe: final pose + ghosts of earlier poses) on a white
    # floor with a light grid; no frame and no legend, the caption names path and target
    pan_w = SIM["w"] - 4.0
    pan_h = pan_w / still_aspect
    px = SIM["x"] + 2.0
    pan_y = by1 + ((SIM["y"] + SIM["h"] - by1) - pan_h) / 2      # centred in the space below the matrix
    sim = group(
        "Simulation",
        block(**SIM, fill=FILL_SIM),
        text(SIM["x"] + SIM["w"] / 2, SIM["y"] + TITLE_DY, "MuJoCo Simulation Environment", fs=FS_MAIN,
             anchor="middle", weight="bold"),
        image(fish_x, fish_y, fish_w, fish_h, fish_uri),
        *coeffs,
        f'<clipPath id="still_clip"><rect x="{f(px)}" y="{f(pan_y)}" width="{f(pan_w)}" '
        f'height="{f(pan_h)}" rx="1.2"/></clipPath>',
        image(px, pan_y, pan_w, pan_h, still_uri, extra='clip-path="url(#still_clip)"'),
    )

    # ---- TG block -----------------------------------------------------------
    # chain on top (right to left, at the height of the incoming action and outgoing theta_m):
    #   u* -> [first-order smoothing] -> u -> [harmonic oscillator] -> theta_m
    # the parameter vector below it, and how the oscillator composes theta_m from its parts.
    # each box is its bold label plus the same air on both sides (label widths measured on
    # the rendered PNG: "Harmonic oscillator" 17.4 mm, "First-order smoothing" see below)
    osc_w, lpf_w = 25.2, 28.2
    OSC = dict(x=TG["x"] + 3.0, y=ch_y, w=osc_w, h=ch_h)
    LPF = dict(x=cpg_r - 3.0 - lpf_w, y=ch_y, w=lpf_w, h=ch_h)
    hp_w, hp_h, hp_gap = 11.8, 7.0, 4.0
    hp_x0 = TG["x"] + (TG["w"] - (4 * hp_w + 3 * hp_gap)) / 2
    hp_y = TG["y"] + 18.2
    parts = harmonic_parts(hp_x0, hp_y, hp_w, hp_h, hp_gap)
    cpg = group(
        "TG",
        block(**TG, fill=FILL_TG),
        text(TG["x"] + TG["w"] / 2, TG["y"] + TITLE_DY, "Trajectory Generator (TG)", fs=FS_MAIN,
             anchor="middle", weight="bold"),
        block(**LPF, fill=FILL_SUB, rx=0.9),
        text(LPF["x"] + lpf_w / 2, y_act - 0.8, "First-order smoothing", fs=FS_SMALL, anchor="middle",
             weight="bold"),
        eq(LPF["x"] + lpf_w / 2, y_act + 2.7, "\\b{u}* → \\b{u}", fs=FS_SMALL, anchor="middle"),
        block(**OSC, fill=FILL_SUB, rx=0.9),
        text(OSC["x"] + osc_w / 2, y_act - 0.8, "Harmonic oscillator", fs=FS_SMALL, anchor="middle",
             weight="bold"),
        eq(OSC["x"] + osc_w / 2, y_act + 2.7, "dφ/dt = ω", fs=FS_SMALL, anchor="middle"),
        # filter -> oscillator (the action arrow into the filter and theta_m out of the
        # oscillator are drawn in the Arrows group so they run straight across the box edges)
        line(LPF["x"], y_act, OSC["x"] + osc_w + 0.4, y_act),
        eq(TG["x"] + TG["w"] / 2, TG["y"] + 16.2,
           "\\b{u} = (ω, θ_{c}, a_{1}, b_{1}, a_{2}, b_{2})", fs=FS_SMALL, anchor="middle"),
        parts,
    )

    # ---- reward block --------------------------------------------------------
    # "Reward" centred on top; below it the three terms of eq. (reward) as pictograms with the
    # operators between them, no equation text: the arrival bonus (the fish on the 5 cm ring
    # around the target), minus the distance to the target (fish, dashed d, target), minus a
    # fish sweeping its tail (the action penalty on the TG parameters). Bonus first, so both
    # penalties carry their minus and no sign leads the row.
    g_cy = REW["y"] + 8.0                          # pictogram row centre
    y_lab = g_cy + 4.6                             # one label line under the row
    r_arr = 2.3                                    # 5 cm arrival radius (not to scale)
    # the row is centred in the box with one gap G between every element (pictogram, minus,
    # pictogram, minus, pictogram); the element widths are their drawn extents
    W_ARR, W_MIN, W_DIST, W_FLAP = 1.7 + 1.05 + 0.15 + 2 * r_arr, 1.5, 1.7 + 11.0 + 0.7, 3.85 + 1.45
    G = (REW["w"] - 20.0 - (W_ARR + 2 * W_MIN + W_DIST + W_FLAP)) / 4   # 10 mm side margins for the labels
    x0 = REW["x"] + (REW["w"] - (W_ARR + 2 * W_MIN + W_DIST + W_FLAP + 4 * G)) / 2
    x_f3 = x0 + 1.7                                # arrival pictogram: fish centre ...
    x_t = x_f3 + 1.05 + 0.15 + r_arr               #   ... target centre, the fish's nose on the ring
    x_m1 = x_t + r_arr + G + W_MIN / 2             # minus
    x_d0 = x_m1 + W_MIN / 2 + G + 1.7              # distance pictogram: fish centre
    x_d1 = x_d0 + 11.0                             #   ... target
    x_m2 = x_d1 + 0.7 + G + W_MIN / 2              # minus
    x_fl = x_m2 + W_MIN / 2 + G + 3.85             # flapping fish (body centre)
    rew = group(
        "Reward",
        block(**REW, fill=FILL_REW),
        eq(REW["x"] + REW["w"] / 2, REW["y"] + TITLE_DY, r"\b{Reward }\r{(}r_{t}\r{)}", fs=FS_MAIN, anchor="middle"),
        # arrival bonus
        f'<circle cx="{f(x_t)}" cy="{f(g_cy)}" r="{f(r_arr)}" fill="#009E73" fill-opacity="0.15" '
        f'stroke="#009E73" stroke-width="0.2" stroke-dasharray="0.5 0.4"/>',
        f'<circle cx="{f(x_t)}" cy="{f(g_cy)}" r="0.7" fill="#009E73" stroke="{INK}" '
        f'stroke-width="0.12"/>',
        tp.fish_top(x_f3, g_cy, 3.4, heading_deg=90),
        eq((x_f3 + x_t) / 2, y_lab, "\\r{Bonus at }5\\r{ cm}", fs=FS_SMALL, anchor="middle",
           fill="#009E73"),
        # distance term
        text(x_m1, g_cy + 1.0, "−", fs=FS_MAIN, anchor="middle"),
        tp.fish_top(x_d0, g_cy, 3.4, heading_deg=90),
        f'<circle cx="{f(x_d1)}" cy="{f(g_cy)}" r="0.7" fill="#009E73" stroke="{INK}" '
        f'stroke-width="0.12"/>',
        line(x_d0 + 2.1, g_cy, x_d1 - 1.3, g_cy, sw=0.22, dash="0.5 0.4", marker=True),
        eq((x_d0 + 2.1 + x_d1 - 1.3) / 2, g_cy - 1.0, "d", fs=FS_SMALL, anchor="middle"),
        eq((x_d0 + x_d1) / 2, y_lab, "\\r{Distance }d", fs=FS_SMALL, anchor="middle", fill=INK_SOFT),
        # action penalty
        text(x_m2, g_cy + 1.0, "−", fs=FS_MAIN, anchor="middle"),
        flap_fish(x_fl, g_cy),
        eq(x_fl, y_lab, "\\r{Action }‖\\b{a}‖", fs=FS_SMALL, anchor="middle", fill=INK_SOFT),
    )

    # ---- SAC block -----------------------------------------------------------
    net_w, net_h = 12.5, 5.0
    net_x = ACT["x"] + (sub_w - net_w) / 2         # glyph centred (its s / a labels hang either side)
    buf_row_w = 4.8 + 1.4 + 20.0                   # icon, gap, "Replay Buffer"
    buf_x0 = BUF["x"] + (sub_w - buf_row_w) / 2
    act_r = ACT["x"] + ACT["w"]
    buf_r = BUF["x"] + BUF["w"]
    sac_r = SAC["x"] + SAC["w"]
    bus_x = sac_r - 2.3                            # action bus: actor -> TG and -> buffer
    net_dy = ACT["h"] - net_h - 1.3                # glyph sits under the title line
    sac = group(
        "SAC",
        block(**SAC, fill=FILL_SAC),
        text(SAC["x"] + SAC["w"] / 2, SAC["y"] + TITLE_DY, "Soft Actor-Critic (SAC)", fs=FS_MAIN, anchor="middle",
             weight="bold"),
        block(**ACT, fill=FILL_SUB, rx=1.1),
        eq(ACT["x"] + sub_w / 2, ACT["y"] + 3.7, "\\b{Actor  }π(\\b{a} | \\b{s})", fs=FS_MAIN, anchor="middle"),
        tp.network(net_x, ACT["y"] + net_dy, net_w, net_h, [5, 6, 6, 3], "\\b{s}", "\\b{a}"),
        block(**CRI, fill=FILL_SUB, rx=1.1),
        eq(CRI["x"] + sub_w / 2, CRI["y"] + 3.7, "\\b{Critic  }Q(\\b{s}, \\b{a})", fs=FS_MAIN, anchor="middle"),
        tp.network(net_x, CRI["y"] + net_dy, net_w, net_h, [6, 6, 6, 1], "\\b{s},\\b{a}", "Q"),
        block(**BUF, fill=FILL_SUB, rx=1.1),
        icon("database", buf_x0, BUF["y"] + 3.4, 4.8, color=INK),
        text(buf_x0 + 6.2, BUF["y"] + 4.8, "Replay Buffer", fs=FS_MAIN, weight="bold"),
        eq(BUF["x"] + sub_w / 2, BUF["y"] + 8.9, "(\\b{s}_{t}, \\b{a}_{t}, r_{t}, \\b{s}_{t+1})", fs=FS_SMALL,
           anchor="middle"),
        # actor -> action bus (right lane) -> down into the buffer
        path(f"M {f(act_r)},{f(y_actor)} H {f(bus_x)} V {f(y_buf)} H {f(buf_r + 0.4)}",
             marker=True),
        # ... and up, over the actor along the free lane, out of SAC and into the TG's filter
        path(f"M {f(bus_x)},{f(y_actor)} V {f(y_act)} H {f(LPF['x'] + lpf_w + 0.4)}", marker=True),
        # buffer -> critic (minibatch), critic -> actor (policy gradient)
        line(BUF["x"] + BUF["w"] * 0.3, BUF["y"], BUF["x"] + BUF["w"] * 0.3, CRI["y"] + CRI["h"] + 0.4,
             sw=0.28),
        line(CRI["x"] + CRI["w"] * 0.3, CRI["y"], CRI["x"] + CRI["w"] * 0.3, ACT["y"] + ACT["h"] + 0.4,
             sw=0.28, dash="0.7 0.5"),
    )

    # ---- observation strip (icons) under the state arrow ------------------------
    y_state = (TG["y"] + TG["h"] + REW["y"]) / 2 - 3.2   # band centred between TG and reward
    fan_x = SAC["x"] - 3.2                # state fan-out point
    y_rew = REW["y"] + REW["h"] / 2
    obs = [  # (icon, width share, line 1, line 2) -- names only, no dimensions
        ("engine",    13.4, "Motor angle",  "& velocity"),
        ("rotate-3d", 12.4, "IMU",          "yaw rate"),
        ("wave-sine", 11.6, "TG phase",     "cos φ, sin φ"),
        ("camera",    16.6, "Camera",       "target pose"),
        ("history",   13.6, "Past actions", ""),
    ]
    obs = [(n, 1.0, l1, l2) for n, _, l1, l2 in obs]  # equal widths: evenly spaced in the band
    strip_x0, strip_x1 = TG["x"] + 2.5, TG["x"] + TG["w"] - 2.5   # the chips fill the band
    scale = (strip_x1 - strip_x0) / sum(w for _, w, _, _ in obs)
    ico, y0 = 4.0, y_state + 0.2
    row_h = ico
    chips, cx = [], strip_x0
    for name, w, l1, l2 in obs:
        w *= scale
        chips.append(icon(name, cx + w / 2 - ico / 2, y0 + (row_h - ico) / 2, ico, color=INK))
        chips.append(text(cx + w / 2, y0 + row_h + 3.0, l1, fs=FS_SMALL, anchor="middle"))
        chips.append(text(cx + w / 2, y0 + row_h + 6.2, l2, fs=FS_SMALL, anchor="middle"))
        cx += w
    # soft band that gathers the state label, its arrow and the observation strip into one
    # visual unit between the simulator and SAC
    band = dict(x=TG["x"], y=y_state - 5.2, w=TG["w"],                  # same span as the TG and reward boxes
                h=(y0 + row_h + 6.2 + 1.2) - (y_state - 5.2))
    y_mid = band["y"] + band["h"] / 2                 # the state arrow's height: the band's middle

    # ---- arrows between blocks -------------------------------------------------
    arrows = group(
        "Arrows",
        block(**band, fill=FILL_BAND),
        # TG oscillator -> sim: motor setpoint theta_m, straight out of the chain
        line(OSC["x"], y_act, sim_r + 0.4, y_act),
        eq((sim_r + TG["x"]) / 2, y_act - 1.3, "θ_{m}", fs=FS_MAIN, anchor="middle"),
        # SAC actor -> TG filter: action (the line itself is drawn from the action bus in the SAC group)
        eq((cpg_r + SAC["x"]) / 2, y_act - 1.3, "\\b{u}*", fs=FS_MAIN, anchor="middle"),
        # sim -> SAC: state (fans out to the actor, and into the replay buffer)
        line(sim_r, y_mid, band["x"] - 0.4, y_mid),                       # into the band ...
        line(band["x"] + band["w"], y_mid, fan_x, y_mid, marker=False),   # ... and out of it
        path(f"M {f(fan_x)},{f(y_mid)} V {f(y_actor)} H {f(ACT['x'] - 0.4)}", marker=True),
        path(f"M {f(fan_x)},{f(y_mid)} V {f(BUF['y'] + 3.0)} H {f(BUF['x'] - 0.4)}", marker=True),
        eq((sim_r + fan_x) / 2, y_state - 1.2, "\\b{State }\\r{(}\\b{s}_{t}\\r{)}", fs=FS_MAIN, anchor="middle"),
        *chips,
        # sim -> reward -> replay buffer, one straight line through the reward box
        line(sim_r, y_rew, REW["x"] - 0.4, y_rew),
        line(rew_r, y_rew, BUF["x"] - 0.4, y_rew),
    )

    defs = (
        '<defs>'
        # refX sits well behind the tip so the (butt-ended) line is hidden inside the head
        '<marker id="arrow" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4.5" '
        'markerHeight="4.5" orient="auto-start-reverse">'
        f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{INK}"/></marker>'
        '</defs>'
    )

    main = "\n  ".join([sim, cpg, rew, sac, arrows])
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by figures/fig2_training_pipeline.py (swim2real-icra2026) - edit freely in Inkscape. -->
<svg
   width="{f(W * tp.PRINT_SCALE)}mm" height="{f(Hm * tp.PRINT_SCALE)}mm" viewBox="0 0 {f(W)} {f(Hm)}"
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
    return svg, Hm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--fish", default=str(STATIC / "40_hull_hinges_fluid.png"))
    p.add_argument("--still", default=str(STATIC / "sim_collage_strobe.png"),
                   help="overhead strobe still (see the module docstring for how it was rendered)")
    p.add_argument("--still-aspect", type=float, default=4.2,
                   help="w/h crop aspect for the still (raw render is 2:1)")
    p.add_argument("--grid-refine", type=int, default=2,
                   help="subdivide the still's floor grid this many times (1 = as rendered)")
    p.add_argument("--grid-grey", type=int, default=165,
                   help="grey level (0-255) of the floor grid lines; the render has 212")
    p.add_argument("--no-pdf", action="store_true", help="skip the PDF export next to the SVG")
    p.add_argument("--no-png", action="store_true", help="skip the PNG preview")
    args = p.parse_args(argv)

    still = Path(args.still)
    if not still.is_file():
        sys.exit(f"Missing input {still}; see the module docstring for how it is rendered.")

    fish_uri = tp.png_data_uri(tp.load_fish(Path(args.fish)))
    still_img, _box = load_panel_fine(still, args.still_aspect, args.grid_refine, args.grid_grey)
    still_uri = tp.png_data_uri(still_img)

    svg, height = build_svg(fish_uri, still_uri, args.still_aspect)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Saved {out}  ({W:g} x {height:g} mm, {out.stat().st_size / 1024:.0f} kB)")
    k = tp.PRINT_SCALE                        # the page is k x the design; LaTeX scales it back
    if not args.no_pdf:
        cf.export_pdf(out, OUT / (out.stem + ".pdf"), W * k, height * k)
    if not args.no_png:
        cf.export_png(out, OUT / (out.stem + ".png"), W * k, height * k, px_per_mm=2100 / W / k)


if __name__ == "__main__":
    main()
