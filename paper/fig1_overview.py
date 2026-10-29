"""Fig. 1 (overview, single column): the three contributions in one picture, no technical detail.

Three tinted step boxes read like text: (1) staged system identification of the hardware fish
(tail stiffness, fluid, motor) gives the MuJoCo twin; (2) the policy is learned in simulation only
(CPG action space, domain randomisation, curriculum); (3) it is deployed zero-shot, unchanged,
with the pool strobe of the figure-eight pursuit inside the observation loop.
Every label is real SVG text (9 pt and 8 pt, the paper's sizes); rasters are embedded as base64
PNG so the SVG is self-contained and editable in Inkscape.

Inputs:
  static/real_sofi.png                  hardware CAD render (AprilTag mast, tail right)
  static/40_hull_hinges_fluid.png       ellipsoid/hinge twin render (render_hull_hinges_fluid.py)
  static/sim_collage_panel2_t3.82s.png  overhead MuJoCo still of a training episode
  fishdeploy/data/02_deploy_20260520/analysis/report_strobes/*.png   the deployment strobes
Output: out/fig1_overview.pdf/.png; the editable SVG goes to out/svg/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svg as tp                                # noqa: E402  (SVG primitives, icons, export)

from _paths import STATIC as _STATIC, OUT as _OUT, ANALYSIS as _ANALYSIS
STATIC = _STATIC                                   # renders, stills, montage
OUT = _OUT                                         # generated figures, gitignored
SVG_DIR = _OUT / "svg"                             # editable SVG sources, disposable
DEFAULT_OUT = SVG_DIR / "fig1_overview.svg"       # PDF and PNG twins go to OUT/<stem>

W = tp.W                          # 86.8 mm = \columnwidth, placed unscaled
# type sizes at the printed size (the page is the column width, included unscaled):
# 9 pt (FS), 8 pt (FS2), 10 pt (FS_STEP), paper_style's tiers. K stays 1.
K = tp.PRINT_SCALE
FS = tp.FS                        # 9 pt: titles, headers, step names
FS2 = 8.0 / K / 72.27 * 25.4      # 8 pt: detail lines, chips, callouts (the floor)
TINT_SIM, TINT_REAL = "#EEF3FA", "#F7F2EB"       # side panels: simulation (cool) / hardware (warm)
INK, INK_SOFT = tp.INK, tp.INK_SOFT
ACC = ["#0072B2", "#D55E00", "#009E73"]          # Okabe-Ito: identify / learn / deploy
TINT = ["#EEF3FA", "#FFF4E6", "#EEF7EC"]
TIP = 0.7                                         # arrowhead overshoot (see marker refX)
FS_STEP = 10.0 / K / 72.27 * 25.4                 # 10 pt: the three numbered step titles (PANEL_PT)
COL_SIM_TITLE, COL_HW_TITLE = "#0072B2", "#C0562A" # cool / warm, matching the panel tints
DEPLOY_ASPECT = 2.3                               # w/h of the deployment strobe when it is not the figure-eight
_STROBE_DIR = _ANALYSIS / "report_strobes"
HIRES_STROBES = {                                  # 1920 x 1080 originals of the montage panels (paper repo)
    "a": _STROBE_DIR / "hope_didnt_work_2_highres_traceover_strobe.png",
    "b": _STROBE_DIR / "insane_run_8_middle_bg_highres_traceover_strobe.png",
    "c": _STROBE_DIR / "insane_run_2_highres_traceover_strobe.png",
    "d": _STROBE_DIR / "finetune_run_7_highres_traceover_strobe.png",
    "e": _STROBE_DIR / "finetune_run_8_highres_traceover_strobe.png",
    "f": _STROBE_DIR / "figure_eight_6_clean_highres_traceover_strobe.png",
}
# The deployment collage (fig7_deploy_collage.py, box B) crops the same figure-eight
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

# ---------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------

TEXT_BLACK = "#000000"

def blacken_text(svg: str) -> str:
    """Set every <text> that uses the ink or soft-ink grey to plain black. Strokes, icons and
    the white numerals in the step badges keep their colours."""
    import re
    return re.sub(r'(<text\b[^>]*?fill=")(?:%s|%s)(")' % (INK, INK_SOFT),
                  lambda m: m.group(1) + TEXT_BLACK + m.group(2), svg)

SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by paper/fig1_overview.py - edit freely in Inkscape. -->
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

# ---------------------------------------------------------------------------------------
# minimal (title-page) variant: same Z layout, icons instead of text
# ---------------------------------------------------------------------------------------

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

def mini_axes(x, y, w, h):
    """L-shaped axes for a schematic mini plot."""
    return tp.path(f"M {tp.f(x)},{tp.f(y)} V {tp.f(y + h)} H {tp.f(x + w)}", stroke=INK_SOFT, sw=0.18)

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

    Split out of main() so another consumer (a video, a slide) builds the same picture rather than
    a lookalike. render_filter, if given, is applied to the renders before they are encoded.
    """
    hw = load_render(STATIC / "real_sofi.png")
    twin = load_render(STATIC / "40_hull_hinges_fluid.png")
    if render_filter is not None:
        hw, twin = render_filter(hw), render_filter(twin)
    sim_path = sorted(STATIC.glob("sim_collage_panel2_t*.png"))
    if not sim_path:
        sys.exit("need static/sim_collage_panel2_t*.png (the overhead MuJoCo still of the training episode)")
    sim = Image.open(sim_path[-1]).convert("RGB").crop((600, 120, 1560, 720))
    sim = crop_aspect(sim, photo_aspect)
    hires = HIRES_STROBES[rich_panel]              # the 1920 x 1080 strobe composite of fishdeploy
    if not hires.is_file():
        sys.exit(f"need {hires} (fishdeploy/data/.../analysis/report_strobes)")
    src = Image.open(hires).convert("RGB")
    if rich_panel == "f" and src.size == (1920, 1080):
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
    p.add_argument("--panel", default="f", choices=list("abcdef"),
                   help="which deployment strobe to show (default f: the figure-eight pursuit, as in the paper)")
    p.add_argument("--photo-aspect", type=float, default=1.6)
    p.add_argument("--plain-cards", action="store_true", help="identification cards without their white backing")
    p.add_argument("--fish-w", type=float, default=17.5, help="CAD render width in mm")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)

    svg, H = build_svg_boxed(**boxed_inputs(args.photo_aspect, args.panel),
                             white_cards=not args.plain_cards, fish_w=args.fish_w)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Saved {out}  ({W:.1f} x {H:.1f} mm, {out.stat().st_size / 1024:.0f} kB)")
    if not args.no_pdf:
        tp.export_pdf(out, OUT / (out.stem + ".pdf"), W * K, H * K)
    if not args.no_png:
        tp.export_png(out, OUT / (out.stem + ".png"), W * K, H * K, px_per_mm=10.0 / K)


if __name__ == "__main__":
    main()
