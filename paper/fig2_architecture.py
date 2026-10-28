"""Full-width (page 2 top) architecture figure: training pipeline + curriculum in one picture.

Top band  (the SAC / CPG / simulation pipeline, after make_training_pipeline_fig.py):

  [MuJoCo simulation environment] <-theta_m- [Trajectory generator (CPG) + LPF] <-u*- [SAC actor]
     ellipsoid/hinge twin render                 chain: u* -> LPF -> u -> oscillator -> theta_m
     3 overhead stills                           waveform for a step in u*
     --s_t (24-D) --------- icon strip ------------------------------------------> [actor, buffer]
     --d-> [Reward r_t] ---------------------------------------------------------> [replay buffer]

Bottom band (the ten-stage, three-phase curriculum, after make_curriculum_fig.py, laid out as
one horizontal timeline):

  Phase 1 Sensor Noise      Phase 2 Domain Randomization     Phase 3 Target Widening
  [1] -> [2] -> [3]   ->    [4] -> [5] -> [6]        ->      [7] -> [8] -> [9] -> [10]

with two arrows up into the top band: the stage configures the environment (SIM) and sets the
learning rate (SAC).

Text is 7 pt (2.46 mm) for titles, equations and stage titles; secondary annotations
(callouts, units, icon captions) are 6 pt (2.1 mm) -- flagged in the script header so they can
be raised if Mike wants the strict 7 pt floor everywhere. Everything is real SVG text, editable
in Inkscape; rasters are embedded base64 PNG.

Inputs:  figures/static/40_hull_hinges_fluid.png, figures/static/sim_collage_panel{0,1,2}_t*.png
         (rendered by icra_report/Images/scripts/render_sim_collage.py --times 0 2 4 --target-deg 20
          --no-start-ring --no-trail --save-panels: no swum-path trail on the stills)
Output:  out/architecture_fig2.svg (+ .pdf, .png)
Run via `make figures`, or from .gen/figures: python fig2_architecture.py [--no-pdf] [--no-png]
The PDF export needs Inkscape or Chrome/Edge on the PATH.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _svg_pipeline as tp                  # noqa: E402
import _svg_curriculum as cf                # noqa: E402

from _paths import STATIC as _STATIC, OUT as _OUT
STATIC = _STATIC                                   # renders, stills, montage
OUT = _OUT                                         # generated figures, gitignored
FILL_TG = getattr(tp, "FILL_TG", None) or getattr(tp, "FILL_CPG")   # the helper renamed this box tint
DEFAULT_OUT = OUT / "architecture_fig2.svg"

W = 177.8                    # \textwidth in mm; placed unscaled with width=\textwidth
FS = cf.FS                   # 7 pt
FS2 = 6.0 / 72.27 * 25.4     # 6 pt secondary annotations
INK, INK_SOFT = tp.INK, tp.INK_SOFT
TIP = 0.7

text, eq, rect, line, path, image, icon, group = (tp.text, tp.eq, tp.rect, tp.line, tp.path,
                                                   tp.image, tp.icon, tp.group)
f = tp.f


# ------------------------------------------------------------------------------------------
# pieces
# ------------------------------------------------------------------------------------------

def network(x, y, w, h, layers, in_label, out_label, fs=FS2):
    xs = np.linspace(x, x + w, len(layers))
    spacing = min(h / max(layers), 2.0)
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
        fill = tp.COL_IN if i == 0 else tp.COL_OUT if i == len(cols) - 1 else tp.COL_HID
        for (cx, cy) in col:
            out.append(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="0.55" fill="{fill}" '
                       f'stroke="{INK}" stroke-width="0.12"/>')
    out.append(eq(x - 1.3, y + h / 2 + 0.7, in_label, fs=fs, anchor="end"))
    out.append(eq(x + w + 1.3, y + h / 2 + 0.7, out_label, fs=fs, anchor="start"))
    return "\n    ".join(out)


def cpg_waveform(px0, py0, pw, ph):
    """theta_m(t) for a step in the commanded amplitude at t = 1 s, first-order filtered
    (tau = 0.1 s); the filtered (orange) and commanded (grey dashed) envelopes are drawn too."""
    t = np.linspace(0, 2.6, 600)
    f_hz, tau, theta_c = 1.15, 0.1, 0.12
    k_des = np.where(t < 1.0, 0.45, 0.9)
    k = np.where(t < 1.0, 0.45, 0.9 - 0.45 * np.exp(-(t - 1.0) / tau))
    phi = 2 * np.pi * f_hz * t
    theta = theta_c + k * (np.cos(phi) + 0.25 * np.sin(2 * phi))
    lo, hi = -1.5, 1.6

    def X(tt): return px0 + (tt / t[-1]) * pw
    def Y(v): return py0 + ph - (v - lo) / (hi - lo) * ph
    def poly(vals): return "M " + " L ".join(f"{X(tt):.2f},{Y(v):.2f}" for tt, v in zip(t, vals))

    return "\n    ".join([
        line(px0, Y(theta_c), px0 + pw, Y(theta_c), stroke=tp.COL_DES, sw=0.15, dash="0.6 0.5", marker=False),
        line(px0, py0 + ph, px0 + pw + 1.5, py0 + ph, stroke=INK, sw=0.18),
        eq(px0 + pw + 2.3, py0 + ph + 0.8, "t", fs=FS),
        eq(px0 - 0.7, Y(theta_c) + 0.7, "θ_{c}", fs=FS2, anchor="end", fill=INK_SOFT),
        path(poly(theta_c + k_des), stroke=tp.COL_DES, sw=0.18, dash="0.8 0.5"),
        path(poly(theta_c - k_des), stroke=tp.COL_DES, sw=0.18, dash="0.8 0.5"),
        path(poly(theta_c + k), stroke=tp.COL_FILT, sw=0.24),
        path(poly(theta_c - k), stroke=tp.COL_FILT, sw=0.24),
        path(poly(theta), stroke=tp.COL_SIGNAL, sw=0.34),
        line(X(1.0), py0 + ph, X(1.0), py0 + 0.4, stroke=INK_SOFT, sw=0.12, dash="0.4 0.4", marker=False),
        eq(px0 + pw + 1.3, Y(theta[-1]) + 2.4, "θ_{m}", fs=FS, fill=tp.COL_SIGNAL),
        # legend (top-left inside the plot)
        line(X(0.02), Y(hi) + 1.0, X(0.02) + 3.0, Y(hi) + 1.0, stroke=tp.COL_FILT, sw=0.3, marker=False),
        eq(X(0.02) + 3.8, Y(hi) + 1.75, "\\b{u}", fs=FS2, fill=INK_SOFT),
        line(X(0.02) + 7.4, Y(hi) + 1.0, X(0.02) + 10.4, Y(hi) + 1.0, stroke=tp.COL_DES, sw=0.24,
             dash="0.8 0.5", marker=False),
        eq(X(0.02) + 11.2, Y(hi) + 1.75, "\\b{u}*", fs=FS2, fill=INK_SOFT),
    ])


def load_fish(path: Path, width_px: int = 1000):
    """Twin render on white -> white faded to alpha (self-contained: the shared module's
    loaders change signature from time to time)."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).copy()
    a[..., 3] = tp.white_to_alpha(a)
    im = Image.fromarray(a)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS)


def load_panel(path: Path, aspect: float, width_px: int = 900):
    """Overhead still cropped to `aspect`, a band centred on the camera's centre row."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    box = tp.crop_box((im.height, im.width), 40, im.width - 40, aspect, y_center=im.height // 2)
    im = im.crop(box)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS)


def target_glyph(cx, cy, R, theta, r, color):
    k = R / 4.0
    r_in, r_out = r[0] * k, r[1] * k
    return [f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(R)}" fill="none" stroke="{INK_SOFT}" '
            f'stroke-width="0.15" stroke-dasharray="0.45 0.45"/>',
            cf.path(cf.sector_path(cx, cy, r_in, r_out, theta), stroke=color, sw=0.2, fill=color,
                    extra='fill-opacity="0.38" fill-rule="evenodd"'),
            cf.fish_glyph(cx, cy, 2.0)]


# ------------------------------------------------------------------------------------------
# figure
# ------------------------------------------------------------------------------------------

def build_svg(fish_uri, panel_uris, panel_aspect, panel_times) -> tuple[str, float]:
    PH = 66.0                                   # bottom edge of the pipeline band
    M = 0.4

    SIM = dict(x=M, y=1.0, w=61.5, h=PH - 1.0)
    CPG = dict(x=70.0, y=1.0, w=54.0, h=32.0)
    REW = dict(x=70.0, y=49.5, w=54.0, h=PH - 49.5)
    SAC = dict(x=133.0, y=1.0, w=W - M - 133.0, h=PH - 1.0)
    sub_w = SAC["w"] - 4 - 4.5
    ACT = dict(x=SAC["x"] + 2, y=9.5, w=sub_w, h=15.0)
    CRI = dict(x=SAC["x"] + 2, y=27.0, w=sub_w, h=15.0)
    BUF = dict(x=SAC["x"] + 2, y=45.0, w=sub_w, h=12.0)

    sim_r, cpg_r, rew_r, sac_r = (SIM["x"] + SIM["w"], CPG["x"] + CPG["w"],
                                  REW["x"] + REW["w"], SAC["x"] + SAC["w"])
    y_act = 7.2                                 # action lane (over the actor, into the CPG)

    # ---- simulation ----------------------------------------------------------------------
    # twin render, then the 5 x 3 fluid-coefficient matrix under it, with a brace from each
    # matrix column up to the fish zone it belongs to (head/body, tail, caudal fin)
    fish_w = 46.0
    fish_h = fish_w * 760 / 1347
    fish_x = SIM["x"] + (SIM["w"] - fish_w) / 2
    fish_y = 5.8
    fish_b = fish_y + fish_h
    # zone extents as fractions of the render width (blue ellipsoid / hinges+green / orange fin)
    zones = [("Head", 0.08, 0.53), ("Tail", 0.54, 0.76), ("Fin", 0.77, 0.94)]
    y_br = fish_b + 0.7                          # underbrace baseline
    y_hdr = y_br + 4.6                           # matrix header baseline
    ROW = 2.2
    rows = ["blunt drag", "slender drag", "angular drag", "Kutta lift", "Magnus lift"]
    subs = ["b", "s", "a", "K", "M"]
    col_x = [SIM["x"] + 30.0, SIM["x"] + 38.0, SIM["x"] + 46.0]    # even matrix columns
    y_rows = [y_hdr + ROW * (i + 1) for i in range(len(rows))]
    mat = []
    for (name, f0, f1), cx in zip(zones, col_x):
        x0, x1 = fish_x + f0 * fish_w, fish_x + f1 * fish_w
        xm = (x0 + x1) / 2
        # underbrace: ticks up towards the fish, stem down, elbow across to the matrix column
        mat.append(path(f"M {f(x0)},{f(y_br - 1.0)} V {f(y_br)} H {f(x1)} V {f(y_br - 1.0)}",
                        stroke=INK_SOFT, sw=0.22))
        mat.append(path(f"M {f(xm)},{f(y_br)} V {f(y_br + 1.6)} H {f(cx)} V {f(y_hdr - 2.0)}",
                        stroke=INK_SOFT, sw=0.22))
        mat.append(text(cx, y_hdr, name, fs=FS2, anchor="middle", weight="bold"))
        for yr, sb in zip(y_rows, subs):
            mat.append(eq(cx, yr, f"c_{{{sb},{name[0]}}}", fs=FS2, anchor="middle"))
    for yr, name in zip(y_rows, rows):
        mat.append(text(col_x[0] - 4.6, yr, name, fs=FS2, anchor="end", fill=INK_SOFT))
    mat.append(text(SIM["x"] + 2.0, y_hdr, "Fluid coefficients (15)", fs=FS2, weight="bold"))
    # matrix brackets
    bx0, bx1 = col_x[0] - 3.2, col_x[-1] + 3.2
    by0, by1 = y_hdr + 0.9, y_rows[-1] + 0.7
    mat.append(path(f"M {f(bx0 + 0.9)},{f(by0)} H {f(bx0)} V {f(by1)} H {f(bx0 + 0.9)}", sw=0.25))
    mat.append(path(f"M {f(bx1 - 0.9)},{f(by0)} H {f(bx1)} V {f(by1)} H {f(bx1 - 0.9)}", sw=0.25))

    n = len(panel_uris)
    gap = 1.5
    pan_w = (SIM["w"] - 4.0 - gap * (n - 1)) / n
    pan_h = pan_w / panel_aspect
    pan_y = by1 + 3.6
    panels = []
    for i, (uri, tl) in enumerate(zip(panel_uris, panel_times)):
        px = SIM["x"] + 2.0 + i * (pan_w + gap)
        panels += [text(px + pan_w / 2, pan_y - 0.9, tl, fs=FS2, anchor="middle", fill=INK_SOFT),
                   image(px, pan_y, pan_w, pan_h, uri),
                   rect(px, pan_y, pan_w, pan_h, "none", rx=0.3, stroke=INK_SOFT, sw=0.15)]

    legend_y = pan_y + pan_h + 2.4
    legend_cx = SIM["x"] + SIM["w"] / 2
    sim = group(
        "Simulation",
        rect(**SIM, fill=tp.FILL_SIM),
        text(SIM["x"] + SIM["w"] / 2, 4.7, "MuJoCo Simulation Environment", fs=FS, anchor="middle", weight="bold"),
        image(fish_x, fish_y, fish_w, fish_h, fish_uri),
        *mat,
        *panels,
        # legend: target only (the stills are rendered without the swum-path trail)
        f'<circle cx="{f(legend_cx - 3.4)}" cy="{f(legend_y)}" r="0.8" fill="#19BF19" stroke="{INK}" stroke-width="0.12"/>',
        text(legend_cx - 1.8, legend_y + 0.75, "target", fs=FS2, fill=INK_SOFT),
        text(legend_cx, PH - 2.0, "Per-episode DR: mass, CoM, tail k & c, fluid coefficients ±20 %",
             fs=FS2, anchor="middle", fill=INK_SOFT),
    )

    # ---- trajectory generator (CPG) ------------------------------------------------------
    ch_y, ch_h = 8.0, 6.6
    ch_cy = ch_y + ch_h / 2
    OSC = dict(x=CPG["x"] + 2.0, y=ch_y, w=26.0, h=ch_h)
    LPF = dict(x=cpg_r - 3.0 - 19.5, y=ch_y, w=19.5, h=ch_h)
    osc_r, lpf_r = OSC["x"] + OSC["w"], LPF["x"] + LPF["w"]
    plot = dict(px0=CPG["x"] + 5.5, py0=CPG["y"] + 18.6, pw=CPG["w"] - 11.0, ph=10.0)
    cpg = group(
        "CPG",
        rect(**CPG, fill=FILL_TG),
        text(CPG["x"] + CPG["w"] / 2, 4.7, "Trajectory Generator (CPG) + Low-Pass Filter", fs=FS,
             anchor="middle", weight="bold"),
        rect(**LPF, fill=tp.FILL_SUB, rx=0.8),
        text(LPF["x"] + LPF["w"] / 2, ch_cy - 0.4, "Low-pass filter", fs=FS, anchor="middle", weight="bold"),
        eq(LPF["x"] + LPF["w"] / 2, ch_cy + 2.3, "\\b{u}\\r{ tracks }\\b{u}*\\r{,  }τ\\r{ = 0.1 s}",
           fs=FS2, anchor="middle", fill=INK_SOFT),
        rect(**OSC, fill=tp.FILL_SUB, rx=0.8),
        text(OSC["x"] + OSC["w"] / 2, ch_cy - 0.4, "Harmonic oscillator", fs=FS, anchor="middle", weight="bold"),
        eq(OSC["x"] + OSC["w"] / 2, ch_cy + 2.3, "θ_{m}(φ; \\b{u})\\r{,  }\\r{d}φ\\r{/d}t = ω",
           fs=FS2, anchor="middle", fill=INK_SOFT),
        # u* comes in on the action lane, drops to the chain height, enters the filter
        path(f"M {f(cpg_r - 0.15)},{f(y_act)} H {f(cpg_r - 1.5)} V {f(ch_cy)} H {f(lpf_r + TIP)}", marker=True),
        line(LPF["x"], ch_cy, osc_r + TIP, ch_cy),
        eq((LPF["x"] + osc_r) / 2, ch_cy - 1.3, "\\b{u}", fs=FS, anchor="middle"),
        cpg_waveform(**plot),
    )

    # ---- reward ---------------------------------------------------------------------------
    g_x0, g_w = REW["x"] + 1.5, 14.0
    g_cy = REW["y"] + REW["h"] / 2
    tgt = (g_x0 + g_w - 3.0, g_cy)
    r_arr = 2.3
    fish_c = (g_x0 + 2.6, g_cy)
    txt_cx = (g_x0 + g_w + 1.0 + rew_r - 1.0) / 2
    rew = group(
        "Reward",
        rect(**REW, fill=tp.FILL_REW),
        f'<circle cx="{f(tgt[0])}" cy="{f(tgt[1])}" r="{f(r_arr)}" fill="#009E73" fill-opacity="0.15" '
        f'stroke="#009E73" stroke-width="0.2" stroke-dasharray="0.5 0.4"/>',
        f'<circle cx="{f(tgt[0])}" cy="{f(tgt[1])}" r="0.7" fill="#009E73" stroke="{INK}" stroke-width="0.12"/>',
        text(tgt[0], tgt[1] + r_arr + 2.1, "5 cm", fs=FS2, anchor="middle", fill="#009E73"),
        tp.fish_top(fish_c[0], fish_c[1], 3.4, heading_deg=90),
        line(fish_c[0] + 2.1, g_cy, tgt[0] - r_arr - 0.5, g_cy, sw=0.22, dash="0.5 0.4", marker=True),
        eq((fish_c[0] + 2.1 + tgt[0] - r_arr) / 2, g_cy - 1.0, "d", fs=FS, anchor="middle"),
        text(txt_cx, REW["y"] + 3.5, "Reward", fs=FS, anchor="middle", weight="bold"),
        eq(txt_cx, REW["y"] + 7.4, "r_{t} = −d − \\r{0.1}‖\\b{a}_{t}‖ + \\r{1000}·\\b{1}[d < \\r{5 cm}]", fs=FS * 0.92,
           anchor="middle"),
        text(txt_cx, REW["y"] + 10.6, "d: exact simulator distance (unobserved)", fs=FS2, anchor="middle", fill=INK_SOFT),
    )

    # ---- SAC ------------------------------------------------------------------------------
    net_w, net_h = 14.0, 7.0
    net_x = ACT["x"] + 5.5
    act_r, cri_r, buf_r = ACT["x"] + ACT["w"], CRI["x"] + CRI["w"], BUF["x"] + BUF["w"]
    bus_x = sac_r - 2.3
    q_node = (net_x + net_w, CRI["y"] + 4.0 + net_h / 2)
    y_act_mid = ACT["y"] + 4.0 + net_h / 2
    sac = group(
        "SAC",
        rect(**SAC, fill=tp.FILL_SAC),
        text(SAC["x"] + SAC["w"] / 2, 4.7, "Soft Actor-Critic (SAC)", fs=FS, anchor="middle", weight="bold"),
        rect(**ACT, fill=tp.FILL_SUB, rx=1.0),
        eq(ACT["x"] + 1.6, ACT["y"] + 3.0, "\\b{Actor  }π(\\b{a} | \\b{s})", fs=FS),
        text(act_r - 1.6, ACT["y"] + 3.0, "MLP 3 × 356", fs=FS2, anchor="end", fill=INK_SOFT),
        network(net_x, ACT["y"] + 4.0, net_w, net_h, [5, 6, 6, 3], "\\b{s}", "\\b{a}"),
        eq(ACT["x"] + ACT["w"] / 2, ACT["y"] + 13.6, "\\b{a} = \\b{u}* = (ω, θ_{c}, a_{1}, b_{1}, a_{2}, b_{2})",
           fs=FS2, anchor="middle"),
        rect(**CRI, fill=tp.FILL_SUB, rx=1.0),
        eq(CRI["x"] + 1.6, CRI["y"] + 3.0, "\\b{Critic  }Q(\\b{s}, \\b{a})", fs=FS),
        text(cri_r - 1.6, CRI["y"] + 3.0, "MLP 3 × 356", fs=FS2, anchor="end", fill=INK_SOFT),
        network(net_x, CRI["y"] + 4.0, net_w, net_h, [6, 6, 6, 1], "\\b{s},\\b{a}", "Q"),
        path(f"M {f(q_node[0] + 3.4)},{f(q_node[1] - 0.4)} C {f(q_node[0] + 9)},{f(q_node[1] - 3.2)} "
             f"{f(q_node[0] + 13)},{f(q_node[1] + 4.8)} {f(net_x + net_w * 0.55)},{f(q_node[1] + 5.2)}",
             sw=0.26, dash="0.7 0.5", marker=True),
        eq(q_node[0] + 11.0, q_node[1] - 1.2, "\\r{TD error }δ", fs=FS2, anchor="middle", fill=INK_SOFT),
        text(q_node[0] + 11.0, q_node[1] + 7.0, "updates Q", fs=FS2, anchor="middle", fill=INK_SOFT),
        rect(**BUF, fill=tp.FILL_SUB, rx=1.0),
        icon("database", BUF["x"] + 1.6, BUF["y"] + 3.0, 5.0, color=INK_SOFT),
        text(BUF["x"] + 8.6, BUF["y"] + 3.6, "Replay Buffer", fs=FS, weight="bold"),
        eq(BUF["x"] + 8.6, BUF["y"] + 6.9, "(\\b{s}_{t}, \\b{a}_{t}, r_{t}, \\b{s}_{t+1})", fs=FS2),
        text(BUF["x"] + 8.6, BUF["y"] + 9.6, "kept across curriculum stages", fs=FS2, fill=INK_SOFT),
        # action bus: actor -> buffer (down) and -> CPG (up, along the free lane)
        path(f"M {f(act_r)},{f(y_act_mid)} H {f(bus_x)} V {f(BUF['y'] + 4.5)} H {f(buf_r + TIP)}", marker=True),
        path(f"M {f(bus_x)},{f(y_act_mid)} V {f(y_act)} H {f(cpg_r + TIP)}", marker=True),
        eq(bus_x - 0.9, CRI["y"] - 1.0, "\\b{a}_{t}", fs=FS, anchor="end"),
        line(BUF["x"] + BUF["w"] * 0.3, BUF["y"], BUF["x"] + BUF["w"] * 0.3, CRI["y"] + CRI["h"] + TIP, sw=0.28),
        text(BUF["x"] + BUF["w"] * 0.3 + 1.0, CRI["y"] + CRI["h"] + 2.0, "minibatch", fs=FS2, fill=INK_SOFT),
        line(CRI["x"] + CRI["w"] * 0.3, CRI["y"], CRI["x"] + CRI["w"] * 0.3, ACT["y"] + ACT["h"] + TIP, sw=0.28,
             dash="0.7 0.5"),
        text(CRI["x"] + CRI["w"] * 0.3 + 1.0, CRI["y"] - 1.0, "policy gradient", fs=FS2, fill=INK_SOFT),
    )

    # ---- state strip + inter-block arrows --------------------------------------------------
    y_state = 36.4
    fan_x = SAC["x"] - 3.2
    y_rew = REW["y"] + REW["h"] / 2
    obs = [("engine", 14.0, "Motor angle", "& velocity (3)"),
           ("rotate-3d", 10.5, "IMU", "yaw rate (1)"),
           ("wave-sine", 12.0, "CPG phase", "cos, sin (2)"),
           ("camera", 15.5, "3 camera frames", "pose + age (12)"),
           ("history", 15.0, "Last 3 actions", "(ω, θc) (6)")]
    strip_x0, strip_x1 = sim_r + 1.0, fan_x - 1.0
    scale = (strip_x1 - strip_x0) / sum(w for _, w, _, _ in obs)
    ico, y0 = 3.8, y_state + 1.9
    chips, cx = [], strip_x0
    for name, w, l1, l2 in obs:
        w *= scale
        chips += [icon(name, cx + w / 2 - ico / 2, y0, ico, color=INK),
                  text(cx + w / 2, y0 + ico + 2.3, l1, fs=FS2, anchor="middle"),
                  text(cx + w / 2, y0 + ico + 4.6, l2, fs=FS2, anchor="middle", fill=INK_SOFT)]
        cx += w

    arrows = group(
        "Arrows",
        # CPG -> sim: theta_m setpoint, straight out of the oscillator
        line(OSC["x"], ch_cy, sim_r + TIP, ch_cy),
        eq((sim_r + CPG["x"]) / 2, ch_cy - 1.4, "θ_{m}", fs=FS, anchor="middle"),
        text((sim_r + CPG["x"]) / 2, ch_cy + 3.0, "50 Hz", fs=FS2, anchor="middle", fill=INK_SOFT),
        # SAC -> CPG: action lane label
        eq((cpg_r + SAC["x"]) / 2, y_act - 1.4, "\\b{a}_{t} = \\b{u}*", fs=FS, anchor="middle"),
        text((cpg_r + SAC["x"]) / 2, y_act + 3.0, "6-D", fs=FS2, anchor="middle", fill=INK_SOFT),
        # sim -> SAC: state, fanning out to the actor and the buffer
        line(sim_r, y_state, fan_x, y_state, marker=False),
        f'<circle cx="{f(fan_x)}" cy="{f(y_state)}" r="0.45" fill="{INK}"/>',
        path(f"M {f(fan_x)},{f(y_state)} V {f(y_act_mid)} H {f(SAC['x'] + TIP)}", marker=True),
        path(f"M {f(fan_x)},{f(y_state)} V {f(BUF['y'] + 3.0)} H {f(SAC['x'] + TIP)}", marker=True),
        eq(sim_r + 2.0, y_state - 1.3, "\\b{State  s}_{t}", fs=FS, anchor="start"),
        text(fan_x - 1.5, y_state - 1.3, "24-D, hardware-measurable only", fs=FS2, anchor="end", fill=INK_SOFT),
        *chips,
        # sim -> reward -> buffer
        line(sim_r, y_rew, REW["x"] - TIP, y_rew),
        eq((sim_r + REW["x"]) / 2, y_rew - 1.2, "d", fs=FS, anchor="middle"),
        path(f"M {f(rew_r)},{f(y_rew)} H {f(SAC['x'] - 1.6)} V {f(BUF['y'] + 7.0)} H {f(SAC['x'] + TIP)}", marker=True),
        eq(rew_r + 4.2, y_rew - 1.2, "r_{t}", fs=FS, anchor="middle"),
    )

    # ---- curriculum strip -------------------------------------------------------------------
    y_band0 = PH                                   # gap band: PH .. y_head
    band_h = 7.6
    y_head = y_band0 + band_h
    head_h = 4.8
    y_box = y_head + head_h + 2.2
    box_h = 15.0
    H = y_box + box_h + M

    w1, w3, g, G = 14.8, 14.8, 2.9, 4.2
    per_phase = [sum(1 for s in cf.STAGES if s["phase"] == i) for i in range(3)]
    widths = [w1 if s["phase"] < 2 else w3 for s in cf.STAGES]
    xs, x = [], M
    for k, s in enumerate(cf.STAGES):
        if k > 0:
            x += widths[k - 1] + (G if s["phase"] != cf.STAGES[k - 1]["phase"] else g)
        xs.append(x)
    lr = cf.lr_schedule(cf.STAGES)

    strip = []
    # headers
    for i, ph in enumerate(cf.PHASES):
        ks = [k for k, s in enumerate(cf.STAGES) if s["phase"] == i]
        x0, x1 = xs[ks[0]], xs[ks[-1]] + widths[ks[-1]]
        strip.append(rect(x0, y_head, x1 - x0, head_h, ph["accent"], stroke="none", rx=1.2))
        strip.append(text((x0 + x1) / 2, y_head + head_h / 2 + 0.85, f"{ph['title']}  ·  {ph['name']}",
                          fs=FS, anchor="middle", weight="bold", fill="#FFFFFF"))
    # stage boxes
    R = 4.2
    titles = {0: ["Regular", "Swimming"], 1: ["Encoder &", "IMU Noise"], 2: ["AprilTag", "Noise"],
              3: ["Initial", "Velocity DR"], 4: ["Body &", "Tail DR"], 5: ["Fluid", "Coeff. DR"]}
    for k, s in enumerate(cf.STAGES):
        ph = cf.PHASES[s["phase"]]
        x, w, y = xs[k], widths[k], y_box
        el = [rect(x, y, w, box_h, ph["tint"], rx=1.2,
                   stroke=cf.DEPLOYED_EDGE if s.get("deployed") else INK_SOFT,
                   sw=0.55 if s.get("deployed") else tp.EDGE)]
        el.append(text(x + 1.3, y + 2.7, f"{k + 1}", fs=FS, weight="bold", fill=ph["accent"]))
        el.append(text(x + w - 1.2, y + 2.7, "LR " + cf.fmt_lr(lr[k]), fs=FS2, anchor="end", fill=INK_SOFT))
        if "theta" in s:
            cx, cy = x + 1.4 + R, y + 3.4 + R
            el += target_glyph(cx, cy, R, s["theta"], s["r"], ph["accent"])
            el.append(text(x + w - 1.2, y + box_h - 1.5, f"±{s['theta']}°", fs=FS, anchor="end", weight="bold"))
        else:
            if "fixed" in s:
                Rf = 2.9
                lab = f"±{s['fixed']['theta']}°"
                unit_w = 2 * Rf + 0.8 + len(lab) * FS * 0.5
                cx, cy = x + w / 2 - unit_w / 2 + Rf, y + 3.2 + Rf
                el += target_glyph(cx, cy, Rf, s["fixed"]["theta"], s["fixed"]["r"], ph["accent"])
                el.append(text(cx + Rf + 0.8, cy + 0.85, lab, fs=FS, weight="bold"))
            else:
                ic = 5.0
                el.append(icon(s["icon"], x + w / 2 - ic / 2, y + 3.4, ic, color=ph["accent"]))
            t1, t2 = titles[k]
            el.append(text(x + w / 2, y + box_h - 4.3, t1, fs=FS, anchor="middle", weight="bold"))
            el.append(text(x + w / 2, y + box_h - 1.5, t2, fs=FS, anchor="middle", weight="bold"))
        # promotion arrow to the next stage
        if k < len(cf.STAGES) - 1:
            nx = xs[k + 1]
            same = cf.STAGES[k + 1]["phase"] == s["phase"]
            el.append(line(x + w + 0.3, y + box_h / 2, nx - 0.3 - TIP, y + box_h / 2,
                           stroke=ph["accent"] if same else INK, sw=0.35 if same else 0.4, marker=False)
                      .replace("/>", f' marker-end="url(#{"arrow_" + str(s["phase"]) if same else "arrow"})"/>'))
        strip.append(group(f"Stage {k + 1} {s['name']}", *el))
    curriculum = group("Curriculum", *strip)

    # gap band: title + two arrows up into the pipeline
    xa, xb = 31.0, 155.0
    band = group(
        "Curriculum links",
        text(W / 2, y_band0 + 3.0, "Training Curriculum", fs=FS, anchor="middle", weight="bold"),
        text(W / 2, y_band0 + 5.9, "promotion at 95 % rolling target acquisition", fs=FS2, anchor="middle",
             fill=INK_SOFT),
        line(xa, y_head - 0.3, xa, PH + TIP, sw=0.35),
        text(xa + 1.4, y_band0 + band_h / 2 + 0.75, "stage sets noise, randomization, targets",
             fs=FS2, fill=INK_SOFT),
        line(xb, y_head - 0.3, xb, PH + TIP, sw=0.35),
        text(xb - 1.4, y_band0 + band_h / 2 + 0.75, "stage sets learning rate", fs=FS2, anchor="end",
             fill=INK_SOFT),
    )

    def marker(mid, color):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4.5" '
                'markerHeight="4.5" orient="auto-start-reverse">'
                f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{color}"/></marker>')
    defs = ("<defs>" + marker("arrow", INK)
            + "".join(marker(f"arrow_{i}", ph["accent"]) for i, ph in enumerate(cf.PHASES)) + "</defs>")

    body = "\n  ".join([sim, cpg, rew, sac, arrows, curriculum, band])
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by figures/fig2_architecture.py - edit freely in Inkscape. -->
<svg
   width="{f(W)}mm" height="{f(H)}mm" viewBox="0 0 {f(W)} {f(H)}"
   version="1.1" id="architecture_fig2"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   font-family="{tp.FONT}">
  <sodipodi:namedview id="namedview" pagecolor="#ffffff" bordercolor="#000000"
     borderopacity="0.25" inkscape:document-units="mm" inkscape:pagecheckerboard="0"/>
  {defs}
  <g id="layer1" inkscape:groupmode="layer" inkscape:label="Figure">
  {body}
  </g>
</svg>
'''
    return svg, H


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--fish", default=str(STATIC / "40_hull_hinges_fluid.png"))
    p.add_argument("--panels", nargs="+",
                   default=sorted(str(q) for q in STATIC.glob("sim_collage_panel*.png")))
    p.add_argument("--panel-aspect", type=float, default=2.65)
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)
    if len(args.panels) < 2:
        sys.exit("need draft/sim_collage_panel*.png (render_sim_collage.py --save-panels)")

    fish_uri = tp.png_data_uri(load_fish(Path(args.fish)))
    panel_uris = [tp.png_data_uri(load_panel(Path(q), args.panel_aspect)) for q in args.panels]
    times = []
    for q in args.panels:
        tval = float(Path(q).stem.rsplit("_t", 1)[1].rstrip("s"))
        times.append(f"t = {tval:.1f} s (reached)" if q == args.panels[-1] else f"t = {tval:g} s")

    svg, H = build_svg(fish_uri, panel_uris, args.panel_aspect, times)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Saved {out}  ({W:.1f} x {H:.1f} mm, {out.stat().st_size / 1024:.0f} kB)")
    if not args.no_pdf:
        cf.export_pdf(out, out.with_suffix(".pdf"), W, H)
    if not args.no_png:
        cf.export_png(out, out.with_suffix(".png"), W, H, px_per_mm=10.0)


if __name__ == "__main__":
    main()
