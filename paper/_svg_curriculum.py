"""Generate the one-column "training curriculum" figure as an editable SVG (+ PDF).

Layout, following the hand sketch in icra_report/Figure plans.jpeg ("Figure - Curriculum"):

   [Sensor Noise]          ->  [Domain Randomization]        ->  [Target Widening]
     1 clean sensors (glyph)     4 initial velocity DR             7  +-60 deg  (polar sector)
     2 encoder & IMU noise       5 body & tail DR                  8  +-85 deg
     3 AprilTag noise            6 fluid coefficient DR            9  +-120 deg  (deployed)
                                                                  10  +-180 deg

Three columns, one per phase; the stages of a phase stack downwards inside its column.
Phase 3 draws the target distribution of each stage as an annular sector (fish at the centre,
heading up): half-angle theta_max and radial band r_min..r_max, all at one common scale.

Two variants (--variant):
  lean  (default) compact two-line phase headers, stage number + icon + title, and the relative
        learning rate top-right; numeric ranges and sigma values are left to the running text.
        All text 7 pt; only the "Phase n" header line is bold.
  full  the earlier, denser draft: two-line headers ("Phase n" + name), per-stage detail
        lines (noise sigmas, DR ranges, radial bands) and the relative learning rate in the
        top-right corner of every box (x0.3 per stage inside a phase, warm-restarted to 1x at
        each phase boundary). Kept so the denser layout can be brought back.

The stage list mirrors rl/environments/env_sofi_full_curriculum.py (CURRICULUM_STAGES) and
rl/train_rl_curriculum.py (LR_SCALE = 0.3). Icons are Tabler Icons (MIT), see icons/LICENSE.md.
Every element is a plain SVG element (real text), so the file can be edited in Inkscape.

Output:
  icra_report/Images/draft/curriculum.svg        lean  (+ .pdf, and a .png preview)
  icra_report/Images/draft/curriculum_full.svg   full  (+ .pdf, .png)

Usage (plain Python, no third-party packages):
  python icra_report/Images/scripts/make_curriculum_fig.py [--variant lean|full|both]
                                                           [--out PATH] [--no-pdf] [--no-png]
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

# Helper module for fig1_overview.py / fig2_architecture.py (page geometry, curriculum stage
# list, target-sector glyph, SVG -> PDF/PNG export). Its own CLI builds the older stand-alone
# curriculum figure.
from _paths import STATIC as IMG_DIR, OUT as _OUT
DEFAULT_OUT = {"lean": _OUT / "curriculum.svg",
               "full": _OUT / "curriculum_full.svg"}
ICON_DIR = Path(__file__).resolve().parent / "icons"

# -- page (mm). One IEEE column = 245.718 pt (paper_style.COL_PT) = 86.4 mm ------------------
W = 245.718 / 72.27 * 25.4          # = \columnwidth, so the SVG is placed unscaled
FS = 9.0 / 72.27 * 25.4             # 9 pt: the paper's label size (paper_style.LABEL_PT)
PRINT_SCALE = 1.0                   # page / design: the page is the design (see _svg_pipeline)

# the same family as the plots (paper_style: Times through LaTeX); Chrome on Windows sets
# Times New Roman, on Linux Liberation Serif, and embeds it as TrueType
FONT = "'Times New Roman', Times, 'Nimbus Roman', 'Liberation Serif', serif"
INK = "#2B3440"
INK_SOFT = "#5B6573"
EDGE = 0.25
ARROW_W = 0.3

# Okabe-Ito accents (paper_style.CB) + very light tints for the boxes
PHASES = [
    dict(title="Phase 1", name="Sensor Noise", accent="#0072B2", tint="#EEF3FA"),
    dict(title="Phase 2", name="Domain Randomization", accent="#D55E00", tint="#FFF4E6"),
    dict(title="Phase 3", name="Target Widening", accent="#009E73", tint="#EEF7EC"),
]
DEPLOYED_EDGE = PHASES[2]["accent"]  # outline of the last stage the deployed policy completed
# Arrowheads: the marker is anchored (refX) well behind its tip so the line's blunt end is
# hidden inside the head; the tip then extends TIP mm past the line's endpoint.
TIP = 0.7

# Stage list = env CURRICULUM_STAGES, in order. theta/r only for the target-widening phase.
# DR ranges: rl/environments/env_sofi_phase1_2_physicsDR.py; sensor stages: _FULL_SENSOR.
STAGES = [
    # detail: one item per line, each starting with a capital; sigma values are the 1-sigma
    # noise levels in rl/models/sensor_model.py
    # `fixed`: the env default target band (r 1-2 m, +-30 deg) that holds through stages 1-6,
    # drawn once in stage 1 at the same scale as the phase-3 glyphs
    dict(name="clean+delay", phase=0, icon="fish", fixed=dict(theta=30, r=(1.0, 2.0)),
         title="Regular Swimming", detail="Clean sensors\nVision delay 10–50 ms"),
    dict(name="enc+imu+delay", phase=0, icon="compass",
         title="Encoder & IMU Noise",
         detail="Encoder σ 0.001 rad\nEnc. rate σ 0.01 rad/s\nIMU σ 0.006 rad/s"),
    dict(name="full_noise", phase=0, icon="camera",
         title="AprilTag Noise",
         detail="Pose σ 2 mm, 1.5°\nVelocity σ 0.02 m/s\n10 % frame drops"),
    dict(name="init_vel", phase=1, icon="wind", reset_lr=True,
         title="Initial Velocity DR", detail="Linear ±0.15 m/s\nYaw rate ±0.3 rad/s"),
    dict(name="physical", phase=1, icon="weight",
         title="Body & Tail DR", detail="Mass, MoI ±5 %\nCoM ±5 mm\nTail k, c ±15 %"),
    dict(name="full_dr", phase=1, icon="ripple",
         title="Fluid Coefficient DR", detail="15 coefficients ±20 %"),
    dict(name="fwd_wide", phase=2, reset_lr=True, theta=60, r=(0.5, 2.5)),
    dict(name="semi_lateral", phase=2, theta=85, r=(0.5, 2.5)),
    dict(name="lateral", phase=2, theta=120, r=(0.5, 2.5), deployed=True),
    dict(name="full_range", phase=2, theta=180, r=(0.5, 4.0)),
]
LR_SCALE = 0.3                       # rl/train_rl_curriculum.py

# Per-variant geometry / content switches (mm). The phase-3 box height h_p3 sets the common
# glyph scale; phases 1-2 (three boxes) are stretched so that every column spans the same
# height.
VARIANTS = {
    "lean": dict(head_h=7.2, head_two_line=True, h_p3=9.6, glyph_pad=2.0,
                 show_lr=True, lr_reset_tag=False, show_detail=False, show_range=False,
                 bold=False),
    "full": dict(head_h=7.8, head_two_line=True, h_p3=12.8, glyph_pad=5.2,
                 show_lr=True, lr_reset_tag=True, show_detail=True, show_range=True,
                 bold=True),
}
# No footer: the promotion rule, the lr convention and the meaning of the green outline are
# stated in the LaTeX caption.


# -- pure helpers (unit-tested) ---------------------------------------------------

def lr_schedule(stages, scale=LR_SCALE):
    """Learning rate relative to the initial one, per stage: x`scale` at every intra-phase
    promotion, reset to 1 at a stage flagged reset_lr (a phase boundary)."""
    out, lr = [], 1.0
    for i, s in enumerate(stages):
        if i == 0 or s.get("reset_lr"):
            lr = 1.0
        else:
            lr *= scale
        out.append(lr)
    return out


def fmt_lr(v: float) -> str:
    s = "1" if v == 1 else f"{v:.3g}".rstrip("0").rstrip(".")
    return s + "×"


def wrap_lines(s: str, max_chars: int) -> list[str]:
    """Greedy word wrap (a word longer than max_chars gets its own line). A newline in `s`
    forces a break."""
    if "\n" in s:
        return [ln for part in s.split("\n") for ln in wrap_lines(part, max_chars)]
    lines, cur = [], ""
    for w in s.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _fit(s: str, col_w: float, enabled: bool, pad: float = 1.6) -> str:
    """SVG attributes that condense `s` into `col_w`, or "" if it already fits (or is off).

    The same average-advance estimate wrap_lines() uses (~0.42 em regular in Times at this
    size); lengthAdjust="spacingAndGlyphs" then squeezes letterforms and tracking together,
    which at 9 pt reads as a slightly narrow cut rather than as damage.
    """
    avail = col_w - pad
    if not enabled or len(s) * FS * 0.42 <= avail:
        return ""
    return f'textLength="{f(avail)}" lengthAdjust="spacingAndGlyphs" '


def _pol(cx, cy, r, deg):
    """Point at radius r, `deg` degrees clockwise from straight up (-y)."""
    a = math.radians(deg)
    return cx + r * math.sin(a), cy - r * math.cos(a)


def sector_path(cx, cy, r_in, r_out, half_deg) -> str:
    """SVG path of the annular sector |angle| <= half_deg about 'up', radii r_in..r_out.
    A +-180 deg sector is emitted as two half-annuli (a single arc with coincident endpoints
    would vanish)."""
    def p(x, y):
        return f"{x:.3f},{y:.3f}"
    if half_deg >= 179.99:
        # full annulus: outer and inner circles as two arcs each (drawn with fill-rule evenodd)
        rings = []
        for r in (r_out, r_in):
            top, bot = _pol(cx, cy, r, 0), _pol(cx, cy, r, 180)
            rings.append(f"M {p(*top)} A {r:.3f} {r:.3f} 0 0 1 {p(*bot)} "
                         f"A {r:.3f} {r:.3f} 0 0 1 {p(*top)} Z")
        return " ".join(rings)
    large = 1 if 2 * half_deg > 180 else 0
    o0, o1 = _pol(cx, cy, r_out, -half_deg), _pol(cx, cy, r_out, half_deg)
    i0, i1 = _pol(cx, cy, r_in, -half_deg), _pol(cx, cy, r_in, half_deg)
    return (f"M {p(*o0)} A {r_out:.3f} {r_out:.3f} 0 {large} 1 {p(*o1)} "
            f"L {p(*i1)} A {r_in:.3f} {r_in:.3f} 0 {large} 0 {p(*i0)} Z")


# -- SVG primitives ----------------------------------------------------------------

def f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def text(x, y, s, fs=FS, anchor="start", weight="normal", style="normal", fill=INK, extra=""):
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{f(fs)}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" {extra}>{escape(s)}</text>')


def rect(x, y, w, h, fill, rx=1.2, stroke=INK, sw=EDGE, extra=""):
    return (f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{f(rx)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" {extra}/>')


def line(x1, y1, x2, y2, stroke=INK, sw=ARROW_W, dash="", marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" '
            f'stroke-width="{f(sw)}" stroke-linecap="butt"{d}{m}/>')


def path(d, stroke=INK, sw=ARROW_W, fill="none", dash="", extra=""):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da} {extra}/>')


def icon_paths(svg_text: str) -> list[str]:
    ds = re.findall(r'<path[^>]*\bd="([^"]+)"', svg_text)
    return [d for d in ds if d.replace(" ", "") not in ("M00h24v24H0z", "M0,0h24v24H0z")]


def icon(name, x, y, size, color=INK, sw=1.6):
    """24x24 Tabler outline icon, top-left at (x, y), `size` mm wide."""
    ds = icon_paths((ICON_DIR / f"{name}.svg").read_text(encoding="utf-8"))
    s = size / 24.0
    body = "".join(f'<path d="{d}"/>' for d in ds)
    return (f'<g transform="translate({f(x)},{f(y)}) scale({s:.4f})" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'inkscape:label="icon:{name}">{body}</g>')


def group(label, *children):
    gid = re.sub(r"[^a-z0-9]+", "_", label.lower())
    body = "\n    ".join(c for c in children if c)
    return f'<g id="{gid}" inkscape:label="{label}">\n    {body}\n  </g>'


def fish_glyph(cx, cy, length, color=INK):
    """Tiny top-view fish heading up: ellipse body + triangular tail."""
    b = length * 0.62
    w = length * 0.30
    d = (f"M {cx:.3f},{cy - b / 2:.3f} "
         f"C {cx + w / 2:.3f},{cy - b / 2:.3f} {cx + w / 2:.3f},{cy + b / 2:.3f} {cx:.3f},{cy + b / 2:.3f} "
         f"C {cx - w / 2:.3f},{cy + b / 2:.3f} {cx - w / 2:.3f},{cy - b / 2:.3f} {cx:.3f},{cy - b / 2:.3f} Z "
         f"M {cx:.3f},{cy + b / 2 - 0.1:.3f} L {cx - w * 0.75:.3f},{cy + length / 2:.3f} "
         f"L {cx + w * 0.75:.3f},{cy + length / 2:.3f} Z")
    return path(d, stroke=color, sw=0.15, fill=color)


# -- figure -------------------------------------------------------------------------

def build_svg(variant: str = "lean", *, fit_headers: bool = False) -> tuple[str, float]:
    """The curriculum figure. `fit_headers` condenses a phase name too wide for its column.

    Off by default, so the figure this module writes is unchanged: at one IEEE column the
    phase-2 header "Domain Randomization" is about 3 mm wider than its column and spills over
    the neighbouring arrow. Set it for a re-use at another size -- the talk video's
    tools/make_curriculum_animation.py does -- and any header name whose estimated advance
    exceeds the column gets an SVG textLength, which condenses that one line to fit instead.
    """
    V = VARIANTS[variant]
    wt = "bold" if V["bold"] else "normal"   # emphasis weight (none in lean)
    lr = lr_schedule(STAGES)
    n_col = len(PHASES)
    M = 0.5                     # page margin
    GAP_X = 4.6                 # between columns (holds the phase / transition arrows)
    GAP_Y = 2.7                 # between stage boxes (holds a promotion arrow)
    col_w = (W - 2 * M - GAP_X * (n_col - 1)) / n_col
    HEAD_H = V["head_h"]
    y_head = M
    y_stack = y_head + HEAD_H + GAP_Y

    # Phase 3 has four stage boxes; phases 1-2 have three. All columns span the same height.
    per_phase = [sum(1 for s in STAGES if s["phase"] == i) for i in range(n_col)]
    h_p3 = V["h_p3"]
    stack_h = per_phase[2] * h_p3 + (per_phase[2] - 1) * GAP_Y
    box_h = [(stack_h - (n - 1) * GAP_Y) / n for n in per_phase]
    LINE = FS * 1.12            # baseline spacing
    H = y_stack + stack_h + M

    col_x = [M + i * (col_w + GAP_X) for i in range(n_col)]

    parts = []

    # ---- headers + phase arrows ------------------------------------------------------
    heads = []
    for i, ph in enumerate(PHASES):
        x = col_x[i]
        heads.append(rect(x, y_head, col_w, HEAD_H, ph["accent"], stroke="none"))
        if V["head_two_line"]:
            # two lines ("Phase n" / name) vertically centred in the header
            y0 = y_head + HEAD_H / 2 - LINE / 2 + 0.85
            heads.append(text(x + col_w / 2, y0, ph["title"], anchor="middle",
                              weight="bold", fill="#FFFFFF"))      # "Phase n" always bold
            heads.append(text(x + col_w / 2, y0 + LINE, ph["name"], anchor="middle",
                              fill="#FFFFFF", extra=_fit(ph["name"], col_w, fit_headers)))
        else:
            # one line, bold, optically centred (7 pt caps are ~1.7 mm tall)
            heads.append(text(x + col_w / 2, y_head + HEAD_H / 2 + 0.85, ph["name"],
                              anchor="middle", weight=wt, fill="#FFFFFF"))
        if i < n_col - 1:
            heads.append(line(x + col_w + 0.35, y_head + HEAD_H / 2,
                              x + col_w + GAP_X - 0.3 - TIP, y_head + HEAD_H / 2,
                              stroke=INK, sw=ARROW_W, marker="arrow"))
    parts.append(group("Phase headers", *heads))

    # ---- stage boxes -------------------------------------------------------------------
    # Text wrapping (Times New Roman at 7 pt, measured: ~0.45 em average advance bold,
    # ~0.42 em regular). Pre-pass: line counts per stage, so that the title baseline can be
    # shared across a row (e.g. stage 6 lines up with stage 3 even though it has fewer lines).
    avail = col_w - 2.6
    wrapped = {}
    for k, s in enumerate(STAGES):
        if "title" in s:
            wrapped[k] = (wrap_lines(s["title"], int(avail / (FS * 0.45))),
                          wrap_lines(s["detail"], int(avail / (FS * 0.42)))
                          if V["show_detail"] else [])
    row_of = {}
    _cnt = [0] * n_col
    for k, s in enumerate(STAGES):
        row_of[k] = _cnt[s["phase"]]
        _cnt[s["phase"]] += 1
    row_lines = {}
    for k, (t, d) in wrapped.items():
        row_lines[row_of[k]] = max(row_lines.get(row_of[k], 0), len(t) + len(d))

    # polar target-distribution glyph: 4 m dashed reference circle of radius R_REF (one
    # common scale for every glyph in the figure), sector = reachable band of that stage
    R_MAX_M = 4.0
    R_REF = (h_p3 - V["glyph_pad"]) / 2        # fits inside a phase-3 box

    def target_glyph(cx, cy, theta, r, color):
        k_scale = R_REF / R_MAX_M
        r_in, r_out = (v * k_scale for v in r)
        return [f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(R_REF)}" fill="none" '
                f'stroke="{INK_SOFT}" stroke-width="0.15" stroke-dasharray="0.45 0.45"/>',
                path(sector_path(cx, cy, r_in, r_out, theta), stroke=color, sw=0.2,
                     fill=color, extra='fill-opacity="0.38" fill-rule="evenodd"'),
                fish_glyph(cx, cy, 2.0)]

    def range_txt(r):
        return f"{r[0]:g}–{r[1]:g} m"

    row = [0] * n_col
    boxes = []                                   # (x, y, w, h) per stage, for the phase arrows
    for k, (s, lr_k) in enumerate(zip(STAGES, lr)):
        i = s["phase"]
        ph = PHASES[i]
        x, w, h = col_x[i], col_w, box_h[i]
        y = y_stack + row[i] * (h + GAP_Y)
        boxes.append((x, y, w, h))
        if s.get("deployed"):
            el = [rect(x, y, w, h, ph["tint"], stroke=DEPLOYED_EDGE, sw=0.55)]
        else:
            el = [rect(x, y, w, h, ph["tint"], stroke=INK_SOFT, sw=EDGE)]
        # corner labels: stage number (left) and, in the full variant, the relative learning
        # rate (right, grey)
        el.append(text(x + 1.3, y + 2.75, f"{k + 1}", weight=wt, fill=ph["accent"]))
        if V["show_lr"]:
            lr_txt = fmt_lr(lr_k)
            if V["lr_reset_tag"] and k > 0 and s.get("reset_lr"):
                lr_txt += " (reset)"
            el.append(text(x + w - 1.3, y + 2.75, "LR " + lr_txt, anchor="end", fill=INK_SOFT))
        # promotion arrow into the next box of the same column
        if row[i] < per_phase[i] - 1:
            el.append(line(x + w / 2, y + h + 0.3, x + w / 2, y + h + GAP_Y - 0.3 - TIP,
                           stroke=ph["accent"], sw=ARROW_W, marker=f"arrow_{i}"))

        if "theta" in s:
            if V["show_range"]:
                cx, cy = x + 1.8 + R_REF, y + 4.2 + R_REF
                el += target_glyph(cx, cy, s["theta"], s["r"], ph["accent"])
                tx = cx + R_REF + 1.8
                el.append(text(tx, cy - 0.4, f"±{s['theta']}°", weight=wt))
                el.append(text(tx, cy - 0.4 + LINE, range_txt(s["r"]), fill=INK_SOFT))
            else:
                # glyph vertically centred, right of the stage number; the angle sits
                # right-aligned under the LR label so the two never touch the glyph
                cx, cy = x + 4.6 + R_REF, y + h / 2
                el += target_glyph(cx, cy, s["theta"], s["r"], ph["accent"])
                el.append(text(x + w - 1.3, y + h - 1.6, f"±{s['theta']}°", anchor="end",
                               weight=wt))
        elif V["show_detail"]:
            # icon in the corner-label row, right after the stage number (clear of the lr label),
            # then full-width title + detail lines
            ic = 4.8
            el.append(icon(s["icon"], x + 3.6, y + 1.1, ic, color=ph["accent"]))
            if "fixed" in s:
                # the fixed target band of phases 1-2, next to the icon in the label row; it
                # reaches down to y + 0.8 + 2 R_REF, which the bottom-aligned text clears
                cx, cy = x + 9.6 + R_REF, y + 0.8 + R_REF
                el += target_glyph(cx, cy, s["fixed"]["theta"], s["fixed"]["r"], ph["accent"])
                lx = cx + R_REF + 1.3           # labels stay put (glyph moved 0.6 mm left)
                el.append(text(lx, cy + 1.0, f"±{s['fixed']['theta']}°", fs=FS * 0.8,
                               weight=wt, fill=INK))
                el.append(text(lx, cy + 1.0 + LINE * 0.8, range_txt(s["fixed"]["r"]),
                               fs=FS * 0.8, fill=INK_SOFT))
            tx = x + 1.3
            title_lines, detail_lines = wrapped[k]
            # title baseline shared across the row: the block of the row's longest box sits at
            # the bottom (clear of the glyph in stage 1); shorter boxes start at the same line
            yy = y + h - 1.4 - (row_lines[row[i]] - 1) * LINE
            for ln in title_lines:
                el.append(text(tx, yy, ln, weight=wt))
                yy += LINE
            for ln in detail_lines:
                el.append(text(tx, yy, ln, fill=INK_SOFT))
                yy += LINE
        else:
            # lean card: number top-left, a centred pictogram, the title centred at the bottom.
            # Stage 1 shows the fixed forward target band (fish included) instead of an icon.
            if "fixed" in s:
                lab = f"±{s['fixed']['theta']}°"
                lab_w = len(lab) * FS * 0.42
                unit_w = 2 * R_REF + 1.0 + lab_w
                cx, cy = x + w / 2 - unit_w / 2 + R_REF, y + 2.0 + R_REF
                el += target_glyph(cx, cy, s["fixed"]["theta"], s["fixed"]["r"], ph["accent"])
                el.append(text(cx + R_REF + 1.0, cy + 0.85, lab, weight=wt))
            else:
                ic = 6.0
                # starts below the corner-label row so it clears the LR text
                el.append(icon(s["icon"], x + w / 2 - ic / 2, y + 3.4, ic, color=ph["accent"]))
            title_lines, _ = wrapped[k]
            yy = y + h - 1.8 - (len(title_lines) - 1) * LINE
            for ln in title_lines:
                el.append(text(x + w / 2, yy, ln, anchor="middle", weight=wt))
                yy += LINE
        parts.append(group(f"Stage {k + 1} {s['name']}", *el))
        row[i] += 1

    # ---- phase-to-phase arrows: last stage of a phase -> first stage of the next -----------
    trans = []
    for i in range(n_col - 1):
        k_from = max(k for k, s in enumerate(STAGES) if s["phase"] == i)
        k_to = min(k for k, s in enumerate(STAGES) if s["phase"] == i + 1)
        xa, ya, wa, ha = boxes[k_from]
        xb, yb, wb, hb = boxes[k_to]
        x_mid = xa + wa + GAP_X / 2
        d = (f"M {f(xa + wa + 0.3)},{f(ya + ha / 2)} H {f(x_mid)} V {f(yb + hb / 2)} "
             f"H {f(xb - 0.3 - TIP)}")
        trans.append(path(d, stroke=INK, sw=ARROW_W, extra='marker-end="url(#arrow)"'))
    parts.append(group("Phase transitions", *trans))

    # refX sits well behind the tip so the butt-ended line is hidden inside the head
    # (with refX at the tip the line's square end pokes out either side of the point).
    def marker(mid, color):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="6.5" refY="5" markerWidth="4.5" '
                'markerHeight="4.5" orient="auto-start-reverse">'
                f'<path d="M 0,0.5 L 10,5 L 0,9.5 z" fill="{color}"/></marker>')
    markers = [marker("arrow", INK)] + [marker(f"arrow_{i}", ph["accent"])
                                        for i, ph in enumerate(PHASES)]
    defs = "<defs>" + "".join(markers) + "</defs>"

    body = "\n  ".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Generated by icra_report/Images/scripts/make_curriculum_fig.py ({variant} variant) - edit freely in Inkscape. -->
<svg
   width="{f(W)}mm" height="{f(H)}mm" viewBox="0 0 {f(W)} {f(H)}"
   version="1.1" id="curriculum"
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
    return svg, H


# -- export (no Inkscape on this machine: headless Chrome/Edge print & screenshot) -------

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


def _find(names, candidates):
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


def _wrapper_html(svg: Path, w_mm: float, h_mm: float, px_per_mm: float | None = None, chrome=None) -> str:
    if px_per_mm:
        size = f"width:{w_mm * px_per_mm:.0f}px;height:{h_mm * px_per_mm:.0f}px"
    else:
        size = f"width:{w_mm}mm;height:{h_mm}mm"
    return ("<!doctype html><html><head><meta charset='utf-8'><style>"
            f"@page{{size:{w_mm}mm {h_mm}mm;margin:0}} html,body{{margin:0;padding:0;background:#fff}}"
            f"img{{display:block;{size}}}"
            f"</style></head><body><img src='{_uri(svg, chrome)}'></body></html>")


def export_pdf(svg: Path, pdf: Path, w_mm: float, h_mm: float) -> None:
    ink = _find(["inkscape"], _INKSCAPE_CANDIDATES)
    if ink:
        subprocess.run([ink, _bpath(svg, ink), "--export-type=pdf", f"--export-filename={_bpath(pdf, ink)}"],
                       check=True, timeout=300)
        print(f"Saved {pdf}  (Inkscape)")
        return
    chrome = _find(["chrome", "google-chrome", "chromium", "msedge"], _CHROME_CANDIDATES)
    if not chrome:
        print("WARNING: no Inkscape or Chrome found; PDF not written.")
        return
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "print.html"
        wrapper.write_text(_wrapper_html(svg, w_mm, h_mm, chrome=chrome), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={_bpath(pdf, chrome)}", _uri(wrapper, chrome)],
                       check=True, timeout=120, capture_output=True)
    if not pdf.is_file():
        print("WARNING: Chrome did not produce a PDF.")
        return
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
    size = f", page {float(m[1]) * 25.4 / 72:.1f} x {float(m[2]) * 25.4 / 72:.1f} mm" if m else ""
    print(f"Saved {pdf}  (Chrome print{size})")


def export_png(svg: Path, png: Path, w_mm: float, h_mm: float, px_per_mm: float = 8.0) -> None:
    chrome = _find(["chrome", "google-chrome", "chromium", "msedge"], _CHROME_CANDIDATES)
    if not chrome:
        return
    wpx, hpx = round(w_mm * px_per_mm), round(h_mm * px_per_mm)
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "shot.html"
        wrapper.write_text(_wrapper_html(svg, w_mm, h_mm, px_per_mm, chrome=chrome), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={wpx},{hpx}", f"--screenshot={_bpath(png, chrome)}",
                        _uri(wrapper, chrome)], check=True, timeout=120, capture_output=True)
    if png.is_file():
        print(f"Saved {png}  ({wpx}x{hpx} px preview)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["lean", "full", "both"], default="both",
                   help="lean = current draft, full = earlier dense draft (default: both)")
    p.add_argument("--out", default=None,
                   help="output .svg (single variant only; default: draft/curriculum[_full].svg)")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)

    variants = ["lean", "full"] if args.variant == "both" else [args.variant]
    if args.out and len(variants) > 1:
        p.error("--out needs a single --variant")
    for v in variants:
        svg, H = build_svg(v)
        out = Path(args.out) if args.out else DEFAULT_OUT[v]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg, encoding="utf-8")
        print(f"[{v}] Saved {out}  ({W:.1f} x {H:.1f} mm, {out.stat().st_size / 1024:.0f} kB)")
        if not args.no_pdf:
            export_pdf(out, out.with_suffix(".pdf"), W, H)
        if not args.no_png:
            export_png(out, out.with_suffix(".png"), W, H)


if __name__ == "__main__":
    main()
