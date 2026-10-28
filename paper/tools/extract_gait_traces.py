#!/usr/bin/env python3
"""Recover the panel-(b) motor-angle traces of the 13 July gait figure from its vector PDF.

The script that drew `10_gaits_combined.pdf` was never saved and the rollouts behind it
(the exploit re-analysis, incl. a DA-LPF1 checkpoint) are not in this repo, so the plotted
traces are the only surviving record. The PDF is vector: every trace is a polyline whose
vertices are the plotted samples (minus the collinear ones matplotlib's path simplifier
dropped, which does not move the drawn line). This walks the content stream with a CTM
stack, keeps every stroked polyline with the clip rectangle it was drawn under, calibrates
page space to (time, angle) from the tick marks (x = 0..6 s, y = -1, 0, 1 rad) and snaps
time to the 50 Hz sample grid. The snap residual is printed as a check: it is 0.00 ms.

Usage:  python tools/extract_gait_traces.py <10_gaits_combined.pdf> [data/gait_traces.npz]
Needs:  pypdf, numpy
"""
import re
import sys
import numpy as np
import pypdf

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/gait_traces.npz"
DT = 0.02                                         # rollout sample period (50 Hz)

# panel (b) axes, as clip rectangles in page points (x0, y0, x1, y1), top row first
AXES = {
    "tg":      (336.4, 331.9, 695.5, 391.7),      # labelled "CPG" in the July figure
    "da":      (336.4, 239.6, 695.5, 299.4),      # "DA-raw"
    "da_lpf1": (336.4, 147.4, 695.5, 207.1),      # "DA-LPF1"
}
CMD_GREY = (0.4, 0.4, 0.4)                        # the dashed "commanded" line

# ---- 1. walk the content stream --------------------------------------------------
page = pypdf.PdfReader(SRC).pages[0]
stream = page.get_contents().get_data().decode("latin1")
tok = re.compile(r"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>]+|[-+.\d]+|[A-Za-z'\"*]+")


def mul(a, b):
    return (a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3], a[2]*b[0] + a[3]*b[2],
            a[2]*b[1] + a[3]*b[3], a[4]*b[0] + a[5]*b[2] + b[4], a[4]*b[1] + a[5]*b[3] + b[5])


def apply(m, x, y):
    return (m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5])


ctm, stack = (1, 0, 0, 1, 0, 0), []
clip, clip_stack = None, []
rect, want_clip = None, False
path, paths, operands = [], [], []
rgb = None
for t in tok.findall(stream):
    if re.fullmatch(r"[-+.\d]+", t) or t[0] in "/(<[]":
        operands.append(t)
        continue
    nums = [float(o) for o in operands if re.fullmatch(r"[-+.\d]+", o)]
    if t == "q":
        stack.append(ctm); clip_stack.append(clip)
    elif t == "Q":
        ctm = stack.pop(); clip = clip_stack.pop()
    elif t == "cm" and len(nums) == 6:
        ctm = mul(tuple(nums), ctm)
    elif t == "m" and len(nums) >= 2:
        path = [apply(ctm, nums[-2], nums[-1])]
    elif t in ("l", "c") and len(nums) >= 2:
        path.append(apply(ctm, nums[-2], nums[-1]))
    elif t == "re" and len(nums) == 4:
        x, y, w, h = nums
        rect = [apply(ctm, x, y), apply(ctm, x + w, y + h)]
        path = rect
    elif t == "W":
        want_clip = True
    elif t == "n":
        if want_clip and rect is not None:
            (xa, ya), (xb, yb) = rect
            clip = (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))
        want_clip, rect, path = False, None, []
    elif t in ("S", "s", "B", "b", "f", "F", "f*", "B*", "b*"):
        if len(path) >= 2:
            paths.append(dict(pts=np.array(path), clip=clip, rgb=rgb))
        path, rect = [], None
    elif t == "RG" and len(nums) == 3:
        rgb = tuple(round(v, 3) for v in nums)
    elif t == "G" and nums:
        rgb = (round(nums[-1], 3),) * 3
    operands = []

# ---- 2. calibrate from the tick marks --------------------------------------------
stubs = [p["pts"] for p in paths if p["clip"] is None and len(p["pts"]) == 2]
x_ticks = sorted({round(p[0, 0], 3) for p in stubs
                  if abs(p[0, 0] - p[1, 0]) < 1e-3 and 2 < abs(p[1, 1] - p[0, 1]) < 6
                  and abs(max(p[:, 1]) - AXES["da_lpf1"][1]) < 0.5 and 336 <= p[0, 0] <= 696})
assert len(x_ticks) == 7, x_ticks
to_t = np.polyfit(x_ticks, np.arange(7), 1)        # page x -> seconds, ticks are 0..6 s


def same(c, ref):
    return c is not None and all(abs(a - b) < 0.2 for a, b in zip(c, ref))


out = {}
for name, ax in AXES.items():
    y_ticks = sorted({round(p[0, 1], 3) for p in stubs
                      if abs(p[0, 1] - p[1, 1]) < 1e-3 and 2 < abs(p[1, 0] - p[0, 0]) < 6
                      and abs(max(p[:, 0]) - ax[0]) < 0.5 and ax[1] - 1 < p[0, 1] < ax[3] + 1})
    assert len(y_ticks) == 3, (name, y_ticks)
    to_a = np.polyfit(y_ticks, [-1, 0, 1], 1)        # page y -> rad
    lines = [p for p in paths if same(p["clip"], ax) and len(p["pts"]) > 20]
    assert len(lines) == 2, [len(p["pts"]) for p in lines]
    for p in lines:
        kind = "cmd" if p["rgb"] == CMD_GREY else "act"
        t = np.polyval(to_t, p["pts"][:, 0]); a = np.polyval(to_a, p["pts"][:, 1])
        k = np.round(t / DT); resid = np.abs(t - k * DT).max() * 1e3
        print(f"{name:8s} {kind}: {len(t):4d} vertices, 0..{t.max():.2f} s, "
              f"snap residual {resid:.2f} ms, range {a.min():+.3f}..{a.max():+.3f} rad")
        assert resid < 1.0, "calibration off: vertices do not sit on the sample grid"
        out[f"{name}_{kind}_t"] = k * DT
        out[f"{name}_{kind}_a"] = a
out["dt"] = DT
out["source"] = np.array(f"vertices of panel (b) of {SRC.split('/')[-1]} (13 July 2026 figure)")
np.savez(OUT, **out)
print("wrote", OUT, sorted(k for k in out if k.endswith("_a")))
