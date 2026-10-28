#!/usr/bin/env python3
"""Fig 10 (reach collage) regenerated at the PAPER's final printed size.

Derived from the report generator 05_Report/.../figures/scripts/reach_redetect.py.
The report asset is 12.6 in wide; dropped into the paper's 3.38 in column that is a
0.27 scale, so its 13 pt titles print at 3.5 pt. This build targets a SINGLE-COLUMN
figure (3.42 in): ONE dial holding all 30 fixed-target runs (22 static + 8 perturbed
starts), after the advisor's note that the two-panel full-width version was far too large.
Colour is the outcome; every target is a filled circle; undisturbed runs have solid paths, disturbed runs
dotted paths. The legend carries both: outcome counts on the first row, the run
key (line style, circle at the end like a real run, in neutral grey) on the second. Type is 7 pt
everywhere (tick labels, legend); paths stay >= 1 pt. Built at final size, nothing is scaled.

The earlier full-width build used subplots_adjust + a 10 pt title pad, which pushed both
"(a)"/"(b)" titles off the top of the page -- the printed figure had no panel labels.
This build uses layout="constrained" and save(expect="col"), so a clipped title now fails.

Writes to 06_Publication/paper/Images/draft/ -- the report's assets are untouched.
Run via `make figures`, or from .gen/figures:  python fig_reach_collage_paper.py [--split]
`--split` writes fig10_reach_collage_split_paper instead: two half-column dials side by side in
one column, undisturbed runs left and disturbed runs right. `--half` writes
fig10_reach_collage_half_paper: the single dial with about half of the runs (3 reached and 1
missed disturbed run, about half of each undisturbed outcome), thinned evenly over bearing.
Neither variant is in the paper.
"""
import csv, math, os, sys, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
try:
    from dt_apriltags import Detector
except ImportError:      # not needed when the heading cache is present
    Detector = None
import cv2

from _paths import ANALYSIS as _ANALYSIS, OUT as _OUT
from paper_style import use_paper_style, size, save, LW, LABEL_PT, TICK_PT
use_paper_style()
COLLAGE_LW = 1.7                  # so traj lines (0.6x below) land at ~1.0 pt
SMALL_PT = TICK_PT                # tick labels and the metre labels (paper_style tier)
RLAB_ANGLE = 140                  # data-frame angle of the metre labels: the empty sector between 120 and 150 deg

ANA = str(_ANALYSIS)
RUNS = os.path.join(ANA, "..", "runs")
REDET = os.path.join(ANA, "cache_redetect")
CACHE = os.path.join(ANA, "cache")
DEST = str(_OUT)
os.makedirs(DEST, exist_ok=True)
CAM = [1055.0202, 1054.2105, 945.8357, 489.0351]
FISH, ANCHOR = 0, 1
# Colour = outcome. REACH_COLORS=okabe (default, colourblind-safe) or greenred (literal).
if os.environ.get("REACH_COLORS", "okabe") == "greenred":
    COL = {"success": "#2ca02c", "fail": "#d62728"}
    SHADE, _SUFFIX = "#2ca02c", "_greenred"
else:
    COL = {"success": "#009E73", "fail": "#D55E00"}
    SHADE, _SUFFIX = "#009E73", ""
MK, MS = "o", 4.0                 # legend marker (static runs)
MARK = {"static": ("o", 4.0), "disturbance": ("o", 4.0)}   # circles for every run
HOLLOW = {"static": False, "disturbance": False}           # every run a filled circle; disturbed runs differ by their dotted path only
MEW = 0.9                                                   # ring width of the hollow circle
LS = {"static": "-", "disturbance": (0, (0.6, 2.8))}       # ... and their paths are dotted (round caps -> dots)
ALPHA = {"static": 0.5, "disturbance": 0.75}               # dots read lighter; compensate a little
_DET = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=1) if Detector else None

# Headings were precomputed from the raw deployment video (tools/make_heading_cache.py)
# so this figure does not need the 1.5 GB of .avi.  Delete the JSON to recompute.
import json as _json
_HCACHE_PATH = os.path.join(ANA, "cache_heading", "initial_heading.json")
_HCACHE = _json.load(open(_HCACHE_PATH)) if os.path.exists(_HCACHE_PATH) else {}


def velocity_gate(P, T, v_max=1.0, floor=0.12):
    keep = np.ones(len(P), bool); last = 0
    for k in range(1, len(P)):
        dt = max(T[k] - T[last], 1e-3)
        if math.hypot(P[k, 0] - P[last, 0], P[k, 1] - P[last, 1]) > v_max * dt + floor:
            keep[k] = False
        else:
            last = k
    return keep


def smooth_xy(P, w=5):
    if len(P) < w:
        return P
    k = np.ones(w) / w; out = P.copy()
    for c in range(2):
        out[:, c] = np.convolve(P[:, c], k, mode="same")
        out[:w, c] = P[:w, c]; out[-w:, c] = P[-w:, c]
    return out


def initial_heading(run, n_want=12):
    if run in _HCACHE:
        return _HCACHE[run]
    if _DET is None:
        raise RuntimeError(f'no cached heading for {run} and dt_apriltags is unavailable')
    vids = glob.glob(os.path.join(RUNS, f"*{run}*", "raw", "video.avi"))
    if not vids:
        return None
    cap = cv2.VideoCapture(vids[0]); yaws = []; fi = 0
    while len(yaws) < n_want and fi < 400:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        tm = {t.tag_id: t for t in _DET.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
              estimate_tag_pose=True, camera_params=CAM, tag_size=0.08)}
        if FISH in tm and ANCHOR in tm:
            fwd = (tm[ANCHOR].pose_R.T @ tm[FISH].pose_R)[:, 1]
            yaws.append(math.atan2(fwd[1], fwd[0]))
    cap.release()
    return None if not yaws else math.atan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))


def _to_start_frame(P, T, yaw0, tw_med, lat_sign, truncate):
    keep = velocity_gate(P, T)
    if keep.sum() >= 5:
        idx = np.arange(len(P))
        for c in range(2):
            P[~keep, c] = np.interp(idx[~keep], idx[keep], P[keep, c])
    x0, y0 = np.median(P[:8, 0]), np.median(P[:8, 1])
    c, s = math.cos(yaw0), math.sin(yaw0)
    tob = lambda x, y: (c * (x - x0) + s * (y - y0), lat_sign * (-s * (x - x0) + c * (y - y0)))
    traj = smooth_xy(np.array([tob(px, py) for px, py in P]), 5)
    tf, tl = tob(tw_med[0], tw_med[1])
    if truncate:                                        # truncate at closest approach
        dist = np.hypot(traj[:, 0] - tf, traj[:, 1] - tl)
        traj = traj[:int(np.argmin(dist)) + 1]
    return traj, tf, tl


def geom(run, outcome):
    # truncate at closest for successes; also for finetune_run_4 (its full path wanders on partial video)
    trunc = (outcome == "success") or (run == "finetune_run_4")
    c = np.load(os.path.join(CACHE, f"{run}.npz"), allow_pickle=True)
    r = np.load(os.path.join(REDET, f"{run}_redetect.npz"), allow_pickle=True)
    if len(c["fish_world"]) == len(r["fish_world"]):        # re-detect reliable
        fw, tw, ft = r["fish_world"].astype(float), r["tgt_world"].astype(float), r["frame_time"].astype(float)
        vf = ~np.isnan(fw[:, 0]); vt = ~np.isnan(tw[:, 0])
        if vf.sum() < 10 or vt.sum() < 3:
            return None
        yaw0 = initial_heading(run)
        if yaw0 is None:
            return None
        gi = np.where(vf)[0]
        return _to_start_frame(fw[gi], ft[gi], yaw0, np.median(tw[vt], 0), -1.0, trunc)
    # fallback: old cache (partial/dropped video makes re-detect unreliable)
    fw, yaw, tw, ft = c["fish_world"], c["fish_yaw"], c["tgt_world"], c["frame_time"]
    good = ~np.isnan(fw[:, 0]) & ~np.isnan(yaw)
    if good.sum() < 10:
        return None
    gi = np.where(good)[0]
    yaw0 = math.atan2(np.mean(np.sin(yaw[gi[:8]])), np.mean(np.cos(yaw[gi[:8]])))
    tg = tw[~np.isnan(tw[:, 0])]
    if len(tg) < 3:
        return None
    return _to_start_frame(fw[gi].astype(float), ft[gi].astype(float), yaw0, np.median(tg, 0), +1.0, trunc)


def separate(P, min_sep=0.17, iters=80):
    """Nudge overlapping target markers apart so both stay visible (just touching)."""
    P = np.array(P, float)
    for _ in range(iters):
        moved = False
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                d = P[j] - P[i]; dist = math.hypot(*d)
                if 1e-6 < dist < min_sep:
                    u = d / dist; push = (min_sep - dist) / 2
                    P[i] -= u * push; P[j] += u * push; moved = True
        if not moved:
            break
    return P


# strict-threshold reclassification (reached only 246 mm > 50 mm) -- plot-level, source CSV untouched
FORCE_OUTCOME = {"finetune_run_4": "fail"}

# compute geometry once per run
ITEMS = []
for r in csv.DictReader(open(os.path.join(ANA, "reachability/run_geometry.csv"))):
    if r["group"] not in ("static", "disturbance"):
        continue
    out = FORCE_OUTCOME.get(r["run"], r["outcome"])
    g = geom(r["run"], out)
    if g is None:
        print("skip", r["run"]); continue
    traj, tf, tl = g
    ITEMS.append({"grp": r["group"], "out": out, "traj": traj, "tf": tf, "tl": tl})


def draw_polar(ax, items, subtitle=None, theta_step=30, uniform=False, ms_scale=1.0):
    """uniform=True draws every run in the undisturbed style (filled circle, solid path), for
    layouts where the panel already says which group a run belongs to; ms_scale shrinks the
    target circles."""
    ax.set_theta_zero_location("N"); ax.set_theta_direction(1)
    # shade the TRAINED region light green: |bearing| <= 120 deg AND 0.5-2.5 m (stage-8 curriculum).
    # anything outside (in angle or range) is out-of-distribution.
    tt = np.radians(np.linspace(-120, 120, 240))
    ax.fill_between(tt, 0.5, 2.5, color=SHADE, alpha=0.09, zorder=0)
    ax.plot(tt, np.full_like(tt, 0.5), color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(tt, np.full_like(tt, 2.5), color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=1)
    for a in (120, -120):
        ax.plot([math.radians(a)] * 2, [0.5, 2.5], color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=1)
    tgt = separate([(it["tf"], it["tl"]) for it in items])
    for it, (tf, tl) in zip(items, tgt):
        col, grp = COL[it["out"]], ("static" if uniform else it["grp"])
        tr = it["traj"]
        ax.plot(np.arctan2(tr[:, 1], tr[:, 0]), np.hypot(tr[:, 0], tr[:, 1]),
                color=col, lw=COLLAGE_LW*0.6, ls=LS[grp], alpha=ALPHA[grp], zorder=2,
                solid_capstyle="round", dash_capstyle="round")
        tr_rad = min(math.hypot(tf, tl), 2.95)          # clamp to the rim so far targets (>3 m) stay visible
        mk, ms = MARK[grp]
        ms *= ms_scale
        hollow = HOLLOW[grp]
        ax.plot(math.atan2(tl, tf), tr_rad, mk, ms=ms, mfc="none" if hollow else col, mec=col,
                mew=MEW if hollow else 0, zorder=4)
    ax.plot(0, 0, "o", color="#1f4e9c", ms=3.6, zorder=6)
    ax.set_rlim(0, 3.0)
    # metre labels: drawn by hand in an empty sector, each on a white patch that cuts its ring
    # (the rim included) so the text never sits on a line or a path.
    ax.set_rgrids([1, 2, 3], labels=["", "", ""])
    for r_, lab in ((1, "1 m"), (2, "2 m"), (3, "3 m")):
        ax.text(math.radians(RLAB_ANGLE), r_, lab, fontsize=SMALL_PT, ha="center", va="center",
                zorder=7, clip_on=False, bbox=dict(fc="white", ec="none", pad=0.8))
    angles = list(range(0, 360, theta_step))
    ax.set_thetagrids(angles, labels=[f"{a if a <= 180 else a - 360:d}°".replace("-", "−") for a in angles])
    ax.tick_params(labelsize=SMALL_PT, pad=0.5)
    ax.grid(True, color="0.5", lw=0.5, ls=":")
    if subtitle:
        ax.set_title(subtitle, fontsize=LABEL_PT, pad=3)


KEY = "0.25"          # neutral grey: the run key is independent of the outcome colour
SPLIT = "--split" in sys.argv     # two half-column dials side by side (undisturbed | disturbed)
HALF = "--half" in sys.argv       # the single dial, but with about half of the runs (see subsample)


def subsample(items, keep):
    """Deterministic thinning: for each (group, outcome), sort the runs by target bearing and
    take `keep[(group, outcome)]` of them at evenly spaced ranks, so the kept runs still span
    the same bearings as the full set. Order of the returned list follows `items`."""
    chosen = set()
    for key, n_keep in keep.items():
        idx = [i for i, it in enumerate(items) if (it["grp"], it["out"]) == key]
        idx.sort(key=lambda i: math.atan2(items[i]["tl"], items[i]["tf"]))
        if n_keep >= len(idx):
            chosen.update(idx); continue
        picks = np.round(np.linspace(0, len(idx) - 1, n_keep)).astype(int)
        chosen.update(idx[k] for k in picks)
    return [it for i, it in enumerate(items) if i in chosen]


# --half: 3 reached + 1 missed disturbed run, about half of each undisturbed outcome
HALF_KEEP = {("disturbance", "success"): 3, ("disturbance", "fail"): 1,
             ("static", "success"): 7, ("static", "fail"): 4}
if HALF:
    ITEMS = subsample(ITEMS, HALF_KEEP)
    print("half: kept", {k: sum((it["grp"], it["out"]) == k for it in ITEMS) for k in HALF_KEEP})


class EndMarker(HandlerLine2D):
    """Legend handle drawn like a run: the path, with the target circle at its END."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line, = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        # the circle, on its own, at the end of the path
        dot = Line2D([x[-1]], [y[-1]], ls="", transform=trans)
        dot.update_from(orig_handle); dot.set_linestyle("")
        # the path, stopped at the rim of the circle so a dotted line never shows through a hollow one
        r_px = (0.5 * orig_handle.get_markersize() + 0.5 * orig_handle.get_markeredgewidth()) * legend.figure.dpi / 72
        x_end = x[-1] - r_px
        line.set_data([x[0], x_end], [y[0], y[-1]])
        line.set_marker("")
        return [line, dot]


def outcome_handles(items):
    n = lambda out: sum(it["out"] == out for it in items)
    return [Line2D([0], [0], marker=MK, ls="", mfc=COL[out], mec="none", ms=MS + 0.6,
                   label=f"{word} ({n(out)}/{len(items)})")
            for out, word in (("success", "Reached"), ("fail", "Missed"))]


def run_handle(grp, lab):
    return Line2D([0], [0], marker=MARK[grp][0], ms=MARK[grp][1] + 0.6, ls=LS[grp], color=KEY,
                  mfc="none" if HOLLOW[grp] else KEY, mec=KEY, mew=MEW if HOLLOW[grp] else 0,
                  lw=COLLAGE_LW * 0.6, label=lab)


LEGEND_KW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=LABEL_PT, frameon=False,
                 handletextpad=0.8, columnspacing=1.5, handlelength=2.6)

if not SPLIT:
    # one dial, both fixed-target groups, legend below (the LaTeX caption carries the title)
    fig = plt.figure(figsize=size(1.0, aspect=1.08), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    draw_polar(ax, ITEMS)
    run_handles = [run_handle("static", "Undisturbed run"), run_handle("disturbance", "Disturbed run")]
    h = outcome_handles(ITEMS) + run_handles
    # column-major fill -> row 1: reached | missed, row 2: undisturbed | disturbed
    handles = [h[0], h[2], h[1], h[3]]
    ax.legend(handles=handles, ncol=2, handler_map={rh: EndMarker() for rh in run_handles}, **LEGEND_KW)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    out = save(fig, f"fig8_reach_collage{'_half' if HALF else ''}{_SUFFIX}", expect="col")
else:
    # two half-column dials side by side, undisturbed | disturbed, drawn by the same draw_polar
    # as the single dial. Angle labels every 60 deg (7 pt type does not fit every 30 deg on a
    # 40 mm dial); per-panel counts in the panel titles; one shared outcome legend below.
    GROUPS = [("static", "(a) Undisturbed", "Undisturbed run"),
              ("disturbance", "(b) Disturbed", "Disturbed run")]
    fig = plt.figure(figsize=size(1.0, aspect=0.60), layout="constrained")
    for k, (grp, title, lab) in enumerate(GROUPS):
        items = [it for it in ITEMS if it["grp"] == grp]
        n_ok = sum(it["out"] == "success" for it in items)
        ax = fig.add_subplot(1, 2, k + 1, projection="polar")
        # the panel says which group a run is in, so both groups share one style: filled
        # circle, solid path; circles a little smaller than on the single dial
        draw_polar(ax, items, subtitle=f"{title}: {n_ok}/{len(items)} reached", theta_step=60,
                   uniform=True, ms_scale=0.8)
    fig.legend(handles=outcome_handles(ITEMS), loc="outside lower center", ncol=2, fontsize=LABEL_PT,
               frameon=False, handletextpad=0.8, columnspacing=1.5)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    out = save(fig, f"fig8_reach_collage_split{_SUFFIX}", expect="col")
print("saved", out)
