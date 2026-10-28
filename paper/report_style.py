"""One figure recipe for the whole report. See FIGURE_RECIPE.md.

Vector PDF, build-at-final-size (set_size), never scale in LaTeX, one font/line spec
with a `dense` tier for collages, framed/airy variant switch, usetex serif so figure
text == body text.

Usage in a generator:
    from report_style import use_report_style, set_size, DATA_LW, EMPH_LW, savefig
    use_report_style(dense=False)                 # or dense=True for >=4-panel collages
    fig, ax = plt.subplots(figsize=set_size(0.62))
    ...
    savefig(fig, ".../figures/restyled/xx")       # writes xx.pdf (+ xx.png preview)

Env switches:  REPORT_VARIANT=framed|airy   REPORT_USETEX=0 (STIX fallback)
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXTWIDTH_PT = 418.25368          # report \textwidth (measured); re-measure if class changes
_IN_PER_PT = 1.0 / 72.27
_GOLDEN = (5 ** 0.5 - 1) / 2      # ~0.618

DATA_LW = 1.1                     # normal data lines
EMPH_LW = 1.6                     # the one truth/reference line per plot

# Okabe-Ito, fixed order (identity follows entity, never cycled). Validated CB-safe.
CB = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
      "sky": "#56B4E9", "amber": "#E69F00", "grey": "#666666", "black": "#111111"}
CB_CYCLE = [CB["blue"], CB["orange"], CB["green"], CB["purple"], CB["sky"], CB["amber"]]
ROLE = {"cpg": CB["blue"], "da": CB["orange"], "da_lpf": CB["green"],
        "setpoint": CB["grey"], "encoder": CB["black"],
        "identified": CB["blue"], "old_model": CB["orange"],
        "forward": CB["grey"], "backward": CB["orange"]}


def set_size(frac=1.0, subplots=(1, 1), aspect=None):
    """Figure size in inches, built at final on-page size (no LaTeX scaling).
    frac = fraction of \\textwidth; height by golden ratio x rows/cols, or by `aspect`."""
    w = frac * TEXTWIDTH_PT * _IN_PER_PT
    h = w * (aspect if aspect is not None else _GOLDEN * subplots[0] / subplots[1])
    return (w, h)


# One collage spec (used by EVERY multi-panel "collage" figure so they never drift):
# big roomy canvas, its own font/weights; LaTeX scales the vector PDF down (crisp).
COLLAGE_LW = 2.4                  # collage data lines (on-page ~1.1 after LaTeX scaling)


def use_report_style(dense=False, collage=False):
    usetex = os.environ.get("REPORT_USETEX", "1") != "0"
    framed = os.environ.get("REPORT_VARIANT", "framed") != "airy"
    if collage:                            # roomy multi-panel result figures (nav, reach, measurements)
        base, tick = 11.0, 10.5
    else:
        base = 7.5 if dense else 9.0       # dense grid vs single-plot tier
        tick = 6.5 if dense else 8.0
    rc = {
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": tick,
        "lines.linewidth": DATA_LW, "lines.markersize": 4.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": framed, "axes.spines.right": framed,
        "axes.grid": framed, "grid.linewidth": 0.5, "grid.alpha": 0.30,
        "legend.frameon": False, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CB_CYCLE),
    }
    if collage:                            # the neat, uniform collage look (thicker spines/ticks, medium weight)
        rc.update({"axes.titleweight": "medium", "axes.labelweight": "medium",
                   "axes.linewidth": 1.3,
                   "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                   "xtick.major.size": 5.0, "ytick.major.size": 5.0})
    if usetex:
        rc.update({"text.usetex": True, "font.family": "serif",
                   "text.latex.preamble": r"\usepackage{lmodern}"})
    else:
        rc.update({"text.usetex": False, "font.family": "STIXGeneral",
                   "mathtext.fontset": "stix"})
    mpl.rcParams.update(rc)


def variant_suffix():
    """'' for framed (report default), '_airy' for the airy build."""
    return "" if os.environ.get("REPORT_VARIANT", "framed") != "airy" else "_airy"


def savefig(fig, stem):
    """Save `stem`.pdf (for the report) + `stem`.png (for quick preview). `stem` has no
    extension. The variant suffix is appended automatically."""
    s = stem + variant_suffix()
    fig.savefig(s + ".pdf")
    fig.savefig(s + ".png")
    plt.close(fig)
    return s + ".pdf"
