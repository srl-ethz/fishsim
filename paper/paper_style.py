r"""One figure recipe for the whole paper. Every generator in this folder imports it.

The rules, and where they come from (Mike's review, September 2026):

  * ONE family everywhere: Times, set by LaTeX (mathptmx: Times text and maths), so the plots
    match the Times New Roman of the SVG diagrams (_svg.FONT) and the body text of
    the paper (ieeeconf sets Times). No default DejaVu in one panel and a sans in another.
  * Size: nothing under 8 pt at final print size (the caption size is the floor). Axis labels,
    titles, legends and any text on a panel are LABEL_PT = 9 pt; tick labels TICK_PT = 8 pt;
    bold panel letters PANEL_PT = 10 pt. There is no smaller tier.
  * >= 1 pt data lines.
  * Built at final on-page size and NEVER scaled by LaTeX. This is the only way the two
    floors above survive: \includegraphics[width=\columnwidth] on an asset that is not exactly
    \columnwidth silently rescales all of its type. The widths are read out of the compiled
    document (see COL_PT), not taken from the class documentation.
  * Fonts embedded, never Type 3 (Type 3 is rejected by submission systems). usetex embeds the
    Type 1 Nimbus Roman (Times); `pdffonts` on the paper must show every font "emb yes".
  * Framed axes with a light grid -- the "classic" look, matching the SRL house style.
  * Vector PDF for the paper; a PNG twin is written alongside for quick preview.

Usage:
    from paper_style import use_paper_style, size, save, CB, ROLE, LW, EMPH_LW
    use_paper_style()
    fig, ax = plt.subplots(figsize=size(1.0, aspect=0.62), layout="constrained")
    ...
    save(fig, "fig9_velspectrum", expect="col")        # -> out/fig9_velspectrum.pdf + .png

Do NOT pass bbox_inches="tight" anywhere. It re-measures the drawn content and
silently returns a figure of a different width, which is what breaks the size floor.
`layout="constrained"` does the same job (nothing clipped) while keeping the size
exactly as requested; `save()` asserts that it did.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- the paper's real geometry, read out of the compiled document -------------
#   \AtBeginDocument{\typeout{\the\columnwidth \the\textwidth}} on the Overleaf source
#   (ieeeconf, letterpaper, 10pt, conference, \overrideIEEEmargins) gives:
COL_PT = 245.718          # \columnwidth  (3.400 in; the class documentation's 3.5 in is not this document)
FULL_PT = 505.89          # \textwidth    (7.000 in; figure* spans this)
PT_PER_IN = 72.27
COL = COL_PT / PT_PER_IN          # 3.400 in
FULL = FULL_PT / PT_PER_IN        # 7.000 in

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# ---- the size tiers -------------------------------------------------------------
LABEL_PT = 9.0            # axis labels, titles, legends, annotations, any text on a panel
TICK_PT = 8.0             # tick labels: the smallest type anywhere in a figure
PANEL_PT = 10.0           # bold panel letters (A, B)
BASE_PT = LABEL_PT        # kept for older scripts that import it
LW = 1.0                  # every data line
EMPH_LW = 1.5             # the one reference/"truth" line in a plot, if there is one

# ---- colour ------------------------------------------------------------------
# Okabe-Ito, colour-blind safe, fixed order. Colour carries identity, so a given
# entity keeps its hue across every figure in the paper (see ROLE).
CB = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
      "sky": "#56B4E9", "amber": "#E69F00", "grey": "#666666", "black": "#111111"}
CYCLE = [CB["blue"], CB["orange"], CB["green"], CB["purple"], CB["sky"], CB["amber"]]

ROLE = {
    "measured": CB["black"],      # anything recorded on the real fish
    "sim": CB["orange"],          # anything produced by the simulator
    "twin": CB["orange"],
    "setpoint": CB["grey"],       # commanded / reference signals
    "identified": CB["blue"],     # the identified model
    "old_model": CB["orange"],    # the inherited model it replaces
    "cpg": CB["blue"],
    "fit": CB["orange"],          # regression lines, fitted curves
}

# Times for text AND maths, through LaTeX. mathptmx (psnfss) is the Type 1 Times clone every
# TeX install ships (Nimbus Roman) and sets the maths in it too, so a theta on an axis is the
# same face as the word next to it, and the same face as the paper's body text.
LATEX_PREAMBLE = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{mathptmx}",
])


def size(frac=1.0, aspect=0.62, full=False):
    r"""Figure size in inches at final on-page size.

    frac   fraction of the target width (1.0 = fill it)
    aspect height / width
    full   measure against \textwidth (for figure*) instead of \columnwidth
    """
    w = frac * (FULL if full else COL)
    return (w, w * aspect)


def use_paper_style():
    mpl.rcParams.update({
        # type: the three tiers, nothing below TICK_PT
        "font.size": LABEL_PT,
        "axes.titlesize": LABEL_PT, "axes.labelsize": LABEL_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": LABEL_PT, "legend.title_fontsize": LABEL_PT,
        "figure.labelsize": LABEL_PT, "figure.titlesize": LABEL_PT,
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "text.usetex": True, "font.family": "serif",
        "text.latex.preamble": LATEX_PREAMBLE,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # never Type 3 (applies to any non-TeX glyph)

        # ink
        "lines.linewidth": LW, "lines.markersize": 3.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.direction": "out", "ytick.direction": "out",

        # the classic framed look: all four spines, light grid behind the data
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.4,
        "grid.linestyle": "-", "axes.axisbelow": True,

        "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.borderpad": 0.2, "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5, "legend.borderaxespad": 0.4,

        "axes.prop_cycle": mpl.cycler(color=CYCLE),

        # size discipline: save exactly what was asked for
        "figure.dpi": 130, "savefig.dpi": 400,
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "savefig.facecolor": "white",
    })


def save(fig, stem, dest=None, expect=None, quiet=False):
    """Write <stem>.pdf and <stem>.png at exactly the requested figure size.

    expect  'col' | 'full' | None -- if given, assert the PDF really came out at
            that width, so a stray bbox_inches or a runaway legend cannot quietly
            put us back under the size floor.
    """
    d = os.path.abspath(dest or DEST)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, stem + ".pdf")
    fig.savefig(path)
    fig.savefig(os.path.join(d, stem + ".png"))
    plt.close(fig)

    got_pt = fig.get_size_inches()[0] * PT_PER_IN
    if expect is not None:
        want = COL_PT if expect == "col" else FULL_PT
        err = abs(got_pt - want)
        flag = "OK " if err < 0.5 else "OFF"
        if err >= 0.5:
            raise SystemExit(
                f"{stem}: figure is {got_pt:.1f} pt but {expect} width is {want:.1f} pt. "
                f"{LABEL_PT:.0f} pt type would print at {LABEL_PT * want / got_pt:.1f} pt. Fix the figsize.")
        if not quiet:
            print(f"  {flag} {stem}.pdf  {got_pt:.1f} pt ({expect}), labels {LABEL_PT:.0f} pt, ticks {TICK_PT:.0f} pt")
    elif not quiet:
        print(f"      {stem}.pdf  {got_pt:.1f} pt")
    return path
