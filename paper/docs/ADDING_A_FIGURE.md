# Adding a figure to this repo

A checklist for people and agents. It exists because a figure is "in the repo" only when all of
the following are true at once: it rebuilds from `make figures` on a fresh checkout, it reads
nothing outside the checkout, it is listed in the README table, the byte-identity check knows
about it, and the PDF in the Overleaf project is the one this repo produced. Miss one and the
figure quietly rots.

Read `README.md` (layout, token convention) and `docs/FIGURE_RECIPE.md` (the type-size
standard) first. This document only tells you where things go and in what order.

## The mental model in five lines

1. `figures/`, `upstream/`, `data/`, `tools/`, `docs/` are **source**. Commit them.
2. `.gen/` is a **generated mirror** of `figures/`, `upstream/` and `data/` in which the literal
   token `@@SOFI_ROOT@@` has been replaced by the absolute path of `.gen/` on this machine.
   Scripts are copied with the token resolved; every other file is symlinked. Never edit `.gen/`.
3. Scripts **run from `.gen/figures`**, never from `figures/`. `make` does this for you.
4. `out/` is where every figure lands, as `fig<N>_<name>.pdf` plus a `.png` twin and nothing
   else: `make figures` empties the folder first and builds only the variants the paper uses.
   N is the figure's number in the report. It is gitignored. The paper's copy lives in the
   Overleaf project (`Images/` or `Images/draft/`), and `make check` compares the two.
5. A data file belongs in git if a figure or a number in the paper reads it, and in the
   archive (`docs/ARCHIVE.md`) otherwise. Nothing is in both.

## Checklist

Work top to bottom. Each step names the file it touches.

### 1. Decide the kind of figure

| kind | example | template |
|---|---|---|
| matplotlib plot | `figures/fig12_spectrum_paper.py` | section "A plot" below |
| SVG diagram (boxes, arrows, icons, embedded images) | `figures/fig1_overview.py`, `figures/fig2_architecture.py` | section "A diagram" |
| hand-drawn TikZ | `figures/tikz/deployloop.tex` | section "A TikZ figure" |
| photograph or render with no generator | `figures/static/path_overlay.png` | drop the file in `figures/static/`, add a README table row, done |

### 2. Put the inputs where they belong

- **Data a figure reads** that comes from `fishCREATE` or `sofi_control`: copy it into
  `upstream/<repo>/<same relative path>`. Keep the parent repo's directory structure so a path
  here maps 1:1 to a path there. Copy only the files the figure reads, not the folder.
- **Derived caches** (something computed once from data too large to ship): `data/`, plus a row
  in `data/README.md` naming the tool that regenerates it.
- **Rasters with no generator** (photos, CAD renders, simulator stills): `figures/static/`.
- **Nothing** with a home directory, drive letter or machine name in it. If a copied upstream
  script has one, replace it with `@@SOFI_ROOT@@/...`.
- Do not add raw video, policy checkpoints beyond the two in `data/policies/`, or training
  logs. Those are archive material.
- Line endings: `.gitattributes` has `* -text`, so CSVs keep their CRLF. Do not "fix" them.

### 3. Write the generator in `figures/`

Naming: `fig<N>_<topic>.py` if it has a paper figure number, `fig_<topic>.py` if not. Helper
modules that are not figures start with an underscore (`_segfit.py`, `_svg_pipeline.py`).
The **output stem** is `fig<N>_<name>` with the report's figure number and no other prefix or
suffix (`save(fig, "fig6_motor_overlay", expect="col")`); a variant that is not in the paper
takes a suffix (`fig8_reach_collage_split`). SVG generators write the editable `.svg` to
`.gen/svg/` and only the `.pdf` and `.png` to `out/`. When the paper's figure numbering changes,
the stems change with it, in the script, the `Makefile` comment, `PAIRS` and the README table.

Every path in the script is either relative to the token or derived from `__file__`:

```python
ROOT = Path("@@SOFI_ROOT@@")                 # -> .gen/ after configure
sys.path.insert(0, str(ROOT / "figures"))    # for paper_style and helpers
DATA = ROOT / "upstream" / "sofi_control" / "..."
```

The token is replaced textually, so it must appear as a plain string. Do not build it from
parts or read it from an environment variable.

### 4. Register it

- `Makefile`, target `figures`: one line, `@cd $(GEN) && $(PY) fig<N>_<topic>.py [flags]`, with
  the output stem as the trailing comment. Any flag the paper needs (like `--pair`,
  `--variant compact`) goes here, so `make` is the record of how the paper's version was built.
  Only the paper's variant is built; other variants are listed in the README's second table.
- `README.md`, table "Figure to script to data": one row, paper figure, script with its flags,
  what it reads. If it is not in the paper yet, use the second table ("Not currently in the
  paper").
- `tools/check_against_paper.py`, list `PAIRS`: the output stem, so `make check` compares it.
- `docs/FIGURE_RECIPE.md`: nothing to add if the figure follows `paper_style`; note any
  deliberate deviation there.
- If it needs a package not in `environment.yml`, add it there with a comment saying which
  figure needs it.

### 5. Build and verify

```bash
python tools/configure.py            # or just: make figures
cd .gen/figures && python fig<N>_<topic>.py
```

Then check, in this order:

1. The script printed `OK  <stem>.pdf  246.9 pt (col)` (or `505.9 pt (full)`). If it printed
   `OFF`, or nothing, the width is wrong and the type will be rescaled on the page.
2. `pdffonts out/<stem>.pdf` shows embedded fonts and no Type 3.
3. Open the PNG twin and look at it at roughly print size before looking at the PDF at 400 %.
4. Rebuild the paper with the new PDF **on the page** and look there. That is the only view
   that counts.

### 6. Hand the PDF to the paper

Copy `out/<stem>.pdf` into the Overleaf project's `Images/draft/` (drafts) or `Images/`
(final); `\graphicspath` covers both, so the `.tex` includes it by bare filename:

```latex
\includegraphics[width=\columnwidth]{<stem>.pdf}     % or width=\textwidth for figure*
```

Never any other width, and never a scale factor. Afterwards `make check` should say
`IDENTICAL` for the new stem.

### 7. Commit

Source only: the script, its inputs under `upstream/`, `data/` or `figures/static/`, and the
four registrations. `out/` and `.gen/` are ignored and stay that way.

## A plot

The rules are in `FIGURE_RECIPE.md`; the skeleton that satisfies them is:

```python
#!/usr/bin/env python3
"""Fig N (<what it shows>) at the paper's column width. <One sentence: the question it answers.>"""
import sys
from pathlib import Path
ROOT = Path("@@SOFI_ROOT@@")
sys.path.insert(0, str(ROOT / "figures"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import use_paper_style, size, save, CB, ROLE, LW, EMPH_LW
use_paper_style()

# ... load from ROOT / "upstream" / ... or ROOT / "data" / ...

fig, ax = plt.subplots(figsize=size(1.0, aspect=0.62), layout="constrained")
# ... plot with colours from CB / ROLE; never pass fontsize= or linewidth= literals
save(fig, "fig<N>_<topic>_paper", expect="col")     # expect="full" for a figure*
```

Things people get wrong: `bbox_inches="tight"` anywhere (changes the width), a legend that
spills outside the axes without `loc="outside ..."`, a second font size for tick labels, and
running the script from `figures/` instead of `.gen/figures` (the `save()` destination is
derived from where `paper_style.py` lives, so the output lands next to the source tree).

## A diagram

Figures 1 and 2 are drawn as SVG by Python, not plotted. Follow their pattern rather than the
plot skeleton:

- Import the helpers: `_svg_pipeline.py` (text, equations with `_{}`/`^{}`/`\r{}`/`\b{}` markup,
  boxes, arrows, Tabler icons from `figures/icons/`, network and waveform glyphs) and
  `_svg_curriculum.py` (page geometry constants `W`, `FS`, the curriculum stage list,
  `export_pdf`, `export_png`).
- Lay the figure out in millimetres at the final width (`cf.W` = the column, `177.8` for the
  text width), with `width`/`height` and `viewBox` equal, and include it at that width. A font
  size in the script is then the printed size; there is no scaling step to reason about.
- Text: `FS_MAIN` / `cf.FS` (9 pt) for titles and labels, `FS_SMALL` / `FS2` (8 pt) for
  annotations; use real `<text>` elements, never outlines, so the SVG stays editable in
  Inkscape. The `font-family` is `tp.FONT`, the shared Times stack.
- Rasters go in as base64 PNG data URIs (`tp.png_data_uri`), so the SVG is self-contained.
- Export: `cf.export_pdf(svg, pdf, W, H)` uses Inkscape if it is on the PATH, else a headless
  Chrome/Edge print sized to the page; `cf.export_png` writes the preview. Without either tool
  you get only the SVG, so say so in the README caveat if a new tool dependency appears.
- Register exactly as for a plot (Makefile, README, `PAIRS`, FIGURE_RECIPE), and add a line to
  the "diagram figures" section of `FIGURE_RECIPE.md` if the new one deviates from the plot
  rules in a new way.

## A TikZ figure

`figures/tikz/<name>.tex` is a thin standalone wrapper around `_body_<name>.tex`, and the body
is byte-identical to the block in the paper's `Chapters/`. Edit the body, paste it back into
the paper, and `make tikz` builds the standalone PDF into `out/` for checking. Add the name to
the `for n in ...` loop in the `tikz` target of the `Makefile` and a README row that says
"none, hand-drawn" under "reads".

## Pulling a fix from a parent repo

`upstream/` is a pruned mirror, so a fix in `fishCREATE` or `sofi_control` does not arrive by
itself. Copy the changed file to the same relative path under `upstream/`, replace any absolute
path with `@@SOFI_ROOT@@/...`, and note in the commit message which parent commit it came from.
If the fix changes a figure, `make check` will show it as `MISMATCH` until the Overleaf PDF is
refreshed too.

## Platform notes

- **Windows.** `tools/configure.py` creates symlinks for data files; that needs Developer Mode
  or an elevated shell. Python's default text encoding is not UTF-8 on Windows; the scripts here
  are UTF-8 and `configure.py` reads and writes them as such. If a new tool of yours reads
  scripts or Markdown, pass `encoding="utf-8"`, or run with `PYTHONUTF8=1`.
- **LaTeX.** `usetex` is on for all plots, so a TeX installation with `dvipng` and `mathptmx`
  is needed; the first draw fails without them.
- **MuJoCo.** Figures 5 and 6 simulate; they take a few minutes and write a temporary XML into
  `.gen/`, which is disposable.
- **Inkscape / Chrome.** Only the diagram figures need one of them, for the PDF export.

## For agents specifically

- Before touching anything, run `make figures` (or the single script) once and confirm the
  existing outputs build, so you know the environment works and what "unchanged" looks like.
- Do the registration steps in the same change as the script. A generator that is not in the
  `Makefile` and `PAIRS` is invisible to `make check` and will be missed.
- Never write into `.gen/` or `out/` by hand, never commit them, and never introduce an absolute
  path; grep your diff for `C:/`, `/home/`, `/Users/` and `icra_report` before finishing.
- Report the printed width line and the page-on-paper check in your summary, not just "it
  built".
- If a helper you import was renamed or changed by someone else (it happens; `_svg_pipeline.py`
  is shared), fix the caller, not the helper, unless the helper is yours.
