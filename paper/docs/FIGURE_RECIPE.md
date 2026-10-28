# Figure standard

Every plot generator in `paper/` imports `paper_style.py`. Nothing sets its own fonts, line
widths, colours or figure size. If a figure needs something the module does not provide, the
module gets extended rather than the figure overriding it. The three SVG diagrams take the same
family and sizes from `_svg.py` (see the last section).

## Per figure

1. `from paper_style import use_paper_style, size, save, CB, ROLE, LW, EMPH_LW`
2. `use_paper_style()`
3. `fig, ax = plt.subplots(figsize=size(1.0, aspect=0.62), layout="constrained")`,
   or `size(..., full=True)` for a `figure*` spanning `\textwidth`
4. Plot. Take line widths and colours from the module; never pass `fontsize=` or
   `linewidth=` literals. If a panel needs a size, import the tier (`LABEL_PT`, `TICK_PT`,
   `PANEL_PT`) rather than writing a number.
5. `save(fig, "fig9_velspectrum", expect="col")`, or `expect="full"`
6. Rebuild the paper and look at the figure **on the page**, not at the standalone PDF.

`save(..., expect=)` refuses to write a figure whose width is not the target width, and
prints the width it wrote. That check is the point of the module: it catches the failure
described below before it reaches the PDF.

## The rules

| rule | value | source |
|---|---|---|
| font, everywhere | one family: Times in the plots (LaTeX `mathptmx`, text and maths), Times New Roman in the SVG diagrams; the paper's body text is Times too | review: no DejaVu in one panel and another face in the next |
| axis labels, titles, legends, panel text | `LABEL_PT` = 9 pt | review: 9 to 10 pt |
| tick labels | `TICK_PT` = 8 pt, the smallest type anywhere | review: nothing under 8 pt (caption size is the floor; ticks may go to 7) |
| bold panel letters | `PANEL_PT` = 10 pt | |
| data lines | ≥ 1 pt | 1 pt ideally, or thicker |
| format | vector PDF, fonts embedded, never Type 3 | Type 3 is rejected by PaperCept |
| size | built at the measured column / text width, never rescaled by LaTeX | see below |
| axes | four spines, light grid behind the data | `ax.grid()`, `set_axisbelow(True)` |
| colour | Okabe-Ito, fixed per entity | colour-blind safe; an entity keeps its hue across figures |

There is no smaller tier. A figure with a 7 pt legend or a 6 pt annotation has a bug.

## Two failure modes worth knowing about

**Build at final size. Never let LaTeX rescale.** `\includegraphics[width=\columnwidth]` on
an asset that is not exactly `\columnwidth` wide rescales every glyph in it. A 325 pt asset in
a 246 pt column is a 0.76 scale, so its 9 pt type prints at 6.8 pt. Widening the include does
not fix it, because the asset is what is wrong. The geometry, read out of the compiled
document (`\typeout{\the\columnwidth}` on the Overleaf source, `ieeeconf` letterpaper 10pt
conference with `\overrideIEEEmargins`):

```
\columnwidth = 245.718 pt (3.400 in)   \textwidth = 505.89 pt (7.000 in)   \columnsep = 14.45 pt
```

These are not the 3.5 in / 7.16 in the class documentation quotes; use the measured values
(`paper_style.COL_PT`, `FULL_PT`). If the class options change, measure again.

**`bbox_inches="tight"` silently changes the width.** It re-measures the drawn content after
layout, so a legend spilling outside the axes makes the saved figure wider than requested, and
LaTeX then scales it back down. Use `layout="constrained"` instead: it prevents clipping
without touching the size. For a legend outside the axes use
`fig.legend(loc="outside lower center")`, which constrained layout reserves space for.

## Checking a build

1. Every script prints `OK  <stem>.pdf  245.7 pt (col), labels 9 pt, ticks 8 pt` (or
   `505.9 pt (full)`).
2. `pdffonts out/<stem>.pdf`: every row `emb yes`, no `Type 3`. The plots embed
   `NimbusRomNo9L-*` (Times) and, for a few maths symbols, `CMSY`/`CMEX`; the diagrams embed
   `TimesNewRomanPS*` as TrueType.
3. After uploading, `pdffonts` on the compiled paper: every font embedded.
4. Print one page at 100 % and check that no figure text is smaller than the caption text.

## The diagram figures (Fig. 1 overview, Fig. 2 training pipeline, Fig. 3 hardware)

`fig1_overview.py`, `fig2_training_pipeline.py` and
`fig3_experimental_setup.py` are not matplotlib plots and do not go through `paper_style`. They write SVG directly (boxes, arrows, icons, embedded PNG
renders and photos), so every label is real, Inkscape-editable text, and are exported to PDF
with Chrome/Edge (or Inkscape if present).

They follow the same family (Times New Roman via SVG `font-family`) and the same sizes. Each
is laid out by hand in millimetres at the column or text width and included at that width, so
a font size in the script is the printed size: `FS_MAIN` / `FS` = 9 pt for titles, box
labels and signal labels, `FS_SMALL` / `FS2` = 8 pt for equations, notes and icon captions,
`FS_STEP` = 10 pt for Fig. 1's numbered step titles. The layouts were redrawn for these sizes
in September 2026 (wider boxes, taller pages, a narrower simulator block in Fig. 2).

A warning from that rework: do not try to get larger type by drawing at a smaller size on a
page scaled up by k and letting LaTeX scale it back. The include scales the type by 1/k as
well, so nothing is gained; the diagrams printed at 7 pt for a day that way. `PRINT_SCALE`
in `_svg.py` is kept at 1 so the page always equals the layout.

| rule | plots | diagrams |
|---|---|---|
| type | 9 pt labels, 8 pt ticks | 9 pt titles and labels, 8 pt annotations, 10 pt step titles |
| font | Times via LaTeX (`mathptmx`) | Times New Roman, Times, Liberation Serif via SVG `font-family`; Chrome embeds the one it finds as TrueType. A `Segoe UI Symbol` fallback covers the arrows and double bars Times lacks (else Chrome would set them from Cambria Math as Type 3) |
| export | `fig.savefig` | Inkscape if on the PATH, else a headless Chrome/Edge print sized to the SVG page; without either you get the SVG only |
| axes, grid | four spines, grid | schematic mini plots with L-shaped axes and no ticks |

They take their inputs from `static/` (renders, stills, the CAD cutaway) and, for the
figure-eight strobe of Fig. 1, from `fishdeploy/data/02_deploy_20260520/analysis/report_strobes/`.
`make figures` builds `out/fig1_overview.pdf`, `out/fig2_training_pipeline.pdf` and
`out/fig3_experimental_setup.pdf` with their `.png` twins; the editable `.svg` sources land in
`out/svg/`. Fig. 3 is the CAD cutaway with its labels and arrows drawn by the script (the tips
taken from the earlier Inkscape drawing), so the type follows the standard; its panel B is a
photograph of the built fish, `static/real_fish_cut.png`, cut out of `real_fish.jpeg`
and relit by `tools/cut_real_fish.py` (needs rembg, see its docstring; the cutout is committed). The SVG
helper `_svg.py` (primitives, equation markup, icons, PDF export) and the Tabler icons in `icons/` (MIT,
licence included) belong to them.

## History

The figures were first rebuilt in September 2026 to a 7 pt Latin Modern standard, from assets
that printed as small as 2 pt (the reach collage) and 4 pt (the fluid fit, motor overlay),
after being authored wide and scaled down by LaTeX. The review then set the present standard:
one family and a floor at the caption size; the family was Helvetica for a day and is Times,
matching the body text, which is what `paper_style.py` encodes now.
