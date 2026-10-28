# paper: the figures of the ICRA 2026 submission

Every figure of the paper, rebuilt from this repository and the measured data in
[fishdeploy](https://github.com/Ramchi99/fishdeploy).

```bash
git clone https://github.com/Ramchi99/fishdeploy ../fishdeploy      # or export FISHDEPLOY_ROOT=/path/to/it
cd paper
make figures        # -> out/fig<N>_<name>.pdf + .png, the ten figures (about 5 min; Figs 4-6 run MuJoCo)
make check          # text + byte-size comparison against reference/, the PDFs as submitted
```

Needs the `fishsim` environment plus a LaTeX with `dvipng` and the `mathptmx` package (matplotlib
`usetex`). Figures 1 to 3 are SVG generators and need Inkscape or Chrome/Edge for the PDF export; under
WSL the Windows installations are found automatically.

## One script per figure

The script is named like its output and like the reference PDF: `fig<N>_<name>.py` writes
`out/fig<N>_<name>.pdf` and is compared against `reference/fig<N>_<name>.pdf`.

| Figure | Script | Reads |
|---|---|---|
| 1 overview | `fig1_overview.py` | `static/` renders and stills, the figure-eight strobe in `fishdeploy/data/02_deploy_20260520/analysis/report_strobes/` |
| 2 training pipeline | `fig2_training_pipeline.py` | `static/40_hull_hinges_fluid.png`, `static/sim_collage_strobe.png` |
| 3 hardware | `fig3_experimental_setup.py` | `static/hw_cad_cutaway.png`, `static/real_fish_cut.png` (cut from `real_fish.jpeg` by `tools/cut_real_fish.py`) |
| 4 tail stiffness | `fig4_tailstiff.py` | `../sysid/tail_stiffness/`, `../geometry/SOFI.xml`, the 100 `r17.5` air sweeps in `fishdeploy/data/03_sysid/tail_benchmark/air/`, a MuJoCo static equilibrium |
| 5 fluid fit | `fig5_fluidfit.py --pair` | the twelve `03_field2sim_ready.csv` in `fishdeploy/data/01_field_openloop_20260401/`, the model from `../geometry/sofi.py` |
| 6 motor overlay | `fig6_motor_overlay.py` | `fishdeploy/data/02_deploy_20260520/…full_success_2/raw/run.csv`, `SOFI.xml`, `SOFI_motorid.xml` |
| 7 deployment collage | `fig7_deploy_collage.py` | five strobe composites, `fishdeploy/data/02_deploy_20260520/analysis/{cache,cache_redetect}/`, `fishdeploy/analysis/plot_nav_collage.py`; the figure-eight panel is `_fig7_figure8_panel.py` |
| 8 reach collage | `fig8_reach_collage.py` | `fishdeploy/data/02_deploy_20260520/analysis/{cache,cache_redetect,cache_heading}/`, `reachability/run_geometry.csv` |
| 9 tail-velocity power | `fig9_velspectrum.py` (`--no-hardware` drops the deployed runs) | the twelve field runs, the 43 deployment `run.csv` files (`paper_runs.txt`), `data/sim_rollouts.npz` |
| 10 gait traces | `fig10_gait_traces.py` | `data/gait_traces.npz` |

Shared modules: `paper_style.py` is the one size and font standard of the plots (`docs/FIGURE_RECIPE.md`),
`_svg.py` the toolkit of the three SVG diagrams (primitives, icons from `icons/`, PDF export), `_paths.py`
resolves where the inputs are. `tools/compress_figures.py` re-encodes the embedded photographs as JPEG so
the paper stays under the submission size limit; it runs as the last step of `make figures`.
`render_hull_hinges_fluid.py` makes the simulated-fish render in `static/` (re-run only when the model changes).

## Cached simulation inputs (`data/`)

| File | What | Regenerate |
|---|---|---|
| `sim_rollouts.npz` | commanded and measured motor traces of 25 episodes each of the CPG and the direct-action policy, under domain randomisation | `python tools/probe_sim_rollouts.py` (needs `../policies/` and the environments; the published Fig. 9 rests on the cached file, so regenerating moves its numbers slightly) |
| `gait_traces.npz` | the three motor-angle traces of Fig. 10, recovered from the vector PDF of the original figure (`tools/extract_gait_traces.py`), because the rollouts behind it were not kept | not regenerable |

## Documents

- `docs/RUNS.md`: the twelve field runs, their roles in the fit, and where the identified domain lies.
- `docs/DEPLOY_RUNS.md`: the 43 deployment runs with outcome and target geometry.
- `docs/TABLES.md`: the paper's tables and where their numbers come from.
- `docs/FIGURE_RECIPE.md`: the figure standard (sizes, fonts, the build-at-final-size rule).

## Adding a figure

Copy the closest script, keep the five lines of the recipe (`use_paper_style()`, `size()`, plot with the
module's colours and widths, `save(fig, "fig<N>_<name>", expect="col"|"full")`), add the line to the
`Makefile` and the stem to `tools/check_against_paper.py`, drop the submitted PDF into `reference/`, and
look at the figure on the page, not at the standalone PDF.

## Things that will catch you out

- `paper_style.save()` writes next to `paper_style.py` (`out/`), whatever the working directory.
- `make check` compares byte sizes. The SVG figures are printed by Chrome, and a Chrome update changes a
  few bytes of font metadata without changing the picture; the reference PDFs were printed by Chrome 152.
- Never run the figures and a fluid fit at the same time: both write temporary XML models into
  `../geometry/`.

Figures that were tried and are not in the paper (report and presentation variants, the identification
domain maps, the filtered-spectrum variant of Fig. 9) are kept privately, not here.
