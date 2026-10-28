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

## Figure, script, inputs

| Paper figure | Output | Script | Reads |
|---|---|---|---|
| 1 overview | `fig1_overview` | `fig1_overview.py` (boxed variant; `--variant compact\|rich\|text\|minimal` for the others) | `static/` renders and stills, the six strobe composites in `fishdeploy/data/02_deploy_20260520/analysis/report_strobes/` |
| 2 training pipeline | `fig2_training_pipeline` | `fig2_training_pipeline.py` | `static/40_hull_hinges_fluid.png`, `static/sim_collage_strobe.png` (+ `.json` sidecar) |
| 3 hardware | `fig3_experimental_setup` | `fig3_experimental_setup.py` | `static/hw_cad_cutaway.png`, `static/real_fish_cut.png` (cut from `real_fish.jpeg` by `tools/cut_real_fish.py`) |
| 4 tail stiffness | `fig4_tailstiff` | `fig_tailstiff_paper.py` | `../sysid/tail_stiffness/`, `../geometry/SOFI.xml`, the 100 `r17.5` air sweeps in `fishdeploy/data/03_sysid/tail_benchmark/air/`, a MuJoCo static equilibrium |
| 5 fluid fit | `fig5_fluidfit` | `fig5_fluidfit_paper.py --pair` | the twelve `03_field2sim_ready.csv` in `fishdeploy/data/01_field_openloop_20260401/`, the model from `../geometry/sofi.py` |
| 6 motor overlay | `fig6_motor_overlay` | `fig6_motor_overlay_paper.py` | `fishdeploy/data/02_deploy_20260520/…full_success_2/raw/run.csv`, `SOFI.xml`, `SOFI_motorid.xml` |
| 7 deployment collage | `fig7_deploy_collage` | `fig_deploy_collage_paper.py` (divider variant) | five strobe composites, `fishdeploy/data/02_deploy_20260520/analysis/{cache,cache_redetect}/`, `fishdeploy/analysis/plot_nav_collage.py` |
| 8 reach collage | `fig8_reach_collage` | `fig_reach_collage_paper.py` | `fishdeploy/data/02_deploy_20260520/analysis/{cache,cache_redetect,cache_heading}/`, `reachability/run_geometry.csv` |
| 9 tail-velocity power | `fig9_velspectrum` | `fig_velspectrum.py` (`--no-hardware` drops the deployed runs) | the twelve field runs, the deployment `run.csv` files, `data/sim_rollouts.npz` |
| 10 gait traces | `fig10_gait_traces` | `fig_gaits_traces_paper.py` | `data/gait_traces.npz` |

`paper_style.py` holds the one size and font standard (`docs/FIGURE_RECIPE.md`); `_paths.py` resolves
where the inputs are. `tools/compress_figures.py` re-encodes the embedded photographs as JPEG so the paper
stays under the submission size limit; it runs as the last step of `make figures`.

Not in the paper, kept because they were used along the way: `fig4c_keff_paper.py` (the k_eff intercept
panel), `fig12_spectrum_paper.py` (command spectrum of the 43 runs), `fig_velspectrum_lpf.py` (adds the
low-pass-filtered direct-action policy; needs `data/sim_rollouts_lpf.npz`, regenerated with
`tools/probe_sim_rollouts.py`), `fig_domain.py` / `fig_domain_deploy.py` (the identified domain),
`fig_figure8_paper.py` (figure-eight pursuit alone), `fig2_architecture.py` (superseded Fig. 2),
`render_hull_hinges_fluid.py` (the simulated-fish render in `static/`), and the two TikZ drawings in `tikz/`
(`make tikz`).

## Cached simulation inputs (`data/`)

| File | What | Regenerate |
|---|---|---|
| `sim_rollouts.npz` | commanded and measured motor traces of 25 episodes each of the CPG and the direct-action policy, under domain randomisation | `python tools/probe_sim_rollouts.py` (needs `../policies/` and the environments; the published Fig. 9 rests on the cached file, so regenerating moves its numbers slightly) |
| `gait_traces.npz` | the three motor-angle traces of Fig. 10, recovered from the vector PDF of the original figure (`tools/extract_gait_traces.py`), because the rollouts behind it were not kept | not regenerable |

## Documents

- `docs/RUNS.md`: the twelve field runs, their roles in the fit, and where the identified domain lies.
- `docs/DEPLOY_RUNS.md`: the 43 deployment runs with outcome and target geometry.
- `docs/TABLES.md`: the paper's tables and where their numbers come from.
- `docs/FIGURE_RECIPE.md`, `docs/ADDING_A_FIGURE.md`: the figure standard and the checklist for a new one.

## Things that will catch you out

- `paper_style.save()` writes next to `paper_style.py` (`out/`), whatever the working directory.
- `.gitattributes` in fishdeploy keeps the CSVs CRLF; normalising them would change Fig. 9's bytes.
- The fluid-fit figure writes a temporary MuJoCo XML into `../geometry/` and removes it.
- `make check` compares extracted text and byte size. The two figures with embedded photographs (5 and 7)
  can differ by a few bytes between versions of the PDF library that re-encodes the images; a pixel
  comparison of the rendered pages is the tie-breaker.
