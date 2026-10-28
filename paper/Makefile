# fishsim/paper: rebuild every figure of the paper from this repository plus the fishdeploy data.
#
#   make figures        -> out/fig<N>_<name>.pdf (+ .png), the ten figures, numbered as in the paper
#   make check          -> compare out/ against reference/ (the PDFs as submitted): text + byte size
#   make tikz           -> the two hand-drawn TikZ figures (not in the paper's figure list)
#
# Measured data is read from ../../fishdeploy (or $FISHDEPLOY_ROOT), see _paths.py.
# Figs 4-6 run MuJoCo; Figs 1-3 are SVG and need Inkscape or Chrome/Edge for the PDF export.
PY ?= python

.PHONY: all figures check tikz clean
all: figures

figures:
	@rm -rf out && mkdir -p out
	@$(PY) fig1_overview.py                            # fig1_overview (boxed variant, the default)
	@$(PY) fig2_training_pipeline.py                   # fig2_training_pipeline
	@$(PY) fig3_experimental_setup.py                  # fig3_experimental_setup
	@$(PY) fig_tailstiff_paper.py                      # fig4_tailstiff
	@$(PY) fig5_fluidfit_paper.py --pair               # fig5_fluidfit
	@$(PY) fig6_motor_overlay_paper.py                 # fig6_motor_overlay
	@$(PY) fig_deploy_collage_paper.py                 # fig7_deploy_collage (divider variant, the default)
	@$(PY) fig_reach_collage_paper.py                  # fig8_reach_collage
	@$(PY) fig_velspectrum.py                          # fig9_velspectrum (with the deployed hardware runs)
	@$(PY) fig_gaits_traces_paper.py                   # fig10_gait_traces
	@$(PY) tools/compress_figures.py out/fig*.pdf      # JPEG-encode the embedded photos (paper must stay < 6 MB)
	@echo "--- figures written to out/ ---" && ls out

check:                  ## compare against reference/ (or PAPER_IMAGES=<dir>)
	@$(PY) tools/check_against_paper.py $(PAPER_IMAGES)

tikz:
	@mkdir -p out
	@cd tikz && for n in tailstiff deployloop; do \
	   pdflatex -interaction=batchmode -halt-on-error -output-directory=../out $$n.tex >/dev/null \
	     && echo "  OK  out/$$n.pdf" || echo "  FAILED $$n"; done
	@rm -f out/*.aux out/*.log

clean:
	@rm -rf out tikz/*.aux tikz/*.log
