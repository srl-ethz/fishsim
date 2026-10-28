# The deployment runs, 2026-05-20

Every closed-loop hardware run of the trajectory-generator policy from the deployment session, with its
outcome, target geometry and tracking quality. This is the deployment counterpart to
`RUNS.md`, which covers the twelve open-loop runs the fluid model was identified on.

Per-run logs: `upstream/sofi_control/rl_deploy/runs/<timestamp>_<run>/raw/run.csv`.
Vision caches: `upstream/sofi_control/rl_deploy/analysis_20260520/cache/`.
Raw video is not in this repo; see `ARCHIVE.md`.

Hardware closed-loop swimming runs of the 24-D trajectory-generator policy. Every run has a real-time trace
video in the archive. Static- and disturbance-target runs feed the fish-frame reachability
plot (`reachability/run_geometry.csv`, and the reach-collage figure).

**Raw data is read-only** — nothing here was written into the `runs/.../raw/` folders.

Legend: drops = recorder dropped frames (timing recovered by two-phase reconstruction, see the note at the end); miss% = frames where the fish tag was not detected (gaps linearly interpolated); ⚠️ marks miss% > 10%.


## Static-target runs

| run | outcome | target r (m) | target θ (°) | min dist (m) | drops | miss% | notes |
|---|---|---|---|---|---|---|---|
| hope_didnt_work_2 | ✅ success | 1.61 | -33 | 0.045 | — | 9% | FIRST EVER successful RL run |
| hope_didnt_work_3 | ❌ fail | 1.76 | +57 | 1.410 | — | ⚠️22% |  |
| hope_didnt_work_5 | ✅ success | 1.80 | +10 | 0.060 | 264 (14%) | 4% | has frame drops |
| hope_didnt_work_6 | ✅ success | 1.76 | +34 | 0.026 | — | ⚠️10% |  |
| polaris_2 | ✅ success | 1.90 | +21 | 0.008 | — | ⚠️23% |  |
| crazy_run_1 | ❌ fail | 1.71 | +59 | 0.685 | — | ⚠️21% |  |
| crazy_run_3 | ❌ fail | 1.72 | +63 | 0.860 | — | ⚠️22% |  |
| crazy_run_4 | ✅ success | 2.12 | -65 | 0.017 | — | ⚠️17% |  |
| crazy_run_5 | ✅ success | 1.96 | +51 | 0.026 | — | ⚠️15% |  |
| full_success_1 | ✅ success | 1.88 | +5 | 0.045 | — | ⚠️23% |  |
| full_success_2 | ✅ success | 1.79 | +72 | 0.023 | — | 6% |  |
| full_success_3 | ✅ success | 1.71 | -84 | 0.017 | 614 (24%) | 4% | has frame drops |
| finetune_run_1 | ✅ success | 2.79 | -83 | 0.023 | — | 7% |  |
| finetune_run_2 | ❌ fail | 3.15 | +66 | 1.461 | — | 5% | target moved by hand at the end |
| finetune_run_3 | ❌ fail | 2.59 | +103 | 2.583 | — | ⚠️26% |  |
| finetune_run_4 | ✅ success | 2.14 | -83 | 0.246 | 2251 (50%) | 3% | heavy frame drops (~50%) |
| finetune_run_5 | ✅ success | 2.82 | -49 | 0.064 | — | ⚠️11% |  |
| finetune_run_6 | ✅ success | 1.49 | -175 | 0.016 | 115 (6%) | 3% | target out-of-distribution (behind) |
| finetune_run_7 | ✅ success | 1.73 | +158 | 0.021 | — | 7% | target out-of-distribution (behind) |
| finetune_run_8 | ❌ fail | 1.66 | +112 | 1.656 | — | 2% |  |
| finetune_run_9 | ❌ fail | 0.96 | -110 | 0.725 | 2579 (48%) | 0% | heavy frame drops (~48%) |
| finetune_run_10 | ❌ fail | 0.93 | +99 | 0.922 | — | ⚠️14% |  |

## Disturbance runs (perturbed)

| run | outcome | target r (m) | target θ (°) | min dist (m) | drops | miss% | notes |
|---|---|---|---|---|---|---|---|
| insane_run_1 | ❌ fail | 1.75 | +72 | 1.740 | — | ⚠️16% | initial backward velocity |
| insane_run_2 | ✅ success | 1.74 | +39 | 0.047 | 141 (7%) | 3% | initial backward velocity; has frame drops |
| insane_run_3 | ✅ success | 2.04 | -48 | 0.002 | — | ⚠️10% | initial rotational velocity |
| insane_run_4 | ✅ success | 2.17 | -5 | 0.039 | — | ⚠️18% | slight stick disturbance |
| insane_run_5 | ✅ success | 2.19 | +1 | 0.084 | — | ⚠️14% | heavy stick disturbance |
| insane_run_6 | ✅ success | 2.14 | -34 | 0.185 | — | ⚠️15% | fluid disturbance |
| insane_run_7 | ✅ success | 1.64 | +42 | 0.031 | — | ⚠️13% | fluid disturbance |
| insane_run_8 | ✅ success | 2.05 | -9 | 0.015 | — | ⚠️15% | heavy fluid disturbance |

## Special

| run | outcome | frames | drops | miss% | notes |
|---|---|---|---|---|---|
| polaris | 🔍 review | 1639 | — | 9% | SPECIAL — flagged for later review |

## Moving-target runs

| run | outcome | frames | drops | miss% | notes |
|---|---|---|---|---|---|
| amazing_run_1 | ✅ success | 920 | — | ⚠️14% | moving simple target |
| amazing_run_2 | ✅ success | 1284 | — | 10% | moving target, circle |

## Figure-eight runs

| run | outcome | frames | drops | miss% | notes |
|---|---|---|---|---|---|
| amazing_run_3 | ❌ fail | 1205 | — | ⚠️10% | figure-eight attempt 1 |
| amazing_run_4 | ❌ fail | 0 | – | – | **CORRUPT video (0 frames)** — figure-eight attempt 2 — CORRUPT VIDEO (0 frames) |
| amazing_run_5 | ❌ fail | 1020 | — | ⚠️14% | figure-eight attempt 3 |
| amazing_run_6 | 🟡 semi | 1793 | — | 6% | figure-eight attempt 4 |
| amazing_run_7 | ❌ fail | 809 | — | ⚠️15% | figure-eight attempt 5 |
| amazing_run_8 | 🟡 semi | 917 | — | ⚠️15% | figure-eight attempt 6 |
| figure_eight_2 | ❌ fail | 1532 | — | 6% | figure-eight attempt 7 |
| figure_eight_3 | ❌ fail | 1440 | — | 8% | figure-eight attempt 8 |
| figure_eight_5 | 🟡 semi | 1787 | 399 (18%) | 3% | figure-eight attempt 9; has frame drops |
| figure_eight_6 | ✅ success | 2805 | 729 (21%) | 4% | figure-eight attempt 10 — SUCCESS; has frame drops |

## Assumptions & things to know

- **Real-time playback:** videos are variable-frame-rate; each frame is held for its true inter-frame time. Runs with recorder frame-drops were time-corrected (two-phase reconstruction) so the tail does not speed up. Affected: hope_didnt_work_5, full_success_3, finetune_run_4, finetune_run_6, finetune_run_9, insane_run_2, figure_eight_5, figure_eight_6.
- **Detection gaps (miss% > 10%):** the fish tag is intermittently lost to motion blur when the fish swims fast (lowering the AprilTag decimation did **not** help). Those frames are linearly interpolated for both the trace and the reachability path. Flagged runs: amazing_run_1, amazing_run_3, amazing_run_5, amazing_run_7, amazing_run_8, crazy_run_1, crazy_run_3, crazy_run_4, crazy_run_5, finetune_run_10, finetune_run_3, finetune_run_5, full_success_1, hope_didnt_work_3, hope_didnt_work_6, insane_run_1, insane_run_3, insane_run_4, insane_run_5, insane_run_6, insane_run_7, insane_run_8, polaris_2.
- **Corrupt recording:** amazing_run_4 has 0 frames in video.avi (the old single-VideoWriter pipeline produced an empty file). No video or geometry possible; listed for completeness.
- **Fish frame:** right-handed top-down — origin at the fish's initial position, +x = its initial swim direction, +y = its left. Target angle/radius are in this frame (0° ahead, +90° left, −90° right, ±180° behind). Verified against the two 'target behind' runs (finetune_run_6/7 → ±175°/+158°). Initial pose = position median + circular-mean yaw over the first 8 detected frames.
- **Outlier rejection:** the AprilTag pose occasionally flips (ambiguity / anchor jitter), producing world-position jumps up to ~2 m. The fish world-track is velocity-gated (>~1 m/s steps dropped & re-interpolated) then lightly smoothed before the fish-frame transform. Most affected: finetune_run_4 (40 pts), finetune_run_6 (19), finetune_run_7 (3), finetune_run_3 (1).
- **Trajectory truncation:** successful runs are drawn up to the point of closest approach to the target; failed runs are drawn to the end of the run.
- **Reachability scope:** only static + disturbance runs (target fixed). Moving-target and figure-eight runs are video-only. `polaris` is excluded.
- **Reached ratios are descriptive only** — targets were placed by hand, not sampled uniformly, so the ratios are not an unbiased reachability rate.
