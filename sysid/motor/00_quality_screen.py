"""Module 00 — Run Quality Screener.

Produces run_index.json (gitignored). User must review before running modules 01-05.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import (
    load_field_test_runs, load_rl_deploy_runs, parse_run_name_field,
    FIELD_TEST_ROOT, RL_DEPLOY_ROOT,
)
from sysid.motor._quality_utils import (
    compute_rl_run_signals, assign_quality_tier, compute_usable_window,
)

OUT_PATH = Path(__file__).parent / "run_index.json"


def screen_field_test():
    """All 15 field test runs marked ok. Just index paths and metadata."""
    runs_data = load_field_test_runs()
    result = {}
    for run_name, df in runs_data.items():
        freq_hz, amp_mm = parse_run_name_field(run_name)
        volt_vals = df["Volt"].dropna().values
        result[run_name] = {
            "status":        "ok",
            "flags":         [],
            "freq_Hz":       freq_hz,
            "amp_mm":        amp_mm,
            "voltage_range": [float(volt_vals.min()), float(volt_vals.max())]
            if len(volt_vals) else [None, None],
        }
    return result


def screen_rl_deploy():
    """Screen all May 20 RL deploy runs."""
    runs_data = load_rl_deploy_runs("20260520_*")
    result = {}
    for run_name, df in runs_data.items():
        if "motor_setpoint" not in df.columns:
            result[run_name] = {"quality": "X", "flags": ["no_motor_setpoint_column"],
                                "usable_window": [0, 0], "reached_target": False}
            continue

        sigs = compute_rl_run_signals(df)
        quality = assign_quality_tier(sigs)
        start, end = compute_usable_window(df, sigs["vision_lost_end_row"])

        flags = []
        if sigs["motor_frozen"]:          flags.append("motor_frozen")
        if sigs["vision_lost_early"]:     flags.append("vision_lost_early")
        if sigs["jitter_fraction"] > 0.10: flags.append(f"jitter_{sigs['jitter_fraction']:.2f}")
        if sigs["spurious_zero_rows"]:    flags.append(f"spurious_zeros_{len(sigs['spurious_zero_rows'])}")

        result[run_name] = {
            "quality":         quality,
            "flags":           flags,
            "usable_window":   [start, end],
            "reached_target":  sigs["reached_target"],
            "run_duration_s":  round(sigs["run_duration_s"], 1),
            "jitter_fraction": round(sigs["jitter_fraction"], 3),
            "vision_lost_end_row": sigs["vision_lost_end_row"],
        }
    return result


def main():
    index = {
        "field_test": screen_field_test(),
        "rl_deploy":  screen_rl_deploy(),
    }

    rl = index["rl_deploy"]
    n_total = len(rl)
    counts = {t: sum(1 for v in rl.values() if v.get("quality") == t) for t in ("A","B","C","X")}
    print(f"\nRL deploy: {n_total} runs — A:{counts['A']} B:{counts['B']} C:{counts['C']} X:{counts['X']}")
    print(f"Field test: {len(index['field_test'])} runs (all ok)")
    print(f"\nWriting {OUT_PATH}")
    print("*** REVIEW run_index.json before running modules 01-05 ***")

    with open(OUT_PATH, "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
