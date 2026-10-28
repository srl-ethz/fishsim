"""Module 05 — Sensor Noise Estimation.

Uses May 20 RL deploy runs (quality A+B within usable_window) to update
SensorModel noise parameters. Compares to May 7 shadow_zero baselines.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import (
    load_rl_deploy_runs, load_frame_timestamps,
    load_run_index, get_usable_rl_df,
)
from sysid.motor._noise_utils import (
    estimate_stationary_std, extract_fps_from_frame_timestamps,
    compute_frame_drop_rate,
)

OUT_JSON  = Path(__file__).parent / "results" / "05_noise_estimation.json"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MAY7_DEFAULTS = {
    "encoder_pos_noise_std_rad": 0.00057,
    "imu_wz_noise_std_rad_s":    0.002,
    "vision_pos_noise_std_m":    0.0013,
    "vision_vel_noise_std_m_s":  0.016,
    "vision_latency_min_s":      0.010,
    "vision_latency_max_s":      0.050,
    "frame_drop_rate":           0.15,
    "camera_fps_mean_hz":        36.8,
}

SIM_VALUES = {
    "encoder_pos_noise_std_rad": 0.001,
    "encoder_vel_noise_std_rad_s": 0.01,
    "imu_wz_noise_std_rad_s":    0.006,
    "imu_bias_std_rad_s":        0.005,
    "vision_pos_noise_std_m":    0.002,
    "vision_vel_noise_std_m_s":  0.02,
    "vision_latency_min_s":      0.010,
    "vision_latency_max_s":      0.050,
    "frame_drop_rate":           0.10,
    "camera_fps_mean_hz":        30.0,
}


def _within_30pct(may20_val, may7_val):
    if may7_val == 0:
        return may20_val < 1e-6
    return abs(may20_val - may7_val) / abs(may7_val) < 0.30


def estimate_encoder_noise(runs, run_index):
    """Std of motor_pos_rad in stationary segment (first rows of usable window)."""
    stds = []
    for run_name, df in runs.items():
        usable = get_usable_rl_df(run_name, df, run_index)
        if usable is None or len(usable) < 20:
            continue
        # First 5 rows of usable window = motor at rest before first move
        static_pos = usable["motor_pos_rad"].iloc[:5].values
        if np.std(static_pos) > 0.05:
            continue  # not actually stationary
        stds.append(estimate_stationary_std(static_pos))
    return float(np.median(stds)) if stds else None


def estimate_fps_and_drops(runs, run_index):
    """Camera FPS and drop rate from frame_seq column and frame_timestamps."""
    all_fps, all_jitter, all_drop_rates = [], [], []
    for run_name in sorted(runs.keys()):
        df = runs[run_name]
        usable = get_usable_rl_df(run_name, df, run_index)
        if usable is None or "frame_seq" not in usable.columns:
            continue

        drop_rate = compute_frame_drop_rate(usable["frame_seq"].values)
        all_drop_rates.append(drop_rate)

        # Try frame_timestamps.csv for FPS — use laptop_time column
        ts_df = load_frame_timestamps(run_name)
        if ts_df is not None and len(ts_df) >= 2:
            # Use laptop_time column if available, otherwise fall back to second column
            if "laptop_time" in ts_df.columns:
                timestamps = ts_df["laptop_time"].values
            else:
                timestamps = ts_df.iloc[:, 1].values
            fps, jitter = extract_fps_from_frame_timestamps(timestamps)
            if 10 < fps < 60:
                all_fps.append(fps)
                all_jitter.append(jitter)

    return (
        float(np.mean(all_fps))       if all_fps       else None,
        float(np.std(all_fps))        if all_fps       else None,
        float(np.mean(all_jitter))    if all_jitter    else None,
        float(np.mean(all_drop_rates)) if all_drop_rates else None,
    )


def estimate_vision_latency(runs, run_index):
    """Latency distribution from vision_age_ms on fresh-frame rows."""
    all_latencies = []
    for run_name, df in runs.items():
        usable = get_usable_rl_df(run_name, df, run_index)
        if usable is None or "frame_seq" not in usable.columns:
            continue
        fresh_mask = np.diff(usable["frame_seq"].values, prepend=-1) != 0
        latencies_ms = usable["vision_age_ms"].values[fresh_mask]
        all_latencies.append(latencies_ms[latencies_ms > 0])

    if not all_latencies:
        return None, None, None, None
    all_lat = np.concatenate(all_latencies) / 1000.0  # → seconds
    return (
        float(np.percentile(all_lat, 50)),
        float(np.percentile(all_lat, 95)),
        float(all_lat.min()),
        float(all_lat.max()),
    )


def main():
    run_index = load_run_index()
    runs = load_rl_deploy_runs("20260520_*")

    enc_std   = estimate_encoder_noise(runs, run_index)
    fps_mean, fps_std, jitter_std, drop_rate = estimate_fps_and_drops(runs, run_index)
    lat_p50, lat_p95, lat_min, lat_max = estimate_vision_latency(runs, run_index)

    # Print comparison table
    def compare(label, may20, may7):
        if may20 is None:
            print(f"  {label:45s}: May20=N/A   May7={may7}")
        else:
            flag = "" if _within_30pct(may20, may7) else " *** DIFFERS >30%"
            print(f"  {label:45s}: May20={may20:.5f}  May7={may7}{flag}")

    print("\n=== Sensor Noise Comparison ===")
    compare("encoder_pos_noise_std_rad",   enc_std,  MAY7_DEFAULTS["encoder_pos_noise_std_rad"])
    compare("camera_fps_mean_hz",          fps_mean, MAY7_DEFAULTS["camera_fps_mean_hz"])
    compare("frame_drop_rate",             drop_rate, MAY7_DEFAULTS["frame_drop_rate"])
    if lat_p50:
        print(f"  {'vision_latency_p50_s':45s}: May20={lat_p50:.4f}  sim_min={SIM_VALUES['vision_latency_min_s']}")

    # --- Plot: FPS distribution ---
    if fps_mean is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        # Collect per-run FPS values for histogram
        fps_vals = []
        for run_name in sorted(runs.keys()):
            ts_df = load_frame_timestamps(run_name)
            if ts_df is not None and len(ts_df) >= 2:
                if "laptop_time" in ts_df.columns:
                    timestamps = ts_df["laptop_time"].values
                else:
                    timestamps = ts_df.iloc[:, 1].values
                fps, _ = extract_fps_from_frame_timestamps(timestamps)
                if 10 < fps < 60:
                    fps_vals.append(fps)
        ax.hist(fps_vals, bins=20, edgecolor="black")
        ax.axvline(30.0,    color="blue",   ls="--", lw=2, label="Sim (30 Hz)")
        ax.axvline(fps_mean, color="red",   ls="-",  lw=2, label=f"Measured mean ({fps_mean:.1f} Hz)")
        ax.set_xlabel("Camera FPS")
        ax.set_ylabel("Count (runs)")
        ax.set_title("Observed camera FPS across May 20 RL deploy runs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "05a_camera_fps_distribution.pdf")
        fig.savefig(PLOTS_DIR / "05a_camera_fps_distribution.png", dpi=150)
        plt.close(fig)

    result = {
        "encoder_pos_noise_std_rad": {
            "may7": MAY7_DEFAULTS["encoder_pos_noise_std_rad"],
            "may20": round(enc_std, 6) if enc_std else None,
            "sim":  SIM_VALUES["encoder_pos_noise_std_rad"],
        },
        "imu_wz_noise_std_rad_s": {
            "may7": MAY7_DEFAULTS["imu_wz_noise_std_rad_s"],
            "may20": "not_estimated_water_disturbance",
            "sim":  SIM_VALUES["imu_wz_noise_std_rad_s"],
            "note": "May7 shadow_zero used — water currents contaminate May20 stationary segments",
        },
        "imu_bias_std_rad_s": {
            "may7": "~5e-5 per_run_constant",
            "may20": "not_estimated",
            "sim":  SIM_VALUES["imu_bias_std_rad_s"],
            "note": "Sim draws per-episode bias from N(0,0.005); measured is ~100x smaller — intentionally conservative",
        },
        "vision_latency_p50_s":  round(lat_p50,  4) if lat_p50  else None,
        "vision_latency_p95_s":  round(lat_p95,  4) if lat_p95  else None,
        "vision_latency_min_s":  round(lat_min,  4) if lat_min  else None,
        "vision_latency_max_s":  round(lat_max,  4) if lat_max  else None,
        "frame_drop_rate": {
            "may7": MAY7_DEFAULTS["frame_drop_rate"],
            "may20": round(drop_rate, 4) if drop_rate is not None else None,
            "sim":  SIM_VALUES["frame_drop_rate"],
        },
        "camera_fps_mean_hz": {
            "may7": MAY7_DEFAULTS["camera_fps_mean_hz"],
            "may20": round(fps_mean, 2) if fps_mean else None,
            "sim":  SIM_VALUES["camera_fps_mean_hz"],
            "recommendation": "update_to_37Hz",
        },
        "camera_fps_dr_range_hz": {
            "may7": [28, 37],
            "may20": [round(fps_mean - 2 * (fps_std or 1), 1),
                      round(fps_mean + 2 * (fps_std or 1), 1)] if fps_mean else None,
        },
        "camera_fps_jitter_std_s": {
            "may20": round(jitter_std, 5) if jitter_std else None,
            "sim":   "none",
            "recommendation": "add_within_episode_jitter",
        },
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
