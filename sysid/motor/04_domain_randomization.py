"""Module 04 — Domain Randomisation Synthesis.

Reads JSONs from modules 01, 02, 03, 05 and produces the final DR table.
No new raw data analysis — pure synthesis.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RESULTS = Path(__file__).parent / "results"
OUT_JSON = RESULTS / "04_domain_randomization.json"

UNCHANGED_DR = {
    "init_vel_trans_m_s":   {"type": "gaussian",  "std": 0.15,  "status": "unchanged"},
    "init_vel_rot_rad_s":   {"type": "gaussian",  "std": 0.30,  "status": "unchanged"},
    "body_mass_frac":       {"type": "gaussian",  "std": 0.05,  "status": "unchanged"},
    "body_inertia_frac":    {"type": "gaussian",  "std": 0.05,  "status": "unchanged"},
    "com_jitter_m":         {"type": "gaussian",  "std": 0.005, "status": "unchanged"},
    "tail_stiffness_frac":  {"type": "gaussian",  "std": 0.15,  "status": "unchanged"},
    "tail_damping_frac":    {"type": "gaussian",  "std": 0.15,  "status": "unchanged"},
    "fluid_coeff_frac":     {"type": "gaussian",  "std": 0.20,  "status": "unchanged"},
}


def load_result(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — run module {name[:2]} first")
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    m01 = load_result("01_motor_torque")
    m03 = load_result("03_pd_gains")
    m05 = load_result("05_noise_estimation")

    # Motor forcerange
    fr_rec = m01.get("recommended_motor_forcerange_Nm", 1.0)
    fr_range = [round(fr_rec * 0.7, 3), round(fr_rec * 1.3, 3)]

    # Motor Kp & Kv — prefer module 06 (2-D period-averaged sweep) over module 03
    m06_path = RESULTS / "06_motor_calibration.json"
    if m06_path.exists():
        with open(m06_path) as f:
            m06 = json.load(f)
        kp_range = m06.get("recommended_kp_dr_range_updated",
                            m03.get("recommended_kp_dr_range", [640.0, 960.0]))
        kv_range = m06.get("recommended_kv_dr_range_updated", [0.0, 0.6])
    else:
        kp_range = m03.get("recommended_kp_dr_range", [640.0, 960.0])
        kv_range = [0.0, 20.0]

    # Camera FPS
    fps_mean = (m05.get("camera_fps_mean_hz", {}) or {}).get("may20", 37.0) or 37.0
    fps_range = (m05.get("camera_fps_dr_range_hz", {}) or {}).get("may20", [28, 38]) or [28, 38]
    jitter_std = (m05.get("camera_fps_jitter_std_s", {}) or {}).get("may20", 0.003) or 0.003

    new_dr = {
        "motor_forcerange_Nm": {
            "type":   "uniform",
            "range":  fr_range,
            "source": "module 01",
            "status": "IMPLEMENT",
            "impl":   "model.actuator_forcerange[motor_idx] = [-fr, fr] per reset",
        },
        "motor_kp_Nm_per_rad": {
            "type":   "uniform",
            "range":  kp_range,
            "source": "module 06" if m06_path.exists() else "module 03",
            "status": "IMPLEMENT",
            "impl":   (
                "model.actuator_gainprm[motor_idx, 0] = kp; "
                "model.actuator_biasprm[motor_idx, 1] = -kp"
            ),
        },
        "motor_kv_Nm_s_per_rad": {
            "type":   "uniform",
            "range":  kv_range,
            "source": "module 06" if m06_path.exists() else "module 03",
            "status": "IMPLEMENT",
            "impl":   "model.actuator_biasprm[motor_idx, 2] = -kv",
        },
        "motor_zero_offset_rad": {
            "type":   "uniform",
            "range":  [-0.05, 0.05],
            "source": "tendon pre-tension variation ≈ ±0.625mm at 12.5mm pulley",
            "status": "IMPLEMENT",
            "impl":   "data.qpos[motor_joint_id] += rng.uniform(-0.05, 0.05) in reset_model()",
        },
        "camera_fps_hz": {
            "type":         "per_episode",
            "mean":         fps_mean,
            "range":        fps_range,
            "jitter_std_s": jitter_std,
            "source":       "module 05",
            "status":       "IMPLEMENT",
            "impl": (
                "SensorModel: randomise camera_period at episode reset; "
                "sample each inter-frame interval from Normal(1/fps_episode, jitter_std)"
            ),
        },
        "tendon_deadzone_rad": {
            "type":   "uniform",
            "range":  [0.0, 0.15],
            "source": "designed — not yet measured",
            "status": "DESIGNED_NOT_IMPLEMENTED",
            "impl":   "asymmetric forcerange on slack side — deferred, requires non-standard MuJoCo workaround",
        },
    }

    result = {
        "unchanged_dr": UNCHANGED_DR,
        "new_dr":        new_dr,
        "summary": {
            "items_to_implement": [k for k, v in new_dr.items() if v["status"] == "IMPLEMENT"],
            "items_deferred":     [k for k, v in new_dr.items() if "NOT_IMPLEMENTED" in v["status"]],
        },
    }

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print("=== Domain Randomisation Synthesis ===")
    print(f"Unchanged DR items: {len(UNCHANGED_DR)}")
    print(f"New DR — IMPLEMENT: {result['summary']['items_to_implement']}")
    print(f"New DR — deferred:  {result['summary']['items_deferred']}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
