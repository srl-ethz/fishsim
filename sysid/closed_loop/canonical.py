"""The curated closed-loop segments: how the fits find and read them.

The segments were curated from the deployment videos in the private fluidID workspace; what is public
is the manifest (`layer2_parts.yml`), the 1 kHz canonical CSVs it produced and the motor-zero
corrections, all in `fishdeploy/data/02_deploy_20260520/canonical/` (see fishdeploy `analysis/canonical/`
for the conversion code). These helpers are the subset of that workspace the two fits need.
"""
from __future__ import annotations
import json
import os
import re
import numpy as np
import yaml
from scipy.interpolate import interp1d


def slug(name: str) -> str:
    """File-name form of a hand-written run name: ASCII letters, digits and underscores only
    ('Weird Run 180° Turn' -> 'Weird_Run_180_Turn'). The manifests keep the human names."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def run_slug(num: int, name: str) -> str:
    return f"{num:03d}_{slug(name)}"


def part_id(num: int, name: str, part: int) -> str:
    return f"{num:03d}_{slug(name)}__p{part}"


def load_layer2(path: str) -> list:
    """The curation manifest: one entry per run with its accepted [t_start, t_end] parts."""
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    return doc if doc is not None else []


def accepted_parts(layer2_doc: list) -> list[dict]:
    flat = []
    for entry in layer2_doc:
        for p in entry.get("parts", []):
            flat.append({"part_id": part_id(entry["num"], entry["name"], p["part"]),
                         "num": entry["num"], "name": entry["name"], "run_id": entry["run_id"],
                         "source": entry["source"], "part": p["part"],
                         "t_start": p["t_start"], "t_end": p["t_end"],
                         "starts_at_rest": p["starts_at_rest"]})
    return sorted(flat, key=lambda d: d["part_id"])


def _grid(t_start: float, t_end: float, sim_dt: float) -> np.ndarray:
    n = int(round((t_end - t_start) / sim_dt)) + 1
    return t_start + np.arange(n) * sim_dt


def _interp(src_t, src_v, grid):
    f = interp1d(src_t, src_v, kind="linear", bounds_error=False, fill_value=(src_v[0], src_v[-1]))
    return f(grid)


def clock_offset_s(run_dir: str) -> float:
    """Pi → laptop clock offset in seconds (laptop = pi + offset), from the run's config; 0 if absent."""
    p = os.path.join(run_dir, "raw", "run_config.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f).get("clock_offset_ms", 0.0)) / 1000.0
    return 0.0
