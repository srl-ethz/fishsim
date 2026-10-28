"""Usage: python paper/tools/make_heading_cache.py OUT.json [PATH/TO/fishdeploy]

Precompute initial_heading() for every reach run, so the figure no longer needs
the 1.5 GB of raw deployment video.  Run ONCE against the full working tree."""
import csv, glob, json, math, os, sys

import numpy as np, cv2
from dt_apriltags import Detector
TREE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FISHDEPLOY_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "fishdeploy"))
ANA  = os.path.join(TREE, "analysis")
RUNS = os.path.join(TREE, "deploy", "runs")      # needs the videos from the archive next to the CSVs
CAM  = [1055.0202, 1054.2105, 945.8357, 489.0351]
FISH, ANCHOR = 0, 1
_DET = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=1)

def initial_heading(run, n_want=12):
    vids = glob.glob(os.path.join(RUNS, f"*{run}*", "raw", "video.avi"))
    if not vids: return None
    cap = cv2.VideoCapture(vids[0]); yaws=[]; fi=0
    while len(yaws) < n_want and fi < 400:
        ret, frame = cap.read()
        if not ret: break
        fi += 1
        tm = {t.tag_id: t for t in _DET.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
              estimate_tag_pose=True, camera_params=CAM, tag_size=0.08)}
        if FISH in tm and ANCHOR in tm:
            fwd = (tm[ANCHOR].pose_R.T @ tm[FISH].pose_R)[:, 1]
            yaws.append(math.atan2(fwd[1], fwd[0]))
    cap.release()
    if not yaws: return None
    return float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))

runs = [r["run"] for r in csv.DictReader(open(os.path.join(ANA, "reachability/run_geometry.csv")))]
out={}
for r in runs:
    h = initial_heading(r)
    out[r] = h
    print(f"  {r:<28s} {'None' if h is None else round(h,6)}")
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(f"\nwrote {len(out)} headings ({sum(v is not None for v in out.values())} resolved)")
