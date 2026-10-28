#!/usr/bin/env python3
"""Cut the built fish out of static/real_fish.jpeg -> static/real_fish_cut.png.

The photo is a white fish leaning on a white foam board, so a plain matting pass loses the tail:
the fin lobes and the tapered tail plate vanish into the board. This does

  1. a BiRefNet-lite matte (rembg) of the whole frame, and a second one of a zoomed tail window
     (230,230)-(760,760) whose alpha is pasted over the first on the tail side of the frame;
  2. two hand-placed exclusions in photo pixels: the shadowed strip of board under the tail
     plate's straight lower edge, and the wedge of cabinet showing under the lower fin lobe;
  3. mask clean-up: largest component, hole fill, 5 px open/close, 1 px soft edge;
  4. lighting on the fish pixels only: gray-world white balance on the bright hull, a levels
     stretch (0.5th..99.3rd luminance percentile -> 4..250), a midtone lift (gamma 0.72), a
     gentle S-curve and a mild unsharp mask;
  5. the grey epoxy smudges on the hull inpainted: hull pixels (below the tag and mast, right of
     the tail plate) darker than their 71 px local median by a margin, but not the black tape,
     screws or seam (luminance and thinness filters), are filled with cv2.inpaint (Telea);
  6. rotation by ROTATE_DEG (14 deg anticlockwise, --rotate to change it). The photo is a
     three-quarter view from above, so no single angle levels everything: matching the CAD's
     principal axis (--rotate auto, about 27 deg) makes the vertical tail fin lean visibly,
     while 0 deg leaves the hull diving. 14 deg splits the difference. Then a tight crop with
     a 10 px margin, RGBA, full photo resolution.

The result is committed (static/real_fish_cut.png) and consumed by
fig3_experimental_setup.py as panel B, so this only needs re-running if the photo or the
processing changes. It needs rembg, which is not in environment.yml:

    python -m venv .venv-rembg && .venv-rembg/Scripts/pip install "rembg[cpu]" opencv-python-headless
    .venv-rembg/Scripts/python tools/cut_real_fish.py

The first run downloads the BiRefNet-general-lite weights (~180 MB) to ~/.rembg/models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "static" / "real_fish.jpeg"
DST = ROOT / "static" / "real_fish_cut.png"
CAD = ROOT / "static" / "hw_cad_cutaway.png"   # head left; Fig. 3 mirrors it
HULL_X0 = 620                                # photo x where the hull starts (tail plate ends)
HULL_Y0 = 500                                # photo y below the AprilTag; the stain filter stays off the tag
ROTATE_DEG = 14.0                            # anticlockwise; see the docstring
STAIN_DIFF = 8.0                             # a hull pixel this much darker than its 71 px median is a stain
HEAD_X0, HEAD_DIFF = 960, 4.5                # the head (right of the tape) gets a more sensitive pass
MODEL = "birefnet-general-lite"
TAIL_WIN = (230, 230, 760, 760)              # x0, y0, x1, y1 of the zoomed tail pass
# exclusions, photo pixels (1600 x 1200): the board in shadow under the tail plate's lower edge
# (372,538)->(626,742), and the cabinet under the lower fin lobe up to the board's left edge
BOARD_SHADOW = [(372, 538), (626, 742), (626, 800), (372, 800)]
CABINET_WEDGE = [(300, 604), (346, 604), (374, 538), (380, 538), (380, 700), (300, 700)]


def mattes(im: Image.Image, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The two model mattes; --cache keeps them in an .npz so the tuning steps re-run without rembg."""
    if cache and cache.is_file():
        z = np.load(cache)
        return z["full"], z["tail"]
    from rembg import new_session, remove
    sess = new_session(MODEL)
    full = remove(im, session=sess, only_mask=True, post_process_mask=False)
    tail = remove(im.crop(TAIL_WIN), session=sess, only_mask=True, post_process_mask=False)
    full, tail = np.asarray(full).astype(np.float32) / 255, np.asarray(tail).astype(np.float32) / 255
    if cache:
        np.savez_compressed(cache, full=full, tail=tail)
    return full, tail


def fish_mask(alpha: np.ndarray, tail: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    x0, y0, x1, y1 = TAIL_WIN
    win = np.zeros_like(alpha)
    win[y0:y1, x0:x1] = tail
    use = np.zeros((h, w), np.uint8)                 # the tail side only, left of the hull
    cv2.fillPoly(use, [np.array([(x0, y0), (640, y0), (640, y1), (x0, y1)], np.int32)], 1)
    alpha = np.where(use.astype(bool), np.maximum(alpha, win), alpha)

    excl = np.zeros((h, w), np.uint8)
    for poly in (BOARD_SHADOW, CABINET_WEDGE):
        cv2.fillPoly(excl, [np.array(poly, np.int32)], 1)
    excl = excl.astype(bool)
    alpha = np.where(excl, 0.0, alpha)

    hard = (alpha > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if n > 1:
        hard = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    flood = hard.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    hard |= (flood == 0).astype(np.uint8)            # screws etc. read dark but are fish
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hard = cv2.morphologyEx(cv2.morphologyEx(hard, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    soft = np.clip(cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.0), 0, 1)
    mask = np.where(hard.astype(bool), np.maximum(soft, alpha), np.minimum(soft, alpha))
    return np.where(excl, 0.0, mask)


def relight(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fish = mask > 0.9
    px = rgb[fish].astype(np.float32)
    bright = px[px.mean(1) > np.percentile(px.mean(1), 60)]
    gain = bright.mean() / bright.mean(0)            # gray-world on the white hull
    bal = np.clip(rgb.astype(np.float32) * gain, 0, 255)
    lum = bal[fish].mean(1)
    lo, hi = np.percentile(lum, 0.5), np.percentile(lum, 99.3)
    lev = np.clip((bal - lo) / (hi - lo), 0, 1)
    lev = lev ** 0.72
    lev = 0.5 + 0.5 * np.tanh(2.0 * (lev - 0.5)) / np.tanh(1.0)
    corr = (np.clip(lev, 0, 1) * 246 + 4).astype(np.uint8)
    blur = cv2.GaussianBlur(corr, (0, 0), 2.0)
    print(f"white balance gain {gain.round(3)}, levels {lo:.0f}..{hi:.0f} -> 4..250")
    return cv2.addWeighted(corr, 1.3, blur, -0.3, 0)


def remove_smudges(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint the grey epoxy stains on the hull. A stain is a hull pixel clearly darker than the
    hull around it (61 px median) yet far brighter than the black tape; thin dark structures
    (the hull seam, tendon shadows) are dropped by a 7 px opening so real detail survives."""
    lum = rgb.astype(np.float32).mean(2)
    inside = cv2.erode((mask > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    hull = inside.copy()
    hull[:, :HULL_X0] = False
    hull[:HULL_Y0, :] = False
    med = cv2.medianBlur(rgb, 71).astype(np.float32).mean(2)
    head = hull.copy()
    head[:, :HEAD_X0] = False
    stain = hull & (med - lum > STAIN_DIFF) & (lum > 90) & (med > 150)
    stain |= head & (med - lum > HEAD_DIFF) & (lum > 90) & (med > 150)
    stain = stain.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, k)
    stain = cv2.dilate(stain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    stain &= inside.astype(np.uint8)
    print(f"inpainting {int(stain.sum())} px of smudges")
    return cv2.inpaint(np.ascontiguousarray(rgb), stain, 7, cv2.INPAINT_TELEA)


def axis_deg(mask: np.ndarray) -> float:
    """Angle of the silhouette's principal axis, degrees anticlockwise from +x (y up)."""
    ys, xs = np.nonzero(mask)
    x, y = xs - xs.mean(), -(ys - ys.mean())
    _, v = np.linalg.eigh(np.cov(np.vstack([x, y])))
    d = v[:, 1] if v[0, 1] >= 0 else -v[:, 1]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--out", default=str(DST))
    p.add_argument("--margin", type=int, default=10)
    p.add_argument("--rotate", default=str(ROTATE_DEG),
                   help="degrees anticlockwise, or 'auto' to match the CAD render's principal axis")
    p.add_argument("--keep-smudges", action="store_true")
    p.add_argument("--cache", help="an .npz file to keep the model mattes in between runs")
    args = p.parse_args(argv)

    im = Image.open(args.src).convert("RGB")
    rgb = np.asarray(im)
    alpha, tail = mattes(im, Path(args.cache) if args.cache else None)
    mask = fish_mask(alpha, tail)
    lit = relight(rgb, mask)
    if not args.keep_smudges:
        lit = remove_smudges(lit, mask)
    out = Image.fromarray(np.dstack([lit, (mask * 255).astype(np.uint8)]), "RGBA")
    if args.rotate == "auto":
        from PIL import ImageOps
        cad = np.asarray(ImageOps.mirror(Image.open(CAD).convert("L"))) < 235
        turn = axis_deg(cad) - axis_deg(mask > 0.5)
        print(f"photo axis {axis_deg(mask > 0.5):.1f} deg, CAD {axis_deg(cad):.1f} deg")
    else:
        turn = float(args.rotate)
    if abs(turn) > 1e-6:
        print(f"rotating {turn:.1f} deg anticlockwise")
        out = out.rotate(turn, resample=Image.BICUBIC, expand=True)
    rgba = np.asarray(out)
    ys, xs = np.where(rgba[..., 3] > 8)
    m = args.margin
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, rgba.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, rgba.shape[1])
    crop = rgba[y0:y1, x0:x1]
    Image.fromarray(crop, "RGBA").save(args.out, optimize=True)
    print(f"Saved {args.out}  {crop.shape[1]}x{crop.shape[0]} px, crop ({x0},{y0})-({x1},{y1})")


if __name__ == "__main__":
    main()
