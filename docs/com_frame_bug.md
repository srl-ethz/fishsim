# COM Frame Bug — head+body center of mass is misplaced by ~92 mm

**Found:** 2026-06-29 (branch `feat/sim2real-gap`)
**Status:** Documented only — **nothing changed in the model/generators yet.**
**Severity:** High for sim2real fidelity (turning dynamics); harmless for self-consistent sim-only comparisons.

---

## TL;DR

The fish's head+body center of mass (`centerOfMass` in all four `geometry/sofi*.py`
generators) is set to **x = −7.97 mm**, which places the COM **~8 mm in front of the nose** —
physically impossible, since all 830 g of head+body mass lives in x ∈ [0, 198] mm.

- **Root cause:** the SolidWorks COM is measured from the **head sealing surface**, but it was
  entered into MuJoCo (whose body origin is at the **nose**) **without the ~92.3 mm origin shift**
  between those two datums. The axis directions/signs were handled correctly; only the X translation
  was dropped.
- **Correct value:** `centerOfMass_x ≈ +84.3 mm` (not −7.97 mm). y and z are fine.
- **Mass is correct** (830 g head+body, 865 g total incl. 35 g tail — matches SolidWorks).
- **Inertia tensor is correct** (inertia about the COM is origin-independent; verified below).
- **Only the COM position is wrong** — but because the head DOFs (including the yaw hinge) are
  anchored *at* `centerOfMass`, the simulated fish currently **pivots about its nose instead of
  mid-body**, giving ~30 % too much turning authority.

---

## The two coordinate frames

**SolidWorks frame (from the SW inertia report):**
- **X** = along-fish axis, **0 at the head sealing surface**, **+X toward the front (nose)**.
- **Y** = vertical, 0 at the o-ring center, +Y **downward**.
- **Z** = left–right, 0 at the centerline, +Z to the fish's **left**.

**MuJoCo body frame (from the generated XML):**
- Body origin is at the **nose** (head mesh world x-range = [−0.1, 92.5] mm).
- **+x points toward the tail** (the env convention is "nose = −x"); head-plate (the sealing
  surface) sits at x ≈ **+92.3 mm**, body at +98…198 mm, tail beyond +200 mm.

So the two X origins are **different points ~92 mm apart** (nose vs. sealing surface), and the X
axes point in **opposite** directions (SW +X = MuJoCo −x).

---

## The numbers

SolidWorks per-component COM (SW frame, mm):

| Component | mass (g) | X | Y | Z |
|-----------|----------|------|------|------|
| Head | 580 | +30.2 | +0.4 | −1.9 |
| Body | 250 | −43.6 | −1.5 | 0.0 |
| Tail (+tensioners/fasteners) | 35 | −216.7 | 0 | 0 |

Combined **head+body** (830 g) COM in the SW frame:

```
X_sw = (580·30.2 + 250·(−43.6)) / 830 = +7.971 mm   (forward of the sealing surface)
Y_sw = (580·0.4  + 250·(−1.5))  / 830 = −0.172 mm
Z_sw = (580·(−1.9)+250·0)       / 830 = −1.328 mm
```

What the model currently has:

```
centerOfMass = [−7.971, +1.328, +0.172] mm
             = (−X_sw, −Z_sw, −Y_sw)
```

So the axis **mapping** was done correctly:
- MuJoCo x ← −X_sw (SW front → MuJoCo −x)  ✓ direction
- MuJoCo y ← −Z_sw (SW left/right)          ✓
- MuJoCo z ← −Y_sw (SW down → MuJoCo up)     ✓

…but the X value was used **as if the SW origin were the MuJoCo origin** — i.e. as if the sealing
surface were the nose. It isn't; they're ~92.3 mm apart.

**Correct X:**
```
centerOfMass_x = (sealing-surface x in MuJoCo) − X_sw
               = 92.3 − 7.971
               = +84.33 mm
```
This matches the **pre-E13 (E9–E10) value of +84.3 mm** (`FLUID_COEFFICIENTS.md`) and the
geometric centroid estimate (~+77 mm). The E13 "inertia fix" improved y/z but **broke x** by
dropping the sealing-surface offset. (The generator comment calling the old "+77 mm auto-computed"
value wrong is itself mistaken — +77/+84 mm was the right ballpark.)

---

## Mass and inertia are NOT affected

- **Total mass:** model = 865.6 g (830 g head+body lump + ~34.6 g tail segments + 1 g motor) =
  SolidWorks 580 + 250 + 35 = 865 g. ✓
- **Inertia tensor:** the model's `fullinertia` is correct. Recomputing the combined head+body
  inertia from the SW per-component MOIs via the parallel-axis theorem (about the combined COM,
  then remapping SW→MuJoCo axes) gives, in g·mm²:

  | | model | recomputed |
  |---|------:|-----------:|
  | Ixx | 1 547 000 | 1 547 082 |
  | Iyy | 2 761 000 | 2 761 095 |
  | Izz | 1 722 000 | 1 722 146 |

  They match. Inertia **about the COM** is independent of where you put the body origin, so the
  nose/sealing-surface mistake never corrupted the tensor — **only the COM `pos` is wrong.**

**Net:** a single-coordinate error — `centerOfMass_x` is off by ~92 mm (−7.97 should be +84.3).
Everything else (mass, MOI, y, z) is fine.

---

## Why it matters (dynamics)

In every generator the head DOFs — slides `headX/Y/Z`, hinges `headRollX/Y`, and the **yaw hinge
`headRollZ`** — are all anchored at `pos = centerOfMass`, and the `<inertial>` lump is the only
mass (all geom masses are 0). So the **pivot point of the whole fish is wherever `centerOfMass`
is.**

- Currently the pivot is at **x = −8 mm (the nose tip)**.
- It should be at **x = +84 mm (mid head/body)**.

The tail thrust acts at the tail (≈ +250…430 mm). The yaw moment arm from COM to thrust is
therefore **~90 mm too long**, i.e. roughly **30 % too much turning authority**. Forward
(translational) swimming is much less affected; the error is mostly in **yaw / turning**.

---

## Do we need to re-tune the fluid coefficients?

**For sim-only, self-consistent work (comparing policies in simulation):** no. The wrong COM was
present identically during every fluid SysID and every policy training run, so those results are
internally consistent. Existing comparisons (CPG vs DA vs LPF, the S-pursuit demos, etc.) remain
valid relative to each other.

**For sim2real fidelity (the goal of this branch):** yes, after fixing the COM you should expect to
redo the downstream fits:

1. **Re-run the field2sim fluid SysID.** The current fluid coefficients (E13a, refitB) were fit to
   reproduce the real fish's trajectory *with the pivot at the nose*. With ~30 % less turning
   authority once the COM is corrected, the optimizer will have leaned on the tail thrust /
   angular-drag terms (c7, c8) to compensate. Those will shift. The fit should be re-run with the
   corrected COM.
2. **Retrain the policies.** CPG (E13a), DA (refitB), and the LPF variants were all trained on the
   nose-pivot model; their turning behavior is tuned to it.

The good news: it's a **one-line model change** (correct `centerOfMass_x` to ≈ +84.3 mm in the four
`sofi*.py` generators; the `fullinertia` and mass stay as-is), after which the SysID + RL
pipeline reruns are mechanical.

### Recommended fix (when ready)

```python
# in geometry/sofi*.py
"centerOfMass": [0.08433, 0.001328, 0.000172],   # was [-0.007971, ...] — added the +92.3 mm
                                                  # nose→sealing-surface offset; SW combined COM
                                                  # = +7.97 mm forward of the sealing surface
# fullinertia and inertialMass are correct as-is.
```
(Confirm the exact sealing-surface x against the head-plate geometry before committing; 92.3 mm is
the head-plate face and reproduces the +84.3 mm E9–E10 value.)

---

## How to reproduce this finding

Render the head mesh with the model's centre-of-mass markers (for example with MuJoCo's viewer on
`geometry/SOFI.xml`): the head-body COM marker sits at x ≈ 0, the nose, instead of mid-body. The
numbers above come from `geometry/sofi.py` (`SYSTEMPARAMETERS`) and the CAD mass properties.
