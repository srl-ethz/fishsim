#!/usr/bin/env python3
"""Fig. 4 (tail stiffness): the transmission geometry, the bend it produces, and the measurement.

Three panels, one column.

  (a) Top view, neutral. Every symbol the text uses is here: the shaft angle theta, the
      pulley radius r, the moment arm d_i at each eyelet, and the joint
      pitch. Lengths are the ones the identification used (tail_sysid/utils.py) and
      the ones in the deployed model (geometry/SOFI.xml); the two agree.

      The eyelets do not sit over the joint axes: each is 11.0 mm (first) or 12.5 mm aft of
      its axis, at the middle of the segment, which is what the model has (jnt_pos is zero,
      so the body origin is the axis). That stagger is also why MuJoCo's effective moment
      arms are not the eyelet offsets -- the tendon reaches each joint at an angle.

      The pair is wrapped in opposing directions on the pulley, so it CROSSES between the
      pulley and the first joint -- in the model tendonLeft runs crank -> bodyRight ->
      tail0left. Tendon segments leave the pulley on a true tangent.

  (b) The same view under load: the tail at the static equilibrium of the identified twin
      with the shaft held at theta. Hinge angles come from MuJoCo, not from a sketch.

      Worth knowing when reading it: MuJoCo takes each moment arm from the real 3D tendon
      path, which is not the eyelet offset. Its effective arms are [14.1, 16.7, 12.5, 8.8,
      5.7] mm against the nominal d_i of [19.5, 14.5, 10.75, 7.0, 4.5], so the twin bends
      almost uniformly rather than mostly at the root, and its shaft stiffness is 0.93
      against the identified 0.98. The drive is drawn as the equivalent pulley of radius r
      because that is what the k_eff mapping uses, and the model agrees with it: the tendon
      Jacobian gives dL/dtheta = 17.35 mm against r = 17.5.

  (c) The measurement, every condition a point at its drive frequency. Re H is affine in
      omega^2, so a line through the drive frequencies
      puts the structural stiffness at the intercept and the effective inertia in the slope.

Inputs
    ../geometry/SOFI.xml                          identified twin, for the bend
    ../sysid/tail_stiffness/ on the r17.5 air-benchmark sweep in fishdeploy/data/03_sysid/tail_benchmark/air
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, FancyArrowPatch

from _paths import REPO, OUT as _OUT
sys.path.insert(0, str(REPO / "sysid" / "tail_stiffness"))
from step1_stiffness import prescreen_condition, process_file, ALL_AMP_FRACS
from utils import fit_spring_inertia
from paper_style import use_paper_style, size, save, CB, LW, EMPH_LW

MODEL = str(REPO / "geometry" / "SOFI.xml")

R_PULLEY = 17.5
D = np.array([19.5, 14.5, 10.75, 7.0, 4.5])
STICK_X = np.array([11.0, 12.5, 12.5, 12.5, 12.5])    # bracket station aft of the axis
STICK_OUT = np.array([24.5, 19.5, 15.75, 12.0, 9.5]) # how far the bracket reaches out
L_FLEX = np.array([9.5, 11.0, 11.0, 11.0, 11.0])     # flexure: axis -> just before the bracket
KJ = np.array([2.988, 2.391, 2.123, 1.767, 1.323])      # identified, SOFI.xml
THETA_B = -0.22         # rad; negative bends the tail up, a slight load case
THETA_DRAW = -0.60      # rad; the theta MARK only. The load case is a 12.6 deg shaft
                        # rotation, whose arc sits almost on the reference spoke and
                        # cannot be read. The mark is drawn enlarged to show the
                        # convention -- zero straight down, growing clockwise -- and the
                        # caption gives the real value. Geometry is untouched by this.

# line art in ink and grey with ONE accent for the tendon, so the schematic reads as a
# drawing; the data panel keeps the house roles (measured black, fit orange).
INK, STRUCT, DIM = "#1A1A1A", "#A2A8AD", "#4D5359"
SOFT = "#7C8288"
GHOST = "#C3C8CC"
TENDON, HINGE_F, HINGE_E = CB["blue"], CB["sky"], CB["blue"]


def twin_geometry(theta):
    """Static equilibrium of the identified twin with the shaft held at theta.

    Head pinned, gravity and fluid off, so all that is left is the tendon pulling the
    hinge springs. Returns coordinates in mm with the motor at the origin.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(MODEL)
    m.opt.gravity[:] = 0; m.opt.density = 0; m.opt.viscosity = 0
    d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    head = ["headX_0", "headY_0", "headZ_0", "headRollX_0", "headRollY_0", "headRollZ_0"]
    pq = [m.jnt_qposadr[jid(n)] for n in head]
    pv = [m.jnt_dofadr[jid(n)] for n in head]
    mq, mv = m.jnt_qposadr[jid("motor_0")], m.jnt_dofadr[jid("motor_0")]
    hq = [m.jnt_qposadr[jid("joint%d_0" % k)] for k in range(5)]
    for j in range(m.njnt):
        a = m.jnt_dofadr[j]
        m.dof_damping[a] = max(m.dof_damping[a], 0.08)
    prev = None
    for it in range(60000):
        for a in pq: d.qpos[a] = 0.0
        for v in pv: d.qvel[v] = 0.0
        d.qpos[mq] = theta; d.qvel[mv] = 0.0
        mujoco.mj_step(m, d)
        if it % 5000 == 0:
            cur = np.array([d.qpos[a] for a in hq])
            if prev is not None and np.max(np.abs(cur - prev)) < 1e-9:
                break
            prev = cur
    for a in pq: d.qpos[a] = 0.0
    d.qpos[mq] = theta
    mujoco.mj_forward(m, d)
    g = twin_nodes(m, d)
    phi = g["phi"]
    sh = np.round(phi / phi.sum() * 100, 1) if abs(phi.sum()) > 1e-12 else phi * 0
    print("    theta=%.2f  phi[deg]=%s  share[%%]=%s"
          % (theta, np.round(np.degrees(phi), 2), sh))
    return g


def twin_nodes(m, d):
    """Every point the transmission drawing needs, in mm, with the motor at the origin.

    Split out of twin_geometry() so the same drawing can be built from a state this module did
    not solve for. twin_geometry() still calls it, so panel (A) is byte-identical; what it buys
    is that tools/make_hardware_sim2real_animation.py can drive the SAME model dynamically --
    head pinned, shaft swept at the frequencies the bench test used -- and read its poses back
    through this one function, instead of keeping a second copy of the geometry that could
    drift away from the printed figure.

    Reads state only: whatever (m, d) is in, after an mj_forward.
    """
    import mujoco
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    hq = [m.jnt_qposadr[jid("joint%d_0" % k)] for k in range(5)]
    org = d.xpos[bid("motor_0")][:2].copy()
    mm = lambda p: (np.asarray(p)[:2] - org) * 1000.0
    phi = np.array([d.qpos[a] for a in hq])
    hb = np.array([mm(d.xpos[m.jnt_bodyid[jid("joint%d_0" % k)]]) for k in range(5)])
    left = np.array([mm(d.site_xpos[sid("tail%dleft_0" % k)]) for k in range(5)])
    right = np.array([mm(d.site_xpos[sid("tail%dright_0" % k)]) for k in range(5)])
    # the tendon anchors on the rigid fin segment, 35 mm aft of the last joint at +-8 mm,
    # and it is NOT a point: the pair terminates apart, on the tensioner.
    ancL = mm(d.site_xpos[sid("tail5left_0")])
    ancR = mm(d.site_xpos[sid("tail5right_0")])
    tip = (ancL + ancR) / 2
    pulL = mm(d.site_xpos[sid("pulleyLeft_0")])
    pulR = mm(d.site_xpos[sid("pulleyRight_0")])
    fin = mm(d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finTail_0")])
    return dict(phi=phi, hinge=hb, left=left, right=right, tip=tip,
                ancL=ancL, ancR=ancR, pulL=pulL, pulR=pulR, fin=fin)


def tangent_point(P, r, s):
    """Where a line from external point P touches the circle of radius r at the origin."""
    P = np.asarray(P, float)
    n2 = float(P @ P)
    perp = np.array([-P[1], P[0]])
    return (r / n2) * (r * P + s * np.sqrt(max(n2 - r * r, 0.0)) * perp)


def draw_ghost(ax, g, theta):
    """The twin's static equilibrium at theta, behind the neutral drawing.

    Only the bent plate and the tendon lying on the pulley: drawing the ghost's tendon
    runs as well tangles them with the neutral ones and reads as a mistake.
    """
    hb, tip = g["hinge"], g["tip"]
    spine = np.vstack([hb, tip])
    ax.plot(spine[:, 0], spine[:, 1], color=GHOST, lw=2.4, solid_capstyle="round", zorder=1)
    ax.scatter(hb[:, 0], hb[:, 1], s=11, facecolor="white", edgecolor=GHOST,
               linewidth=0.6, zorder=2)

    # the phi_i accumulate: the last segment already sits at their sum from the body
    # axis, so the reference is hung on the LAST JOINT and the existing segment is used
    # as the other ray. Nothing is lengthened. Checked against sum(phi) every run.
    j = hb[4]
    v = (tip - j) / np.linalg.norm(tip - j)
    a_seg = np.degrees(np.arctan2(v[1], v[0]))
    tot = np.degrees(abs(np.sum(g["phi"])))
    assert abs(abs(a_seg) - tot) < 0.05, (a_seg, tot)
    EXT = float(np.linalg.norm(tip - j))          # exactly the last segment, no more
    ax.plot([j[0], j[0] + EXT], [j[1], j[1]], color=DIM, lw=0.6 * LW,
            ls=(0, (2.6, 1.9)), zorder=3)
    R2 = EXT - 7.0
    ax.add_patch(Arc(j, 2 * R2, 2 * R2, theta1=min(0.0, a_seg), theta2=max(0.0, a_seg),
                     color=INK, lw=0.9, zorder=6))
    ab = np.deg2rad(a_seg / 2)
    ax.annotate(r"$\sum_i\varphi_i$",
                (j[0] + R2 * np.cos(ab), j[1] + R2 * np.sin(ab)),
                (j[0] - 2.0, j[1] + 12.5),
                ha="center", va="bottom", color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.5 * LW, color=INK, shrinkA=1, shrinkB=2))
    ax.extra_pts = np.array([[j[0] + EXT, j[1]]])

    return spine


def draw_top(ax, g, gb, theta):
    ax.set_aspect("equal")
    ax.axis("off")
    gspine = draw_ghost(ax, gb, theta)

    hb, left, right, tip = g["hinge"], g["left"], g["right"], g["tip"]
    spine = np.vstack([hb, tip])
    back = hb[0] + (hb[0] - hb[1]) / np.linalg.norm(hb[1] - hb[0]) * 13.0
    ax.plot(np.r_[back[0], spine[:, 0]], np.r_[back[1], spine[:, 1]],
            color=STRUCT, lw=2.6, solid_capstyle="round", zorder=2)
    # aft of the last joint the body has no further joint: that stretch is rigid
    # along the SEGMENT AXIS, not off an anchor: the anchors sit +-8 mm to the side, and
    # taking the direction from one of them tilts the fin away from the spine.
    axis = (tip - hb[4]) / np.linalg.norm(tip - hb[4])
    fin_end = tip + axis * 11.0
    ax.plot([hb[4, 0], fin_end[0]], [hb[4, 1], fin_end[1]], color=STRUCT, lw=3.4,
            solid_capstyle="round", zorder=2)


    # guide brackets: each reaches out past its eyelet, which is what holds the tendon
    # off the plate. Drawn at their true station and extent.
    for k in range(5):
        xs_ = hb[k, 0] + STICK_X[k]
        ax.plot([xs_, xs_], [-STICK_OUT[k], STICK_OUT[k]], color=STRUCT, lw=1.9,
                solid_capstyle="round", zorder=2)

    # exactly the model's route: pulleyRight -> tail0left -> ... -> tail5left, and mirror.
    # The pair crosses because each side leaves from the opposite side of the pulley.
    for side, start, anc in ((left, g["pulR"], g["ancL"]), (right, g["pulL"], g["ancR"])):
        pts = np.vstack([start, side, anc])
        ax.plot(pts[:, 0], pts[:, 1], color=TENDON, lw=LW, zorder=3,
                solid_joinstyle="round")
        ax.scatter(side[:, 0], side[:, 1], s=4.0, color=TENDON, zorder=6)
        ax.scatter([anc[0]], [anc[1]], s=9.0, marker="s", color=TENDON, zorder=6)

    # the flexure, axis to bracket, thickened on the plate and named once
    kf = 0
    ax.plot([hb[kf, 0], hb[kf, 0] + L_FLEX[kf]], [0, 0], color=INK, lw=2.9,
            solid_capstyle="butt", zorder=5)
    # labelled, not dimensioned: at this scale L_i is under 4 mm on the page and two
    # arrowheads merge into a blob. The thickened blue run on the plate is the extent.
    ax.annotate(r"$L_i$", (hb[kf, 0] + L_FLEX[kf] / 2, -1.7), (hb[kf, 0] - 11.0, -19.0),
                ha="center", va="top", color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.5 * LW, color=INK,
                                shrinkA=1, shrinkB=2))

    ax.add_patch(Circle((0, 0), R_PULLEY, facecolor="white", edgecolor=INK, lw=LW, zorder=4))
    ax.add_patch(Circle((0, 0), 1.4, facecolor=INK, edgecolor="none", zorder=8))
    ax.scatter(hb[:, 0], hb[:, 1], s=30, facecolor=HINGE_F, edgecolor=HINGE_E,
               linewidth=0.7, zorder=7)
    # theta is zero with the spoke straight down and grows CLOCKWISE, which is the sense
    # that winds the tendon and bends the tail up, as the ghost shows.
    a0 = 270.0
    a1 = a0 + np.degrees(theta)
    for ang, st in ((a0, (0, (2.0, 1.6))), (a1, "-")):
        ax.plot([0, R_PULLEY * np.cos(np.deg2rad(ang))],
                [0, R_PULLEY * np.sin(np.deg2rad(ang))],
                color=INK, lw=0.6 * LW, ls=st, zorder=7)
    RA = 11.5
    ax.add_patch(Arc((0, 0), 2 * RA, 2 * RA, theta1=min(a0, a1), theta2=max(a0, a1),
                     color=INK, lw=0.9, zorder=7))
    am = np.deg2rad((a0 + a1) / 2)
    ax.annotate(r"$\theta$", (RA * np.cos(am), RA * np.sin(am)),
                ((R_PULLEY + 10) * np.cos(am), (R_PULLEY + 10) * np.sin(am)),
                ha="center", va="top", color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.5 * LW, color=INK, shrinkA=1, shrinkB=2))

    ax.annotate("", (0, 0), (0, R_PULLEY), zorder=7,
                arrowprops=dict(arrowstyle="<->", lw=0.75 * LW, color=DIM,
                                shrinkA=0, shrinkB=0))
    ax.text(-1.8, R_PULLEY * 0.55, r"$r$", ha="right", va="center", zorder=8, color=INK)
    ax.text(R_PULLEY + 5.0, R_PULLEY - 2.0, "pulley", ha="left", va="center", color=INK)

    # d_i, once: with the brackets drawn the rest is read off the picture
    kd = 0
    xe, ye = right[kd]
    ax.annotate("", (xe, 0), (xe, ye), zorder=6,
                arrowprops=dict(arrowstyle="<->", lw=0.75 * LW, color=DIM,
                                shrinkA=0, shrinkB=0))
    ax.text(xe + 2.4, ye * 0.5, r"$d_i$", ha="left", va="center", color=INK, zorder=8)

    # scale bar, bottom left, out of the way of every label
    x0 = hb[3, 0] + 2.0
    yb = min(left[:, 1].min(), right[:, 1].min()) - 7.0   # just clear of the lower tendon
    ax.plot([x0, x0 + 50], [yb, yb], color=DIM, lw=LW, solid_capstyle="butt", zorder=6)
    for x in (x0, x0 + 50):
        ax.plot([x, x], [yb - 1.3, yb + 1.3], color=DIM, lw=LW, zorder=6)
    ax.text(x0 + 25, yb - 2.2, "50 mm", ha="center", va="top", color=DIM)

    allp = np.vstack([spine, gspine, left, right, g["ancL"], g["ancR"], fin_end,
                      getattr(ax, "extra_pts", np.empty((0, 2)))])
    ax.set_xlim(-R_PULLEY - 7, allp[:, 0].max() + 9)
    ax.set_ylim(min(allp[:, 1].min(), -R_PULLEY) - 13.0,
                max(allp[:, 1].max(), R_PULLEY) + 8.0)


def panel_c(ax):
    recs = []
    for d_mm in (5.0, 7.0, 9.0):
        for f in (1.0, 2.0, 3.0, 4.0):
            for amp in ALL_AMP_FRACS:
                pk, _, _, _ = prescreen_condition("air", d_mm, f, amp, slack_mm=2.5)
                if not pk:
                    continue
                r = process_file("air", d_mm, f, amp, use_cycle_avg=True)
                if r:
                    r["d_mm"] = d_mm
                    recs.append(r)
    w2 = np.array([r["omega_sq"] for r in recs])
    reH = np.array([r["H"].real for r in recs])
    d_mm_all = np.array([r["d_mm"] for r in recs])
    k_eff, I_eff = fit_spring_inertia(w2, reH)
    print("    n=%d conditions   k_eff=%.3f   I_eff=%.5f" % (len(recs), k_eff, I_eff))
    # every condition a black point at its drive frequency
    ax.scatter(w2, reH, s=6, facecolor=CB["black"], edgecolor="none", zorder=3)
    g = np.linspace(0, w2.max() * 1.08, 60)
    ax.plot(g, k_eff - I_eff * g, color=CB["orange"], lw=EMPH_LW, zorder=4)
    ax.plot([0], [k_eff], "o", ms=4.0, color=CB["orange"], mec="white", mew=0.7, zorder=5)
    # the label in the empty upper right, clear of every box and of the line
    ax.text(0.97, 0.93, r"$k_\mathrm{eff}=%.2f$" % k_eff, color=CB["orange"],
            ha="right", va="top", transform=ax.transAxes)
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel(r"$\omega^2$  (rad$^2$/s$^2$)")
    ax.set_ylabel(r"$\mathrm{Re}\,H$  (N\,m/rad)")
    ax.set_xlim(-14, w2.max() * 1.10)
    ax.set_ylim(-1.05, 1.45)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])


# Guarded only so twin_geometry() can be imported by tools/make_hardware_sim2real_animation.py,
# which needs the SAME static equilibrium the paper draws rather than its own copy of it.
# Running this file as a script is unchanged -- the figure below is byte-identical either way.
if __name__ == "__main__":
    use_paper_style()
    fig = plt.figure(figsize=size(1.0, aspect=0.70), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.80, 0.74], hspace=0.03)
    ax_a, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    print("  twin static equilibrium:")
    draw_top(ax_a, twin_geometry(0.0), twin_geometry(THETA_B), THETA_DRAW)
    ax_a.set_title(r"\textbf{(A)} Transmission geometry, top view", loc="left", pad=1)
    panel_c(ax_c)
    ax_c.set_title(r"\textbf{(B)} Stiffness from the frequency sweep", loc="left", pad=6)

    save(fig, "fig4_tailstiff", expect="col")
