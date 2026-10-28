#!/usr/bin/env python3
"""Shared SysID figure for the report (used by BOTH sec 4.1 and 4.2): the fish hull
with (a) the five identified tail HINGES marked as spheres, and (b) the fluid
ELLIPSOIDS coloured by segment (head/body blue, tail green, fin orange).

Adapted from fishCREATE/digital_twin/diagnostics/scripts/render_fluid_ellipsoids.py.
Changes: WHITE background (opaque backdrop quad so translucent ellipsoids blend on
white), and hinge markers at the tail-joint anchors.

Run from the fishCREATE repo root in the `mujoco` conda env:
    python <this>
Writes: 05_Report/Final_Report/figures/sysid/40_hull_hinges_fluid.png
"""
import os, sys, argparse, numpy as np, mujoco, cv2

from _paths import REPO as _REPO, OUT as _OUT
FISHCREATE = str(_REPO)
sys.path.insert(0, FISHCREATE)
import geometry.auto_SOFI_da as gen
OUT = str(_OUT / "40_hull_hinges_fluid.png")   # the committed copy is paper/static/40_hull_hinges_fluid.png

COL_HEAD = [0.25, 0.45, 1.00, 0.45]   # blue
COL_TAIL = [0.20, 0.85, 0.35, 0.42]   # green
COL_FIN  = [1.00, 0.55, 0.10, 0.50]   # orange
GREY     = [0.55, 0.58, 0.62, 0.22]   # translucent structure
HINGE    = [0.80, 0.10, 0.55, 1.0]    # opaque magenta-red hinge cylinders
# The model already contains a thin cylinder geom at each tail joint (the physical
# hinge, along the z rotation axis); we colour those directly.
HINGE_GEOMS = ["joint0_0", "joint1_0", "joint2_0", "joint3_0", "joint4_0"]
HINGE_RADIUS = 0.0035                  # slightly thickened from the model's 0.002


def fluid_class(name):
    if name.startswith("hull_fluid"):
        return "head"
    if name in ("finTail_0", "tensioner_0"):
        return "fin"
    return "tail"


def fluid_semiaxes(model, g):
    t = int(model.geom_type[g]); s = model.geom_size[g]; G = mujoco.mjtGeom
    if t in (G.mjGEOM_BOX, G.mjGEOM_ELLIPSOID): return s.copy()
    if t == G.mjGEOM_SPHERE:   return np.array([s[0], s[0], s[0]])
    if t == G.mjGEOM_CYLINDER: return np.array([s[0], s[0], s[1]])
    if t == G.mjGEOM_CAPSULE:  return np.array([s[0], s[0], s[1] + s[0]])
    if t == G.mjGEOM_MESH:     return model.geom_aabb[g][3:].copy()
    return s.copy()


def build_model():
    xml = os.path.join(FISHCREATE, "geometry", "_hull_fig_tmp.xml")
    gen.generate_xml(gen.SYSTEMPARAMETERS, filename=xml)
    model = mujoco.MjModel.from_xml_path(xml); data = mujoco.MjData(model)
    os.remove(xml)
    for g in range(model.ngeom):
        model.geom_rgba[g] = GREY
    for hide in ("floor", "hull_fluid_0", "motorConnector_0", "motorWire_0",
                 "motorBlock_0", "motorShaft_0", "motorPulley_0"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, hide)
        if gid >= 0: model.geom_rgba[gid, 3] = 0.0
    model.site_rgba[:, 3] = 0.0

    # Colour the model's own hinge cylinder geoms opaque, and thicken them a little.
    for nm in HINGE_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid >= 0:
            model.geom_rgba[gid] = HINGE
            model.geom_size[gid, 0] = HINGE_RADIUS

    mujoco.mj_forward(model, data)
    return model, data


def collect_ellipsoids(model, data):
    colors = {"head": COL_HEAD, "tail": COL_TAIL, "fin": COL_FIN}
    out = []
    for g in range(model.ngeom):
        if not np.any(model.geom_fluid[g] != 0): continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        semi = fluid_semiaxes(model, g)
        R = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + R @ model.geom_aabb[g][:3]
        out.append((semi.astype(float), center.astype(float),
                    data.geom_xmat[g].astype(float).copy(), colors[fluid_class(name)]))
    return out


def hinge_positions(model, data):
    ps = []
    for nm in TAIL_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid >= 0: ps.append(data.xanchor[jid].astype(float).copy())
    return ps


def add_backdrop(scene, center, view_dir):
    """A large emissive white wall perpendicular to the view axis, behind the fish,
    so translucent geoms blend on white (a camera-facing plane always fills the frame)."""
    n = view_dir / np.linalg.norm(view_dir)
    up = np.array([0., 0., 1.])
    x = np.cross(up, n); x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.column_stack([x, y, n]).flatten()   # local z = n (wall normal)
    i = scene.ngeom
    mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([12., 12., 0.01]), center.astype(float), R,
                        np.array([1., 1., 1., 1.], np.float32))
    scene.geoms[i].emission = 1.0
    scene.ngeom += 1


def add_ellipsoids(scene, ellipsoids):
    for semi, pos, mat, rgba in ellipsoids:
        if scene.ngeom >= scene.maxgeom: break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                            semi, pos, mat, np.array(rgba, np.float32))
        scene.ngeom += 1


def make_camera(az, el, dist):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.17, 0.0, 0.0]; cam.distance = dist
    cam.azimuth = az; cam.elevation = el
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--azimuth", type=float, default=58.0)
    ap.add_argument("--elevation", type=float, default=-22.0)
    ap.add_argument("--fovy", type=float, default=20.0)
    ap.add_argument("--distance", type=float, default=1.25)
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--r-hinge", type=float, default=0.006)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    model, data = build_model()
    model.vis.global_.fovy = a.fovy
    ell = collect_ellipsoids(model, data)
    n = {"head": 0, "tail": 0, "fin": 0}
    for *_, rgba in ell:
        n["head" if rgba is COL_HEAD else "fin" if rgba is COL_FIN else "tail"] += 1
    print(f"ellipsoids head={n['head']} tail={n['tail']} fin={n['fin']}")

    # camera-facing white wall behind the fish (correct translucent blend on white)
    az, el = np.radians(a.azimuth), np.radians(a.elevation)
    off = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    lookat = np.array([0.17, 0.0, 0.0])
    wall_center = lookat + off * 6.0        # behind the fish, on the view axis
    view_dir = off                          # wall normal along the view axis

    scene_opt = mujoco.MjvOption()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    r = mujoco.Renderer(model, a.height, a.width, max_geom=model.ngeom + 64)
    r.update_scene(data, camera=make_camera(a.azimuth, a.elevation, a.distance),
                   scene_option=scene_opt)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    add_backdrop(r.scene, wall_center, view_dir)
    add_ellipsoids(r.scene, ell)
    rgb = r.render(); r.close()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
