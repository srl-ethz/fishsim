################################################################################
# SOFI Field-to-Sim Geometry Generator — standalone variant.
# Architecture mirrors auto_SOFI_twin.py exactly.
# Field2Sim-specific:
#   1. Fluid defaults: segmented head_fluid / tail_fluid / fin_fluid classes
#      driven by a 5/10/15-param vector in SYSTEMPARAMETERS["fluidCoef"].
#   2. Per-joint hinge stiffness/damping (length-5 lists) applied to joint0..joint4.
#   3. Worldbody overhead camera repositioned to cover the 3 m swim path.
#   4. PULLEY-only (CRANK mode removed - field robot is PULLEY-driven).
#   5. <sensor> block included (same as twin).
################################################################################

import numpy as np
import os
import yaml

SYSTEMPARAMETERS = {
    "filename": "geometry/SOFI_field2sim.xml",

    "dt": 0.001,
    "freeZ": False,

    "numberOfFish": 2,

    "fluidShape": "ellipsoid",
    "fluidDensity": 1000,
    "fluidViscosity": 0.0013,

    # Inertial properties derived from SolidWorks 2026-04-29: Head(580g) + Body(250g) = 830g
    # Axis mapping: MuJoCo X=-SW_X, MuJoCo Y=-SW_Z, MuJoCo Z=-SW_Y
    # Combined COM at x=-7.97mm (on the head side of the sealing surface — physically correct;
    # previous auto-computed x=+77mm was wrong due to hollow-mesh centroid vs actual mass dist.)
    # Motor mass is included in the SW Body(250g) budget → motor child body uses near-zero mass.
    "centerOfMass": [-0.007971, 0.001328, 0.000172],  # m, SW-derived combined COM (head+body)
    "inertialMass": 0.830,                             # kg, head(580g) + body(250g)
    "fullinertia":  [0.001547, 0.002761, 0.001722, 0.000022, -0.000017, -0.000004],  # kg·m²

    "PLAdensity": 1250,
    "electricBoxMass": 0.53,
    "bodyMass": 0.6,   # legacy, no longer used for shell masses
    "tailMass": 0.01,

    # hingeStiffness / hingeDamping removed — now read from field2sim.yml structural block
    "tendonStiffness": 20000,
    "tendonDamping": 10,

    "motorShaftLength": 0.035,
    "pulley_radius": 0.0175,
    "motor_kp": 800.0,
    "motor_kv": 20.0,

    "tailSegmentLength": 0.003,

    "body-tail0":  0.019,
    "tail0tendon": 0.0195,

    "tail0-tail1": 0.022,
    "tail1tendon": 0.0145,

    "tail1-tail2": 0.022,
    "tail2tendon": 0.01075,

    "tail2-tail3": 0.022,
    "tail3tendon": 0.007,

    "tail3-tail4": 0.022,
    "tail4tendon": 0.0045,

    "tail4-finTail": 0.01,
    "tail5tendon":  0.008,

    "tendonRouting": [0, 0, 0, 0, 0, 0],
}


def generate_xml(args, config_path="sysid/fluid/field2sim.yml", ghost_ids=frozenset({0})):  # [F2S]
    # --- 1. LOAD CONFIG ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    fluid_vec = config['fluid_tuning']['coefficients']
    env       = config['environment']
    n         = len(fluid_vec)

    # Update args from config (mirrors twin's load_sync_config behaviour)
    args["fluidDensity"]   = env['density']
    args["fluidViscosity"] = env['viscosity']
    args["dt"]             = env['dt']
    args["pulley_radius"]  = config['motor']['pulley_radius']
    args["motor_kp"]       = config['motor']['kp']
    args["motor_kv"]       = config['motor']['kv']

    # Per-joint hinge stiffness/damping from structural config block.
    # Accepts scalar (broadcast to 5) or list-of-5.
    structural = config.get('structural', {})
    if not structural:
        raise ValueError(
            "Config is missing the 'structural' block (hinge_stiffness/hinge_damping). "
            "Run step1_stiffness.py and add the structural block to field2sim.yml."
        )
    ks_raw = structural['hinge_stiffness']
    kd_raw = structural['hinge_damping']
    ks = [float(ks_raw)] * 5 if isinstance(ks_raw, (int, float)) else [float(v) for v in ks_raw]
    kd = [float(kd_raw)] * 5 if isinstance(kd_raw, (int, float)) else [float(v) for v in kd_raw]
    if len(ks) != 5 or len(kd) != 5:
        raise ValueError(
            f"structural.hinge_stiffness must be scalar or list of 5, got len={len(ks)}; "
            f"structural.hinge_damping must be scalar or list of 5, got len={len(kd)}"
        )

    # --- 2. FLUID COEFFICIENT SPLIT (5 / 10 / 15 params) --- [F2S]
    if n == 5:
        c_head = c_tail = c_fin = fluid_vec
    elif n == 10:
        c_head = fluid_vec[0:5]
        c_tail = c_fin = fluid_vec[5:10]
    elif n == 15:
        c_head = fluid_vec[0:5]
        c_tail = fluid_vec[5:10]
        c_fin  = fluid_vec[10:15]
    else:
        raise ValueError(f"fluidCoef must be length 5, 10, or 15. Got {n}.")

    # --- 3. BUILD FISH BODIES ---
    N        = args["numberOfFish"]
    p_radius = args["pulley_radius"]
    f_shape  = args["fluidShape"]

    # Joint stiffness / damping (matches twin: head slide/yaw joints are free)
    stiffness_head  = 0
    damping_head    = 0
    stiffness_motor = 0
    damping_motor   = 0

    # Pulley tangent angles (PULLEY mode only - field robot is always PULLEY)
    tx   = (0.198 + args["body-tail0"] / 2) - 0.12
    ty   = args["tail0tendon"]
    dist = np.sqrt(tx**2 + ty**2)
    phi  = np.arctan2(ty, tx)
    alpha_tangent = np.arccos(p_radius / dist)
    theta_l = -phi + alpha_tangent   # pulleyRight site
    theta_r =  phi - alpha_tangent   # pulleyLeft  site

    fishBodyXMLs = []
    for i in range(N):
        # Ghost fish (ghost_ids) has no fluid forces — it is a reference trajectory, not a physics sim.
        # Applying fluid forces to the ghost would cause its uncontrolled tail joints to go NaN
        # with extreme coefficient values, corrupting the entire MuJoCo world state.
        fs = "none" if i in ghost_ids else f_shape

        # Transparency: ghost fish = 0.3 alpha, physics fish = opaque
        a = 0.3 if i in ghost_ids else 1.0
        c_alu          = f".5 .7 .9 {a}"
        c_dark         = f".1 .1 .1 {a}"
        c_light        = f".9 .9 .9 {a}"
        c_red_joint    = f".9 .2 .2 {a}"
        c_dark_red_joint = f".6 .1 .1 {a}"
        c_blue_tail    = f".5 .7 .9 {a}"
        c_red_tail     = f".9 .2 .2 {0.7 * a if i == 1 else a}"
        c_acrylic      = f".9 .9 .9 {0.2 * a if i == 1 else 0.15 * a}"
        c_marker       = f".9 .4 .1 {0.5 * a if i == 1 else 0.15 * a}"

        fishBodyXMLs.append(f"""
        <body pos="0 0 0" name="fish_{i}" childclass="tail_fluid">
            <camera name="tracking_{i}" mode="track" pos="4 -4 4" xyaxes="1 1 0 -0.5 0.5 1"/>
            <camera name="closeup_{i}"  mode="track" pos="2 -2 2" xyaxes="1 1 0 -0.5 0.5 1"/>

            <joint name="headX_{i}"     pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="slide" axis="1 0 0" stiffness="{stiffness_head}" damping="{damping_head}"/>
            <joint name="headY_{i}"     pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="slide" axis="0 1 0" stiffness="{stiffness_head}" damping="{damping_head}"/>
            <joint name="headZ_{i}"     pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="slide" axis="0 0 1" stiffness="{0 if args["freeZ"] else 500}" damping="{0 if args["freeZ"] else 10}"/>
            <joint name="headRollX_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="hinge" axis="1 0 0" range="-180 180" stiffness="{0 if args["freeZ"] else 10}" damping="{0 if args["freeZ"] else 1}" armature="0.001"/>
            <joint name="headRollY_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="hinge" axis="0 1 0" range="-180 180" stiffness="{0 if args["freeZ"] else 10}" damping="{0 if args["freeZ"] else 1}" armature="0.001"/>
            <joint name="headRollZ_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="hinge" axis="0 0 1" limited="false" stiffness="{stiffness_head}" damping="{damping_head}" armature="0.001"/>

            <site name="COM_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" size="0.02" rgba="0 0 0 0"/>

            <!-- EXPLICIT INERTIAL from SolidWorks 2026-04-29: Head(580g)+Body(250g)=830g combined -->
            <!-- Axes: MuJoCo X=-SW_X, Y=-SW_Z, Z=-SW_Y. COM at x=-7.97mm (inside head section). -->
            <!-- Motor mass is baked into the SW Body(250g) → motor child body has near-zero mass. -->
            <!-- Geom masses below are set to 0 to prevent double-counting with this element.       -->
            <inertial mass="{args['inertialMass']}" pos="{args['centerOfMass'][0]} {args['centerOfMass'][1]} {args['centerOfMass'][2]}" fullinertia="{args['fullinertia'][0]} {args['fullinertia'][1]} {args['fullinertia'][2]} {args['fullinertia'][3]} {args['fullinertia'][4]} {args['fullinertia'][5]}"/>

            <!-- BODY MESHES: fluid disabled, use consolidated hull proxy below -->
            <!-- mass=0: inertia is fully specified by the <inertial> element above -->
            <geom name="head_{i}"       type="mesh" mesh="head"      quat="0 0 -0.7071 0.7071" rgba="{c_light}" mass="0" fluidshape="none"/>
            <geom name="head-plate_{i}" type="mesh" mesh="headPlate" quat="0 0 -0.7071 0.7071" rgba="{c_alu}"   mass="0" fluidshape="none"/>
            <geom name="body_{i}"       type="mesh" mesh="body"      quat="0.5 0.5 -0.5 0.5"   rgba="{c_alu}"   mass="0" fluidshape="none"/>

            <!-- HULL FLUID PROXY — mass=0, ONLY for fluid force computation, not inertia -->
            <geom name="hull_fluid_{i}" type="ellipsoid" pos="0.10 0 0.0082" size="0.10 0.039 0.068" rgba="1 0 0 0.1" mass="0" class="head_fluid" fluidshape="{fs}"/>

            <!-- MOTOR visual geometry — all mass=0: inertia already in root <inertial> element -->
            <geom name="motorConnector_{i}" type="cylinder" pos="0.132 0 -0.055"  size="0.005 0.0125" rgba="{c_dark}" mass="0" fluidshape="none"/>
            <geom name="motorWire_{i}"      type="cylinder" pos="0.132 0 -0.0725" size="0.002 0.006"  rgba="{c_dark}" mass="0" fluidshape="none"/>
            <geom name="motorBlock_{i}"     type="box"      pos="0.12 0 -0.0273"  size="0.02325 0.01425 0.017" rgba="{c_dark}" mass="0" fluidshape="none"/>

            <!--
            DETAILED INTERNAL MASS MODEL (commented out — use mesh masses above for now):
            If mesh-based approximation is insufficient, uncomment these and set
            head/head-plate/body masses back to PLA-shell-only values (~90/50/100g).

            <geom name="head_content_{i}" type="box" pos="0.085 0 0.000" size="0.035 0.040 0.035"
                  mass="0.300" rgba="0.2 0.5 0.8 0.2" fluidshape="none" contype="0" conaffinity="0"/>
            <geom name="oil_{i}" type="ellipsoid" pos="0.080 0 -0.019" size="0.055 0.025 0.040"
                  mass="0.200" rgba="0.9 0.8 0.1 0.15" fluidshape="none" contype="0" conaffinity="0"/>
            <geom name="lead_ballast_{i}" type="sphere" pos="0.155 0 -0.045" size="0.015"
                  mass="0.100" rgba="0.3 0.3 0.3 0.5" fluidshape="none" contype="0" conaffinity="0"/>
            -->
            <body pos="0.12 0 0" name="motor_{i}">
                <joint name="motor_{i}" type="hinge" axis="0 0 1" pos="0 0 0" limited="true" range="-0.6 0.6" stiffness="{stiffness_motor}" damping="{damping_motor}" armature="0.001"/>
                <geom name="motorShaft_{i}"  type="cylinder" pos="0 0 0"                      size="0.005 {args['motorShaftLength']/2}"  rgba="{c_dark}"     mass="0.001" fluidshape="none"/>
                <geom name="motorPulley_{i}" type="cylinder" pos="0 0 {args['motorShaftLength']/2}" size="{p_radius} 0.0025" rgba="{c_red_tail}" mass="0"     fluidshape="none"/>
                <site name="pulleyRight_{i}" pos="{p_radius * np.cos(theta_l)} {p_radius * np.sin(theta_l)} {args['motorShaftLength']/2+0.005}" size="0.005" rgba="0 0 0 0"/>
                <site name="pulleyLeft_{i}"  pos="{p_radius * np.cos(theta_r)} {p_radius * np.sin(theta_r)} {args['motorShaftLength']/2+0.005}" size="0.005" rgba="0 0 0 0"/>
            </body>

            <site name="bodyLeft_{i}"  pos="0.12 -0.0285 {args['motorShaftLength']/2+0.005}" size="0.005" rgba="0 0 0 0"/>
            <site name="bodyRight_{i}" pos="0.12  0.0285 {args['motorShaftLength']/2+0.005}" size="0.005" rgba="0 0 0 0"/>

            <!-- TAIL - inherits childclass="tail_fluid" from fish body -->
            <geom name="body-joint0_{i}" type="box" pos="{0.198+args["body-tail0"]/4} 0 0.01016" size="{args["body-tail0"]/4} 0.0005 0.0558" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
            <body pos="{0.198+args["body-tail0"]/2} 0 0.01016">
                <geom name="joint0_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0558" rgba="{c_red_joint}" mass="0"/>
                <joint name="joint0_{i}" armature="0.001" stiffness="{ks[0]}" damping="{kd[0]}"/>
                <geom name="joint0-tail0_{i}" type="box" pos="{args["body-tail0"]/4} 0 0" size="{args["body-tail0"]/4} 0.0005 0.0558" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                <geom name="tail0_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.001 0.0558" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                <site name="marker4_{i}" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2} 0 0.0558" size="0.005" rgba="{c_marker}"/>
                <geom name="tail0left_{i}"  type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2} -{args["tail0tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail0tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                <geom name="tail0right_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2}  {args["tail0tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail0tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                <site name="tail0left_{i}"  pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2} {-args["tail0tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                <site name="tail0right_{i}" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2}  {args["tail0tendon"]} 0" size="0.004" rgba="0 0 0 0"/>

                <geom name="tail0-joint1_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"] + args["tail0-tail1"]/4} 0 0" size="{args["tail0-tail1"]/4} 0.0005 0.0517" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                <body pos="{args["body-tail0"]/2 + args["tailSegmentLength"] + args["tail0-tail1"]/2} 0 0">
                    <geom name="joint1_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0517" rgba="{c_dark_red_joint}" mass="0"/>
                    <joint name="joint1_{i}" armature="0.001" stiffness="{ks[1]}" damping="{kd[1]}"/>
                    <geom name="joint1-tail1_{i}" type="box" pos="{args["tail0-tail1"]/4} 0 0" size="{args["tail0-tail1"]/4} 0.0005 0.0517" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                    <geom name="tail1_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.0517" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                    <site name="marker5_{i}" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2} 0 0.0517" size="0.005" rgba="{c_marker}"/>
                    <geom name="tail1left_{i}"  type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2} -{args["tail1tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail1tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                    <geom name="tail1right_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2}  {args["tail1tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail1tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                    <site name="tail1left_{i}"  pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2} {-args["tail1tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                    <site name="tail1right_{i}" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2}  {args["tail1tendon"]} 0" size="0.004" rgba="0 0 0 0"/>

                    <geom name="tail1-joint2_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] + args["tail1-tail2"]/4} 0 0" size="{args["tail1-tail2"]/4} 0.0005 0.0459" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                    <body pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] + args["tail1-tail2"]/2} 0 0">
                        <geom name="joint2_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0459" rgba="{c_dark_red_joint}" mass="0"/>
                        <joint name="joint2_{i}" armature="0.001" stiffness="{ks[2]}" damping="{kd[2]}"/>
                        <geom name="joint2-tail2_{i}" type="box" pos="{args["tail1-tail2"]/4} 0 0" size="{args["tail1-tail2"]/4} 0.0005 0.0459" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                        <geom name="tail2_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.0459" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                        <site name="marker6_{i}" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2} 0 0.0459" size="0.005" rgba="{c_marker}"/>
                        <geom name="tail2left_{i}"  type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2} -{args["tail2tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail2tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                        <geom name="tail2right_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2}  {args["tail2tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail2tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                        <site name="tail2left_{i}"  pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2} {-args["tail2tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                        <site name="tail2right_{i}" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2}  {args["tail2tendon"]} 0" size="0.004" rgba="0 0 0 0"/>

                        <geom name="tail2-joint3_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] + args["tail2-tail3"]/4} 0 0" size="{args["tail2-tail3"]/4} 0.0005 0.0382" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                        <body pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] + args["tail2-tail3"]/2} 0 0">
                            <geom name="joint3_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0382" rgba="{c_red_joint}" mass="0"/>
                            <joint name="joint3_{i}" armature="0.001" stiffness="{ks[3]}" damping="{kd[3]}"/>
                            <geom name="joint3-tail3_{i}" type="box" pos="{args["tail2-tail3"]/4} 0 0" size="{args["tail2-tail3"]/4} 0.0005 0.0382" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                            <geom name="tail3_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.0382" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                            <site name="marker7_{i}" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2} 0 0.0382" size="0.005" rgba="{c_marker}"/>
                            <geom name="tail3left_{i}"  type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2} -{args["tail3tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail3tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                            <geom name="tail3right_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2}  {args["tail3tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail3tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                            <site name="tail3left_{i}"  pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2} {-args["tail3tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                            <site name="tail3right_{i}" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2}  {args["tail3tendon"]} 0" size="0.004" rgba="0 0 0 0"/>

                            <geom name="tail3-joint4_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] + args["tail3-tail4"]/4} 0 0" size="{args["tail3-tail4"]/4} 0.0005 0.0286" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                            <body pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] + args["tail3-tail4"]/2} 0 0">
                                <geom name="joint4_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0286" rgba="{c_red_joint}" mass="0"/>
                                <joint name="joint4_{i}" armature="0.001" stiffness="{ks[4]}" damping="{kd[4]}"/>
                                <geom name="joint4-tail4_{i}" type="box" pos="{args["tail3-tail4"]/4} 0 0" size="{args["tail3-tail4"]/4} 0.0005 0.0286" rgba="{c_light}" mass="0.001" fluidshape="{fs}"/>
                                <geom name="tail4_{i}" type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.0286" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                                <site name="marker8_{i}" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2} 0 0.0286" size="0.005" rgba="{c_marker}"/>
                                <geom name="tail4left_{i}"  type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2} -{args["tail4tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail4tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                                <geom name="tail4right_{i}" type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2}  {args["tail4tendon"]/2 + 0.0025} 0" size="0.0015 {args["tail4tendon"]/2 + 0.0025} 0.005" rgba="{c_blue_tail}" density="{args["PLAdensity"]}" fluidshape="{fs}"/>
                                <site name="tail4left_{i}"  pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2} {-args["tail4tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                                <site name="tail4right_{i}" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2}  {args["tail4tendon"]} 0" size="0.004" rgba="0 0 0 0"/>

                                <!-- FIN - overrides childclass to fin_fluid zone [F2S] -->
                                <geom name="finTail_{i}"   type="mesh"     mesh="finTail" pos="-0.312 0 0.01" quat="0.7071 -0.7071 0 0" rgba="{c_red_tail}" mass="{args['tailMass']}" class="fin_fluid" fluidshape="{fs}"/>
                                <geom name="tensioner_{i}" type="cylinder" pos="0.035 0 0" quat="0.7071 0.7071 0 0" size="0.009 0.00975" rgba="{c_blue_tail}" class="fin_fluid" fluidshape="{fs}"/>
                                <site name="tail5left_{i}"  pos="0.035 {-args["tail5tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                                <site name="tail5right_{i}" pos="0.035  {args["tail5tendon"]} 0" size="0.004" rgba="0 0 0 0"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        """)

    # --- 4. ASSEMBLE XML ---
    XML = f"""
<mujoco model="SOFI_field2sim">
    <option gravity="0 0 0" density="{args["fluidDensity"]}" viscosity="{args["fluidViscosity"]}" integrator="implicitfast" timestep="{args["dt"]}" iterations="100" solver="Newton" tolerance="1e-15" ls_tolerance="1e-9" noslip_tolerance="1e-9" o_solimp="0 0.01 0.01 0.1 2">
        <flag contact="disable"/>
    </option>

    <visual>
        <global offheight="2160" offwidth="3840"/>
        <rgba haze=".3 .3 .3 1"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.9" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".6 .6 .6" rgb2=".8 .8 .8" width="512" height="512" mark="cross" markrgb=".9 .9 .9"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        <mesh name="head"     file="meshes/head.obj"/>
        <mesh name="finTail"  file="meshes/croppedTailFin.obj"/>
        <mesh name="body"     file="meshes/body.obj"/>
        <mesh name="headPlate" file="meshes/headPlate.obj"/>
    </asset>

    <default>
        <joint type="hinge" pos="0 0 0" axis="0 0 1" range="-80 80"/>
        <tendon width="0.001" rgba=".9 .6 .5 1" stiffness="{args["tendonStiffness"]}" damping="{args["tendonDamping"]}"/>
        <!-- Segmented fluid zones - coefficients tuned per zone during SysID [F2S] -->
        <default class="head_fluid">
            <geom fluidcoef="{' '.join(map(str, c_head))}" condim="1"/>
        </default>
        <default class="tail_fluid">
            <geom fluidcoef="{' '.join(map(str, c_tail))}" condim="1"/>
        </default>
        <default class="fin_fluid">
            <geom fluidcoef="{' '.join(map(str, c_fin))}" condim="1"/>
        </default>
    </default>

    <worldbody>
        <!-- [F2S] Camera centered on midpoint of ~3 m swim path in -X direction -->
        <camera name="overhead" pos="-1.5 0 5" xyaxes="1 0 0 0 1 0"/>
        <geom name="floor" pos="0 0 -0.2" size="0 0 1" type="plane" material="matplane"/>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
    """
    for i in range(N):
        XML += fishBodyXMLs[i]

    XML += f"""
        <site name="target" type="sphere" pos="1 0 0" size="0.05" rgba=".1 .8 .1 .4"/>
    </worldbody>

    <tendon>
    """
    # Full 6-site tendon routing (matches twin exactly)
    for i in range(N):
        XML += f"""
        <spatial name="tendonLeft_{i}">
            <site site="pulleyRight_{i}"/>
            {f'<site site="tail0left_{i}"/>'  if args["tendonRouting"][0] == 0 else f'<site site="tail0right_{i}"/>'}
            {f'<site site="tail1left_{i}"/>'  if args["tendonRouting"][1] == 0 else f'<site site="tail1right_{i}"/>'}
            {f'<site site="tail2left_{i}"/>'  if args["tendonRouting"][2] == 0 else f'<site site="tail2right_{i}"/>'}
            {f'<site site="tail3left_{i}"/>'  if args["tendonRouting"][3] == 0 else f'<site site="tail3right_{i}"/>'}
            {f'<site site="tail4left_{i}"/>'  if args["tendonRouting"][4] == 0 else f'<site site="tail4right_{i}"/>'}
            {f'<site site="tail5left_{i}"/>'  if args["tendonRouting"][5] == 0 else f'<site site="tail5right_{i}"/>'}
        </spatial>
        <spatial name="tendonRight_{i}">
            <site site="pulleyLeft_{i}"/>
            {f'<site site="tail0right_{i}"/>' if args["tendonRouting"][0] == 0 else f'<site site="tail0left_{i}"/>'}
            {f'<site site="tail1right_{i}"/>' if args["tendonRouting"][1] == 0 else f'<site site="tail1left_{i}"/>'}
            {f'<site site="tail2right_{i}"/>' if args["tendonRouting"][2] == 0 else f'<site site="tail2left_{i}"/>'}
            {f'<site site="tail3right_{i}"/>' if args["tendonRouting"][3] == 0 else f'<site site="tail3left_{i}"/>'}
            {f'<site site="tail4right_{i}"/>' if args["tendonRouting"][4] == 0 else f'<site site="tail4left_{i}"/>'}
            {f'<site site="tail5right_{i}"/>' if args["tendonRouting"][5] == 0 else f'<site site="tail5left_{i}"/>'}
        </spatial>
        """

    XML += """
    </tendon>

    <actuator>
    """
    for i in range(N):
        XML += f"""
        <position name="motor_{i}" joint="motor_{i}" kp="{args['motor_kp']}" kv="{args['motor_kv']}" ctrllimited="true" ctrlrange="-1.14 1.14" forcerange="-300 300"/>
        """

    XML += """
    </actuator>

    <sensor>
    """
    for i in range(N):
        XML += f"""
        <accelerometer name="accel_{i}" site="COM_{i}"/>
        <gyro          name="gyro_{i}"  site="COM_{i}"/>
        """

    XML += """
    </sensor>
</mujoco>
    """

    with open(args["filename"], "w", encoding="utf-8") as f:
        f.write(XML)


if __name__ == "__main__":
    generate_xml(SYSTEMPARAMETERS)
    print(f"XML generated: {SYSTEMPARAMETERS['filename']}")
