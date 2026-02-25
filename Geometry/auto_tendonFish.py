################################################################################
# Automatically generates the XML for the tendon fish model based on system
# parameters, can generate an arbitrary number of robots per environment.
################################################################################

import mujoco
import numpy as np


SYSTEMPARAMETERS = {
    # Simulation parameters
    "dt": 0.001,
    "fixHead": False,
    "freeZ": False, # If True, the head can move freely in Z direction, otherwise it is fixed
    "tendonRouting": [0, 0, 0, 1, 1], # 0 implies the tendon remains on the same side, 1 implies the tendon crosses to the other side.
    "torqueControl": False, # If True, the motor is controlled by torque, otherwise by velocity

    "numberOfFish": 1,
    "bounds": None, # To visualize a bounded swimming environment, visual purpose only.

    # Fluid parameters, based on water at room temperature
    "fluidShape": "ellipsoid",
    "fluidDensity": 1000,
    "fluidViscosity": 0.0013,
    "fluidCoef": [0.4, 7.79, 2.81, 3.84, 0.27], # Blunt drag, Slender drag, Angular drag, Kutta lift, Magnus lift

    # Material parameters
    "acrylicDensity": 1180,
    "PLAdensity": 1250,

    "electricBoxMass": 0.53,
    "bodyMass": 0.6,
    "tailMass": 0.01,

    "hingeStiffness": 0.65,
    "hingeDamping": 0.0,
    "tendonStiffness": 20000,
    "tendonDamping": 10,

    # Geometric parameters, measured from hardware
    "motorShaftLength": 0.12,
    "motorArmLength": 0.0395,

    "tailSegmentLength": 0.015,
    "body-tail0": 0.014,
    "tail0tendon": 0.045,
    "tail0-tail1": 0.015,
    "tail1tendon": 0.035,
    "tail1-tail2": 0.017,
    "tail2tendon": 0.025,
    "tail2-tail3": 0.014,
    "tail3tendon": 0.015,
    "tail3-tail4": 0.006,
    "tail4tendon": 0.005,
    "tail4-finTail": 0.01,

    ### No need to change: will be recomputed each time the model is changed
    "centerOfMass": [0.02456792, 0.0, 0.00080073], # [x, y, z] in m
}


def generate_xml (args, filename="tendonFish.xml"):
    generate_xml_(args, filename)

    # Automatically assigns the center of mass correctly for visualization purposes for single fish.
    if args["numberOfFish"] == 1:
        prevCom = np.zeros(3, dtype=np.float32)
        for _ in range(10):
            # Compute COM
            model = mujoco.MjModel.from_xml_path(filename)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            com = np.zeros(3, dtype=np.float32)
            totalMass = 0.0
            for i in range(model.nbody):
                x = data.body(i).xpos.copy()
                m = model.body_mass[i]
                com += x * m
                totalMass += m

            com /= totalMass
            # print(f"Center of Mass: {com}")
            args["centerOfMass"] = com
            generate_xml_(args, filename)

            if np.linalg.norm(com - prevCom) < 1e-6:
                return
            prevCom = com.copy()
        
        print(f"Could not converge on COM after 10 iterations, final COM: {com}")
    

def generate_xml_ (args, filename):
    N = args["numberOfFish"]
    fishBodyXMLs = []
    for i in range(N):
        fishBodyXMLs.append(f"""
        <body pos="0 {i*0.5} 0" name="fish_{i}">
            <camera name="tracking_{i}" mode="track" pos="4 -4 4" xyaxes="1 1 0 -0.5 0.5 1"/>
            <camera name="closeup_{i}" mode="track" pos="2 -2 2" xyaxes="1 1 0 -0.5 0.5 1"/>

            {f'<joint name="headX_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="slide" axis="1 0 0" stiffness="0" damping="0"/>' if not args["fixHead"] else ''}
            {f'<joint name="headY_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="slide" axis="0 1 0" stiffness="0" damping="0"/>' if not args["fixHead"] else ''}
            {f'<joint name="headZ_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" type="slide" axis="0 0 1" stiffness="{0 if args["freeZ"] else 500}" damping="{0 if args["freeZ"] else 10}"/>' if not args["fixHead"] else ''}

            {f'<joint name="headRollX_{i}" type="hinge" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" axis="1 0 0" range="-180 180" stiffness="{0 if args["freeZ"] else 10}" damping="{0 if args["freeZ"] else 1}"/>' if not args["fixHead"] else ''}
            {f'<joint name="headRollY_{i}" type="hinge" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" axis="0 1 0" range="-180 180" stiffness="{0 if args["freeZ"] else 10}" damping="{0 if args["freeZ"] else 1}"/>' if not args["fixHead"] else ''}
            {f'<joint name="headRollZ_{i}" type="hinge" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" axis="0 0 1" limited="false" stiffness="0" damping="0"/>' if not args["fixHead"] else ''}
            
            <site name="COM_{i}" pos="{args["centerOfMass"][0]} {args["centerOfMass"][1]} {args["centerOfMass"][2]}" size="0.02" material="marker"/>

            
            <geom name="headPlate_{i}" type="box" pos="0 0 0" size="0.077 0.0025 0.05" material="acrylic" density="{args["acrylicDensity"]}" fluidshape="{args["fluidShape"]}"/>
            <geom name="headTopPlate_{i}" type="box" pos="0.01 0 0.0515" size="0.08 0.02 0.0015" rgba=".9 .9 .9 0" density="{args["acrylicDensity"]}"/>
            <geom name="headTopElectric_{i}" type="box" pos="0.005 0 0.083" size="0.06 0.04 0.03" rgba=".9 .9 .2 0." mass="{args['electricBoxMass']}"/>
            <geom name="headAttachment_{i}" type="box" pos="0.1 0 0" size="0.023 0.015 0.05" rgba=".1 .1 .1 1" mass="0.08" fluidshape="{args["fluidShape"]}"/>

            <geom name="body_{i}" type="box" pos="0.2005 0 0" size="0.0775 0.01 0.0525" material="alu" mass="{args['bodyMass']}" fluidshape="{args["fluidShape"]}"/>
            <geom name="body-joint0_{i}" type="box" pos="{0.278+args["body-tail0"]/4} 0 0" size="{args["body-tail0"]/4} 0.0005 0.0375" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>

            <geom name="finTop_{i}" type="mesh" mesh="finTop" pos="0.26 0 0.04" material="acrylic" density="{args["acrylicDensity"]}"/>

            <site name="marker0_{i}" pos="-0.077 0 0.05" size="0.005" material="marker"/>
            <site name="marker1_{i}" pos="0.065 0 0.05" size="0.005" material="marker"/>
            <site name="marker2_{i}" pos="0.1655 -0.0775 0.05" size="0.005" material="marker"/>
            <site name="marker3_{i}" pos="0.1655 0.0775 0.05" size="0.005" material="marker"/>
            <site name="bodyFin_{i}" pos="0.28 0 0.1" size="0.005" material="marker"/>


            <body pos="0.1655 0 0" name="motor_{i}">
                <joint name="motor_{i}" type="hinge" axis="0 1 0" pos="0 0 0" limited="false" stiffness="0" damping="0"/>
                <geom name="motorShaft_{i}" type="box" pos="0 0 0" size="0.0075 {args['motorShaftLength']/2} 0.0075" rgba=".1 .1 .1 1" mass="0.05" fluidshape="{args["fluidShape"]}"/>
                <geom name="motorArmLeft_{i}" type="box" pos="0 {-args['motorShaftLength']/2+0.005} {-args["motorArmLength"]/2}" size="0.0075 0.005 {args["motorArmLength"]/2}" rgba=".1 .1 .1 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                <geom name="motorArmRight_{i}" type="box" pos="0 {args['motorShaftLength']/2-0.005} {args["motorArmLength"]/2}" size="0.0075 0.005 {args["motorArmLength"]/2}" rgba=".1 .1 .1 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>

                <site name="bodyLeft_{i}" pos="0 {-args['motorShaftLength']/2+0.005} {-args["motorArmLength"]+0.005}" size="0.005" rgba=".9 .6 .5 .5"/>
                <site name="bodyRight_{i}" pos="0 {args['motorShaftLength']/2-0.005} {args["motorArmLength"]-0.005}" size="0.005" rgba=".9 .6 .5 .5"/>
            </body>
            

            <body pos="{0.278+args["body-tail0"]/2} 0 0">
                <geom name="joint0_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0376" rgba=".9 .2 .2 1" mass="0"/>
                <joint name="joint0_{i}"/>
                <geom name="joint0-tail0_{i}" type="box" pos="{args["body-tail0"]/4} 0 0" size="{args["body-tail0"]/4} 0.0005 0.0375" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>
                <geom name="tail0_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.05" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                <site name="marker4_{i}" pos="{args["body-tail0"]/2 + args["tailSegmentLength"]/2} 0 0.05" size="0.005" material="marker"/>

                <geom name="tail0left_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"] - 0.003} -0.02575 0" size="0.003 0.02425 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                <geom name="tail0right_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"] - 0.003} 0.02575 0" size="0.003 0.02425 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                <site name="tail0left_{i}" pos="{args["body-tail0"]/2 + args["tailSegmentLength"] - 0.003} {-args["tail0tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>
                <site name="tail0right_{i}" pos="{args["body-tail0"]/2 + args["tailSegmentLength"] - 0.003} {args["tail0tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>

                <geom name="tail0-joint1_{i}" type="box" pos="{args["body-tail0"]/2 + args["tailSegmentLength"] + args["tail0-tail1"]/4} 0 0" size="{args["tail0-tail1"]/4} 0.0005 0.035" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>

                <body pos="{args["body-tail0"]/2 + args["tailSegmentLength"] + args["tail0-tail1"]/2} 0 0">
                    <geom name="joint1_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0351" rgba=".6 .1 .1 1" mass="0"/>
                    <joint name="joint1_{i}"/>
                    <geom name="joint1-tail1_{i}" type="box" pos="{args["tail0-tail1"]/4} 0 0" size="{args["tail0-tail1"]/4} 0.0005 0.035" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>
                    <geom name="tail1_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.0475" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                    <site name="marker5_{i}" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"]/2} 0 0.05" size="0.005" material="marker"/>

                    <geom name="tail1left_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] - 0.003} -0.02075 0" size="0.003 0.01925 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                    <geom name="tail1right_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] - 0.003} 0.02075 0" size="0.003 0.01925 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                    <site name="tail1left_{i}" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] - 0.003} {-args["tail1tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>
                    <site name="tail1right_{i}" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] - 0.003} {args["tail1tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>

                    <geom name="tail1-joint2_{i}" type="box" pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] + args["tail1-tail2"]/4} 0 0" size="{args["tail1-tail2"]/4} 0.0005 0.0325" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>

                    <body pos="{args["tail0-tail1"]/2 + args["tailSegmentLength"] + args["tail1-tail2"]/2} 0 0">
                        <geom name="joint2_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0326" rgba=".6 .1 .1 1" mass="0"/>
                        <joint name="joint2_{i}"/>
                        <geom name="joint2-tail2_{i}" type="box" pos="{args["tail1-tail2"]/4} 0 0" size="{args["tail1-tail2"]/4} 0.0005 0.0325" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>
                        <geom name="tail2_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.045" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                        <site name="marker6_{i}" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"]/2} 0 0.05" size="0.005" material="marker"/>

                        <geom name="tail2left_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] - 0.003} -0.01575 0" size="0.003 0.01425 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                        <geom name="tail2right_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] - 0.003} 0.01575 0" size="0.003 0.01425 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                        <site name="tail2left_{i}" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] - 0.003} {-args["tail2tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>
                        <site name="tail2right_{i}" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] - 0.003} {args["tail2tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>

                        <geom name="tail2-joint3_{i}" type="box" pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] + args["tail2-tail3"]/4} 0 0" size="{args["tail2-tail3"]/4} 0.0005 0.03" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>


                        <body pos="{args["tail1-tail2"]/2 + args["tailSegmentLength"] + args["tail2-tail3"]/2} 0 0">
                            <geom name="joint3_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0301" rgba=".9 .2 .2 1" mass="0"/>
                            <joint name="joint3_{i}"/>
                            <geom name="joint3-tail3_{i}" type="box" pos="{args["tail2-tail3"]/4} 0 0" size="{args["tail2-tail3"]/4} 0.0005 0.03" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>
                            <geom name="tail3_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.0425" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                            <site name="marker7_{i}" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"]/2} 0 0.05" size="0.005" material="marker"/>

                            <geom name="tail3left_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] - 0.003} -0.01075 0" size="0.003 0.00925 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                            <geom name="tail3right_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] - 0.003} 0.01075 0" size="0.003 0.00925 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                            <site name="tail3left_{i}" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] - 0.003} {-args["tail3tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>
                            <site name="tail3right_{i}" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] - 0.003} {args["tail3tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>

                            <geom name="tail3-joint4_{i}" type="box" pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] + args["tail3-tail4"]/4} 0 0" size="{args["tail3-tail4"]/4} 0.0005 0.0275" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>


                            <body pos="{args["tail2-tail3"]/2 + args["tailSegmentLength"] + args["tail3-tail4"]/2} 0 0">
                                <geom name="joint4_{i}" type="cylinder" pos="0 0 0" size="0.002 0.0276" rgba=".9 .2 .2 1" mass="0"/>
                                <joint name="joint4_{i}"/>
                                <geom name="joint4-tail4_{i}" type="box" pos="{args["tail3-tail4"]/4} 0 0" size="{args["tail3-tail4"]/4} 0.0005 0.0275" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>
                                <geom name="tail4_{i}" type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2} 0 0" size="{args["tailSegmentLength"]/2} 0.0015 0.04" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                                <site name="marker8_{i}" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"]/2} 0 0.05" size="0.005" material="marker"/>

                                <geom name="tail4left_{i}" type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] - 0.003} -0.00575 0" size="0.003 0.00425 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                                <geom name="tail4right_{i}" type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] - 0.003} 0.00575 0" size="0.003 0.00425 0.007" rgba=".5 .7 .9 1" density="{args["PLAdensity"]}" fluidshape="{args["fluidShape"]}"/>
                                <site name="tail4left_{i}" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] - 0.003} {-args["tail4tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>
                                <site name="tail4right_{i}" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] - 0.003} {args["tail4tendon"]} 0" size="0.004" rgba=".9 .6 .5 .5"/>

                                <!-- Tail Fin -->
                                <geom name="tail4-finTail_{i}" type="box" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] + args["tail4-finTail"]/2} 0 0" size="{args["tail4-finTail"]/2} 0.0005 0.0225" rgba=".9 .9 .9 1" mass="0.001" fluidshape="{args["fluidShape"]}"/>
                                <geom name="finTail_{i}" type="mesh" mesh="finTail" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] + args["tail4-finTail"] - 0.012} 0 0" rgba=".9 .2 .2 0.7" mass="{args["tailMass"]}" fluidshape="{args["fluidShape"]}"/>
                                <site name="marker9_{i}" pos="{args["tail3-tail4"]/2 + args["tailSegmentLength"] + args["tail4-finTail"] + 0.045} 0 0.1" size="0.005" material="marker"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        """)

    XML = f"""
<mujoco model="tendonFish">
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

        <mesh name="finTop" file="Meshes/finTop.obj"/>
        <mesh name="finTail" file="Meshes/finTail.obj"/>

        <material name="acrylic" rgba=".9 .9 .9 0.2"/>
        <material name="alu" reflectance="0.9" specular="0.9" shininess="0.9" rgba=".7 .7 .7 1"/>
        <material name="marker" rgba=".9 .4 .1 .5"/>
    </asset>

    <default>
        <joint type="hinge" pos="0 0 0" axis="0 0 1" range="-80 80" stiffness="{args["hingeStiffness"]}" damping="{args["hingeDamping"]}"/>
        <geom fluidcoef="{args['fluidCoef'][0]} {args['fluidCoef'][1]} {args['fluidCoef'][2]} {args['fluidCoef'][3]} {args['fluidCoef'][4]}" condim="1"/>
        <tendon width="0.001" rgba=".9 .6 .5 1" stiffness="{args["tendonStiffness"]}" damping="{args["tendonDamping"]}"/>
    </default>

    <worldbody>
        <camera name="fixedTopCircle" pos="0.0 1.0 6.0" xyaxes="1 0 0 0 1 0"/>
        <camera name="fixedTop" pos="-1.0 -2.0 3.0" xyaxes="1 0 0 0 1 0.6"/>
        <camera name="fixedFrontCircle" pos="0 -2 2" xyaxes="1 0 0 0 1 1.25"/>
        <camera name="fixedFront" pos="0 -1 1" xyaxes="1 0 0 0 1 1.25"/>
        <camera name="aquarium" pos="0 -5 4" xyaxes="1 0 0 0 1 1"/>
        <camera name="frontTarget" pos="-2 -5 4" xyaxes="1 0 0 0 1 1.5"/>
        <camera pos="0.625 -0.592 0.720" xyaxes="0.758 0.653 -0.000 -0.468 0.544 0.697"/>


        <geom name="floor" pos="0 0 -0.2" size="0 0 1" type="plane" material="matplane"/>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>

        {f'<geom name="boundingboxXmin" type="box" pos="{args["bounds"][0][0]} {(args["bounds"][1][1]+args["bounds"][1][0])/2} -0.1" size="0.01 {(args["bounds"][1][1]-args["bounds"][1][0])/2} 0.1" rgba=".5 .5 .9 .25"/>' if args["bounds"] is not None else ''}
        {f'<geom name="boundingboxYmin" type="box" pos="{(args["bounds"][0][1]+args["bounds"][0][0])/2} {args["bounds"][1][0]} -0.1" size="{(args["bounds"][0][1]-args["bounds"][0][0])/2} 0.01 0.1" rgba=".5 .5 .9 .25"/>' if args["bounds"] is not None else ''}
        {f'<geom name="boundingboxYmax" type="box" pos="{(args["bounds"][0][1]+args["bounds"][0][0])/2} {args["bounds"][1][1]} -0.1" size="{(args["bounds"][0][1]-args["bounds"][0][0])/2} 0.01 0.1" rgba=".5 .5 .9 .25"/>' if args["bounds"] is not None else ''}
        {f'<geom name="boundingboxXmax" type="box" pos="{args["bounds"][0][1]} {(args["bounds"][1][1]+args["bounds"][1][0])/2} -0.1" size="0.01 {(args["bounds"][1][1]-args["bounds"][1][0])/2} 0.1" rgba=".5 .5 .9 .25"/>' if args["bounds"] is not None else ''}

    """
    for i in range(N):
        XML += fishBodyXMLs[i]

    XML += f"""
        <site name="target" type="sphere" pos="1 0 0" size="0.05" rgba=".1 .8 .1 .4"/>
    """

    ### Add Tendon Routing
    if args["tendonRouting"] is None:
        print("No tendon routing specified, skipping tendon and actuator generation.")
    else:
        tendonXMLs = []
        for i in range(N):
            tendonXMLs.append(f"""
        <spatial name="tendonLeft_{i}">
            <site site="bodyLeft_{i}"/>
            {f'<site site="tail0left_{i}"/>' if args["tendonRouting"][0] == 0 else f'<site site="tail0right_{i}"/>'}
            {f'<site site="tail1left_{i}"/>' if args["tendonRouting"][1] == 0 else f'<site site="tail1right_{i}"/>'}
            {f'<site site="tail2left_{i}"/>' if args["tendonRouting"][2] == 0 else f'<site site="tail2right_{i}"/>'}
            {f'<site site="tail3left_{i}"/>' if args["tendonRouting"][3] == 0 else f'<site site="tail3right_{i}"/>'}
            {f'<site site="tail4left_{i}"/>' if args["tendonRouting"][4] == 0 else f'<site site="tail4right_{i}"/>'}
        </spatial>
        <spatial name="tendonRight_{i}">
            <site site="bodyRight_{i}"/>    
            {f'<site site="tail0right_{i}"/>' if args["tendonRouting"][0] == 0 else f'<site site="tail0left_{i}"/>'}      
            {f'<site site="tail1right_{i}"/>' if args["tendonRouting"][1] == 0 else f'<site site="tail1left_{i}"/>'}
            {f'<site site="tail2right_{i}"/>' if args["tendonRouting"][2] == 0 else f'<site site="tail2left_{i}"/>'}
            {f'<site site="tail3right_{i}"/>' if args["tendonRouting"][3] == 0 else f'<site site="tail3left_{i}"/>'}
            {f'<site site="tail4right_{i}"/>' if args["tendonRouting"][4] == 0 else f'<site site="tail4left_{i}"/>'}
        </spatial>
            """)
        
        actuatorXMLs = []
        for i in range(N):
            if args["torqueControl"]:
                actuatorXMLs.append(f"""
        <motor name="motor_{i}" joint="motor_{i}" ctrllimited="true" ctrlrange="-100 100"/>
                """)
            else:
                actuatorXMLs.append(f"""
        <velocity name="motor_{i}" joint="motor_{i}" kv="100" ctrllimited="true" ctrlrange="-{2*3.14*5} {2*3.14*5}" forcerange="-100 100"/>
                """)

        XML += f"""
    </worldbody>
    

    <tendon>
        """
        for i in range(N):
            XML += tendonXMLs[i]

        XML += f"""
    </tendon>


    <actuator>
        """
        for i in range(N):
            XML += actuatorXMLs[i]

        XML += f"""
    </actuator>

</mujoco>
        """

    with open(filename, "w") as f:
        f.write(XML)



if __name__ == "__main__":
    generate_xml(SYSTEMPARAMETERS, filename="Geometry/tendonFish.xml")
    print("XML file generated: Geometry/tendonFish.xml")
