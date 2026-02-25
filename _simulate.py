################################################################################
# Helper function for forward simulation of the fish model.
################################################################################

import mujoco
import numpy as np
import time

from Geometry.auto_tendonFish import generate_xml


### Forward Simulation
def sim_fish (systemParameters, controlSignal, totaltime, videofps, warmuptime=0):
    pathXML = "Geometry/_tmp.xml"
    generate_xml(systemParameters, pathXML)
    model = mujoco.MjModel.from_xml_path(pathXML)
    data = mujoco.MjData(model)

    # Set position control for tendons based on initial lengths of the tendons, forward once to populate the initial lengths
    mujoco.mj_forward(model, data)
    initLen = data.ten_length.copy()

    dt = model.opt.timestep
    skipframes = int(1 / (dt * videofps))

    totalMass = 0.0
    for i in range(model.nbody):
        m = model.body_mass[i]
        totalMass += m
    # print(f"Total mass of fish: {totalMass:.4f} kg")

    log = {
        "time": [],
        "runtime": [],
        "controls": [],
        "markers": [],
        "tendonLengths": [],
        "motorPower": [],
        "mass": totalMass,
    }
    frameCount = 0
    counter = 0
    while counter < 1e4 and data.time < totaltime+warmuptime:
        ### Forward Step
        data.ctrl = controlSignal(data.time)
        startTime = time.time()
        mujoco.mj_step(model, data)
        counter += 1
        
        # Skip warmup time
        if data.time < warmuptime:
            continue

        ### Track Markers
        if frameCount % skipframes == 0:
            log["time"].append(data.time)
            log["controls"].append(data.ctrl.copy())
            log["runtime"].append(time.time() - startTime)
            log["tendonLengths"].append(data.ten_length.copy())

            motorTorque = data.joint("motor_0").qfrc_actuator[0]
            motorAngularVel = data.joint("motor_0").qvel[0]
            log["motorPower"].append(motorTorque * motorAngularVel)

            ms = []
            for i in range(10):
                if i == 4:
                    ms.append(data.site(f"bodyFin_0").xpos[:2].copy())
                ms.append(data.site(f"marker{i}_0").xpos[:2].copy())
            log["markers"].append(np.array(ms).flatten())
    
        frameCount += 1
    
    if counter >= 1e4:
        print("Warning: Simulation stopped after exceeding MAX STEPS.")
        return None

    for k in log:
        log[k] = np.array(log[k])
    
    if (log['tendonLengths'].max()-log['tendonLengths'].min())/initLen.mean() > 0.1:
        print(f"Warning: Tendon lengthening is too high: {(log['tendonLengths'].max()-log['tendonLengths'].min())/initLen.mean():.4%}")

        print(f"Minimum/Maximum Tendon Lengths: {log['tendonLengths'].min():.4f} / {log['tendonLengths'].max():.4f} \tAmplitude: {log['tendonLengths'].max()-log['tendonLengths'].min():.4f} \tRelative Lengthening {(log['tendonLengths'].max()-log['tendonLengths'].min())/initLen.mean():.4%}")

    return log



### Sine function for control signal
class SineSignal:
    def __init__(self, frequency=None, leftFrequency=None, rightFrequency=None, phaseOffset=0, mode="velocity"):
        self.mode = mode
        assert mode in ["position", "velocity"], f"Invalid mode: {mode}. Choose 'position' or 'velocity'."

        self.frequency = frequency
        self.leftFrequency = leftFrequency
        self.rightFrequency = rightFrequency
        assert frequency is not None or (leftFrequency is not None and rightFrequency is not None), "Either frequency or leftFrequency/rightFrequency must be set."
        assert not (leftFrequency is not None and rightFrequency is not None and frequency is not None), "Only one of frequency, leftFrequency, or rightFrequency should be set."

        self.phaseOffset = phaseOffset

    def __call__(self, t):
        if self.mode == "position":
            return np.array([
                2 * np.pi * t * self.frequency + self.phaseOffset,
            ])
        elif self.mode == "velocity":
            if self.frequency is not None:
                return np.array([
                    2 * np.pi * self.frequency,
                ])
            else:
                return np.array([
                    2 * np.pi * self.leftFrequency if ((t+self.phaseOffset) % (0.5/self.leftFrequency+0.5/self.rightFrequency)) < (0.5/self.leftFrequency) else 2 * np.pi * self.rightFrequency,
                ])

class GlidingSineSignal:
    def __init__(self, swimmingTime, glidingTime, frequency, phaseOffset=0, mode="velocity"):
        self.mode = mode
        assert mode in ["position", "velocity"], f"Invalid mode: {mode}. Choose 'position' or 'velocity'."
        
        self.swimmingTime = swimmingTime
        self.glidingTime = glidingTime
        self.frequency = frequency
        self.phaseOffset = phaseOffset
    

    def __call__(self, t):
        if self.mode == "position":
            nPeriod = t // (self.swimmingTime + self.glidingTime)  # Actively count number of periods so we can continue with previous motor angle

            if t % (self.swimmingTime + self.glidingTime) <= self.swimmingTime:
                return np.array([
                    2 * np.pi * (t-nPeriod*self.glidingTime) * self.frequency + self.phaseOffset,
                ])
            else:
                return np.array([
                    2 * np.pi * ((nPeriod+1)*self.swimmingTime) * self.frequency + self.phaseOffset,
                ])
            
        elif self.mode == "velocity":
            if t % (self.swimmingTime + self.glidingTime) <= self.swimmingTime:
                return np.array([
                    2 * np.pi * self.frequency,
                ])
            else:
                return np.array([
                    0.0,
                ])



def extract_forward_velocity (log, NUMHEAD=2, NUMMARKERS=11, VIDEOFPS=60):
    ### Returns the forward distance and velocity of the fish based on the markers in the log, forward being defined as the direction the fish head is facing.
    assert NUMMARKERS == 11, "Function only implemented for 11 markers."
    markers = log["markers"].reshape(-1, 11, 2).transpose(1, 0, 2)
    if NUMHEAD == 2:
        markers = np.concatenate([markers[:2], markers[5:10]], axis=0)

    # Define all markers w.r.t. the head0
    head0 = markers[0]
    shiftedMarkers = markers - head0

    # Find rotation matrix such that every frame the tail markers are lined up to the line defined by the first two markers
    headVec = np.mean(shiftedMarkers[1:NUMHEAD], axis=0)
    headVec /= np.linalg.norm(headVec, axis=1, keepdims=True)
    rotationMatrices = np.stack([headVec[:,0], headVec[:,1], -headVec[:,1], headVec[:,0]], axis=1).reshape(-1, 2, 2)

    rotatedMarkers = []
    for i in range(shiftedMarkers.shape[0]):
        rotatedMarkers.append(np.matmul(rotationMatrices, shiftedMarkers[i].reshape(-1, 2, 1)).reshape(-1, 2))
    rotatedMarkers = np.stack(rotatedMarkers, axis=0)
    rotatedHead0 = np.matmul(rotationMatrices[0:1], head0.reshape(-1, 2, 1)).reshape(-1, 2)
    rotatedMarkers[:,:,0] += rotatedHead0[:,0]

    rotatedMarkers -= rotatedMarkers[:NUMHEAD].mean(axis=0, keepdims=True)[:,0:1]


    # Find the linear velocity approximation
    timeAxis = np.arange(0, markers.shape[1] / VIDEOFPS, 1/VIDEOFPS)
    coef = np.polyfit(timeAxis, -rotatedMarkers[:NUMHEAD].mean(axis=0)[:,0], 1) # Swims in negative x direction

    forwardDistance = rotatedMarkers[:NUMHEAD].mean(axis=0)[0,0] - rotatedMarkers[:NUMHEAD].mean(axis=0)[-1,0]
    forwardVelocity = coef[0]

    return forwardDistance, forwardVelocity


def visualize_shape (systemParameters):
    pathXML = "Meshes/tendonFish.xml"
    generate_xml(systemParameters, pathXML)
    model = mujoco.MjModel.from_xml_path(pathXML)
    data = mujoco.MjData(model)

    WIDTH, HEIGHT = 2000, 1600
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        mujoco.mj_forward(model, data)
        # Render the fish model with the current system parameters
        renderer.update_scene(data, camera="closeup")
        img = renderer.render()

    return img