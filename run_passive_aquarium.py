
import cv2
import mujoco
import mujoco.viewer

import time
import numpy as np
import matplotlib.pyplot as plt


from Geometry.auto_tendonFish import generate_xml, SYSTEMPARAMETERS
from _simulate import SineSignal, GlidingSineSignal



BOUNDS = [[-2, 2], [-2, 2], [-2, 2]]

pathXML = "Geometry/tendonFish.xml"
# SYSTEMPARAMETERS["fluidShape"] = "none"
SYSTEMPARAMETERS["numberOfFish"] = 16
SYSTEMPARAMETERS["bounds"] = BOUNDS
generate_xml(SYSTEMPARAMETERS, pathXML)
model = mujoco.MjModel.from_xml_path(pathXML)
data = mujoco.MjData(model)

dt = model.opt.timestep
N = SYSTEMPARAMETERS["numberOfFish"]
M = (model.nbody-1)//N
assert M*N+1 == model.nbody, "Number of bodies does not match number of fish and markers!"


resetTimes = []
controlSignals = []
for i in range(N):
    resetTimes.append(np.random.uniform(2.0, 15.0))
    controlSignals.append(SineSignal(frequency=np.random.uniform(0.2, 5.0)))
    # controlSignals.append(SineSignal(frequency=np.random.uniform(0.0, 0.0)))
# control_signal = GlidingSineSignal(swimmingTime=1.0, glidingTime=0.5, frequency=2.0)


with mujoco.Renderer(model, 2000, 2000) as renderer:
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False) as viewer:
        startTime = time.time()
        counter = 0
        while viewer.is_running() and data.time < 3600:
            for i in range(N):
                if data.time > resetTimes[i]:
                    # Give fish new actuation
                    resetTimes[i] = data.time + np.random.uniform(2.0, 15.0)
                    controlSignals[i] = SineSignal(frequency=np.random.uniform(0.2, 5.0))
                data.ctrl[i] = controlSignals[i](data.time)
            mujoco.mj_step(model, data)

            # If any fish is out of bounds, reset that fish
            for i in range(N):
                oldPos = data.site(f"COM_{i}").xpos.copy()
                if oldPos[0] < BOUNDS[0][0] or oldPos[0] > BOUNDS[0][1] or \
                oldPos[1] < BOUNDS[1][0] or oldPos[1] > BOUNDS[1][1] or \
                oldPos[2] < BOUNDS[2][0] or oldPos[2] > BOUNDS[2][1]:
                    # Reset the fish through periodic bounds, it keeps swimming
                    # print(f"Fish {i} reset due to out of bounds at position {oldPos}")

                    newPos = [0, 0, 0]
                    newPos[0] = ((oldPos[0] - BOUNDS[0][0]) % (BOUNDS[0][1] - BOUNDS[0][0])) + BOUNDS[0][0]
                    newPos[1] = ((oldPos[1] - BOUNDS[1][0]) % (BOUNDS[1][1] - BOUNDS[1][0])) + BOUNDS[1][0]
                    newPos[2] = ((oldPos[2] - BOUNDS[2][0]) % (BOUNDS[2][1] - BOUNDS[2][0])) + BOUNDS[2][0]

                    # print(f"Resetting fish {i} to position {newPos}")

                    displacement = np.array(newPos) - oldPos

                    data.joint(f"headX_{i}").qpos += displacement[0]
                    data.joint(f"headY_{i}").qpos += displacement[1]


            viewer.sync()
            

            ### Logging
            counter += 1
            if counter % 500 == 0:
                print(f"Simulation Time: {data.time:.2f}s \t Real Time: {time.time() - startTime:.2f}s")

                if counter > 6000:
                    renderer.update_scene(data, camera="aquarium")
                    img = renderer.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"Outputs/frames/frame_{counter // 500:04d}.png", img)

