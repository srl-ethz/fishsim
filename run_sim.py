
import mujoco
import mujoco.viewer

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Geometry.auto_tendonFish import generate_xml, SYSTEMPARAMETERS
from _simulate import SineSignal, GlidingSineSignal

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

mm = 1 / 25.4

def simulate_and_render (systemParameters, controlSignal, totaltime, warmuptime=0.0, videofps=60, width=2400, height=1600, cameraName="fixed"):
    pathXML = "Geometry/tendonFish.xml"

    generate_xml(systemParameters, pathXML)
    model = mujoco.MjModel.from_xml_path(pathXML)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    skipframes = int(1 / (dt * videofps))

    log = {
        "time": [],
        "runtime": [],
        "images": [],
        "controls": [],
        "markers": [],
    }
    with mujoco.Renderer(model, height, width) as renderer:
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

            ### Logging
            if frameCount % skipframes == 0:
                log["time"].append(data.time)
                log["controls"].append(data.ctrl.copy())
                log["runtime"].append(time.time() - startTime)

                ms = []
                for i in range(10):
                    if i == 4:
                        ms.append(data.site(f"bodyFin_0").xpos[:2].copy())
                    ms.append(data.site(f"marker{i}_0").xpos[:2].copy())
                log["markers"].append(np.array(ms).flatten())

                # Render
                renderer.update_scene(data, camera=cameraName)
                img = renderer.render()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("Outputs/rendered_image_top.png", img)
                
                log["images"].append(img)

            frameCount += 1

            if frameCount % 2000 == 0:
                print(f"Time: {data.time:.4f}")


    for key in log:
        log[key] = np.array(log[key])

    print(f"\033[95mAverage Runtime per Frame: {1e3*np.mean(log['runtime']):.4f}ms ± {1e3*np.std(log['runtime']):.6f}ms \tFactor Realtime: {dt/np.mean(log['runtime']):.2f}x\033[0m")

    headPos = log["markers"].reshape(-1, 11, 2)[:,:5,0].mean(1)
    print(f"Initial/Final Head X: {headPos[0]:.4f}m/{headPos[-1]:.4f}m")

    ### Store Marker Positions
    np.savetxt("Outputs/markers.csv", log["markers"], delimiter=",")
    print("\033[94mStored markers in Outputs/markers.csv.\033[0m")


    ### Plotting
    nPlots = 3
    fig, axs = plt.subplots(nPlots, 1, figsize=(120*mm, nPlots*40*mm), sharex=True)

    axs[0].plot(log["time"], log["controls"])
    axs[0].set_ylabel("Control Signal (rad/s)")
    axs[0].grid()
    axs[0].set_xlim(0, log["time"][-1])
    axs[0].ticklabel_format(style="sci", axis='y', scilimits=(0,0))

    axs[1].plot(log["time"], headPos)
    axs[1].set_ylabel("Head X Position (m)")
    axs[1].grid()
    axs[1].set_xlim(0, log["time"][-1])
    axs[1].ticklabel_format(style="sci", axis='y', scilimits=(0,0))

    for i in range(log["markers"].shape[1]//2):
        axs[2].plot(log["time"], log["markers"][:, 2*i+1], label=f"marker{i}")
    axs[2].set_ylabel("Marker Y Position (m)")
    axs[2].grid()
    axs[2].set_xlim(0, log["time"][-1])
    axs[2].ticklabel_format(style="sci", axis='y', scilimits=(0,0))
    axs[2].legend(loc="upper center", ncol=6, fancybox=True, bbox_to_anchor=(0.5, -0.35))

    axs[2].set_xlabel("Time (s)")

    fig.savefig("Outputs/plot.png", dpi=600, bbox_inches="tight")
    plt.close()



    ### Save Video
    video = cv2.VideoWriter("_tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), videofps, (width, height))
    for img in log["images"]:
        video.write(img)
    video.release()

    os.system("ffmpeg -i _tmp.mp4 -y -hide_banner -loglevel error -c:v libx264 -crf 28 -c:a aac -strict 2 Outputs/movie.mp4")
    os.remove("_tmp.mp4")
    print("\033[94mStored video in Outputs/movie.mp4.\033[0m")


if __name__ == "__main__":
    systemParameters = SYSTEMPARAMETERS.copy()
    systemParameters["fluidCoef"] = [0.4, 7.79, 2.81, 3.84, 0.27]

    ### Choose control signal
    controlSignal = SineSignal(frequency=1.0)
    # control_signal = SineSignal(leftFrequency=0.5, rightFrequency=3.0)
    # control_signal = GlidingSineSignal(swimmingTime=2/0.596, glidingTime=1/0.596, frequency=0.596)
    
    simulate_and_render(systemParameters, controlSignal, totaltime=4.0, warmuptime=0.0, videofps=60, width=2400, height=1600, cameraName="fixedFront")
