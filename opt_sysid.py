################################################################################
# Given ground truth data and a set of system identification parameters, this 
# script will find the optimal parameters to match the data for both actuation 
# as well as fluid parameters using Bayesian Optimization. 
################################################################################

import argparse
import os
import time
import yaml

import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

from Geometry.auto_tendonFish import SYSTEMPARAMETERS
from _simulate import sim_fish as sim
from _simulate import SineSignal
from Data.process_tailTracking import rotate_markers


plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})

mm = 1 / 25.4

seed = 42
np.random.seed(seed)

NUMHEAD = 2


### Actuation Loss
def compute_metric_act (log, visualize=False):
    markers = log["markers"].reshape(-1, 11, 2)
    if log["numHead"] == 2:
        markers = np.concatenate([markers[:,:2], markers[:,4:]], axis=1)
    rotatedMarkers = rotate_markers(markers, numHead=log["numHead"])

    # Have both sim and real data head be origin for ALL timesteps
    mSim = rotatedMarkers - rotatedMarkers[:,:log["numHead"]].mean(axis=1, keepdims=True)
    mReal = log["rotatedMarkersGT"] - log["rotatedMarkersGT"][:,:log["numHead"]].mean(axis=1, keepdims=True)
    # Truncate
    if mSim.shape[0] < mReal.shape[0]:
        mReal = mReal[:mSim.shape[0]]
    else:
        mSim = mSim[:mReal.shape[0]]
        
    # Consider y-position (lateral) displacement for error
    ySim = mSim[:, log["numHead"]:, 1]
    ySim -= ySim.mean(axis=0)
    yReal = mReal[:ySim.shape[0], log["numHead"]:, 1]
    yReal -= yReal.mean(axis=0)

    error = abs(ySim - yReal).mean()

    if visualize:
        fig, ax = plt.subplots(figsize=(40*mm, 32*mm))
        timeAxis = np.linspace(0, ySim.shape[0]/log["videofps"], ySim.shape[0])
        for i in range(log["numHead"], ySim.shape[1]):
            ax.plot(timeAxis, ySim[:, i], c="tab:blue", linewidth=1)
            ax.plot(timeAxis, yReal[:, i], c="tab:orange", linestyle="dashed", linewidth=0.75)
        
        ax.scatter([], [],  c="tab:blue", s=2, label="Sim")
        ax.scatter([], [],  c="tab:orange", s=2, label="Real")
        ax.grid()
        ax.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
        ax.set_xlim(0, 4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y Position (m)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.7, 1.25), handlelength=0.5, columnspacing=0.1, ncols=2)

        fig.savefig(f"Outputs/opt_act_{log['filename']}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"Outputs/opt_act_{log['filename']}.pdf", bbox_inches="tight")
        plt.close(fig)

    return error

### Fluid Loss
def compute_metric_fluid (log, visualize=False):
    markers = log["markers"].reshape(-1, 11, 2)
    if log["numHead"] == 2:
        markers = np.concatenate([markers[:,:2], markers[:,4:]], axis=1)
    rotatedMarkers = rotate_markers(markers, numHead=log["numHead"])

    # Have both sim and real data head be origin for INITIAL timesteps
    mSim = rotatedMarkers - rotatedMarkers[0:1,:log["numHead"]].mean(axis=1, keepdims=True)
    mReal = log["rotatedMarkersGT"][log["filename"]] - log["rotatedMarkersGT"][log["filename"]][0:1,:log["numHead"]].mean(axis=1, keepdims=True)
    # Truncate
    if mSim.shape[0] < mReal.shape[0]:
        mReal = mReal[:mSim.shape[0]]
    else:
        mSim = mSim[:mReal.shape[0]]

    error = np.linalg.norm(mSim - mReal, axis=2).mean()

    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(88*mm, 30*mm))
        plt.subplots_adjust(wspace=0.5)

        ### Forward Swimming
        timeAxis = np.linspace(0, mSim.shape[0]/log["videofps"], mSim.shape[0])
        axs[1].plot(timeAxis, mSim[:,:log["numHead"]].mean(axis=1)[:,0], c="tab:orange", linewidth=1)
        axs[1].plot(timeAxis, mReal[:,:log["numHead"]].mean(axis=1)[:mSim.shape[0], 0], c="tab:blue", linestyle="dashed", linewidth=0.75)
        axs[1].scatter([], [],  c="tab:blue", s=2, label="Sim")
        axs[1].scatter([], [],  c="tab:orange", s=2, label="Real")
        axs[1].grid()
        axs[1].ticklabel_format(style="sci", axis='y', scilimits=(0,0))
        axs[1].set_xlim(timeAxis[0], timeAxis[-1])
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("X Position (m)")
        axs[1].legend()

        ### Envelope
        timesteps = 120
        frameskip = 10
        markerEnvelope = 5 # Last 5 markers visualized for envelope
        mSim = mSim[:timesteps]
        mReal = mReal[:timesteps]

        axs[0].plot(mSim[::frameskip, -markerEnvelope:, 0], mSim[::frameskip, -markerEnvelope:, 1], marker="o", c="tab:blue", markersize=1, linewidth=0.5)
        axs[0].plot(mReal[::frameskip, -markerEnvelope:, 0], mReal[::frameskip, -markerEnvelope:, 1], marker="o", c="tab:orange", markersize=0.75, linewidth=0.5, linestyle="dashed")
        axs[0].grid()
        # axs[0].ticklabel_format(style="sci", axis='both', scilimits=(0,0))
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        # axs[0].set_xlim(-0., 0.45)


        fig.savefig(f"Outputs/opt_fluid_{log['filename']}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"Outputs/opt_fluid_{log['filename']}.pdf", bbox_inches="tight")
        plt.close(fig)

    return error

### EBT Loss
def compute_ebt_cruising_speed (markers, cD, beta, dt):
    # Fixed parameters for our case
    rho, tailHeight = 1e3, 10e-2
    wettedSurfaceArea = 2 * tailHeight * 0.5
    m = 0.25 * np.pi * beta * rho * (tailHeight)**2

    # Compute EBT lateral displacement derivatives
    dhdt = (markers[1:,-1,1] - markers[:-1,-1,1]) / dt
    dhdt -= dhdt.mean()
    dhdx = (markers[:,-1,1] - markers[:,-2,1]) / (markers[:,-1,0] - markers[:,-2,0])
    dhdx -= dhdx.mean()

    # Compute EBT cruising speed
    cruisingSpeed = np.sqrt( (m * (dhdt**2).mean()) / (cD * rho * wettedSurfaceArea + m * (dhdx**2).mean()) )
    
    return cruisingSpeed

def compute_metric_ebt (markers, cD, beta, dt):
    # Compute EBT cruising speed
    cruisingSpeed = compute_ebt_cruising_speed(markers, cD, beta, dt)

    # Error computation based on real velocity (approximated)
    headDispl = markers[:,:2,0].mean(1) - markers[0:1,:2,0].mean(1)
    timeAxis = np.arange(len(headDispl)) * dt
    velReal = np.polyfit(timeAxis, -headDispl, 1)[0]

    error = abs(cruisingSpeed - velReal)

    return error


def f_act (x, filename, rotatedMarkersGT, simTime, videofps, visualize=False):
    systemParameters = SYSTEMPARAMETERS.copy()
    systemParameters["fluidShape"] = "none"
    frequency = x[0]
    phaseOffset = x[1]
    timeOffset = phaseOffset / (2 * np.pi * frequency)

    log = sim(systemParameters, SineSignal(frequency=frequency), simTime, videofps, warmuptime=timeOffset)
    if log is None:
        print("Simulation failed. Skipping this parameter set.")
        return 1e4
    log["filename"] = filename
    log["numHead"] = NUMHEAD
    log["videofps"] = videofps
    log["rotatedMarkersGT"] = rotatedMarkersGT
    loss = compute_metric_act(log, visualize=visualize)

    return loss

def f_fluid (x, filenames, rotatedMarkersGTs, frequencies, phaseOffsets, simTime, videofps, visualize=False):
    systemParameters = SYSTEMPARAMETERS.copy()
    systemParameters["fluidCoef"] = x

    totalLoss = 0.0
    for filename in filenames:
        timeOffset = phaseOffsets[filename] / (2 * np.pi * frequencies[filename])
        log = sim(systemParameters, SineSignal(frequency=frequencies[filename]), simTime, videofps, warmuptime=timeOffset)
        if log is None:
            print("Simulation failed. Skipping this parameter set.")
            return 1e4
        log["filename"] = filename
        log["numHead"] = NUMHEAD
        log["videofps"] = videofps
        log["rotatedMarkersGT"] = rotatedMarkersGTs
        loss = compute_metric_fluid(log, visualize=visualize)
        totalLoss += loss
    totalLoss /= len(filenames)

    return totalLoss

def f_ebt (x, rotatedMarkersGTs, videofps):
    cD = x[0]
    beta = x[1]

    totalLoss = 0.0
    for rotatedMarkerGT in rotatedMarkersGTs.values():
        loss = compute_metric_ebt(rotatedMarkerGT, cD=cD, beta=beta, dt=1/videofps)
        totalLoss += loss
    totalLoss /= len(rotatedMarkersGTs)

    return totalLoss



def main (filenames, optType="act", simTime=4.0, videofps=60, opt_init=50, opt_iter=100):
    frequencies = {}
    phaseOffsets = {}

    ### Tail Actuation Optimization
    if optType == "act":
        # Initial Guess
        x = [1.0, np.pi/2] # [frequency (Hz), phaseOffset (rad)]

    ### Fluid Parameter Optimization
    elif optType == "fluid":
        # Initial Guess
        x = [1.0, 0.5, 1.5, 1.7, 1.0] # Blunt drag, Slender drag, Angular drag, Kutta lift, Magnus lift
        for filename in filenames:
            # Assume we have optimized the actuation already
            if not os.path.exists(f"Data/Optimization/act_{filename}.yml"):
                raise FileNotFoundError(f"Actuation optimization file not found for {filename}. Please run opt_sysid.py for actuation first.")
            with open(f"Data/Optimization/act_{filename}.yml", "r") as f:
                data = yaml.safe_load(f)
                frequencies[filename] = data["frequency"]
                phaseOffsets[filename] = data["phaseOffset"]

            print(f"Frequency for {filename}: {frequencies[filename]}Hz with phase offset {phaseOffsets[filename]}rad")

    elif optType == "ebt":
        # Initial Guess
        x = [0.5, 1.0] # [c_d, beta]

    else:
        raise ValueError(f"Invalid optimization type: {optType}")

    ### Load ground truth data
    rotatedMarkersGT = {}
    for filename in filenames:
        rotatedMarkersGT[filename] = np.loadtxt(f"Data/Markers/rotatedMarkers_{filename}.csv", delimiter=",").reshape(-1, 9, 2)


    startTime = time.time()
    ### Run Bayesian Optimization
    if not os.path.exists("Data/Optimization"):
        os.makedirs("Data/Optimization")
    if optType == "act":
        for filename in filenames:
            print(f"\033[94mOptimizing actuation for {filename}...\033[0m")
                
            optimizer = BayesianOptimization(
                f=lambda freq,phi: -f_act([freq,phi], filename=filename, rotatedMarkersGT=rotatedMarkersGT[filename], simTime=simTime, videofps=videofps),
                pbounds={'freq': [0.25, 3.0], 'phi': [0.0, 2*np.pi]},
                random_state=seed,
            )
            optimizer.maximize(init_points=opt_init, n_iter=opt_iter)
            x = [optimizer.max['params']['freq'], optimizer.max['params']['phi']]

            # Evaluate Best Parameters
            printX = '[' + ', '.join([f'{xi:.4e}' for xi in x]) + ']'
            optLoss = f_act(x, filename=filename, rotatedMarkersGT=rotatedMarkersGT[filename], simTime=simTime, videofps=videofps, visualize=True)

            print(f"\033[95mOptimization finished in {time.time()-startTime:.2f}s with best parameters: {printX} \tLoss: {optLoss:.6f}\033[0m")

            # Save best parameters
            with open(f"Data/Optimization/act_{filename}.yml", "w") as f:
                yaml.dump({
                    "frequency": float(x[0]),
                    "phaseOffset": float(x[1]),
                }, f, default_flow_style=False)

    elif optType == "fluid":
        print(f"\033[94mOptimizing fluid parameters...\033[0m")
            
        optimizer = BayesianOptimization(
            f=lambda c_b, c_s, c_a, c_k, c_m: -f_fluid([c_b, c_s, c_a, c_k, c_m], filenames=filenames, rotatedMarkersGTs=rotatedMarkersGT, frequencies=frequencies, phaseOffsets=phaseOffsets, simTime=simTime, videofps=videofps),
            pbounds={"c_b": [0.0, 10.0], "c_s": [0.0, 10.0], "c_a": [0.0, 10.0], "c_k": [0.0, 10.0], "c_m": [0.0, 10.0]},
            random_state=seed,
        )
        optimizer.maximize(init_points=opt_init, n_iter=opt_iter)
        x = [optimizer.max['params']['c_b'], optimizer.max['params']['c_s'], optimizer.max['params']['c_a'], optimizer.max['params']['c_k'], optimizer.max['params']['c_m']]

        # Evaluate Best Parameters
        printX = '[' + ', '.join([f'{xi:.4e}' for xi in x]) + ']'
        optLoss = f_fluid(x, filenames=filenames, rotatedMarkersGTs=rotatedMarkersGT, frequencies=frequencies, phaseOffsets=phaseOffsets, simTime=simTime, videofps=videofps, visualize=True)

        print(f"\033[95mOptimization finished in {time.time()-startTime:.2f}s with best parameters: {printX} \tLoss: {optLoss:.6f}\033[0m")

        # Save best parameters
        with open(f"Data/Optimization/fluid.yml", "w") as f:
            yaml.dump({
                "filenames": filenames,
                "frequencies": frequencies,
                "phaseOffsets": phaseOffsets,
                "fluidCoef": [float(xi) for xi in x],
            }, f, default_flow_style=False)

    elif optType == "ebt":
        print(f"\033[94mOptimizing EBT parameters...\033[0m")
            
        optimizer = BayesianOptimization(
            f=lambda c_d,beta: -f_ebt([c_d,beta], rotatedMarkersGTs=rotatedMarkersGT, videofps=videofps),
            pbounds={'c_d': [0.0, 10.0], 'beta': [0.9, 1.1]},
            random_state=seed,
        )
        optimizer.maximize(init_points=opt_init, n_iter=opt_iter)
        x = [optimizer.max['params']['c_d'], optimizer.max['params']['beta']]

        # Evaluate Best Parameters
        printX = '[' + ', '.join([f'{xi:.4e}' for xi in x]) + ']'
        optLoss = f_ebt(x, rotatedMarkersGTs=rotatedMarkersGT, videofps=videofps)

        print(f"\033[95mOptimization finished in {time.time()-startTime:.2f}s with best parameters: {printX} \tLoss: {optLoss:.6f}\033[0m")

        # Save best parameters
        with open(f"Data/Optimization/ebt.yml", "w") as f:
            yaml.dump({
                "filenames": filenames,
                "c_d": float(x[0]),
                "beta": float(x[1]),
            }, f, default_flow_style=False)


    else:
        raise ValueError(f"Invalid optimization type: {optType}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tail marker CSV data, rotate markers to align with head direction, store CSV in same folder as data, and plot results in Outputs folder.")

    parser.add_argument("--filenames", '-f', type=str, nargs='+', default=["f1_75"], help="List of filenames that will load rotated markers that are used for optimization metric computation of the real data. The location should correspond to the path Data/Markers/rotatedMarkers_<filename>.csv.")
    parser.add_argument("--optType", choices=["act", "fluid", "ebt"], default="act", help="Type of optimization to run: 'act' for actuation optimization, 'fluid' for fluid parameter optimization, 'ebt' for EBT parameter optimization.")

    parser.add_argument("--simTime", type=float, default=4.0, help="Simulation time in seconds")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second of the loaded marker video")
    parser.add_argument("--numHead", type=int, default=2, help="Number of head markers")

    parser.add_argument("--opt_init", type=int, default=50, help="Number of initial random evaluations for Bayesian Optimization")
    parser.add_argument("--opt_iter", type=int, default=150, help="Number of iterations for Bayesian Optimization after the initial random evaluations")

    args = parser.parse_args()
    main(args.filenames, optType=args.optType, simTime=args.simTime, videofps=args.fps, opt_init=args.opt_init, opt_iter=args.opt_iter)

