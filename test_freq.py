################################################################################
# Run the evaluation of comparing simulated with real markers, as well as an 
# EBT baseline, across a range of frequencies for constant cruising speed.
################################################################################

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from Geometry.auto_tendonFish import SYSTEMPARAMETERS
from _simulate import sim_fish as sim
from _simulate import SineSignal
from Data.process_tailTracking import rotate_markers
from opt_sysid import compute_ebt_cruising_speed

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1 / 25.4


SIMTIME = 5
VIDEOFPS = 60
NUMHEAD = 2
NUMMARKERS = 9

### Load Optimized Fluid Coefficients
systemParameters = SYSTEMPARAMETERS.copy()
if not os.path.exists(f"Data/Optimization/fluid.yml"):
    raise FileNotFoundError(f"Fluid optimization file not found. Please run opt_sysid.py for fluid first.")
with open(f"Data/Optimization/fluid.yml", "r") as f:
    data = yaml.safe_load(f)
    systemParameters["fluidCoef"] = data["fluidCoef"]

### Load Optimized EBT Parameters
if not os.path.exists(f"Data/Optimization/ebt.yml"):
    raise FileNotFoundError(f"EBT optimization file not found. Please run opt_sysid.py for EBT first.")
with open(f"Data/Optimization/ebt.yml", "r") as f:
    data = yaml.safe_load(f)
    c_d = data["c_d"]
    beta = data["beta"]

### Files to be evaluated
filenames = [
    "f0_50",
    "f0_75",
    "f1_00",
    "f1_25",
    "f1_50",
    "f1_75",
    "f2_00",
    "f2_25"
]
frequencies = {}
phaseOffsets = {}
for filename in filenames:
    # Assume we have optimized the actuation already
    if not os.path.exists(f"Data/Optimization/act_{filename}.yml"):
        raise FileNotFoundError(f"Actuation optimization file not found for {filename}. Please run opt_act.py first.")
    with open(f"Data/Optimization/act_{filename}.yml", "r") as f:
        data = yaml.safe_load(f)
        frequencies[filename] = data["frequency"]
        phaseOffsets[filename] = data["phaseOffset"]

# freqsSim = np.linspace(0.25, 2.5, 19)
freqsSim = frequencies.values()

log = {
    "freqReal": frequencies.values(),
    "freqSim": freqsSim,
    "posReal": [],
    "posSim": [],
    "velReal": [],
    "velSim": [],
    "ebtU": [],
}
### Process Real Data and EBT results
for file in filenames:
    filename = f"Data/Markers/rotatedMarkers_{file}.csv"
    print(f"Processing {filename}")

    markerData = np.loadtxt(filename, delimiter=",").reshape(-1, NUMMARKERS, 2)
    headDispl = markerData[:,:2,0].mean(1) - markerData[0:1,:2,0].mean(1)
    log["posReal"].append(headDispl)

    # Find the linear velocity approximation
    timeAxis = np.arange(len(headDispl)) / VIDEOFPS
    coef = np.polyfit(timeAxis, -headDispl, 1) # Swims in negative x direction
    log["velReal"].append(coef[0])

    # Compute EBT cruising speed
    ebtU = compute_ebt_cruising_speed(markerData, c_d, beta, 1 / VIDEOFPS)
    log["ebtU"].append(ebtU)

### Simulate Data
for freq in freqsSim:
    res = sim(systemParameters, SineSignal(frequency=freq), SIMTIME, VIDEOFPS, warmuptime=0.5)

    markers = res["markers"].reshape(-1, 11, 2)
    if NUMHEAD == 2:
        markers = np.concatenate([markers[:,:2], markers[:,4:]], axis=1)
    rotatedMarkers = rotate_markers(markers, numHead=NUMHEAD)

    rotatedMarkers -= rotatedMarkers[0:1,:NUMHEAD].mean(axis=1, keepdims=True)
    headPos = rotatedMarkers[:,:NUMHEAD].mean(axis=1)[:,0]
    log["posSim"].append(headPos)

    # Find the linear velocity approximation
    timeAxis = np.arange(0, markers.shape[0] / VIDEOFPS, 1/VIDEOFPS)
    coef = np.polyfit(timeAxis, -headPos, 1) # Swims in negative x direction
    log["velSim"].append(coef[0])

### Compute Errors
velSim = []
for i, f in enumerate(freqsSim):
    if f in frequencies.values():
        velSim.append(log["velSim"][i])
assert len(velSim) == len(log["velReal"]), "Mismatch in number of simulated velocities and real velocities"

error = abs(np.array(velSim) - np.array(log["velReal"]))
relError = 100*(abs(np.array(velSim) - np.array(log["velReal"])) / np.array(log["velReal"]).mean())
# relError = 100*(abs(np.array(velSim) - np.array(log["velReal"])) / np.array(log["velReal"]))
print(f"Mean absolute error between simulated and real velocities: {error.mean():.6f}m/s with relative error {relError.mean():.2f}%")

# EBT
ebtError = abs(log["ebtU"] - np.array(log["velReal"]))
ebtRelError = 100*(abs(log["ebtU"] - np.array(log["velReal"])) / np.array(log["velReal"]).mean())
print(f"Mean absolute error between EBT and real velocities: {ebtError.mean():.6f}m/s with relative error {ebtRelError.mean():.2f}%")


### Plotting
fig, ax = plt.subplots(figsize=(60*mm, 40*mm))
ax.plot(log["freqSim"], log["velSim"], markersize=4, marker='o', linestyle='--', color="tab:blue", label='Simulated')
ax.scatter(log["freqReal"], log["velReal"], marker='x', color="tab:orange", label='Measured', zorder=5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Velocity (m/s)")
ax.legend(loc="upper center", ncol=2, fancybox=True, bbox_to_anchor=(0.5, -0.35))
ax.grid()
fig.savefig("Outputs/sim2real_fishvel.png", dpi=300, bbox_inches="tight")
plt.close(fig)


### Paper Figure
fig, ax = plt.subplots(figsize=(88*mm, 30*mm))
ax.plot(log["freqSim"], log["velSim"], markersize=4, marker='o', linestyle='-', color="tab:blue", label='Sim')
ax.plot(log["freqSim"], log["ebtU"], markersize=4, marker='o', linestyle='--', color="tab:green", label='EBT')
ax.scatter(log["freqReal"], log["velReal"], marker='x', color="tab:orange", label='Real', zorder=5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Velocity (m/s)")
ax.legend(loc="upper center", ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.375))
ax.grid()
fig.savefig("Outputs/sim2real_freqs.png", dpi=300, bbox_inches="tight")
fig.savefig("Outputs/sim2real_freqs.pdf", bbox_inches="tight")
plt.close(fig)

