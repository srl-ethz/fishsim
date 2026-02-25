################################################################################
# Processes tail tracking data from a .csv file, rotates all frames to align
# with the head direction, then saves an animation of the marker motion, 
# plotting the lateral sinusoidal motion of the tail, performing an FFT on this 
# motion, and saving the envelope of the tail motion.
################################################################################

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft, fftfreq

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

mm = 1 / 25.4


def rotate_markers (markers, numHead=2):
    """
    Args:
        markers: (numFrames, numMarkers, 2) array of marker positions for each frame, where the first numHead markers are the head markers.
    """
    head0 = markers[:, 0:1]
    shiftedMarkers = markers - head0
    headDx = (markers[1:, :numHead].mean(axis=1) - markers[:-1, :numHead].mean(axis=1)) # Displacement at each timestep

    # Find rotation matrix such that for every frame the tail markers are lined up to the line defined by the first two markers
    headVec = np.mean(shiftedMarkers[:,1:numHead], axis=1)
    headVec /= np.linalg.norm(headVec, axis=1, keepdims=True)
    rotationMatrices = np.stack([headVec[:,0], headVec[:,1], -headVec[:,1], headVec[:,0]], axis=1).reshape(-1, 2, 2)

    rotatedMarkers = np.matmul(shiftedMarkers, rotationMatrices.transpose(0,2,1))
    rotatedHeadDx = np.matmul(rotationMatrices[1:], headDx.reshape(-1, 2, 1)).reshape(-1, 2)

    # Add back the head0 X-position to the rotated markers, projected displacements along head vector.
    # rotatedMarkers[1:,:,0] += np.cumsum(rotatedHeadDx[:,0:1], axis=0)
    rotatedMarkers[1:] += np.cumsum(rotatedHeadDx.reshape(-1, 1, 2), axis=0)

    return rotatedMarkers

def plot_fft (rotatedMarkers, fps, numHead, filename):
    numFrames = rotatedMarkers.shape[0]
    # Perform FFT
    yFFT = fft(rotatedMarkers-rotatedMarkers.mean(axis=0, keepdims=True), axis=0)
    # Get the frequencies
    xf = fftfreq(numFrames, 1/fps)[:numFrames//2]

    fig, ax = plt.subplots(figsize=(80*mm, 40*mm))
    for i in range(numHead, rotatedMarkers.shape[1]):
        ax.plot(xf, 2.0/numFrames * np.abs(yFFT[:numFrames//2, i, 1]), linewidth=1.5, label=f"Marker {i}")
    ax.grid()
    ax.legend()
    ax.set_xlim(0, 2)
    ax.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (m)")
    fig.savefig(f"Outputs/fft_{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()

    for i in range(numHead, rotatedMarkers.shape[1]):
        print(f"Marker {i} Frequency: {xf[np.argmax(np.abs(yFFT[:numFrames//2, i, 1]))]:.4f}Hz (resolution {xf[1]-xf[0]:.4f}Hz) at Amplitude {np.max(2.0/numFrames * np.abs(yFFT[:numFrames//2, i, 1])):.4f}m")

def plot_lateral_motion (rotatedMarkers, fps, numHead, filename):
    numFrames = rotatedMarkers.shape[0]
    times = np.linspace(0, numFrames / fps, numFrames)
    fig, ax = plt.subplots(figsize=(80*mm, 50*mm))
    for i in range(numHead, rotatedMarkers.shape[1]):
        ax.plot(times, rotatedMarkers[:,i,1], linewidth=1.5, label=f"Marker {i}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol=3)
    ax.grid()
    ax.set_xlim(times[0], times[-1])
    ax.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y Position (m)")
    fig.savefig(f"Outputs/position_{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()


def animate_markers (markers, numHead, fps, filename):
    t = 0   
    fig, ax = plt.subplots(figsize=(120*mm, 120*mm))
    if numHead == 2:
        headLine = ax.plot(markers[t, :2, 0], markers[t, :2, 1], c='r', marker='o', markersize=10, zorder=10)
        tailLine = ax.plot(markers[t, 1:, 0], markers[t, 1:, 1], c='k', marker='o', markersize=5)
    elif numHead == 5:
        headLineX = ax.plot(
            np.concatenate([markers[t, :2, 0], markers[t, 4:5, 0]], axis=0), 
            np.concatenate([markers[t, :2, 1], markers[t, 4:5, 1]], axis=0), 
            c='r', marker='o', markersize=10, zorder=10
        )
        headLineY = ax.plot(
            markers[t, 2:4, 0],
            markers[t, 2:4, 1], 
            c='r', marker='o', markersize=10, zorder=10
        )
        tailLine = ax.plot(markers[t, numHead-1:, 0], markers[t, numHead-1:, 1], c='k', marker='o', markersize=5)
    else:
        raise ValueError("numHead must be 2 or 5.")
    ax.grid()
    ax.set_aspect('equal')
    ax.ticklabel_format(style="sci", axis='both', scilimits=(0,0))
    ax.set_xlim(markers[:,:,0].min()-0.03, markers[:,:,0].max()+0.03)
    ax.set_ylim(markers[:,:,1].min()-0.03, markers[:,:,1].max()+0.03)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")


    def update (t):
        if numHead == 2:
            headLine[0].set_xdata(markers[t, :2, 0])
            headLine[0].set_ydata(markers[t, :2, 1])
            tailLine[0].set_xdata(markers[t, 1:, 0])
            tailLine[0].set_ydata(markers[t, 1:, 1])

            return headLine + tailLine
        
        elif numHead == 5:
            headLineX[0].set_xdata(np.concatenate([markers[t, :2, 0], markers[t, 4:5, 0]], axis=0))
            headLineX[0].set_ydata(np.concatenate([markers[t, :2, 1], markers[t, 4:5, 1]], axis=0))
            headLineY[0].set_xdata(markers[t, 2:4, 0])
            headLineY[0].set_ydata(markers[t, 2:4, 1])
            tailLine[0].set_xdata(markers[t, numHead-1:, 0])
            tailLine[0].set_ydata(markers[t, numHead-1:, 1])

            return headLineX + headLineY + tailLine
    
    ani = animation.FuncAnimation(fig, update, frames=markers.shape[0], interval=1000/fps, blit=True)
    ani.save(f"Outputs/{filename}.mp4", fps=fps, dpi=300, extra_args=['-vcodec', 'libx264'])
    plt.close()


def main (filepath, fps=60, numHead=2, generateVideos=False):
    ### Load Data
    filename = os.path.basename(filepath).split(".")[0]
    filefolder = os.path.dirname(filepath)
    markerData = np.loadtxt(filepath, delimiter=",") # Shape (numFrames, numMarkers*2)
    markers = markerData.reshape(-1, markerData.shape[1]//2, 2)

    # Use pre-defined headsize to convert pixels to meters.
    headSizePx = np.linalg.norm((markers[:,1] - markers[:,0]), axis=1).mean()
    headSizeM = 0.142 # Measured from hardware
    markers *= headSizeM / headSizePx

    ### ROTATED MARKERS: Define all markers w.r.t. the head0
    rotatedMarkers = rotate_markers(markers, numHead=numHead)
    np.savetxt(f"{filefolder}/rotated{filename[0].upper() + filename[1:]}.csv", rotatedMarkers.reshape(-1, 2), delimiter=",")

    ### Plot Lateral Motion
    plot_lateral_motion(rotatedMarkers, fps, numHead, filename=filename)

    ### Plot the FFT
    plot_fft(rotatedMarkers, fps, numHead, filename=filename)

    print("\033[94mSaved plots.\033[0m")

    if generateVideos:
        ### Save Unrotated Animation
        animate_markers(markers, numHead, fps, filename=filename)
        print("\033[94mSaved unrotated animation.\033[0m")


        ### Save Rotated Animation
        animate_markers(rotatedMarkers, numHead, fps, filename=f"rotated{filename[0].upper() + filename[1:]}")
        print("\033[94mSaved rotated animation.\033[0m")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tail marker CSV data, rotate markers to align with head direction, store CSV in same folder as data, and plot results in Outputs folder.")
    parser.add_argument("--filepath", '-f', type=str, default="Data/Markers/markers_f1_75.csv", help="Path to the markers file")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second of the video")
    parser.add_argument("--numHead", type=int, default=2, help="Number of head markers")
    parser.add_argument("--generateVideos", action="store_true", help="Generate result videos")

    args = parser.parse_args()
    main(filepath=args.filepath, fps=args.fps, numHead=args.numHead, generateVideos=args.generateVideos)

