# Data Accompanying Paper

## Videos

Users can also download the original videos and track markers from scratch: [Google Drive](https://drive.google.com/drive/folders/1bM4Kjv0C_C5ZQyzxJUkwOnjJE2mTtik9). `f1_25.mp4` implies the fish swimming at a constant motor angular velocity of 1.25Hz. Download the videos and place them in `Data/Videos/`.


## Markers

Extracted markers are already provided in the `Data/Markers/` folder, and can directly be used for system identification tasks after processing them into rotated markers by running the script:

```bash
python Data/process_tailTracking.py -f Data/Markers/markers_f1_75.csv
```

By default, the marker CSV is storing the marker locations as an array of shape (numFrames, 2*numMarkers), where the X and Y pixel positions of each marker are stacked in the second dimension, so [X0, Y0, X1, Y1, X2, Y2, ...]. The above script will rotate the markers into the head frame, and optionally store animations on the tracked markers. Rotated markers are then stored in `Data/Markers/`, and the results in `Outputs/`. 