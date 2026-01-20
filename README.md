# Gait-cycle Based Approach for FoG Prediction

Overview
-------
Freezing of Gait (FoG) is a disabling symptom of Parkinson’s disease that increases fall risk. Most data-driven FoG prediction methods use fixed-length sliding windows on wearable-sensor time series. While effective for model training, sliding windows frequently cut through gait cycles and capture only partial stance–swing dynamics, which can reduce model efficiency and prediction performance.

This repository demonstrates a physiologically informed segmentation strategy that aligns segments with complete gait cycles. By starting each segment at a consistent gait event, the method produces more compact, interpretable inputs that better preserve phase dynamics and can improve prediction while reducing computational cost. The demo notebook is `main.ipynb`.

Dataset
-------
The demo uses a subset of the publicly available Beijing Xuanwu Hospital dataset [1]. To keep the repository small, only selected patients are included here. Replace the `Beijing Dataset/` folder with the full dataset locally to reproduce complete results.

Method
------
A gait cycle is the interval between two successive heel strikes of the same foot and contains stance and swing phases [2]. Rather than using fixed-length windows, the proposed pipeline:

- low-pass filters IMU signals (cutoff 2.5 Hz) to remove high-frequency noise,
- detects gait-cycle onset events from the vertical gyroscope (`Gyro_z`) using a positive zero-crossing preceded by a local minimum below a threshold,
- extracts segments that contain a full gait cycle starting at each detected onset, and
- labels segments for FoG prediction (including short pre-FOG intervals).

Implementation notes
--------------------
- The core dataset loader and processing utilities are in the `Utils/` folder.
- The demo flow in `main.ipynb` performs: loading, downsampling to 50 Hz, denoising, pre-FOG label generation, feature normalization, segmentation, and plotting.
- Segmentation API examples from the notebook:

	- `database.construct_segs_and_labels()` — returns gait-cycle based segments and labels.
	- `database.construct_sliding_windows_and_labels()` — sliding-window baseline for comparison.

Usage
-----
Open `main.ipynb` and run the cells. For a quick test the notebook limits processing to a single patient by default.

Citations
---------
[1] Zhang, W.; Yang, Z.; Li, H.; et al. Multimodal Data for the Detection of Freezing of Gait in Parkinson’s Disease. Sci Data 9, 606 (2022). https://doi.org/10.1038/s41597-022-01713-8

[2] Pappas, I.; Popovic, M.; Keller, T.; Dietz, V.; Morari, M. A Reliable Gait Phase Detection System. IEEE Trans. Neural Syst. Rehabil. Eng., 2001. https://doi.org/10.1109/7333.928571 