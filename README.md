# WigglyWoosh Dog Activity Detection Pipeline

This repository contains a high-performance Python script for syncing and fusing dog video classification results with 6-axis collar IMU signals.

---

## 1. Project Structure

```
wigglywoosh-pipeline/
├── run_pipeline.py     # Self-contained executable script
├── solution.md         # Mathematical explanations and threshold calibration
├── requirements.txt    # Pinned python package dependencies
├── README.md           # Project guide and instructions
└── .gitignore          # Git exclusion rules
```

---

## 2. Setup Instructions

To get started, instantiate a virtual environment and install the pinned dependencies:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install packages
pip install -r requirements.txt
```

---

## 3. Running the Pipeline

The pipeline accepts video (`.mp4`) and IMU sensor (`.csv`) file paths via CLI arguments and outputs a validated timeline JSON (`timeline.json`).

### 3.1 CLI Options

| Argument | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `--video` | **Yes** | — | Path to the dog MP4 video. |
| `--imu` | **Yes** | — | Path to the 6-axis IMU CSV log. |
| `--output` | No | `timeline.json` | Destination path for the timeline output JSON. |
| `--verbose` | No | `False` | Enables debug level verbose logging. |

### 3.2 Usage Examples

```bash
# Basic run with default output (timeline.json)
python run_pipeline.py --video Dog_Video.mp4 --imu collar_imu.csv

# Run with custom output path
python run_pipeline.py --video Dog_Video.mp4 --imu collar_imu.csv --output outputs/timeline.json

# Run with verbose debugging
python run_pipeline.py --video Dog_Video.mp4 --imu collar_imu.csv --verbose
```

---

## 4. Input Specifications

### IMU Sensor CSV Log
Must contain the following headers:
- `timestamp_ms` (integer time, e.g., sampled at ~100 Hz)
- `accel_x`, `accel_y`, `accel_z` (in $g$)
- `gyro_x`, `gyro_y`, `gyro_z` (in $^\circ/s$)

### Video File
Standard MP4 video file. The pipeline samples the video at 5 FPS within each 500 ms window to perform dense Gunnar Farneback optical flow estimation on the CPU.

---

## 5. Pipeline Knowledge Base

### 5.1 Observed IMU Regimes
Data analysis of `collar_imu.csv` shows two bimodal regimes:
- **Active Phase (0 – 5,700 ms):** `accel_rms` in range $1.5 - 2.8g$, `gyro_rms` in range $30 - 120^\circ/s$.
- **Static Phase (8,200 – 11,349 ms):** `accel_rms` in range $0.08 - 0.22g$, `gyro_rms` in range $0 - 4^\circ/s$.

### 5.2 Gravity Correction (DC Offset Removal)
The collar hangs on the dog's neck, presenting a constant $\approx 1g$ gravity DC offset on the Y-axis. Raw acceleration magnitude calculation is biased and fails to identify the Static phase (yielding $\approx 0.93g$ even when still). 
We resolve this by calculating the **dynamic acceleration** component (subtracting the per-window mean from each axis before computing magnitude):
$$\text{dyn\_ax} = a_x - \text{mean}(a_x)$$
$$\text{dyn\_ay} = a_y - \text{mean}(a_y)$$
$$\text{dyn\_az} = a_z - \text{mean}(a_z)$$
$$\text{accel\_rms} = \sqrt{\text{mean}(\text{dyn\_ax}^2 + \text{dyn\_ay}^2 + \text{dyn\_az}^2)}$$

### 5.3 Sensor Fusion Override Logic
The pipeline merges video optical flow predictions and IMU classification scores using four priority-ordered override rules:
1. **Strong IMU Active Override (Priority 1):** If the video predicts `Static` but the IMU indicates strong physical movement ($S > 1.2$, $C_{\text{imu}} > 0.7$), the pipeline overrides the state to `Active` (recovers from camera obstructions).
2. **Strong IMU Static Override (Priority 2):** If the video predicts `Active` but the IMU is perfectly still ($S < 0.2$, $C_{\text{imu}} > 0.7$), the pipeline overrides to `Static` (rejects video false-positives like watermarks).
3. **Agreement (Priority 3):** If both sensors agree, the fused confidence is the average of their confidence scores.
4. **Soft Disagreement (Priority 4):** If the sensors disagree but thresholds aren't crossed, the pipeline defaults to the IMU classification (with a confidence penalty) as the physical ground truth.
