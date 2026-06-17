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
