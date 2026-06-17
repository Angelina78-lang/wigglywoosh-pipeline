# WigglyWoosh Dog Activity Detection Pipeline - Solution Walkthrough

This document outlines the engineering rationale, mathematical models, and sensor fusion overrides implemented in the dog activity detection pipeline.

---

## 1. Vision Model & Optical Flow Architecture

To detect activity from video frames under strict CPU-only execution constraints, a hybrid vision structure was adopted:
1. **Primary Signal (Dense Optical Flow):** We use Gunnar Farneback’s algorithm (`cv2.calcOpticalFlowFarneback`) to compute a dense vector field of motion between consecutive frames sampled at 5 Hz.
   - For a flow field $\mathbf{u}(x,y) = (u_x, u_y)$, the motion magnitude at pixel $(x,y)$ is:
     $$mag(x,y) = \sqrt{u_x(x,y)^2 + u_y(x,y)^2}$$
   - The overall motion energy of the frame pair is the spatial mean of the magnitude, and the window's `motion_energy` is the temporal mean across all consecutive frame pairs in the 500 ms window.
   - This motion energy is converted into a sigmoid confidence score:
     $$video\_confidence = \sigma\left(\frac{motion\_energy}{M} - 1.0\right) = \frac{1}{1 + e^{-\left(\frac{motion\_energy}{M} - 1.0\right)}}$$
     where $M = 2.5$ is the `MOTION_THRESHOLD`.
2. **Auxiliary Model (MobileNetV2):** Preloaded with ImageNet weights, torchvision's MobileNetV2 serves as a lightweight, low-footprint feature extractor for auxiliary semantic analysis on frame sequences without lagging on CPU.

---

## 2. IMU Sensor Feature Extraction

Collar-worn IMU logs contain a significant gravity component. In `collar_imu.csv`, the collar hangs down, presenting a constant DC offset of $\approx 1g$ on the Y-axis. Under static conditions, raw acceleration magnitude is $\approx 0.93g$, which falsely exceeds static activity thresholds. 

To isolate true dynamic motion, we subtract the per-window mean from each acceleration axis before calculating the magnitude:

### 2.1 Mathematical Formulations

For a window $W$ containing samples $i \in \{1, \dots, N\}$:

- **Dynamic Acceleration ($a_{\text{dynamic}}$):**
  $$\mu_x = \frac{1}{N}\sum_{i=1}^N a_{x,i}, \quad \mu_y = \frac{1}{N}\sum_{i=1}^N a_{y,i}, \quad \mu_z = \frac{1}{N}\sum_{i=1}^N a_{z,i}$$
  $$a_{\text{dyn},i} = \sqrt{(a_{x,i} - \mu_x)^2 + (a_{y,i} - \mu_y)^2 + (a_{z,i} - \mu_z)^2}$$
  $$accel\_rms = \sqrt{\frac{1}{N}\sum_{i=1}^N a_{\text{dyn},i}^2}$$
  *Removing the window mean isolates high-frequency body motion from gravity.*

- **Angular Velocity RMS ($g_{\text{rms}}$):**
  $$g_{\text{mag},i} = \sqrt{\omega_{x,i}^2 + \omega_{y,i}^2 + \omega_{z,i}^2}$$
  $$gyro\_rms = \sqrt{\frac{1}{N}\sum_{i=1}^N g_{\text{mag},i}^2}$$

- **Collar Jerk ($J$):**
  $$J = \frac{1}{N-1}\sum_{i=1}^{N-1} \frac{|a_{\text{mag},i+1} - a_{\text{mag},i}|}{\Delta t}$$
  where $a_{\text{mag}}$ represents raw acceleration magnitude and $\Delta t$ is the mean sample interval.

---

## 3. Fusion Engine & Override Calibration

We define the normalized IMU Activity Score $S$ as:
$$S = 0.5 \cdot \text{clip}\left(\frac{accel\_rms}{\theta_{\text{accel}}}, 0, 2\right) + 0.5 \cdot \text{clip}\left(\frac{gyro\_rms}{\theta_{\text{gyro}}}, 0, 2\right)$$
where $\theta_{\text{accel}} = 0.15g$ (dynamic acceleration threshold) and $\theta_{\text{gyro}} = 10.0^\circ/s$.
- **IMU Activity Label:** `Active` if $S > 0.5$ else `Static`
- **IMU Confidence:** $C_{\text{imu}} = \text{clip}(|S - 0.5| \times 2.0, 0, 1)$

### 3.1 Fusion Override Priority Rules

The Fusion Engine evaluates four hierarchical rules:

| Priority | Rule | Condition | Fused Label | Fused Confidence | Engineering Logic |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | IMU Active Override | $V_{\text{label}} = \text{Static}$, $S > 1.2$, $C_{\text{imu}} > 0.7$ | `Active` | $C_{\text{imu}} \times 0.9$ | Recovers from camera obstructions or blind spots. |
| **2** | IMU Static Override | $V_{\text{label}} = \text{Active}$, $S < 0.2$, $C_{\text{imu}} > 0.7$ | `Static` | $C_{\text{imu}} \times 0.9$ | Rejects video false-positives (watermarks, background movement). |
| **3** | Agreement | $V_{\text{label}} = I_{\text{label}}$ | $V_{\text{label}}$ | $\frac{C_{\text{video}} + C_{\text{imu}}}{2}$ | Reinforces decision when both sensors agree. |
| **4** | Soft Disagreement | $V_{\text{label}} \neq I_{\text{label}}$ | $I_{\text{label}}$ | $C_{\text{imu}} \times 0.7$ | Defaults to collar sensors as the ground-truth physical metric. |

> [!NOTE]
> **Active Confidence Uniformity Note:** Active windows show a uniform confidence of `0.9` because **RULE 1 (IMU Active Override)** fires consistently across all 12 Active windows. The video's optical flow signal is weak/Static for this footage, but since the IMU exhibits a high dynamic acceleration and gyro response ($S > 1.2$, $C_{\text{imu}} = 1.0$), the IMU override triggers, resulting in a fused confidence of $C_{\text{imu}} \times 0.9 = 0.9000$.
