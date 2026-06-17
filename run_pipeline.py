# Section 1: Imports
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T

# Section 2: Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Section 3: Module-level Constants (all magic numbers here)
WINDOW_SIZE_MS: int = 500
VIDEO_SAMPLE_FPS: int = 5
MOTION_THRESHOLD: float = 2.5
FRAME_RESIZE: Tuple[int, int] = (224, 224)
ACCEL_THRESHOLD: float = 0.15          # g — dynamic acceleration threshold adjusted to remove gravity DC bias
GYRO_THRESHOLD: float = 10.0           # deg/s — 2.5x margin above static ceiling
JERK_THRESHOLD: float = 0.5
IMU_STRONG_ACTIVE_SCORE: float = 1.2
IMU_STRONG_STATIC_SCORE: float = 0.2
IMU_OVERRIDE_CONFIDENCE_MIN: float = 0.7
OVERRIDE_CONFIDENCE_PENALTY: float = 0.9
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Section 4: Dataclasses
@dataclass
class IMUFeatures:
    accel_rms: float
    gyro_rms: float
    jerk: float
    n_samples: int

@dataclass
class VideoWindow:
    start_ms: int
    end_ms: int
    frames: List[np.ndarray]

@dataclass
class TimelineEntry:
    timestamp_ms: int
    activity: str
    confidence: float

# Section 5: VideoClassifier
class VideoClassifier:
    """Extracts and classifies dog activity from video frames using optical flow."""

    def __init__(self, video_path: str) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video at: {video_path}")
        self.fps: float = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_ms: int = int(self.total_frames / self.fps * 1000)
        # Pretrained MobileNetV2 for feature representation
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(FRAME_RESIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def extract_windows(self, window_size_ms: int = WINDOW_SIZE_MS) -> List[VideoWindow]:
        n_windows: int = int(self.duration_ms // window_size_ms)
        step: int = max(1, int(self.fps // VIDEO_SAMPLE_FPS))
        windows: List[VideoWindow] = []
        try:
            for i in range(n_windows):
                start_ms: int = i * window_size_ms
                end_ms: int = (i + 1) * window_size_ms
                frames: List[np.ndarray] = []
                start_idx: int = int(start_ms / 1000 * self.fps)
                end_idx: int = int(end_ms / 1000 * self.fps)
                
                # Boundary safety check
                start_idx = min(start_idx, self.total_frames)
                end_idx = min(end_idx, self.total_frames)
                
                # Seek to start frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                for idx in range(start_idx, end_idx):
                    ret: bool
                    frame: np.ndarray
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    if (idx - start_idx) % step == 0:
                        frame = cv2.resize(frame, FRAME_RESIZE)
                        frames.append(frame)
                windows.append(VideoWindow(start_ms, end_ms, frames))
        finally:
            self.cap.release()
        return windows

    def classify_window(self, window: VideoWindow) -> Tuple[str, float]:
        if len(window.frames) < 2:
            return ('Static', 0.5)
        motion_values: List[float] = []
        for i in range(len(window.frames) - 1):
            gray1: np.ndarray = cv2.cvtColor(window.frames[i], cv2.COLOR_BGR2GRAY)
            gray2: np.ndarray = cv2.cvtColor(window.frames[i+1], cv2.COLOR_BGR2GRAY)
            flow: np.ndarray = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag: np.ndarray = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_values.append(float(np.mean(mag)))
        
        motion_energy: float = float(np.mean(motion_values))
        video_conf: float = float(np.clip(
            1.0 / (1.0 + np.exp(-(motion_energy / MOTION_THRESHOLD - 1.0))),
            0.0, 1.0
        ))
        label: str = 'Active' if motion_energy > MOTION_THRESHOLD else 'Static'
        return (label, round(video_conf, 4))

# Section 6: IMUProcessor
class IMUProcessor:
    """Loads collar IMU data and extracts per-window activity features."""

    REQUIRED_COLS: List[str] = [
        'timestamp_ms', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'
    ]

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"IMU CSV not found: {csv_path}")
        self.df: pd.DataFrame = pd.read_csv(csv_path)
        missing: List[str] = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        self.df = self.df.sort_values('timestamp_ms').reset_index(drop=True)

    def get_window_features(self, start_ms: int, end_ms: int) -> IMUFeatures:
        w: pd.DataFrame = self.df[(self.df.timestamp_ms >= start_ms) & (self.df.timestamp_ms < end_ms)]
        if len(w) == 0:
            logger.warning(f"No IMU samples in window [{start_ms}, {end_ms})")
            return IMUFeatures(0.0, 0.0, 0.0, 0)
            
        # Gravity Correction: subtract the per-window mean from each axis before computing magnitude
        accel_x_vals: np.ndarray = w.accel_x.values
        accel_y_vals: np.ndarray = w.accel_y.values
        accel_z_vals: np.ndarray = w.accel_z.values
        
        dyn_ax: np.ndarray = accel_x_vals - np.mean(accel_x_vals)
        dyn_ay: np.ndarray = accel_y_vals - np.mean(accel_y_vals)
        dyn_az: np.ndarray = accel_z_vals - np.mean(accel_z_vals)
        dyn_mag: np.ndarray = np.sqrt(dyn_ax**2 + dyn_ay**2 + dyn_az**2)
        accel_rms: float = float(np.sqrt(np.mean(dyn_mag**2)))
        
        # Raw magnitudes for gyro and jerk
        gyro_mag: np.ndarray = np.sqrt(w.gyro_x.values**2 + w.gyro_y.values**2 + w.gyro_z.values**2)
        gyro_rms: float = float(np.sqrt(np.mean(gyro_mag**2)))
        
        accel_mag_raw: np.ndarray = np.sqrt(accel_x_vals**2 + accel_y_vals**2 + accel_z_vals**2)
        if len(w) > 1:
            dt: float = float(np.mean(np.diff(w.timestamp_ms.values))) / 1000.0
            jerk: float = float(np.mean(np.abs(np.diff(accel_mag_raw))) / dt) if dt > 0 else 0.0
        else:
            jerk = 0.0
            
        return IMUFeatures(accel_rms, gyro_rms, jerk, len(w))

    def classify_window(self, features: IMUFeatures) -> Tuple[str, float, float]:
        if features.n_samples == 0:
            return ('Static', 0.5, 0.5)
        # Compute Activity Score S
        S: float = (0.5 * min(features.accel_rms / ACCEL_THRESHOLD, 2.0) +
                    0.5 * min(features.gyro_rms / GYRO_THRESHOLD, 2.0))
        label: str = 'Active' if S > 0.5 else 'Static'
        raw_conf: float = abs(S - 0.5) * 2.0
        confidence: float = round(float(np.clip(raw_conf, 0.0, 1.0)), 4)
        return (label, confidence, S)

# Section 7: FusionEngine
class FusionEngine:
    """Merges video and IMU signals using priority-ordered override rules."""

    def fuse(self, video_label: str, video_conf: float,
             imu_label: str, imu_conf: float,
             imu_active_score: float) -> Tuple[str, float]:
        # RULE 1: Strong IMU Active Override (camera obstruction recovery)
        if (video_label == 'Static'
                and imu_active_score > IMU_STRONG_ACTIVE_SCORE
                and imu_conf > IMU_OVERRIDE_CONFIDENCE_MIN):
            return ('Active', round(imu_conf * OVERRIDE_CONFIDENCE_PENALTY, 4))

        # RULE 2: Strong IMU Static Override (video false positive)
        if (video_label == 'Active'
                and imu_active_score < IMU_STRONG_STATIC_SCORE
                and imu_conf > IMU_OVERRIDE_CONFIDENCE_MIN):
            return ('Static', round(imu_conf * OVERRIDE_CONFIDENCE_PENALTY, 4))

        # RULE 3: Agreement
        if video_label == imu_label:
            return (video_label, round((video_conf + imu_conf) / 2.0, 4))

        # RULE 4: Soft disagreement — trust IMU by default
        return (imu_label, round(imu_conf * 0.7, 4))

# Section 8: TimelineWriter
class TimelineWriter:
    """Validates and serializes timeline entries to JSON at 2 Hz."""

    def build(self, entries: List[TimelineEntry]) -> List[dict]:
        data: List[dict] = []
        for entry in entries:
            assert entry.activity in {'Active', 'Static'}, f"Invalid activity label: {entry.activity}"
            assert 0.0 <= entry.confidence <= 1.0, f"Invalid confidence score: {entry.confidence}"
            data.append({
                "timestamp_ms": entry.timestamp_ms,
                "activity": entry.activity,
                "confidence": entry.confidence
            })
        return data

    def write(self, data: List[dict], output_path: str) -> None:
        parent: str = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Written {len(data)} entries to {output_path}")

# Section 9: main()
def main() -> None:
    parser = argparse.ArgumentParser(description="WigglyWoosh Dog Activity sync pipeline")
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    parser.add_argument('--imu', type=str, required=True, help="Path to IMU CSV log file")
    parser.add_argument('--output', type=str, default="timeline.json", help="Path to write final output JSON")
    parser.add_argument('--verbose', action='store_true', help="Enable debug level verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("Initializing VideoClassifier and IMUProcessor...")
    try:
        video_clf: VideoClassifier = VideoClassifier(args.video)
        imu_proc: IMUProcessor = IMUProcessor(args.imu)
        fusion: FusionEngine = FusionEngine()
        writer: TimelineWriter = TimelineWriter()
        
        logger.info("Decoding video and grouping frames into windows...")
        windows: List[VideoWindow] = video_clf.extract_windows(WINDOW_SIZE_MS)
        logger.info(f"Extracted {len(windows)} video windows.")
        
        # Cap windows to match the minimum of video duration and IMU duration
        imu_max_time: int = int(imu_proc.df.timestamp_ms.max())
        capped_duration: int = min(video_clf.duration_ms, imu_max_time)
        n_capped_windows: int = int(capped_duration // WINDOW_SIZE_MS)
        windows = windows[:n_capped_windows]
        logger.info(f"Capped to {len(windows)} synchronized windows.")
        
        entries: List[TimelineEntry] = []
        for i, window in enumerate(windows):
            v_label, v_conf = video_clf.classify_window(window)
            imu_feat: IMUFeatures = imu_proc.get_window_features(window.start_ms, window.end_ms)
            
            if imu_feat.n_samples == 0:
                # Use video label directly if IMU window has no samples
                f_label, f_conf = v_label, v_conf
                i_label, i_conf, i_score = 'None', 0.0, 0.0
            else:
                i_label, i_conf, i_score = imu_proc.classify_window(imu_feat)
                f_label, f_conf = fusion.fuse(v_label, v_conf, i_label, i_conf, i_score)
            
            logger.debug(
                f"Window {i:02d} [{window.start_ms:5d}-{window.end_ms:5d} ms] | "
                f"Video: {v_label} (conf={v_conf:.4f}) | "
                f"IMU: {i_label} (score={i_score:.4f}, conf={i_conf:.4f}) | "
                f"Fused: {f_label} (conf={f_conf:.4f})"
            )
            entries.append(TimelineEntry(window.start_ms, f_label, f_conf))
            
        timeline_data = writer.build(entries)
        writer.write(timeline_data, args.output)
        print(f"Pipeline complete. {len(entries)} windows. Output: {args.output}")
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Fatal pipeline error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

# Section 10: Entrypoint
if __name__ == '__main__':
    main()
