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
