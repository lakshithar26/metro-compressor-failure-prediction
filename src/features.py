"""Feature engineering utilities for metro compressor data."""

import numpy as np
from collections import deque
from typing import Dict, List

ANALOGUE_SIGNALS = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Oil_temperature', 'Motor_current']
WINDOW_SIZE = 60  # 60 seconds rolling window


class FeatureBuffer:
    """Maintains a rolling buffer for computing window features."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.buffers: Dict[str, deque] = {
            signal: deque(maxlen=window_size) for signal in ANALOGUE_SIGNALS
        }

    def update(self, reading: Dict[str, float]) -> None:
        """Add a new reading to the buffer."""
        for signal in ANALOGUE_SIGNALS:
            if signal in reading:
                self.buffers[signal].append(reading[signal])

    def is_ready(self) -> bool:
        """Check if buffer has enough data for feature computation."""
        return all(len(buf) >= self.window_size for buf in self.buffers.values())

    def compute_features(self) -> Dict[str, float]:
        """Compute rolling window features from buffer."""
        features = {}

        for signal in ANALOGUE_SIGNALS:
            data = np.array(self.buffers[signal])

            # Raw value (most recent)
            features[signal] = data[-1]

            # Rolling statistics
            features[f'{signal}_mean'] = np.mean(data)
            features[f'{signal}_std'] = np.std(data)
            features[f'{signal}_min'] = np.min(data)
            features[f'{signal}_max'] = np.max(data)

        return features

    def get_feature_vector(self, feature_cols: List[str]) -> np.ndarray:
        """Get feature vector in specified column order."""
        features = self.compute_features()
        return np.array([features[col] for col in feature_cols])

    def reset(self) -> None:
        """Clear all buffers."""
        for buf in self.buffers.values():
            buf.clear()
