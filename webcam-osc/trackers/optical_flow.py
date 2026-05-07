import math
from typing import Any

import cv2
import numpy as np

from trackers.base import BaseTracker


class OpticalFlowTracker(BaseTracker):
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        grid_enabled: bool = False,
        grid_size: int = 8,
    ):
        self._params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        self._grid_enabled = grid_enabled
        self._grid_size = grid_size
        self._prev_gray: np.ndarray | None = None

    def process(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
    ) -> list[tuple[str, Any]]:
        pairs: list[tuple[str, Any]] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            pairs.append(("/webcam/flow/magnitude", 0.0))
            pairs.append(("/webcam/flow/direction", 0.0))
            return pairs

        flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, **self._params)
        self._prev_gray = gray

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        mean_mag = float(np.mean(mag))
        # Dominant direction: circular mean weighted by magnitude
        rad = np.deg2rad(ang)
        weights = mag + 1e-6
        mean_sin = float(np.average(np.sin(rad), weights=weights))
        mean_cos = float(np.average(np.cos(rad), weights=weights))
        dominant_dir = float(math.degrees(math.atan2(mean_sin, mean_cos)) % 360)

        pairs.append(("/webcam/flow/magnitude", mean_mag))
        pairs.append(("/webcam/flow/direction", dominant_dir))

        # Optional: per-cell grid
        if self._grid_enabled:
            h, w = mag.shape
            gs = self._grid_size
            cell_h, cell_w = h // gs, w // gs
            for gy in range(gs):
                for gx in range(gs):
                    cell = mag[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
                    pairs.append((f"/webcam/flow/grid/{gy}/{gx}", float(np.mean(cell))))

        # Draw overlay: scale flow arrows on preview frame
        step = 16
        h, w = frame.shape[:2]
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                end = (int(x + fx * 2), int(y + fy * 2))
                cv2.arrowedLine(frame, (x, y), end, (0, 180, 255), 1, tipLength=0.3)

        return pairs
