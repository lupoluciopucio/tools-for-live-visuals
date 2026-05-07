from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseTracker(ABC):
    """
    All trackers implement this interface.

    process() returns a list of (osc_address, value) pairs to be sent.
    It also draws overlays onto `frame` in-place for the live preview.
    """

    @abstractmethod
    def process(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
    ) -> list[tuple[str, Any]]:
        ...

    def release(self):
        """Override to clean up resources (e.g. mediapipe objects)."""
        pass
