from typing import List, Callable
import numpy as np


class AudioProcessor:
    """Simple processor chaining for float32 signals in [-1, 1]."""

    def __init__(self, processors: List[Callable[[np.ndarray], np.ndarray]] | None = None):
        self._processors: List[Callable[[np.ndarray], np.ndarray]] = processors or []

    def add(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._processors.append(fn)

    def process(self, signal: np.ndarray) -> np.ndarray:
        out = signal
        for fn in self._processors:
            out = fn(out)
        return out
