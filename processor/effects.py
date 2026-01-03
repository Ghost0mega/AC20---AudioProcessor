from typing import Callable
import numpy as np


def hard_clip(signal: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Hard clip `signal` to ±threshold.

    `signal` is float32 in [-1, 1], shape (N, C).
    `threshold` should be in (0, 1].
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in (0, 1].")
    return np.clip(signal, -threshold, threshold)


def gain(signal: np.ndarray, g: float = 1.0) -> np.ndarray:
    """Apply gain and return clipped output in [-1, 1]."""
    return np.clip(signal * g, -1.0, 1.0)


def soft_clip(
    signal: np.ndarray,
    variant: str = "tanh",
    g: float = 1.0,
    T: float = 1.0,
    a: float = 0.0,
    b: float = 0.0,
) -> np.ndarray:
    """Soft clip with selectable variant.

    Variants:
    - 'tanh':    f(x) = T * tanh(g x)
    - 'atan':    f(x) = (2 T / π) * arctan(g x)
    - 'rational':f(x) = T * g x / (1 + |g x|)
    - 'tube':    f(x) = tanh(g x + a) + b x   (a, b in [0,1])

    Parameters
    - signal: float32 in [-1, 1], shape (N, C)
    - g: input drive (gain before nonlinearity)
    - T: output scaling (saturation level for tanh/atan/rational)
    - a, b: tube shape parameters in [0, 1]
    """
    x = signal.astype(np.float32)
    gx = g * x

    if variant == "tanh":
        y = T * np.tanh(gx)
    elif variant == "atan":
        y = (2.0 * T / np.pi) * np.arctan(gx)
    elif variant == "rational":
        y = (T * gx) / (1.0 + np.abs(gx) + 1e-12)
    elif variant == "tube":
        # Tube model uses tanh(gx + a) then adds b*x (no T scaling per spec)
        # Clamp a, b to [0,1] for safety
        a_clamped = float(np.clip(a, 0.0, 1.0))
        b_clamped = float(np.clip(b, 0.0, 1.0))
        y = np.tanh(gx + a_clamped) + (b_clamped * x)
    else:
        raise ValueError("variant must be one of: tanh, atan, rational, tube")

    # Ensure numeric stability and keep within [-max, max] if applicable
    return y.astype(np.float32)


# Typing helper for processors
Processor = Callable[[np.ndarray], np.ndarray]
