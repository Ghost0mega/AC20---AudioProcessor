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


def comb_filter(
    signal: np.ndarray,
    sample_rate: int,
    delay_ms: float = 10.0,
    feedback: float = 0.5,
    feedforward: float = 0.0,
    mix: float = 1.0,
) -> np.ndarray:
    """Comb filter with delay input.

    - delay_ms: delay in milliseconds (converted to samples)
    - feedback: feedback coefficient (y[n] += feedback * y[n-D])
    - feedforward: feedforward coefficient (y[n] += feedforward * x[n-D])
    - mix: wet mix amount in [0,1]; output = (1-mix)*x + mix*y

    Notes
    - Supports mono/stereo with shape (N, C).
    - For stability, |feedback| should be < 1.
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, c = x.shape
    delay_samples = int(max(1, round(delay_ms * sample_rate / 1000.0)))
    y = np.zeros_like(x, dtype=np.float32)

    for ch in range(c):
        # Process each channel with simple sample loop to handle feedback
        for i in range(n):
            acc = x[i, ch]
            j = i - delay_samples
            if j >= 0:
                acc += feedforward * x[j, ch]
                acc += feedback * y[j, ch]
            y[i, ch] = acc

    out = (1.0 - mix) * x + mix * y
    return out.astype(np.float32)


def phaser(
    signal: np.ndarray,
    sample_rate: int,
    rate_hz: float = 0.5,
    depth: float = 0.7,
    stages: int = 4,
    feedback: float = 0.0,
    mix: float = 0.5,
    center: float = 0.0,
) -> np.ndarray:
    """Simple multi-stage phaser using cascaded first-order all-pass filters.

    Parameters
    - rate_hz: LFO rate in Hz
    - depth: LFO depth controlling all-pass coefficient modulation (0..1)
    - stages: number of all-pass stages (2..12 recommended)
    - feedback: feedback from wet output back to input (-0.99..0.99)
    - mix: wet mix amount in [0,1]
    - center: DC offset for coefficient (shifts the sweep center)

    Implementation notes
    - Per-sample coefficient a[n] = clamp(center + depth * sin(2π f_lfo t), -0.99, 0.99)
    - Each stage uses: y = -a*x + x_prev + a*y_prev
    - Channels processed independently; maintains simple per-stage state.
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, c = x.shape
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    lfo = depth * np.sin(2.0 * np.pi * rate_hz * t) + center
    a = np.clip(lfo, -0.99, 0.99).astype(np.float32)

    stages = max(1, int(stages))
    fb = float(np.clip(feedback, -0.99, 0.99))
    mix = float(np.clip(mix, 0.0, 1.0))

    y = np.zeros_like(x, dtype=np.float32)

    # Process per channel
    for ch in range(c):
        # Initialize per-stage states
        x_prev = np.zeros(stages, dtype=np.float32)
        y_prev = np.zeros(stages, dtype=np.float32)
        wet_prev = 0.0
        for i in range(n):
            xi = x[i, ch] + fb * wet_prev
            xin = xi
            a_i = a[i]
            # Cascade through stages
            for s in range(stages):
                y_s = -a_i * xin + x_prev[s] + a_i * y_prev[s]
                x_prev[s] = xin
                y_prev[s] = y_s
                xin = y_s
            wet = xin
            y[i, ch] = (1.0 - mix) * x[i, ch] + mix * wet
            wet_prev = wet

    return y.astype(np.float32)


# Typing helper for processors
Processor = Callable[[np.ndarray], np.ndarray]
