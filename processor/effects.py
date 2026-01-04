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


def delay(
    signal: np.ndarray,
    sample_rate: int,
    delay_ms: float = 300.0,
    feedback: float = 0.5,
    mix: float = 0.5,
) -> np.ndarray:
    """Simple delay/echo effect with feedback.

    - delay_ms: delay time in milliseconds
    - feedback: feedback coefficient (|feedback| < 1 for stability)
    - mix: wet mix in [0,1]; output = (1-mix)*x + mix*y

    Implementation:
    y[n] = x[n] + feedback * y[n-D]
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, c = x.shape
    D = int(max(1, round(delay_ms * sample_rate / 1000.0)))
    fb = float(np.clip(feedback, -0.99, 0.99))
    mix = float(np.clip(mix, 0.0, 1.0))

    y = np.zeros_like(x, dtype=np.float32)

    for ch in range(c):
        for i in range(n):
            acc = x[i, ch]
            j = i - D
            if j >= 0:
                acc += fb * y[j, ch]
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


def flanger(
    signal: np.ndarray,
    sample_rate: int,
    base_delay_ms: float = 2.0,
    depth_ms: float = 1.5,
    rate_hz: float = 0.25,
    feedback: float = 0.3,
    mix: float = 0.5,
) -> np.ndarray:
    """Classic flanger using variable fractional delay with sinusoidal LFO.

    Parameters
    - base_delay_ms: central delay in ms (typ. 0.5..5 ms)
    - depth_ms: modulation depth in ms (typ. 0.1..3 ms)
    - rate_hz: LFO frequency in Hz
    - feedback: feedback coefficient (-0.99..0.99)
    - mix: wet mix (0..1)

    Implementation
    - Variable delay d[n] = base + depth * sin(2π f_lfo t)
    - Fractional delay via linear interpolation from past samples.
    - Mono/stereo supported; channels processed independently.
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, c = x.shape
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    base_s = max(0.0, base_delay_ms * sample_rate / 1000.0)
    depth_s = max(0.0, depth_ms * sample_rate / 1000.0)
    fb = float(np.clip(feedback, -0.99, 0.99))
    mix = float(np.clip(mix, 0.0, 1.0))

    # LFO delay in samples per time index
    d = base_s + depth_s * np.sin(2.0 * np.pi * rate_hz * t)

    y = np.zeros_like(x, dtype=np.float32)

    for ch in range(c):
        for i in range(n):
            # Compute fractional delay index
            di = d[i]
            k = int(np.floor(di))
            frac = di - k
            idx0 = i - k
            idx1 = i - (k + 1)

            # Helper to sample with linear interpolation from array arr
            def sample(arr):
                s0 = arr[idx0, ch] if idx0 >= 0 else 0.0
                s1 = arr[idx1, ch] if idx1 >= 0 else 0.0
                return (1.0 - frac) * s0 + frac * s1

            delayed_x = sample(x)
            delayed_y = sample(y)

            wet = delayed_x + fb * delayed_y
            y[i, ch] = (1.0 - mix) * x[i, ch] + mix * wet

    return y.astype(np.float32)


def moving_average_lowpass(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Moving-average low-pass filter.

    window_samples: number of samples in the averaging window (>= 1).
    Applies per-channel with 'same' alignment.
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, c = x.shape
    w = max(1, int(window_samples))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    y = np.zeros_like(x, dtype=np.float32)
    for ch in range(c):
        y[:, ch] = np.convolve(x[:, ch], kernel, mode='same')
    return y.astype(np.float32)


def moving_average_filter(
    signal: np.ndarray,
    sample_rate: int,
    mode: str = "lp",
    cutoff_hz: float | None = None,
    low_cutoff_hz: float | None = None,
    high_cutoff_hz: float | None = None,
) -> np.ndarray:
    """Composite filters built from moving-average low-pass.

    Modes:
    - 'lp': low-pass using moving average at cutoff_hz
    - 'hp': high-pass as x - LP(cutoff_hz)
    - 'bp': band-pass as LP(high_cutoff) - LP(low_cutoff)
    - 'notch': band-stop as x - BP(low_cutoff, high_cutoff)

    Notes:
    - Moving-average LP has sinc response; cutoff mapping is approximate.
      We map window N ≈ fs / cutoff_hz for clear, visible changes.
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    def win_for_cutoff(cut: float) -> int:
        # Approximate mapping: first null at fs/N ≈ cutoff
        return max(1, int(round(sample_rate / max(cut, 1e-3))))

    if mode == "lp":
        if cutoff_hz is None:
            raise ValueError("lp requires cutoff_hz")
        N = win_for_cutoff(float(cutoff_hz))
        return moving_average_lowpass(x, N)

    elif mode == "hp":
        if cutoff_hz is None:
            raise ValueError("hp requires cutoff_hz")
        N = win_for_cutoff(float(cutoff_hz))
        lp = moving_average_lowpass(x, N)
        return (x - lp).astype(np.float32)

    elif mode == "bp":
        if low_cutoff_hz is None or high_cutoff_hz is None:
            raise ValueError("bp requires low_cutoff_hz and high_cutoff_hz")
        lo = float(low_cutoff_hz)
        hi = float(high_cutoff_hz)
        if not (lo < hi):
            raise ValueError("For bp, require low_cutoff_hz < high_cutoff_hz")
        Nlo = win_for_cutoff(lo)
        Nhi = win_for_cutoff(hi)
        lp_hi = moving_average_lowpass(x, Nhi)
        lp_lo = moving_average_lowpass(x, Nlo)
        return (lp_hi - lp_lo).astype(np.float32)

    elif mode == "notch":
        if low_cutoff_hz is None or high_cutoff_hz is None:
            raise ValueError("notch requires low_cutoff_hz and high_cutoff_hz")
        lo = float(low_cutoff_hz)
        hi = float(high_cutoff_hz)
        if not (lo < hi):
            raise ValueError("For notch, require low_cutoff_hz < high_cutoff_hz")
        # Notch as x - BP
        bp = moving_average_filter(x, sample_rate, mode="bp", low_cutoff_hz=lo, high_cutoff_hz=hi)
        return (x - bp).astype(np.float32)

    else:
        raise ValueError("mode must be one of: lp, hp, bp, notch")


def _ema_channel(x: np.ndarray, alpha: float) -> np.ndarray:
    """Single-channel exponential moving average low-pass."""
    y = np.empty_like(x, dtype=np.float32)
    y0 = x[0].astype(np.float32)
    y[0] = y0
    a = float(np.clip(alpha, 0.0, 1.0))
    one_minus_a = 1.0 - a
    for i in range(1, x.shape[0]):
        y[i] = one_minus_a * y[i - 1] + a * x[i]
    return y


def running_average_lowpass(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    order: int = 1,
    brickwall: bool = False,
) -> np.ndarray:
    """Exponential moving-average low-pass (EMA).

    - cutoff_hz: approximate -3 dB frequency; mapped via alpha = 1 - exp(-2π fc/fs)
    - order: number of cascaded EMA stages (increases slope)
    - brickwall: apply forward + reverse (zero-phase) filtering for steep response
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, c = x.shape

    fc = float(max(cutoff_hz, 1e-6))
    fs = float(sample_rate)
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc / fs)
    stages = max(1, int(order))

    y = x.copy()
    # Forward cascades
    for _ in range(stages):
        for ch in range(c):
            y[:, ch] = _ema_channel(y[:, ch], alpha)

    if brickwall:
        # Reverse cascades for zero-phase, steeper response
        y = y[::-1]
        for _ in range(stages):
            for ch in range(c):
                y[:, ch] = _ema_channel(y[:, ch], alpha)
        y = y[::-1]

    return y.astype(np.float32)


def running_average_filter(
    signal: np.ndarray,
    sample_rate: int,
    mode: str = "lp",
    cutoff_hz: float | None = None,
    low_cutoff_hz: float | None = None,
    high_cutoff_hz: float | None = None,
    order: int = 1,
    brickwall: bool = False,
) -> np.ndarray:
    """Composite filters (LP/HP/BP/Notch) derived from EMA low-pass.

    - Slope: controlled by `order` (stages). Higher → steeper.
    - Brickwall: forward+reverse for zero-phase and sharper skirts.
    """
    x = signal.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if mode == "lp":
        if cutoff_hz is None:
            raise ValueError("lp requires cutoff_hz")
        return running_average_lowpass(x, sample_rate, float(cutoff_hz), order, brickwall)

    elif mode == "hp":
        if cutoff_hz is None:
            raise ValueError("hp requires cutoff_hz")
        lp = running_average_lowpass(x, sample_rate, float(cutoff_hz), order, brickwall)
        return (x - lp).astype(np.float32)

    elif mode == "bp":
        if low_cutoff_hz is None or high_cutoff_hz is None:
            raise ValueError("bp requires low_cutoff_hz and high_cutoff_hz")
        lo = float(low_cutoff_hz)
        hi = float(high_cutoff_hz)
        if not (lo < hi):
            raise ValueError("For bp, require low_cutoff_hz < high_cutoff_hz")
        lp_hi = running_average_lowpass(x, sample_rate, hi, order, brickwall)
        lp_lo = running_average_lowpass(x, sample_rate, lo, order, brickwall)
        return (lp_hi - lp_lo).astype(np.float32)

    elif mode == "notch":
        if low_cutoff_hz is None or high_cutoff_hz is None:
            raise ValueError("notch requires low_cutoff_hz and high_cutoff_hz")
        lo = float(low_cutoff_hz)
        hi = float(high_cutoff_hz)
        if not (lo < hi):
            raise ValueError("For notch, require low_cutoff_hz < high_cutoff_hz")
        bp = running_average_filter(x, sample_rate, mode="bp", low_cutoff_hz=lo, high_cutoff_hz=hi, order=order, brickwall=brickwall)
        return (x - bp).astype(np.float32)

    else:
        raise ValueError("mode must be one of: lp, hp, bp, notch")


# Typing helper for processors
Processor = Callable[[np.ndarray], np.ndarray]
