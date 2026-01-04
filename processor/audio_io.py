import wave
from typing import Tuple
import numpy as np


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file.

    Returns (data, sample_rate) where data is float32 in [-1, 1]
    with shape (num_samples, num_channels).
    """
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        if sampwidth != 2:
            raise ValueError(f"Only 16-bit PCM supported (found sampwidth={sampwidth}).")
        frames = wf.readframes(n_frames)

    data_i16 = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        data_i16 = data_i16.reshape(-1, n_channels)
    else:
        data_i16 = data_i16.reshape(-1, 1)

    data_f32 = (data_i16.astype(np.float32)) / 32768.0
    return data_f32, framerate


def write_wav(path: str, data: np.ndarray, sample_rate: int) -> None:
    """Write float32 [-1, 1] signal to a 16-bit PCM WAV file.

    `data` should be shape (num_samples, num_channels).
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Clip to [-1, 1]
    data = np.clip(data, -1.0, 1.0)
    # Convert to int16
    data_i16 = (data * 32767.0).astype(np.int16)
    # Interleave channels for writing
    frames = data_i16.reshape(-1)

    n_channels = data.shape[1]

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(frames.tobytes())


def generate_tone(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.9,
    channels: int = 1,
) -> np.ndarray:
    """Generate a sine tone as float32 in [-1, 1]."""
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    sig = amplitude * np.sin(2.0 * np.pi * frequency * t).astype(np.float32)
    if channels == 1:
        return sig.reshape(-1, 1)
    elif channels == 2:
        # Duplicate to stereo
        return np.column_stack([sig, sig])
    else:
        raise ValueError("Only mono or stereo supported.")


def generate_white_noise(
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.9,
    channels: int = 1,
) -> np.ndarray:
    """Generate uniform white noise as float32 in [-amplitude, amplitude]."""
    n = int(duration * sample_rate)
    amp = float(np.clip(amplitude, 0.0, 1.0))
    noise = (np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)) * amp
    if channels == 1:
        return noise.reshape(-1, 1)
    elif channels == 2:
        return np.column_stack([noise, noise])
    else:
        raise ValueError("Only mono or stereo supported.")


def generate_sine_pluck(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.9,
    attack_ms: float = 5.0,
    decay_tau: float = 0.3,
    channels: int = 1,
) -> np.ndarray:
    """Generate a sine "pluck" tone with a fast attack and exponential decay.

    Parameters
    - frequency: Base sine frequency (Hz)
    - duration: Total length (seconds)
    - sample_rate: Sampling rate (Hz)
    - amplitude: Peak amplitude before envelope (0..1)
    - attack_ms: Attack time in milliseconds for ramp-up
    - decay_tau: Exponential decay time constant in seconds
    - channels: 1 (mono) or 2 (stereo)
    """
    n = int(duration * sample_rate)
    if n <= 0:
        return np.zeros((0, max(1, channels)), dtype=np.float32)

    t = np.arange(n, dtype=np.float32) / sample_rate
    sig = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)

    # Envelope: linear attack then exponential decay
    attack_samples = max(1, int((attack_ms / 1000.0) * sample_rate))
    attack = np.linspace(0.0, 1.0, attack_samples, endpoint=False, dtype=np.float32)
    decay_len = max(0, n - attack_samples)
    # Decay starts at 1.0 and falls with time constant decay_tau
    if decay_len > 0:
        k = 1.0 / max(1e-6, decay_tau * sample_rate)
        decay = np.exp(-k * np.arange(decay_len, dtype=np.float32))
        env = np.concatenate([attack, decay], dtype=np.float32)
    else:
        env = attack[:n]

    # Ensure envelope length equals signal length
    if env.size < n:
        env = np.pad(env, (0, n - env.size), mode='constant', constant_values=0.0)
    elif env.size > n:
        env = env[:n]

    out = (amplitude * env * sig).astype(np.float32)

    if channels == 1:
        return out.reshape(-1, 1)
    elif channels == 2:
        return np.column_stack([out, out])
    else:
        raise ValueError("Only mono or stereo supported.")
