# Simple Audio Processor (Python)

A minimal, modular audio processor for WAV files. It provides:

- WAV read/write using 16-bit PCM
- A pluggable processor pipeline (`AudioProcessor`)
- Effects: gain, hard clip, and soft clip variants (tanh/atan/rational/tube)
- A CLI with commands to process files and generate a test tone

## Quick Start

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate a test tone

```bash
python -m processor.cli generate-tone --output tone.wav --frequency 440 --duration 1.5 --amplitude 1.0
```

### Generate white noise

```bash
python -m processor.cli generate-noise --output noise.wav --duration 2.0 --amplitude 0.8
```

## Processing examples

### Gain

```bash
python -m processor.cli process --input tone.wav --output gained.wav --gain 0.5
```

### Hard clip

```bash
python -m processor.cli process --input tone.wav --output clipped.wav --hard-clip 0.3
```

### Soft clip (variants)

Tanh: `f(x) = T · tanh(gx)`

```bash
python -m processor.cli process --input tone.wav --output soft_tanh.wav --soft-clip tanh --soft-g 2.5 --soft-T 0.9
```

Arctan: `f(x) = (2T/π) · arctan(gx)`

```bash
python -m processor.cli process --input tone.wav --output soft_atan.wav --soft-clip atan --soft-g 2.5 --soft-T 0.9
```

Rational: `f(x) = T·gx / (1 + |gx|)`

```bash
python -m processor.cli process --input tone.wav --output soft_rational.wav --soft-clip rational --soft-g 2.5 --soft-T 0.9
```

Tube-like: `f(x) = tanh(gx + a) + b·x` with `a, b ∈ [0,1]`

```bash
python -m processor.cli process --input tone.wav --output soft_tube.wav --soft-clip tube --soft-g 2.5 --soft-a 0.2 --soft-b 0.3
```

### Chaining effects

Gain → Soft clip → Hard clip:

```bash
python -m processor.cli process \
	--input tone.wav --output chain.wav \
	--gain 2.0 \
	--soft-clip tanh --soft-g 3.0 --soft-T 0.9 \
	--hard-clip 0.95
```

### Comb filter

Enable comb filtering with delay (ms), feedback, feedforward, and wet mix:

```bash
python -m processor.cli process --input tone.wav --output comb.wav \
	--comb --comb-delay-ms 10 --comb-feedback 0.6 --comb-feedforward 0.0 --comb-mix 1.0
```

This implements:
- `y[n] = x[n] + feedforward · x[n-D] + feedback · y[n-D]`, where `D` is the delay in samples.
- Output is `(1-mix)·x + mix·y`.

### Phaser

Enable a multi-stage phaser (cascaded all-pass filters) with LFO sweep:

```bash
python -m processor.cli process --input tone.wav --output phaser.wav \
	--phaser --phaser-rate 0.6 --phaser-depth 0.8 --phaser-stages 4 \
	--phaser-feedback 0.2 --phaser-mix 0.6
```

Notes:
- More stages increase the number of notches (classic: 4–8 stages).
- `--phaser-rate` controls sweep speed; `--phaser-depth` controls how wide the notches move.
- `--phaser-feedback` emphasizes the notches for a more pronounced effect.

## Code Structure

- [processor/audio_io.py](processor/audio_io.py): WAV file I/O (16-bit PCM) with NumPy.
- [processor/effects.py](processor/effects.py): Effects `gain(...)`, `hard_clip(...)`, `soft_clip(...)` variants.
	Also: `comb_filter(signal, sample_rate, delay_ms, feedback, feedforward, mix)`.
- [processor/processor.py](processor/processor.py): `AudioProcessor` for chaining effects.
- [processor/cli.py](processor/cli.py): Command-line interface.

## Notes

- Signals are processed as float32, nominally in [-1, 1].
- The current `gain(...)` clamps to [-1, 1]; if you prefer unclipped gain to let soft/hard clip control the limiting, we can expose a `--gain-no-clip` option.
- The WAV writer clamps to [-1, 1] before converting to 16-bit PCM to avoid overflow.
