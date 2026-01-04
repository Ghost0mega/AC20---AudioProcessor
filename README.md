# Simple Audio Processor (Python)

A minimal, modular audio processor for WAV files. It provides:

- WAV read/write using PCM (16/24/32-bit) to float32
- A pluggable processor pipeline (`AudioProcessor`)
- Effects: gain, hard clip, soft clip variants (tanh/atan/rational/tube)
- Time-based: delay/echo, phaser, flanger, Schroeder reverb
- Filters: moving-average and running-average (EMA) LP/HP/BP/Notch
- Convolver: apply impulse responses (IRs) via FFT
- Generators: sine tone, white noise, sine pluck (with optional gate)
- A CLI to generate and process WAVs

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
### Generate a sine pluck (attack + decay, optional gate)

```bash
python -m processor.cli generate-sine-pluck \
	--output sine_pluck.wav --frequency 440 --duration 1.5 \
	--amplitude 0.9 --attack_ms 5 --decay_tau 0.35 [--gate_ms 30]
```

Notes:
- `attack_ms`: linear ramp time (ms). `decay_tau`: exponential decay time constant (s).
- `gate_ms` (optional): hard gate; envelope becomes zero after this time.

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
### Flanger

Classic variable fractional delay with LFO:

```bash
python -m processor.cli process --input tone.wav --output flanger.wav \
	--flanger --flanger-base-ms 2.0 --flanger-depth-ms 1.5 \
	--flanger-rate 0.25 --flanger-feedback 0.3 --flanger-mix 0.5
```

Parameters:
- Base/depth in ms; `rate` in Hz; `feedback` in (-0.99..0.99); `mix` in [0,1].

### Delay / Echo

Simple feedback delay:

```bash
python -m processor.cli process --input tone.wav --output delay.wav \
	--delay --delay-ms 300 --delay-feedback 0.6 --delay-mix 0.5
```

### Schroeder Reverb

Parallel combs → series all-pass filters. Quick room/hall examples:

```bash
# Roomy with predelay
python -m processor.cli process --input sine_pluck.wav --output room.wav \
	--reverb --reverb-mix 0.65 --reverb-predelay-ms 25 \
	--reverb-comb-feedback 0.8 --reverb-damping 0.65

# Strong hall
python -m processor.cli process --input sine_pluck.wav --output hall.wav \
	--reverb --reverb-mix 0.85 --reverb-predelay-ms 0 \
	--reverb-comb-feedback 0.88 --reverb-damping 0.55
```

Notes:
- `--reverb-damping` applies low-pass in comb feedback; higher → darker tail.
- Defaults use balanced delay sets; predelay shifts the wet onset.

### Filters (Moving-average and EMA)

Moving-average (boxcar) and running-average (EMA) suites:

```bash
# Moving-average low-pass at 1000 Hz
python -m processor.cli process --input noise.wav --output ma_lp_1000.wav \
	--ma-filter lp --ma-cutoff 1000

# EMA low-pass order 4 (steeper)
python -m processor.cli process --input noise.wav --output ra_lp_1000_o4.wav \
	--ra-filter lp --ra-cutoff 1000 --ra-order 4

# EMA band-pass 800–2000 Hz with brickwall (zero-phase)
python -m processor.cli process --input noise.wav --output ra_bp_800_2000_bw.wav \
	--ra-filter bp --ra-low 800 --ra-high 2000 --ra-order 6 --ra-brickwall
```

Notes:
- MA cutoff mapping is approximate (sinc response); EMA uses α mapping for -3 dB.
- EMA `order` controls slope; `--ra-brickwall` applies forward+reverse filtering.

### Convolver (Impulse Responses)

Apply IRs via FFT with optional normalization and auto-gain:

```bash
python -m processor.cli process --input melody.wav --output convolved.wav \
	--convolver --conv-ir audio/impulse-responses/Church\ Schellingwoude.wav \
	--conv-mix 1.0 --conv-normalize --conv-auto-gain --conv-target-peak 0.9
```

Notes:
- Mono IR → applied to all channels; stereo IR → per-channel convolution.
- `--conv-normalize` scales IR to unit peak; `--conv-auto-gain` attenuates wet to target peak.
- More stages increase the number of notches (classic: 4–8 stages).
- `--phaser-rate` controls sweep speed; `--phaser-depth` controls how wide the notches move.
- `--phaser-feedback` emphasizes the notches for a more pronounced effect.

## Code Structure

- [processor/audio_io.py](processor/audio_io.py): WAV I/O (16/24/32-bit PCM → float32); generators:
	- `generate_tone(...)`, `generate_white_noise(...)`, `generate_sine_pluck(...)`
- [processor/effects.py](processor/effects.py):
	- Dynamics: `gain(...)`, `hard_clip(...)`, `soft_clip(...)`
	- Time-based: `delay(...)`, `phaser(...)`, `flanger(...)`, `reverb_schroeder(...)`
	- Filters: `moving_average_filter(...)`, `running_average_filter(...)`
	- Convolver: `convolver(signal, ir, ...)`
- [processor/processor.py](processor/processor.py): `AudioProcessor` for chaining effects.
- [processor/cli.py](processor/cli.py): Command-line interface.

## Notes

- Signals are processed as float32 in [-1, 1]; WAV writer clamps before PCM conversion.
- Some effects (soft clip, reverb, convolver) can increase level; use wet `mix`, auto-gain or reduce input to avoid clipping.
