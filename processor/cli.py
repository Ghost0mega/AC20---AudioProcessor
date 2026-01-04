import argparse
from .audio_io import read_wav, write_wav, generate_tone, generate_white_noise, generate_sine_pluck
from .effects import hard_clip, gain, soft_clip, comb_filter, phaser, flanger, moving_average_filter, running_average_filter, delay
from .processor import AudioProcessor


def cmd_generate_tone(args: argparse.Namespace) -> None:
    sig = generate_tone(
        frequency=args.frequency,
        duration=args.duration,
        sample_rate=args.sample_rate,
        amplitude=args.amplitude,
        channels=args.channels,
    )
    write_wav(args.output, sig, args.sample_rate)
    print(f"Generated tone: {args.output}")


def cmd_generate_noise(args: argparse.Namespace) -> None:
    sig = generate_white_noise(
        duration=args.duration,
        sample_rate=args.sample_rate,
        amplitude=args.amplitude,
        channels=args.channels,
    )
    write_wav(args.output, sig, args.sample_rate)
    print(f"Generated white noise: {args.output}")


def cmd_process(args: argparse.Namespace) -> None:
    sig, sr = read_wav(args.input)

    pipeline = AudioProcessor()
    if args.gain is not None:
        pipeline.add(lambda s: gain(s, args.gain))
    if args.delay:
        pipeline.add(lambda s: delay(s, sr, args.delay_ms, args.delay_feedback, args.delay_mix))
    if args.comb:
        pipeline.add(lambda s: comb_filter(s, sr, args.comb_delay_ms, args.comb_feedback, args.comb_feedforward, args.comb_mix))
    if args.phaser:
        pipeline.add(lambda s: phaser(s, sr, args.phaser_rate, args.phaser_depth, args.phaser_stages, args.phaser_feedback, args.phaser_mix, args.phaser_center))
    if args.flanger:
        pipeline.add(lambda s: flanger(s, sr, args.flanger_base_ms, args.flanger_depth_ms, args.flanger_rate, args.flanger_feedback, args.flanger_mix))
    if args.ma_filter is not None:
        pipeline.add(lambda s: moving_average_filter(s, sr, args.ma_filter, args.ma_cutoff, args.ma_low, args.ma_high))
    if args.ra_filter is not None:
        pipeline.add(lambda s: running_average_filter(s, sr, args.ra_filter, args.ra_cutoff, args.ra_low, args.ra_high, args.ra_order, args.ra_brickwall))
    if args.soft_clip is not None:
        pipeline.add(lambda s: soft_clip(s, args.soft_clip, args.soft_g, args.soft_T, args.soft_a, args.soft_b))
    if args.hard_clip is not None:
        pipeline.add(lambda s: hard_clip(s, args.hard_clip))

    out = pipeline.process(sig)
    write_wav(args.output, out, sr)
    print(f"Processed file written to: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple audio processor CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # generate-tone
    pg = sub.add_parser("generate-tone", help="Generate a test sine tone WAV")
    pg.add_argument("--output", required=True, help="Output WAV path")
    pg.add_argument("--frequency", type=float, default=440.0)
    pg.add_argument("--duration", type=float, default=1.0)
    pg.add_argument("--sample_rate", type=int, default=44100)
    pg.add_argument("--amplitude", type=float, default=0.9)
    pg.add_argument("--channels", type=int, choices=[1, 2], default=1)
    pg.set_defaults(func=cmd_generate_tone)

    # generate-noise
    pn = sub.add_parser("generate-noise", help="Generate white-noise WAV")
    pn.add_argument("--output", required=True, help="Output WAV path")
    pn.add_argument("--duration", type=float, default=1.0)
    pn.add_argument("--sample_rate", type=int, default=44100)
    pn.add_argument("--amplitude", type=float, default=0.9)
    pn.add_argument("--channels", type=int, choices=[1, 2], default=1)
    pn.set_defaults(func=cmd_generate_noise)

    # generate-sine-pluck
    psp = sub.add_parser("generate-sine-pluck", help="Generate a sine pluck (attack + exponential decay)")
    psp.add_argument("--output", required=True, help="Output WAV path")
    psp.add_argument("--frequency", type=float, default=440.0)
    psp.add_argument("--duration", type=float, default=1.0)
    psp.add_argument("--sample_rate", type=int, default=44100)
    psp.add_argument("--amplitude", type=float, default=0.9)
    psp.add_argument("--attack_ms", type=float, default=5.0, help="Attack time in milliseconds")
    psp.add_argument("--decay_tau", type=float, default=0.3, help="Exponential decay time constant in seconds")
    psp.add_argument("--channels", type=int, choices=[1, 2], default=1)
    def cmd_generate_sine_pluck(args: argparse.Namespace) -> None:
        sig = generate_sine_pluck(
            frequency=args.frequency,
            duration=args.duration,
            sample_rate=args.sample_rate,
            amplitude=args.amplitude,
            attack_ms=args.attack_ms,
            decay_tau=args.decay_tau,
            channels=args.channels,
        )
        write_wav(args.output, sig, args.sample_rate)
        print(f"Generated sine pluck: {args.output}")
    psp.set_defaults(func=cmd_generate_sine_pluck)

    # process
    pp = sub.add_parser("process", help="Process a WAV file")
    pp.add_argument("--input", required=True, help="Input WAV path")
    pp.add_argument("--output", required=True, help="Output WAV path")
    pp.add_argument("--gain", dest="gain", type=float, default=None,
                    help="Apply linear gain multiplier (e.g., 0.5, 2.0)")
    # Delay
    pp.add_argument("--delay", action="store_true", help="Enable delay/echo")
    pp.add_argument("--delay-ms", dest="delay_ms", type=float, default=300.0, help="Delay time (ms)")
    pp.add_argument("--delay-feedback", dest="delay_feedback", type=float, default=0.5, help="Delay feedback (-0.99..0.99)")
    pp.add_argument("--delay-mix", dest="delay_mix", type=float, default=0.5, help="Delay wet mix (0..1)")
    # Comb filter
    pp.add_argument("--comb", action="store_true", help="Enable comb filter")
    pp.add_argument("--comb-delay-ms", dest="comb_delay_ms", type=float, default=10.0, help="Comb delay in ms")
    pp.add_argument("--comb-feedback", dest="comb_feedback", type=float, default=0.5, help="Comb feedback coefficient (-1..1)")
    pp.add_argument("--comb-feedforward", dest="comb_feedforward", type=float, default=0.0, help="Comb feedforward coefficient (-1..1)")
    pp.add_argument("--comb-mix", dest="comb_mix", type=float, default=1.0, help="Comb wet mix (0..1)")
    # Phaser
    pp.add_argument("--phaser", action="store_true", help="Enable phaser")
    pp.add_argument("--phaser-rate", dest="phaser_rate", type=float, default=0.5, help="Phaser LFO rate (Hz)")
    pp.add_argument("--phaser-depth", dest="phaser_depth", type=float, default=0.7, help="Phaser LFO depth (0..1)")
    pp.add_argument("--phaser-stages", dest="phaser_stages", type=int, default=4, help="Phaser stages (2..12)")
    pp.add_argument("--phaser-feedback", dest="phaser_feedback", type=float, default=0.0, help="Phaser feedback (-0.99..0.99)")
    pp.add_argument("--phaser-mix", dest="phaser_mix", type=float, default=0.5, help="Phaser wet mix (0..1)")
    pp.add_argument("--phaser-center", dest="phaser_center", type=float, default=0.0, help="Phaser sweep center bias (-1..1)")
    # Flanger
    pp.add_argument("--flanger", action="store_true", help="Enable flanger")
    pp.add_argument("--flanger-base-ms", dest="flanger_base_ms", type=float, default=2.0, help="Flanger base delay (ms)")
    pp.add_argument("--flanger-depth-ms", dest="flanger_depth_ms", type=float, default=1.5, help="Flanger modulation depth (ms)")
    pp.add_argument("--flanger-rate", dest="flanger_rate", type=float, default=0.25, help="Flanger LFO rate (Hz)")
    pp.add_argument("--flanger-feedback", dest="flanger_feedback", type=float, default=0.3, help="Flanger feedback (-0.99..0.99)")
    pp.add_argument("--flanger-mix", dest="flanger_mix", type=float, default=0.5, help="Flanger wet mix (0..1)")
    # Moving-average filters
    pp.add_argument("--ma-filter", dest="ma_filter", choices=["lp", "hp", "bp", "notch"], default=None,
                    help="Moving-average filter type")
    pp.add_argument("--ma-cutoff", dest="ma_cutoff", type=float, default=None, help="Cutoff for lp/hp (Hz)")
    pp.add_argument("--ma-low", dest="ma_low", type=float, default=None, help="Low cutoff for bp/notch (Hz)")
    pp.add_argument("--ma-high", dest="ma_high", type=float, default=None, help="High cutoff for bp/notch (Hz)")
    # Running-average (EMA) filters
    pp.add_argument("--ra-filter", dest="ra_filter", choices=["lp", "hp", "bp", "notch"], default=None,
                    help="Running-average (EMA) filter type")
    pp.add_argument("--ra-cutoff", dest="ra_cutoff", type=float, default=None, help="Cutoff for lp/hp (Hz)")
    pp.add_argument("--ra-low", dest="ra_low", type=float, default=None, help="Low cutoff for bp/notch (Hz)")
    pp.add_argument("--ra-high", dest="ra_high", type=float, default=None, help="High cutoff for bp/notch (Hz)")
    pp.add_argument("--ra-order", dest="ra_order", type=int, default=1, help="EMA stages (slope)")
    pp.add_argument("--ra-brickwall", dest="ra_brickwall", action="store_true", help="Enable brickwall (forward+reverse) filtering")
    pp.add_argument("--soft-clip", dest="soft_clip", choices=["tanh", "atan", "rational", "tube"], default=None,
                    help="Soft clip variant: tanh, atan, rational, tube")
    pp.add_argument("--soft-g", dest="soft_g", type=float, default=1.0, help="Soft clip drive g")
    pp.add_argument("--soft-T", dest="soft_T", type=float, default=1.0, help="Soft clip scale T")
    pp.add_argument("--soft-a", dest="soft_a", type=float, default=0.0, help="Tube parameter a in [0,1]")
    pp.add_argument("--soft-b", dest="soft_b", type=float, default=0.0, help="Tube parameter b in [0,1]")
    pp.add_argument("--hard-clip", dest="hard_clip", type=float, default=None,
                    help="Apply hard clip with threshold in (0,1]")
    pp.set_defaults(func=cmd_process)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
