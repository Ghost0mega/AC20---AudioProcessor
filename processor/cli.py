import argparse
from .audio_io import read_wav, write_wav, generate_tone
from .effects import hard_clip, gain, soft_clip, comb_filter
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


def cmd_process(args: argparse.Namespace) -> None:
    sig, sr = read_wav(args.input)

    pipeline = AudioProcessor()
    if args.gain is not None:
        pipeline.add(lambda s: gain(s, args.gain))
    if args.comb:
        pipeline.add(lambda s: comb_filter(s, sr, args.comb_delay_ms, args.comb_feedback, args.comb_feedforward, args.comb_mix))
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

    # process
    pp = sub.add_parser("process", help="Process a WAV file")
    pp.add_argument("--input", required=True, help="Input WAV path")
    pp.add_argument("--output", required=True, help="Output WAV path")
    pp.add_argument("--gain", dest="gain", type=float, default=None,
                    help="Apply linear gain multiplier (e.g., 0.5, 2.0)")
    # Comb filter
    pp.add_argument("--comb", action="store_true", help="Enable comb filter")
    pp.add_argument("--comb-delay-ms", dest="comb_delay_ms", type=float, default=10.0, help="Comb delay in ms")
    pp.add_argument("--comb-feedback", dest="comb_feedback", type=float, default=0.5, help="Comb feedback coefficient (-1..1)")
    pp.add_argument("--comb-feedforward", dest="comb_feedforward", type=float, default=0.0, help="Comb feedforward coefficient (-1..1)")
    pp.add_argument("--comb-mix", dest="comb_mix", type=float, default=1.0, help="Comb wet mix (0..1)")
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
