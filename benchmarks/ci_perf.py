"""CI performance-regression harness.

Measures avtensor decode scenarios (no torchcodec) against small synthetic
fixtures, and compares two result files. CI builds wheels for a PR's
merge-base and head, runs this harness for each in the same job, and fails
when head is slower than base beyond a threshold — a same-machine relative
comparison, so runner speed differences cancel out.

Usage:
    python benchmarks/ci_perf.py run -o head.json
    python benchmarks/ci_perf.py compare base.json head.json --threshold 1.25

The compared statistic is the minimum over repeats: for CPU-bound decode
work, interference from a noisy runner only ever adds time, so the minimum
is the most stable estimate. The median is reported for context.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "ci_media"

# Deterministic synthetic fixtures. Scenario durations are chosen so every
# measurement is >= ~200 ms on a small CI runner; shorter scenarios drown in
# timer and scheduler noise.
FIXTURES = {
    "video.mp4": [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30",
        "-t",
        "8",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ],
    "audio.m4a": [
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        "240",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
    ],
}


def prepare_fixtures() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    for name, args in FIXTURES.items():
        out = FIXTURES_DIR / name
        if not out.exists():
            subprocess.run([ffmpeg, "-y", "-v", "error", *args, str(out)], check=True)


def scenarios():
    import avtensor
    from avtensor import AudioStreamRequest, MediaDecodeRequest, VideoStreamRequest

    video = str(FIXTURES_DIR / "video.mp4")
    audio = str(FIXTURES_DIR / "audio.m4a")

    def decode(path, video_kwargs=None, audio_stream=False, start=None, end=None):
        req = MediaDecodeRequest(path)
        if video_kwargs is not None:
            v = VideoStreamRequest()
            for k, val in video_kwargs.items():
                setattr(v, k, val)
            req.video_stream = v
        if audio_stream:
            req.audio_streams = [AudioStreamRequest()]
        if start is not None:
            req.start_time, req.end_time = start, end
        return avtensor.decode_asset(req)

    return {
        "video_full": lambda: decode(video, video_kwargs={"number_of_threads": 0}),
        "video_resize": lambda: decode(
            video, video_kwargs={"number_of_threads": 0, "width": 256, "height": 144}
        ),
        "video_seek_window": lambda: decode(
            video, video_kwargs={"number_of_threads": 0}, start=3.0, end=6.0
        ),
        "audio_full": lambda: decode(audio, audio_stream=True),
    }


def run(output: Path, repeats: int) -> None:
    prepare_fixtures()
    import avtensor  # noqa: F401  (fail early if not installed)

    results = {}
    for name, fn in scenarios().items():
        fn()  # warmup
        walls = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            walls.append(time.perf_counter() - t0)
        results[name] = {
            "min_s": min(walls),
            "median_s": statistics.median(walls),
            "repeats": repeats,
        }
        print(
            f"{name}: min {min(walls) * 1000:.0f} ms, "
            f"median {statistics.median(walls) * 1000:.0f} ms",
            flush=True,
        )
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")


def compare(base_path: Path, head_path: Path, threshold: float) -> int:
    base = json.loads(base_path.read_text())
    head = json.loads(head_path.read_text())

    rows = []
    regressions = []
    for name in sorted(set(base) & set(head)):
        b, h = base[name], head[name]
        ratio = h["min_s"] / b["min_s"] if b["min_s"] > 0 else float("inf")
        rows.append((name, b["min_s"], h["min_s"], ratio))
        if ratio > threshold:
            regressions.append((name, ratio))

    lines = [
        f"| scenario | base min | head min | head/base (fail > {threshold:.2f}) |",
        "| --- | --- | --- | --- |",
    ]
    for name, b, h, ratio in rows:
        flag = " ⚠️" if ratio > threshold else ""
        lines.append(
            f"| {name} | {b * 1000:.0f} ms | {h * 1000:.0f} ms | {ratio:.2f}{flag} |"
        )
    table = "\n".join(lines)
    print(table)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Performance vs merge-base\n\n" + table + "\n")

    if regressions:
        names = ", ".join(f"{n} ({r:.2f}x)" for n, r in regressions)
        print(f"\nFAIL: performance regression in {names}", file=sys.stderr)
        return 1
    print("\nOK: no scenario regressed beyond the threshold")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="measure and write a results JSON")
    run_p.add_argument("-o", "--output", type=Path, required=True)
    run_p.add_argument("--repeats", type=int, default=9)

    cmp_p = sub.add_parser("compare", help="compare two results JSONs")
    cmp_p.add_argument("base", type=Path)
    cmp_p.add_argument("head", type=Path)
    cmp_p.add_argument(
        "--threshold",
        type=float,
        default=1.25,
        help="fail when head min exceeds base min by this factor",
    )

    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.repeats)
    else:
        sys.exit(compare(args.base, args.head, args.threshold))


if __name__ == "__main__":
    main()
