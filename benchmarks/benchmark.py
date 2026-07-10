"""Benchmark avtensor against torchcodec.

Prepares deterministic test fixtures (cut from Big Buck Bunny with the local
FFmpeg), runs matched decode scenarios through both libraries, and prints
GitHub-flavored markdown tables. See benchmarks/README.md for methodology and
results.

Usage:
    python benchmarks/benchmark.py                  # everything
    python benchmarks/benchmark.py --scenarios single,concurrency
    python benchmarks/benchmark.py --skip-prepare   # fixtures already present

Requirements: avtensor and torchcodec importable, `ffmpeg` on PATH (or set
$FFMPEG) for fixture preparation, network access for the one-time source
download (~276 MB).
"""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchcodec
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize

import avtensor
from avtensor import AudioStreamRequest, MediaDecodeRequest, VideoStreamRequest

SOURCE_URL = (
    "https://download.blender.org/demo/movies/BBB/"
    "bbb_sunflower_1080p_30fps_normal.mp4.zip"
)

FIXTURES = {
    # name: (ffmpeg args after -i <source>,)
    "clip_1080p_h264_30s.mp4": [
        "-ss",
        "180",
        "-t",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
    ],
    "clip_1080p_hevc_30s.mp4": [
        "-ss",
        "180",
        "-t",
        "30",
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "hvc1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
    ],
    "clip_720p_h264_10s.mp4": [
        "-ss",
        "180",
        "-t",
        "10",
        "-vf",
        "scale=1280:720",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
    ],
}


def prepare_media(media_dir: Path) -> None:
    media_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in FIXTURES if not (media_dir / n).exists()]
    if not missing:
        return
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    if shutil.which(ffmpeg) is None:
        sys.exit(
            f"fixture preparation needs FFmpeg; `{ffmpeg}` not found "
            "(set $FFMPEG or use --skip-prepare with existing media)"
        )
    source = media_dir / "bbb_sunflower_1080p_30fps_normal.mp4"
    if not source.exists():
        print(f"downloading source video ({SOURCE_URL}) ...", flush=True)
        zip_path = media_dir / "bbb.zip"
        urllib.request.urlretrieve(SOURCE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extract("bbb_sunflower_1080p_30fps_normal.mp4", media_dir)
        zip_path.unlink()
    for name in missing:
        print(f"encoding fixture {name} ...", flush=True)
        # -ss placement (before -i) and encode settings are fixed so fixtures
        # are bit-identical across runs on the same FFmpeg build.
        args = FIXTURES[name]
        seek, rest = args[:4], args[4:]
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                *seek,
                "-i",
                str(source),
                *rest,
                str(media_dir / name),
            ],
            check=True,
        )


def _verify_same_ffmpeg(media_dir: Path) -> str:
    """Decodes a tiny window with both libraries, then checks which libavcodec
    each actually loaded (avtensor links at build time, torchcodec dlopens at
    runtime). Aborts if they resolved different builds — the comparison would
    be invalid."""
    clip = media_dir / "clip_720p_h264_10s.mp4"
    av_video(clip, threads=1, start=0.0, end=0.2)
    tc_video(clip, threads=1, start=0.0, end=0.2)
    with open("/proc/self/maps") as f:
        loaded = sorted({line.split()[-1] for line in f if "libavcodec" in line})
    if len(loaded) != 1:
        sys.exit(
            "avtensor and torchcodec loaded different libavcodec builds: "
            f"{loaded} — put exactly one FFmpeg on LD_LIBRARY_PATH and rebuild "
            "avtensor against it."
        )
    return loaded[0]


def _ffmpeg_version() -> str:
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    if shutil.which(ffmpeg):
        out = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, text=True
        ).stdout.splitlines()[0]
        return out.split()[2] if out.startswith("ffmpeg version") else out
    return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# ---------------- timing ----------------


def _cpu_time() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def timeit(fn, repeats: int, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    walls, cpus = [], []
    for _ in range(repeats):
        gc.collect()
        c0, t0 = _cpu_time(), time.perf_counter()
        out = fn()
        t1, c1 = time.perf_counter(), _cpu_time()
        walls.append(t1 - t0)
        cpus.append(c1 - c0)
        del out
    return {
        "wall_median_s": statistics.median(walls),
        "wall_min_s": min(walls),
        "cpu_median_s": statistics.median(cpus),
    }


# ---------------- decode helpers ----------------


def av_video(
    path,
    threads=None,
    width=None,
    height=None,
    start=None,
    end=None,
    hw=None,
    device=None,
):
    req = MediaDecodeRequest(str(path))
    v = VideoStreamRequest()
    if threads is not None:
        v.number_of_threads = threads
    if width:
        v.width, v.height = width, height
    if hw:
        v.hardware_acceleration = True
    if device:
        v.device = device
    req.video_stream = v
    if start is not None:
        req.start_time = start
        req.end_time = end
    (res,) = avtensor.decode_asset(req)
    return res["data"]


def tc_video(
    path, threads=1, resize=None, start=None, end=None, device=None, seek_mode="exact"
):
    transforms = [Resize((resize[1], resize[0]))] if resize else None
    d = VideoDecoder(
        str(path),
        num_ffmpeg_threads=threads,
        transforms=transforms,
        device=device,
        seek_mode=seek_mode,
    )
    if start is not None:
        if end is None:
            raise ValueError("start and end must be given together")
        return d.get_frames_played_in_range(start, end).data
    return d[:]


def av_audio(path, sample_rate=None):
    req = MediaDecodeRequest(str(path))
    a = AudioStreamRequest()
    if sample_rate:
        a.sample_rate = sample_rate
    req.audio_streams = [a]
    (res,) = avtensor.decode_asset(req)
    return res["data"]


def tc_audio(path, sample_rate=None):
    return AudioDecoder(str(path), sample_rate=sample_rate).get_all_samples().data


def av_video_audio(path):
    req = MediaDecodeRequest(str(path))
    v = VideoStreamRequest()
    v.number_of_threads = 0
    req.video_stream = v
    req.audio_streams = [AudioStreamRequest()]
    return [s["data"] for s in avtensor.decode_asset(req)]


def tc_video_audio(path):
    v = VideoDecoder(str(path), num_ffmpeg_threads=0)[:]
    a = AudioDecoder(str(path)).get_all_samples().data
    return [v, a]


# ---------------- scenarios ----------------


def bench_single(media: Path, repeats: int, results: dict) -> None:
    c1080 = media / "clip_1080p_h264_30s.mp4"
    c1080h = media / "clip_1080p_hevc_30s.mp4"
    c720 = media / "clip_720p_h264_10s.mp4"
    cases = [
        (
            "full 1080p H.264, 1 thread",
            "full_1080p_h264_threads1",
            lambda: av_video(c1080, threads=1),
            lambda: tc_video(c1080, threads=1),
        ),
        (
            "full 1080p H.264, auto threads",
            "full_1080p_h264_threadsauto",
            lambda: av_video(c1080, threads=0),
            lambda: tc_video(c1080, threads=0),
        ),
        (
            "full 1080p HEVC, auto threads",
            "full_1080p_hevc_threadsauto",
            lambda: av_video(c1080h, threads=0),
            lambda: tc_video(c1080h, threads=0),
        ),
        (
            "full 720p H.264, 1 thread",
            "full_720p_h264_threads1",
            lambda: av_video(c720, threads=1),
            lambda: tc_video(c720, threads=1),
        ),
        (
            "1080p -> 256x144, 1 thread",
            "resize256x144_1080p_threads1",
            lambda: av_video(c1080, threads=1, width=256, height=144),
            lambda: tc_video(c1080, threads=1, resize=(256, 144)),
        ),
        (
            "1080p -> 256x144, auto threads",
            "resize256x144_1080p_threadsauto",
            lambda: av_video(c1080, threads=0, width=256, height=144),
            lambda: tc_video(c1080, threads=0, resize=(256, 144)),
        ),
        (
            "5 s window @15 s, 1080p",
            "seek_5s_window_1080p",
            lambda: av_video(c1080, threads=0, start=15.0, end=20.0),
            lambda: tc_video(c1080, threads=0, start=15.0, end=20.0),
        ),
        (
            "audio 48 kHz (30 s)",
            "audio_48k",
            lambda: av_audio(c1080),
            lambda: tc_audio(c1080),
        ),
        (
            "audio resampled to 16 kHz",
            "audio_resample16k",
            lambda: av_audio(c1080, sample_rate=16000),
            lambda: tc_audio(c1080, sample_rate=16000),
        ),
        (
            "video + audio, one asset",
            "video_plus_audio_1080p",
            lambda: av_video_audio(c1080),
            lambda: tc_video_audio(c1080),
        ),
    ]
    for label, name, av_fn, tc_fn in cases:
        is_video = "audio" not in name and "plus" not in name
        n_frames = int(av_fn().shape[0]) if is_video else None
        measurements = {
            "avtensor": timeit(av_fn, repeats),
            "torchcodec": timeit(tc_fn, repeats),
        }
        if n_frames:
            for m in measurements.values():
                m["fps"] = n_frames / m["wall_median_s"]
        results["single"][name] = {"label": label, **measurements}
        print(
            f"[{name}] avtensor {measurements['avtensor']['wall_median_s']:.3f}s | "
            f"torchcodec {measurements['torchcodec']['wall_median_s']:.3f}s",
            flush=True,
        )


def bench_concurrency(media: Path, results: dict) -> None:
    """Aggregate decode throughput with W worker threads (1 FFmpeg thread per
    decode), the shape of a threaded data-loading pipeline."""
    c720 = media / "clip_720p_h264_10s.mp4"
    out = {}
    for workers in (1, 2, 4, 8, 16, 32):
        jobs = max(workers * 3, 8)
        for lib, fn in (
            ("avtensor", lambda: av_video(c720, threads=1)),
            ("torchcodec", lambda: tc_video(c720, threads=1)),
        ):
            fn()  # warmup
            gc.collect()
            with ThreadPoolExecutor(max_workers=workers) as ex:
                c0, t0 = _cpu_time(), time.perf_counter()
                futs = [ex.submit(fn) for _ in range(jobs)]
                shapes = [f.result().shape for f in futs]
                t1, c1 = time.perf_counter(), _cpu_time()
            frames = sum(s[0] for s in shapes)
            out.setdefault(lib, {})[workers] = {
                "jobs": jobs,
                "clips_per_s": jobs / (t1 - t0),
                "frames_per_s": frames / (t1 - t0),
                "cpu_s": c1 - c0,
                "wall_s": t1 - t0,
            }
            print(
                f"[concurrency w={workers}] {lib}: "
                f"{out[lib][workers]['frames_per_s']:.0f} frames/s",
                flush=True,
            )
    results["concurrency"] = out


def bench_concurrency_resize(media: Path, results: dict) -> None:
    """Aggregate decode+downscale throughput with W worker threads (1 FFmpeg
    thread per decode, 1080p -> 256x144), the shape of a downscaling
    data-loading pipeline."""
    c1080 = media / "clip_1080p_h264_30s.mp4"
    out = {}
    for workers in (1, 8, 32):
        jobs = max(workers * 2, 4)
        for lib, fn in (
            ("avtensor", lambda: av_video(c1080, threads=1, width=256, height=144)),
            ("torchcodec", lambda: tc_video(c1080, threads=1, resize=(256, 144))),
        ):
            fn()  # warmup
            gc.collect()
            with ThreadPoolExecutor(max_workers=workers) as ex:
                c0, t0 = _cpu_time(), time.perf_counter()
                futs = [ex.submit(fn) for _ in range(jobs)]
                shapes = [f.result().shape for f in futs]
                t1, c1 = time.perf_counter(), _cpu_time()
            frames = sum(s[0] for s in shapes)
            out.setdefault(lib, {})[workers] = {
                "jobs": jobs,
                "clips_per_s": jobs / (t1 - t0),
                "frames_per_s": frames / (t1 - t0),
                "cpu_s": c1 - c0,
                "wall_s": t1 - t0,
            }
            print(
                f"[concurrency_resize w={workers}] {lib}: "
                f"{out[lib][workers]['frames_per_s']:.0f} frames/s",
                flush=True,
            )
    results["concurrency_resize"] = out


def bench_nvdec(media: Path, repeats: int, results: dict) -> None:
    if not torch.cuda.is_available():
        print("[nvdec] skipped: no CUDA device", flush=True)
        return
    c1080 = media / "clip_1080p_h264_30s.mp4"

    def sync(fn):
        def inner():
            out = fn()
            torch.cuda.synchronize()
            return out

        return inner

    def av_gpu(**kwargs):
        def inner():
            out = av_video(c1080, hw=True, device="cuda", **kwargs)
            torch.cuda.synchronize()
            return out

        return inner

    cases = [
        (
            "NVDEC full 1080p (GPU-resident)",
            "nvdec_full_1080p_gpu",
            av_gpu(),
            sync(lambda: tc_video(c1080, device="cuda")),
        ),
        (
            "NVDEC 1080p -> 256x144 (GPU-resident)",
            "nvdec_resize256x144_1080p_gpu",
            av_gpu(width=256, height=144),
            sync(
                lambda: torch.nn.functional.interpolate(
                    tc_video(c1080, device="cuda").float(),
                    size=(144, 256),
                    mode="bilinear",
                    antialias=True,
                ).to(torch.uint8)
            ),
        ),
        (
            "NVDEC full 1080p",
            "nvdec_full_1080p",
            lambda: av_video(c1080, hw=True),
            sync(lambda: tc_video(c1080, device="cuda")),
        ),
        # torchcodec does not support decoder transforms on CUDA, so the
        # comparison point is the realistic pipeline: NVDEC decode to GPU
        # tensors + F.interpolate.
        (
            "NVDEC 1080p -> 256x144",
            "nvdec_resize256x144_1080p",
            lambda: av_video(c1080, hw=True, width=256, height=144),
            sync(
                lambda: torch.nn.functional.interpolate(
                    tc_video(c1080, device="cuda").float(),
                    size=(144, 256),
                    mode="bilinear",
                    antialias=True,
                ).to(torch.uint8)
            ),
        ),
    ]
    for label, name, av_fn, tc_fn in cases:
        r = {
            "label": label,
            "avtensor": timeit(av_fn, repeats),
            "torchcodec_cuda": timeit(tc_fn, repeats),
        }
        results["nvdec"][name] = r
        print(
            f"[{name}] avtensor {r['avtensor']['wall_median_s']:.3f}s | "
            f"torchcodec(cuda) {r['torchcodec_cuda']['wall_median_s']:.3f}s",
            flush=True,
        )


def check_correctness(media: Path, results: dict) -> None:
    c720 = media / "clip_720p_h264_10s.mp4"
    av = av_video(c720, threads=1).float()
    tc = tc_video(c720, threads=1).float()
    n = min(av.shape[0], tc.shape[0])
    diff = (av[:n] - tc[:n]).abs()
    results["correctness"] = {
        "avtensor_frames": int(av.shape[0]),
        "torchcodec_frames": int(tc.shape[0]),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "pct_pixels_diff_gt2": float((diff > 2).float().mean() * 100),
    }
    print(
        f"[correctness] frames {av.shape[0]} vs {tc.shape[0]}, "
        f"mean|diff|={results['correctness']['mean_abs_diff']:.3f}",
        flush=True,
    )


# ---------------- reporting ----------------


def markdown_report(results: dict) -> str:
    lines = []
    env = results["env"]
    lines.append(
        f"Environment: {env['cpu']} ({env['cpu_count']} cores), "
        f"GPU {env['gpu'] or 'n/a'}, torch {env['torch']}, "
        f"torchcodec {env['torchcodec']}, avtensor {env['avtensor']}, "
        f"FFmpeg {env['ffmpeg']}\n"
    )
    if results.get("single"):
        lines.append("| scenario | avtensor | torchcodec | speedup |")
        lines.append("| --- | --- | --- | --- |")
        for r in results["single"].values():
            a, t = r["avtensor"]["wall_median_s"], r["torchcodec"]["wall_median_s"]
            fps = f" ({r['avtensor']['fps']:.0f} fps)" if "fps" in r["avtensor"] else ""
            tfps = (
                f" ({r['torchcodec']['fps']:.0f} fps)"
                if "fps" in r["torchcodec"]
                else ""
            )
            lines.append(
                f"| {r['label']} | {a:.3f} s{fps} | {t:.3f} s{tfps} | {t / a:.2f}x |"
            )
        lines.append("")
    if results.get("concurrency"):
        lines.append("| worker threads | avtensor frames/s | torchcodec frames/s |")
        lines.append("| --- | --- | --- |")
        av, tc = (
            results["concurrency"]["avtensor"],
            results["concurrency"]["torchcodec"],
        )
        for w in av:
            lines.append(
                f"| {w} | {av[w]['frames_per_s']:.0f} | {tc[w]['frames_per_s']:.0f} |"
            )
        lines.append("")
    if results.get("concurrency_resize"):
        lines.append(
            "| worker threads (1080p → 256×144) | avtensor frames/s | torchcodec frames/s |"
        )
        lines.append("| --- | --- | --- |")
        av = results["concurrency_resize"]["avtensor"]
        tc = results["concurrency_resize"]["torchcodec"]
        for w in av:
            lines.append(
                f"| {w} | {av[w]['frames_per_s']:.0f} | {tc[w]['frames_per_s']:.0f} |"
            )
        lines.append("")
    if results.get("nvdec"):
        lines.append("| scenario | avtensor | torchcodec device='cuda' (GPU tensors) |")
        lines.append("| --- | --- | --- |")
        for r in results["nvdec"].values():
            lines.append(
                f"| {r['label']} | {r['avtensor']['wall_median_s']:.3f} s "
                f"| {r['torchcodec_cuda']['wall_median_s']:.3f} s |"
            )
        lines.append("")
    if results.get("correctness"):
        c = results["correctness"]
        lines.append(
            f"Correctness: {c['avtensor_frames']} vs "
            f"{c['torchcodec_frames']} frames, mean |diff| "
            f"{c['mean_abs_diff']:.3f}/255, max {c['max_abs_diff']:.0f}, "
            f"{c['pct_pixels_diff_gt2']:.2f}% of pixels differ by >2."
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--media-dir", type=Path, default=Path(__file__).parent / "media")
    p.add_argument(
        "--scenarios", default="single,concurrency,concurrency_resize,nvdec,correctness"
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument(
        "-o", "--output", type=Path, default=None, help="write raw results JSON here"
    )
    args = p.parse_args()

    if not args.skip_prepare:
        prepare_media(args.media_dir)

    libavcodec = _verify_same_ffmpeg(args.media_dir)

    results = {
        "env": {
            "torch": torch.__version__,
            "torchcodec": torchcodec.__version__,
            "avtensor": importlib.metadata.version("avtensor"),
            "ffmpeg": _ffmpeg_version(),
            "libavcodec_loaded_by_both": libavcodec,
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "single": {},
        "nvdec": {},
    }
    todo = args.scenarios.split(",")
    if "single" in todo:
        bench_single(args.media_dir, args.repeats, results)
    if "concurrency" in todo:
        bench_concurrency(args.media_dir, results)
    if "concurrency_resize" in todo:
        bench_concurrency_resize(args.media_dir, results)
    if "nvdec" in todo:
        bench_nvdec(args.media_dir, args.repeats, results)
    if "correctness" in todo:
        check_correctness(args.media_dir, results)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"raw results -> {args.output}")
    print("\n" + markdown_report(results))


if __name__ == "__main__":
    main()
