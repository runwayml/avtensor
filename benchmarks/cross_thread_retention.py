"""Measure memory retained after decoded tensors are freed on another thread.

Decodes one clip repeatedly, first allocating and dropping on the main thread,
then decoding on worker threads and dropping on the main thread (the shape of a
typical data-loading pipeline), and reports the process's resident memory after
each phase from ``/proc/self/smaps_rollup``. ``LazyFree`` counts pages that were
``MADV_FREE``'d but are still resident, which is how a thread-caching allocator
keeps a cross-thread free alive.

Run it twice to compare the output allocators::

    python benchmarks/cross_thread_retention.py --clip clip.mp4
    AVTENSOR_OUTPUT_ALLOCATOR=torch python benchmarks/cross_thread_retention.py --clip clip.mp4

Without ``--clip`` a synthetic 1280x720 H.264 clip with audio is generated with
``ffmpeg`` (``$FFMPEG`` or the one on ``PATH``). Linux only (reads ``/proc``).
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import avtensor
import torch

GIB = 2**30


def smaps_rollup() -> dict[str, float]:
    """Rss / Anonymous / LazyFree of this process in GiB."""
    wanted = {"Rss", "Anonymous", "LazyFree"}
    out: dict[str, float] = dict.fromkeys(wanted, 0.0)
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in wanted:
                out[key] = int(rest.split()[0]) * 1024 / GIB
    return out


def generate_clip(path: Path, seconds: float, size: str, fps: int) -> None:
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found (set FFMPEG or PATH), or pass --clip instead")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={fps}:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=48000:duration={seconds}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def decode(clip: str, with_audio: bool) -> list[torch.Tensor]:
    """Decode the clip and return its output tensors (so the caller frees them)."""
    request = avtensor.MediaDecodeRequest(
        clip,
        video_stream=avtensor.VideoStreamRequest(),
        audio_streams=[avtensor.AudioStreamRequest()] if with_audio else None,
    )
    return [s["data"] for s in avtensor.decode_asset(request)]


def nbytes(tensors: list[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def report(
    tag: str, base: dict[str, float], times: list[float], bytes_out: int
) -> None:
    m = smaps_rollup()
    mean_ms = 1000 * sum(times) / len(times) if times else float("nan")
    print(
        f"{tag:<34} rss={m['Rss']:6.2f} anon={m['Anonymous']:6.2f} "
        f"lazyfree={m['LazyFree']:6.2f} GiB  "
        f"(+{m['Rss'] - base['Rss']:.2f} GiB over start)  "
        f"decode {mean_ms:7.1f} ms  output {bytes_out / GIB:.2f} GiB/decode",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure memory retained after cross-thread frees of decoded tensors."
    )
    parser.add_argument("--clip", type=Path, help="media file to decode")
    parser.add_argument("--decodes", type=int, default=12, help="decodes per phase")
    parser.add_argument("--threads", type=int, default=4, help="worker threads")
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="synthetic clip length"
    )
    parser.add_argument("--size", default="1280x720", help="synthetic clip size")
    parser.add_argument("--fps", type=int, default=24, help="synthetic clip frame rate")
    parser.add_argument(
        "--settle", type=float, default=0.3, help="seconds to wait after each free"
    )
    args = parser.parse_args()

    allocator = os.environ.get("AVTENSOR_OUTPUT_ALLOCATOR", "system (default)")
    print(f"AVTENSOR_OUTPUT_ALLOCATOR={allocator}", flush=True)

    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    clip = args.clip
    if clip is None:
        tmpdir = tempfile.TemporaryDirectory()
        clip = Path(tmpdir.name) / "clip.mp4"
        generate_clip(clip, args.seconds, args.size, args.fps)
    clip = str(clip)

    with_audio = bool(avtensor.probe_asset(clip)["audio_streams"])

    # Warm up (codec init, allocator arenas) so the baseline is steady.
    bytes_out = nbytes(decode(clip, with_audio))
    gc.collect()
    base = smaps_rollup()
    report("start", base, [], bytes_out)

    times: list[float] = []
    for _ in range(args.decodes):
        t0 = time.monotonic()
        tensors = decode(clip, with_audio)
        times.append(time.monotonic() - t0)
        del tensors
        time.sleep(args.settle)
    report(f"{args.decodes}x decode+free on main thread", base, times, bytes_out)

    pool = ThreadPoolExecutor(max_workers=args.threads)
    times = []
    for _ in range(args.decodes):
        t0 = time.monotonic()
        tensors = pool.submit(decode, clip, with_audio).result()
        times.append(time.monotonic() - t0)
        del tensors  # freed on the main thread, not the decoding thread
        time.sleep(args.settle)
    report(f"{args.decodes}x decode in pool, free on main", base, times, bytes_out)

    time.sleep(5.0)
    report("after 5 s idle", base, times, bytes_out)

    pool.shutdown(wait=True)
    time.sleep(0.5)
    report("after worker threads exited", base, times, bytes_out)

    if tmpdir is not None:
        tmpdir.cleanup()


if __name__ == "__main__":
    main()
