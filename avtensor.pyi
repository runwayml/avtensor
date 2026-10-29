import sys
from enum import Enum
from typing import Literal, TypedDict

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
else:
    from typing import NotRequired
import torch
from typing_extensions import NotRequired

class StreamType(Enum):
    Video = 0
    Audio = 1

class VideoStreamRequest:
    index: int | None
    width: int | None
    height: int | None
    fps: float | None
    # Threads for FFmpeg decoding. Defaults to 1 (best when many decodes run
    # concurrently); 0 picks automatically (faster single decode). Also sizes
    # the RGB conversion on the direct conversion path (README: software
    # decode, rgb24, no resize/fps, width a multiple of 32, SDR or supported
    # raw HDR); other decodes use the filter graph's pool of up to 16 threads.
    number_of_threads: int | None
    hardware_acceleration: bool | None
    # "NCHW" (default, [T, C, H, W], a non-contiguous view) or "NHWC"
    # ([T, H, W, C], contiguous).
    dimension_order: str | None
    # "cuda" or "cuda:N": keep decoded frames on the GPU. Implies
    # hardware_acceleration (an explicit False is an error). The returned
    # tensor is CUDA-resident; frames are never copied to system memory.
    device: str | None
    # "uint8" (default) or "float32". float32 decodes to planar float in
    # [0, 1] (NCHW-contiguous), preserving the depth of 10/12-bit sources.
    dtype: str | None
    # HDR handling for PQ/HLG or wide-gamut sources: "tonemap" (default)
    # tone maps to an SDR BT.709 preview; "raw" preserves the source's code
    # values (tagged matrix/range only — transfer function untouched). Use
    # "raw" when you need the actual HDR signal, e.g. training on PQ
    # masters or colorimetric measurement.
    hdr_mode: Literal["tonemap", "raw"] | None
    # YUV->RGB conversion: "fast" (libswscale's default path) or "accurate"
    # (zimg: exact float output, round-to-nearest uint8). Defaults to "fast"
    # for uint8 (bit-identical to earlier releases) and "accurate" for
    # float32 (falling back to "fast" with a warning when FFmpeg lacks
    # zscale/libzimg). The fast path biases uint8 low by ~0.5-1.4 code values
    # per channel on 4:2:0 sources and scales float32 by 255/256. "accurate"
    # is not supported with device="cuda".
    color_conversion: Literal["fast", "accurate"] | None
    # How `fps` resampling builds each output frame: "nearest" (default)
    # keeps the nearest source frame (uneven spacing for e.g. 30 -> 24 fps);
    # "blend" cross-fades neighbouring frames; "motion" interpolates along
    # estimated block motion (about 1.2-1.6x the cost of "nearest").
    fps_conversion: Literal["nearest", "blend", "motion"] | None
    # Apply the stream's display-matrix rotation as the ffmpeg CLI's
    # -autorotate does (90/270 degrees swap width and height). width/height
    # refer to the rotated frame. Default False: frames come out as coded.
    # Stream-level matrix only: orientation carried per frame (JPEG EXIF
    # Orientation, H.264/HEVC display-orientation SEI) is not applied, unlike
    # the CLI. Not supported with device="cuda" on a rotated stream.
    autorotate: bool | None
    # "filter" (default): `fps` resamples with FFmpeg's fps filter, or every
    # decoded frame is returned. "cli": the ffmpeg CLI's constant-frame-rate
    # output at the stream's guessed rate (frames duplicated/dropped by
    # timestamp drift, as `ffmpeg -i in -f rawvideo -` emits them), including
    # its timestamp-discontinuity correction for MPEG-TS/PS/HLS. "cli"
    # requires fps unset and a whole-file decode, and is not supported with
    # device or hardware_acceleration=True. Differences from the CLI: it
    # autorotates only with autorotate=True (the CLI rotates by the display
    # matrix by default), and plan_asset is an estimate, not an upper bound
    # (the output grows past it, up to the output size limit).
    fps_mode: Literal["filter", "cli"] | None

    def __init__(
        self,
        *,
        index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        number_of_threads: int | None = None,
        hardware_acceleration: bool | None = None,
        dimension_order: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        hdr_mode: Literal["tonemap", "raw"] | None = None,
        color_conversion: Literal["fast", "accurate"] | None = None,
        fps_conversion: Literal["nearest", "blend", "motion"] | None = None,
        autorotate: bool | None = None,
        fps_mode: Literal["filter", "cli"] | None = None,
    ): ...

class LoudnessNormalization:
    integrated_loudness_target: float | None
    loudness_range_target: float | None
    true_peak_level_target: float | None
    measured_integrated_loudness: float | None
    measured_loudness_range: float | None
    measured_true_peak_level: float | None
    measured_threshold: float | None
    offset_gain: float | None
    linear: bool | None
    dual_mono: bool | None

    def __init__(
        self,
        *,
        integrated_loudness_target: float | None = None,
        loudness_range_target: float | None = None,
        true_peak_level_target: float | None = None,
        measured_integrated_loudness: float | None = None,
        measured_loudness_range: float | None = None,
        measured_true_peak_level: float | None = None,
        measured_threshold: float | None = None,
        offset_gain: float | None = None,
        linear: bool | None = None,
        dual_mono: bool | None = None,
    ): ...

class AudioStreamRequest:
    index: int | None
    sample_rate: int | None
    loudness_normalization: LoudnessNormalization | None

    def __init__(
        self,
        *,
        index: int | None = None,
        sample_rate: int | None = None,
        loudness_normalization: LoudnessNormalization | None = None,
    ): ...

class S3Config:
    endpoint_url: str | None
    region: str | None
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None
    credentials: str | None
    force_path_style: bool | None

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        credentials: str | None = None,
        force_path_style: bool | None = None,
    ): ...

class MediaDecodeRequest:
    input: str | bytes
    start_time: float | None
    end_time: float | None
    video_stream: VideoStreamRequest | None
    audio_streams: list[AudioStreamRequest] | None
    # Largest output tensor (bytes) each stream may allocate. Default:
    # AVTENSOR_MAX_OUTPUT_TENSOR_BYTES when set, else 16 GiB. Per stream, so
    # video plus N audio streams can total (1 + N)x it; 0 refuses every
    # non-empty stream. It bounds buffers sized from file metadata, which an
    # untrusted file controls: raise it per request rather than process-wide.
    max_output_bytes: int | None

    def __init__(
        self,
        input: str | bytes,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        video_stream: VideoStreamRequest | None = None,
        audio_streams: list[AudioStreamRequest] | None = None,
        max_output_bytes: int | None = None,
    ): ...

DurationSource = Literal["stream", "container", "none"]

class AudioStreamMetadata(TypedDict):
    index: int
    sample_rate: int
    channels: int
    codec_name: str
    # Stream duration in seconds, falling back to the container's (Matroska/
    # WebM streams report none).
    duration_s: float | None
    duration_source: DurationSource

class VideoStreamMetadata(TypedDict):
    index: int
    # Coded dimensions: what the decoder outputs unless a size is requested
    # or autorotate turns the frame by 90/270 degrees (swapping them).
    width: int
    height: int
    fps: float
    codec_name: str
    duration_s: float | None
    duration_source: DurationSource
    # Frame count from the container header, if it stores one.
    nb_frames: int | None
    # Stream display-matrix rotation in degrees (counterclockwise, ffprobe's
    # `rotation`); applied only with VideoStreamRequest(autorotate=True).
    # None when the orientation is carried per frame (JPEG EXIF Orientation,
    # H.264/HEVC display-orientation SEI), which is not applied either.
    rotation: float | None
    # Sample aspect ratio (num, den), (0, 1) when unknown. The decoder does
    # not apply it.
    sample_aspect_ratio: tuple[int, int]

class MediaMetadata(TypedDict):
    # Demuxer name(s), e.g. "mov,mp4,m4a,3gp,3g2,mj2" or "matroska,webm".
    format_name: str
    duration_s: float | None
    video_streams: list[VideoStreamMetadata]
    audio_streams: list[AudioStreamMetadata]

class PlannedStream(TypedDict):
    stream_type: StreamType
    stream_index: int
    # Shape decode_asset allocates; the decoded tensor is trimmed to what was
    # actually decoded, so this is an upper bound, except for video with
    # fps_mode="cli", whose output grows past it when the CLI emits more
    # frames (e.g. files with no stored duration plan 1 frame). Video is in
    # the requested dimension_order; audio is [C, T].
    shape: list[int]
    dtype: Literal["uint8", "float32"]
    # Only set for Video Streams
    fps: NotRequired[float]
    # Only set for Audio Streams
    sample_rate: NotRequired[int]
    # Duration being decoded after the seek window (None for a still image).
    duration_s: float | None
    duration_source: DurationSource

class DecodeResult(TypedDict):
    # Decoded frame data
    data: torch.Tensor
    # Stream Type
    stream_type: StreamType
    # Index of the stream in the media container
    stream_index: int
    # Sample Rate - only set for Audio Streams
    sample_rate: NotRequired[int]
    # Frame Rate - only set for Video Streams
    fps: NotRequired[float]
    # Presentation timestamp (seconds) of each frame, as a float64 Tensor of
    # shape [T] - only set for Video Streams
    pts: NotRequired[torch.Tensor]

def decode_asset(
    request: MediaDecodeRequest, *, s3_config: S3Config | None = None
) -> list[DecodeResult]:
    """Decodes the requested streams from the media asset."""
    ...

def probe_asset(
    input: str | bytes, *, s3_config: S3Config | None = None
) -> MediaMetadata:
    """Probes the asset's stream layout (dimensions, frame rate, sample rate)
    without decoding it."""
    ...

def plan_asset(
    request: MediaDecodeRequest, *, s3_config: S3Config | None = None
) -> list[PlannedStream]:
    """Returns the buffers decode_asset(request) would allocate, without
    decoding: same stream selection, duration, seek window, fps and size
    handling."""
    ...

def interpolate_frames(
    prev: torch.Tensor, next: torch.Tensor, alphas: list[float]
) -> torch.Tensor:
    """Frames at phases ``alphas`` (each strictly between 0 and 1; 0 = prev,
    1 = next) between two RGB frames, by the motion-compensated
    interpolation behind ``VideoStreamRequest(fps_conversion="motion")``.

    ``prev`` / ``next``: uint8 CPU tensors of one shape, ``[3, H, W]`` or
    ``[H, W, 3]``, any strides. Returns uint8 ``[len(alphas), 3, H, W]``.
    Motion is estimated once per pair. Phases are quantised to 1/256 and
    clamped inside (0, 1): 1e-4 becomes 1/256, anything >= 255/256 becomes
    255/256. Frames pass through BT.709 limited-range yuv444p and back, so
    only rounding is lost; frames that need no interpolation (a phase of
    exactly 0 or 1) should be taken from the inputs directly rather than
    passed here. No torch op runs on the pixel data (the pixels are moved
    by plain copy loops), so the cost does not depend on torch's intra-op
    thread settings; the GIL is released for the duration."""
    ...
