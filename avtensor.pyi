import sys
from enum import Enum
from typing import TypedDict

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
    # FFmpeg decoder threads. Defaults to 1 (best when many decodes run
    # concurrently); 0 lets FFmpeg pick automatically (faster single decode).
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
    hdr_mode: str | None

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
        hdr_mode: str | None = None,
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

    def __init__(
        self,
        input: str | bytes,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        video_stream: VideoStreamRequest | None = None,
        audio_streams: list[AudioStreamRequest] | None = None,
    ): ...

class AudioStreamMetadata(TypedDict):
    index: int
    sample_rate: int

class VideoStreamMetadata(TypedDict):
    index: int
    width: int
    height: int
    fps: float

class MediaMetadata(TypedDict):
    video_streams: list[VideoStreamMetadata]
    audio_streams: list[AudioStreamMetadata]

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
