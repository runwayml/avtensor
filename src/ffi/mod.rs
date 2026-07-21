use log::LevelFilter;
/// Python interface for the avtensor crate.
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
mod tensor;
use anyhow::Context;
use pyo3_log::Logger;
use rsmpeg::ffi::{self, av_log_set_level};
pub use tensor::PyTensor;

use crate::decoder::{self, decode_media, DecodedStream, Frame};

/// Different stream types that can be decoded.
#[pyclass(eq, eq_int)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StreamType {
    /// Video stream type
    Video,
    /// Audio stream type
    Audio, // TODO: Data / caption streams
}

impl From<decoder::StreamType> for StreamType {
    fn from(stream_type: decoder::StreamType) -> Self {
        match stream_type {
            decoder::StreamType::Video => StreamType::Video,
            decoder::StreamType::Audio => StreamType::Audio,
        }
    }
}

/// A request to decode a video stream.
#[pyclass(eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct VideoStreamRequest {
    /// The index of the video stream in the media file.
    ///
    /// If the index is not provided this will default to the first video stream in the media file.
    #[pyo3(get, set)]
    index: Option<usize>,
    /// The desired width of the video frames to decode.
    ///
    /// If None is provided then the width of the frames returned will correspond to the original width.
    #[pyo3(get, set)]
    width: Option<u32>,
    /// The desired height of the video frames to decode.
    ///
    /// If None is provided then the height of the frames returned will correspond to the orginal height.
    #[pyo3(get, set)]
    height: Option<u32>,
    /// The desired frame rate of the video stream to decode, this will cause video frames to be subsampled, or duplicated depending
    /// on the frame rate of the original video stream.
    ///
    /// If the frame rate is not provided this will default to the frame rate of the video stream.
    #[pyo3(get, set)]
    fps: Option<f64>,
    /// The number of FFmpeg threads to use for decoding.
    ///
    /// Defaults to 1, which is the right choice when many decodes run
    /// concurrently (e.g. a threaded data loader). Set to 0 to let FFmpeg
    /// pick automatically, which is faster for a single decode.
    #[pyo3(get, set)]
    number_of_threads: Option<usize>,
    /// Decode on GPU hardware (NVDEC via FFmpeg's `*_cuvid` decoders).
    ///
    /// Fails if the FFmpeg build has no hardware decoder for the stream's
    /// codec or no GPU is available.
    #[pyo3(get, set)]
    hardware_acceleration: Option<bool>,
    /// Dimension order of the returned video tensor: "NCHW" (the default,
    /// `[T, C, H, W]`, a non-contiguous view) or "NHWC" (`[T, H, W, C]`,
    /// contiguous — frames are decoded in this layout, so no permute is
    /// applied).
    #[pyo3(get, set)]
    dimension_order: Option<String>,
    /// Keep decoded frames on the GPU: "cuda" or "cuda:N". Requires
    /// `hardware_acceleration=True`. The returned tensor is CUDA-resident
    /// (NV12 -> RGB conversion runs on the GPU via NPP); frames are never
    /// copied to system memory. `width`/`height` must both be set to an
    /// even, strictly-downscaled size (NVDEC's scaler) or both unset; `fps`
    /// and HDR tone mapping are not supported on this path.
    #[pyo3(get, set)]
    device: Option<String>,
    /// Element type of the decoded video tensor: "uint8" (the default) or
    /// "float32". With "float32", frames are converted to 16-bit RGB by
    /// FFmpeg and returned as float32 in [0, 1], preserving the full depth
    /// of 10/12-bit sources instead of quantizing them to 8 bits.
    #[pyo3(get, set)]
    dtype: Option<String>,
    /// HDR handling for PQ/HLG or wide-gamut sources: "tonemap" (the
    /// default) tone maps to an SDR BT.709 preview; "raw" preserves the
    /// source's code values — YUV→RGB uses the tagged matrix/range only
    /// and the transfer function is left untouched. Use "raw" whenever the
    /// consumer needs the actual HDR signal (training on PQ masters,
    /// colorimetric measurement); the tone-mapped default is display-
    /// oriented and substantially alters both luminance and chroma.
    #[pyo3(get, set)]
    hdr_mode: Option<String>,
}

impl VideoStreamRequest {
    /// Whether the caller asked for NHWC output; errors on any value other
    /// than "NCHW"/"NHWC".
    fn wants_nhwc(&self) -> Result<bool, anyhow::Error> {
        match self.dimension_order.as_deref() {
            None | Some("NCHW") => Ok(false),
            Some("NHWC") => Ok(true),
            Some(other) => Err(anyhow::anyhow!(
                "dimension_order must be \"NCHW\" or \"NHWC\", got {other:?}"
            )),
        }
    }
}

#[pymethods]
impl VideoStreamRequest {
    #[new]
    #[pyo3(signature = (*, index=None, width=None, height=None, fps=None, number_of_threads=None, hardware_acceleration=None, dimension_order=None, device=None, dtype=None, hdr_mode=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        index: Option<usize>,
        width: Option<u32>,
        height: Option<u32>,
        fps: Option<f64>,
        number_of_threads: Option<usize>,
        hardware_acceleration: Option<bool>,
        dimension_order: Option<String>,
        device: Option<String>,
        dtype: Option<String>,
        hdr_mode: Option<String>,
    ) -> Self {
        VideoStreamRequest {
            index,
            width,
            height,
            fps,
            number_of_threads,
            hardware_acceleration,
            dimension_order,
            device,
            dtype,
            hdr_mode,
        }
    }
}

impl VideoStreamRequest {
    /// Parses `device` into a CUDA ordinal.
    fn device_ordinal(&self) -> Result<Option<i32>, anyhow::Error> {
        match self.device.as_deref() {
            None => Ok(None),
            Some("cuda") => Ok(Some(0)),
            Some(s) => match s.strip_prefix("cuda:").map(str::parse) {
                Some(Ok(ordinal)) => Ok(Some(ordinal)),
                _ => Err(anyhow::anyhow!(
                    "device must be \"cuda\" or \"cuda:N\", got {s:?}"
                )),
            },
        }
    }

    /// Parses `dtype` into the decoder enum.
    fn dtype_parsed(&self) -> Result<decoder::OutputDtype, anyhow::Error> {
        match self.dtype.as_deref() {
            None | Some("uint8") => Ok(decoder::OutputDtype::Uint8),
            Some("float32") => Ok(decoder::OutputDtype::Float32),
            Some(other) => Err(anyhow::anyhow!(
                "dtype must be \"uint8\" or \"float32\", got {other:?}"
            )),
        }
    }

    fn to_decoder_request(&self) -> Result<decoder::VideoStreamRequest, anyhow::Error> {
        Ok(decoder::VideoStreamRequest {
            index: self.index,
            width: self.width,
            height: self.height,
            frame_rate: self.fps,
            number_of_threads: self.number_of_threads,
            hardware_acceleration: self.hardware_acceleration,
            device: self.device_ordinal()?,
            dtype: self.dtype_parsed()?,
            hdr_mode: decoder::HdrMode::parse(self.hdr_mode.as_deref())?,
        })
    }
}

#[pyclass(eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessNormalization {
    /// Integrated loudness target. Range is -70.0 - -5.0. Default value is -24.0.
    #[pyo3(get, set)]
    integrated_loudness_target: Option<f32>,
    /// loudness range target. Range is 1.0 - 50.0. Default value is 7.0.
    #[pyo3(get, set)]
    loudness_range_target: Option<f32>,
    /// true peak level. Default value is -2.0.
    #[pyo3(get, set)]
    true_peak_level_target: Option<f32>,
    /// Measured integrated loudness of the input audio. Range is -99.0 - +0.0.
    #[pyo3(get, set)]
    measured_integrated_loudness: Option<f32>,
    /// Measured loudness range of input file. Range is 0.0 - 99.0.
    #[pyo3(get, set)]
    measured_loudness_range: Option<f32>,
    /// Measured true peak level of input file. Range is -99.0 - +99.0
    #[pyo3(get, set)]
    measured_true_peak_level: Option<f32>,
    /// Measured threshold of input file. Range is -99.0 - +0.
    #[pyo3(get, set)]
    measured_threshold: Option<f32>,
    /// Offset gain to apply to input audio. Gain is applied before the true-peak limiter. Range is -99.0 - +99.0. Default is +0.0.
    #[pyo3(get, set)]
    offset_gain: Option<f32>,
    /// Normalize by linearly scaling the source audio. measured_integrated_loudness, measured_loudness_range, measured_true_peak_level, and measured_threshold must all be specified. `loudness_range_target` shouldn’t be lower than source LRA and the change in integrated loudness shouldn’t result in a true peak which exceeds the target TP. If any of these conditions aren’t met, normalization mode will revert to dynamic. Options are true or false. Default is true.
    #[pyo3(get, set)]
    linear: Option<bool>,
    /// Treat mono input files as "dual-mono". If a mono file is intended for playback on a stereo system, its EBU R128 measurement will be perceptually incorrect. If set to true, this option will compensate for this effect. Multi-channel input files are not affected by this option. Options are true or false. Default is false.
    #[pyo3(get, set)]
    dual_mono: Option<bool>,
}

#[pymethods]
impl LoudnessNormalization {
    #[new]
    #[pyo3(signature = (*, integrated_loudness_target=None, loudness_range_target=None, true_peak_level_target=None, measured_integrated_loudness=None, measured_loudness_range=None, measured_true_peak_level=None, measured_threshold=None, offset_gain=None, linear=None, dual_mono=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        integrated_loudness_target: Option<f32>,
        loudness_range_target: Option<f32>,
        true_peak_level_target: Option<f32>,
        measured_integrated_loudness: Option<f32>,
        measured_loudness_range: Option<f32>,
        measured_true_peak_level: Option<f32>,
        measured_threshold: Option<f32>,
        offset_gain: Option<f32>,
        linear: Option<bool>,
        dual_mono: Option<bool>,
    ) -> Self {
        LoudnessNormalization {
            integrated_loudness_target,
            loudness_range_target,
            true_peak_level_target,
            measured_integrated_loudness,
            measured_loudness_range,
            measured_true_peak_level,
            measured_threshold,
            offset_gain,
            linear,
            dual_mono,
        }
    }
}

impl From<LoudnessNormalization> for decoder::LoudnessNormalization {
    fn from(value: LoudnessNormalization) -> Self {
        decoder::LoudnessNormalization {
            integrated_loudness_target: value.integrated_loudness_target,
            loudness_range_target: value.loudness_range_target,
            true_peak_level_target: value.true_peak_level_target,
            measured_integrated_loudness: value.measured_integrated_loudness,
            measured_loudness_range: value.measured_loudness_range,
            measured_true_peak_level: value.measured_true_peak_level,
            measured_threshold: value.measured_threshold,
            offset_gain: value.offset_gain,
            linear: value.linear,
            dual_mono: value.dual_mono,
        }
    }
}

/// A request to decode an audio stream.
///
/// Nested request objects are held by reference: mutating
/// `request.loudness_normalization` after assignment is observed by
/// `decode_asset`.
#[pyclass(eq)]
#[derive(Debug)]
pub struct AudioStreamRequest {
    /// The index of the audio stream in the media file.
    ///
    /// If the index is not provided this will default to the first audio stream in the media file.
    #[pyo3(get, set)]
    index: Option<usize>,
    /// The sample rate of the audio stream to decode. If this is different than the original sample rate of the audio stream, then the audio will be resampled.
    #[pyo3(get, set)]
    sample_rate: Option<u32>,
    /// Loudness normalization configuration.
    #[pyo3(get, set)]
    loudness_normalization: Option<Py<LoudnessNormalization>>,
}

impl PartialEq for AudioStreamRequest {
    fn eq(&self, other: &Self) -> bool {
        if self.index != other.index || self.sample_rate != other.sample_rate {
            return false;
        }
        match (&self.loudness_normalization, &other.loudness_normalization) {
            (None, None) => true,
            (Some(l), Some(r)) => Python::with_gil(|py| *l.borrow(py) == *r.borrow(py)),
            _ => false,
        }
    }
}

#[pymethods]
impl AudioStreamRequest {
    #[new]
    #[pyo3(signature = (*, index=None, sample_rate=None, loudness_normalization=None))]
    pub fn py_new(
        index: Option<usize>,
        sample_rate: Option<u32>,
        loudness_normalization: Option<Py<LoudnessNormalization>>,
    ) -> Self {
        AudioStreamRequest {
            index,
            sample_rate,
            loudness_normalization,
        }
    }
}

impl AudioStreamRequest {
    fn to_decoder_request(&self, py: Python<'_>) -> decoder::AudioStreamRequest {
        decoder::AudioStreamRequest {
            index: self.index,
            sample_rate: self.sample_rate,
            loudness_normalization: self
                .loudness_normalization
                .as_ref()
                .map(|ln| ln.borrow(py).clone().into()),
        }
    }
}

#[derive(FromPyObject, IntoPyObject, Clone)]
pub enum MediaInput {
    /// Input is a URI to a media file.
    Uri(String),
    /// Input is raw bytes of media data
    Bytes(Vec<u8>), // N.b. this conversion requires copying the input when calling from Python -> Rust, we can't borrow without holding the GIL.
}

/// A request to decode media from a given input URI.
///
/// The stream request objects are held by reference: mutating
/// `request.video_stream` or an element of `request.audio_streams` after
/// assignment is observed by `decode_asset`.
#[pyclass]
pub struct MediaDecodeRequest {
    /// The URI of the media to decode.
    #[pyo3(get)]
    input: MediaInput,
    /// The start time in the presentation timeline to start yielding decoded frames/samples from, in seconds.
    #[pyo3(get, set)]
    start_time: Option<f64>,
    /// The end time in the presentation timeline to stop yielding decoded frames/samples from, in seconds.
    #[pyo3(get, set)]
    end_time: Option<f64>,
    /// The video stream to decode (if any).
    #[pyo3(get, set)]
    video_stream: Option<Py<VideoStreamRequest>>,
    /// The audio streams to decode (if any).
    #[pyo3(get, set)]
    audio_streams: Option<Vec<Py<AudioStreamRequest>>>,
}

#[pymethods]
impl MediaDecodeRequest {
    #[new]
    #[pyo3(signature = (input, *, start_time=None, end_time=None, video_stream=None, audio_streams=None))]
    pub fn py_new(
        input: MediaInput,
        start_time: Option<f64>,
        end_time: Option<f64>,
        video_stream: Option<Py<VideoStreamRequest>>,
        audio_streams: Option<Vec<Py<AudioStreamRequest>>>,
    ) -> Self {
        MediaDecodeRequest {
            input,
            start_time,
            end_time,
            video_stream,
            audio_streams,
        }
    }
}

impl MediaDecodeRequest {
    /// Snapshots the request into plain Rust structs.
    ///
    /// `decode_asset` calls this under the GIL before decoding starts, so an
    /// in-flight decode can never observe Python-side mutation — the nested
    /// request objects being held by reference (for natural Python attribute
    /// semantics) does not weaken that guarantee.
    fn to_decoder_request(
        &self,
        py: Python<'_>,
    ) -> Result<decoder::MediaDecodeRequest, anyhow::Error> {
        let source = match &self.input {
            MediaInput::Uri(uri) => decoder::MediaSource::Uri(uri.clone()),
            MediaInput::Bytes(bytes) => decoder::MediaSource::Bytes(bytes.clone()),
        };

        Ok(decoder::MediaDecodeRequest {
            source,
            start_time: self.start_time,
            end_time: self.end_time,
            video_stream: self
                .video_stream
                .as_ref()
                .map(|v| v.borrow(py).to_decoder_request())
                .transpose()?,
            audio_streams: self.audio_streams.as_ref().map(|streams| {
                streams
                    .iter()
                    .map(|s| s.borrow(py).to_decoder_request(py))
                    .collect()
            }),
        })
    }
}

/// Explicit configuration for avtensor's S3 client.
///
/// Fields set here are authoritative — the environment is not consulted for
/// them; unset fields fall back to the standard AWS environment. Requests
/// with distinct configs get distinct cached clients, so one process can
/// read from several S3-compatible stores (with different endpoints and
/// credentials) at the same time.
// No Debug/__repr__: `secret_access_key`/`session_token` must not leak into
// logs or tracebacks.
#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub struct S3Config {
    /// Endpoint URL of the store.
    #[pyo3(get, set)]
    endpoint_url: Option<String>,
    /// Signing region.
    #[pyo3(get, set)]
    region: Option<String>,
    /// Static credentials; must be set together with `secret_access_key`.
    #[pyo3(get, set)]
    access_key_id: Option<String>,
    #[pyo3(get, set)]
    secret_access_key: Option<String>,
    /// Session token accompanying the static credentials.
    #[pyo3(get, set)]
    session_token: Option<String>,
    /// Credentials mode: "default" (the standard provider chain) or
    /// "container" (container credentials exclusively). Mutually exclusive
    /// with static credentials.
    #[pyo3(get, set)]
    credentials: Option<String>,
    /// Use path-style addressing (required by MinIO and some other stores).
    #[pyo3(get, set)]
    force_path_style: Option<bool>,
}

#[pymethods]
impl S3Config {
    #[new]
    #[pyo3(signature = (*, endpoint_url=None, region=None, access_key_id=None, secret_access_key=None, session_token=None, credentials=None, force_path_style=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        endpoint_url: Option<String>,
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        credentials: Option<String>,
        force_path_style: Option<bool>,
    ) -> Self {
        S3Config {
            endpoint_url,
            region,
            access_key_id,
            secret_access_key,
            session_token,
            credentials,
            force_path_style,
        }
    }
}

impl S3Config {
    /// Snapshots the config into the decoder-facing struct.
    fn to_decoder_config(&self) -> crate::util::s3::S3Config {
        crate::util::s3::S3Config {
            endpoint_url: self.endpoint_url.clone(),
            region: self.region.clone(),
            access_key_id: self.access_key_id.clone(),
            secret_access_key: self.secret_access_key.clone(),
            session_token: self.session_token.clone(),
            credentials: self.credentials.clone(),
            force_path_style: self.force_path_style,
        }
    }
}

/// Metadata for an audio stream.
#[derive(IntoPyObject, Debug, Clone)]
pub struct AudioStreamMetadata {
    /// The index of the audio stream in the media container.
    index: usize,
    /// The sample rate of the audio stream.
    sample_rate: usize,
}

/// Metadata for a video stream.
#[derive(IntoPyObject, Debug, Clone)]
pub struct VideoStreamMetadata {
    /// The index of the video stream in the media container.
    index: usize,
    /// The width of the video stream in pixels.
    width: usize,
    /// The height of the video stream in pixels.
    height: usize,
    /// The average frame rate of the video stream.
    fps: f64,
}

#[derive(IntoPyObject, Debug, Clone)]
pub struct MediaMetadata {
    /// The video streams in the media container.
    video_streams: Vec<VideoStreamMetadata>,
    /// The audio streams in the media container.
    audio_streams: Vec<AudioStreamMetadata>,
}

/// Output of a decoding operation
#[derive(IntoPyObject)]
pub enum DecodeResult {
    Video {
        /// The decoded data.
        data: PyTensor,
        /// The type of the stream that was decoded.
        stream_type: StreamType,
        /// The index of the stream that was decoded.
        stream_index: usize,
        /// The frame rate
        fps: f64,
        /// Presentation timestamp (seconds) of each frame in `data`, as a
        /// float64 Tensor of shape [T].
        pts: PyTensor,
    },
    Audio {
        /// The decoded data.
        data: PyTensor,
        /// The type of the stream that was decoded.
        stream_type: StreamType,
        /// The index of the stream that was decoded.
        stream_index: usize,
        /// The sample rate
        sample_rate: usize,
    },
}

impl DecodeResult {
    /// Converts a decoded stream into the Python-facing result. uint8 video
    /// is decoded in `[T, H, W, C]` layout (NHWC contiguous, NCHW a view);
    /// float32 video is decoded planar, `[T, C, H, W]` (NCHW contiguous,
    /// NHWC a view). The permutes below make `dimension_order` hold either
    /// way.
    fn from_stream(stream: DecodedStream, nhwc: bool) -> Result<Self, anyhow::Error> {
        let stream_type = stream.stream_type().into();
        Ok(match stream {
            DecodedStream {
                src_stream_index: stream_index,
                metadata: decoder::StreamMetadata::Video { frame_rate },
                data,
                decoded_frames,
                frame_pts,
            } => DecodeResult::Video {
                data: PyTensor(
                    data.map(|d| {
                        let channels_first = d.kind() == tch::Kind::Float;
                        match (channels_first, nhwc) {
                            (true, false) | (false, true) => Ok(d),
                            (true, true) => d
                                .f_permute([0, 2, 3, 1])
                                .context("Failed to permute video tensor"), // [T, C, H, W] -> [T, H, W, C]
                            (false, false) => d
                                .f_permute([0, 3, 1, 2])
                                .context("Failed to permute video tensor"), // [T, H, W, C] -> [T, C, H, W]
                        }
                    })
                    .unwrap_or_else(|| {
                        // Video Frames are captured in [T, C, H, W] format, concat along the the time dimension.
                        let data = decoded_frames
                            .into_iter()
                            .map(|frame| match frame {
                                Frame::Video { data, .. } => Ok(data),
                                Frame::Audio { .. } => {
                                    Err(anyhow::anyhow!("Expected video frame, got audio frame"))
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        tch::Tensor::f_cat(&data, 0)
                            .context("Failed to stack video frames into Tensor")
                    })?,
                ),
                stream_type,
                stream_index,
                fps: frame_rate,
                pts: PyTensor(tch::Tensor::from_slice(&frame_pts)),
            },
            DecodedStream {
                src_stream_index: stream_index,
                metadata: decoder::StreamMetadata::Audio { sample_rate },
                data,
                decoded_frames,
                frame_pts: _,
            } => DecodeResult::Audio {
                data: PyTensor(data.map(Ok).unwrap_or_else(|| {
                    // Audio Frames are captured in [C, T] format, concat along the temporal dimension.
                    let data = decoded_frames
                        .into_iter()
                        .map(|frame| match frame {
                            Frame::Audio { data, .. } => Ok(data),
                            Frame::Video { .. } => {
                                Err(anyhow::anyhow!("Expected audio frame, got video frame"))
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    tch::Tensor::f_cat(&data, 1).context("Failed to stack audio frames into Tensor")
                })?),
                stream_type,
                stream_index,
                sample_rate: sample_rate as usize,
            },
        })
    }
}

#[pyfunction]
#[pyo3(signature = (input, *, s3_config=None))]
fn probe_asset(
    py: Python<'_>,
    input: MediaInput,
    s3_config: Option<PyRef<'_, S3Config>>,
) -> PyResult<MediaMetadata> {
    let source = match input {
        MediaInput::Uri(uri) => decoder::MediaSource::Uri(uri),
        MediaInput::Bytes(bytes) => decoder::MediaSource::Bytes(bytes),
    };
    let s3_config = s3_config.map(|c| c.to_decoder_config());

    // Release the GIL while probing (it may perform network I/O).
    let probed = py
        .allow_threads(move || decoder::probe_media(source, s3_config))
        .map_err(|e| PyValueError::new_err(format!("Failed to probe media: {:?}", e)))?;

    Ok(MediaMetadata {
        video_streams: probed
            .video_streams
            .into_iter()
            .map(|s| VideoStreamMetadata {
                index: s.index,
                width: s.width.max(0) as usize,
                height: s.height.max(0) as usize,
                fps: s.fps,
            })
            .collect(),
        audio_streams: probed
            .audio_streams
            .into_iter()
            .map(|s| AudioStreamMetadata {
                index: s.index,
                sample_rate: s.sample_rate.max(0) as usize,
            })
            .collect(),
    })
}

#[pyfunction]
#[pyo3(signature = (request, *, s3_config=None))]
fn decode_asset(
    py: Python<'_>,
    request: PyRef<'_, MediaDecodeRequest>,
    s3_config: Option<PyRef<'_, S3Config>>,
) -> PyResult<Vec<DecodeResult>> {
    let decoder_request = request
        .to_decoder_request(py)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert request: {:?}", e)))?;
    // Snapshot into a plain Rust struct before releasing the GIL.
    let s3_config = s3_config.map(|c| c.to_decoder_config());
    let nhwc = match &request.video_stream {
        Some(v) => v
            .borrow(py)
            .wants_nhwc()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?,
        None => false,
    };
    // Release the borrow on the Python object before dropping the GIL.
    drop(request);
    let request = decoder_request;

    // Release the GIL while decoding.
    py.allow_threads(move || {
        // Decode media.
        let decoded_streams = decode_media(request, s3_config)
            .map_err(|e| PyValueError::new_err(format!("Failed to decode media: {:?}", e)))?
            .into_iter()
            .inspect(|stream| {
                log::debug!(
                    "Decoded stream: {:?}, type: {:?}, index: {}",
                    stream.src_stream_index,
                    stream.stream_type(),
                    stream.src_stream_index
                );
            })
            .map(|stream| {
                DecodeResult::from_stream(stream, nhwc).map_err(|e| {
                    PyValueError::new_err(format!("Failed to convert decoded stream: {:?}", e))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        log::debug!(
            "Decoded {} streams, returning results to Python",
            decoded_streams.len()
        );
        Ok(decoded_streams)
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn avtensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // This extension links against torch's shared libraries; importing torch
    // here guarantees they are loaded even when the user imports avtensor
    // first, which would otherwise crash on the first tensor conversion.
    m.py().import("torch")?;

    // Configure the avtensor crate to only forward error logs to Python.
    Logger::default()
        .filter(LevelFilter::Error)
        .install()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize logging: {}", e)))?;

    m.add_class::<VideoStreamRequest>()?;
    m.add_class::<AudioStreamRequest>()?;
    m.add_class::<MediaDecodeRequest>()?;
    m.add_class::<S3Config>()?;
    m.add_class::<StreamType>()?;
    m.add_class::<LoudnessNormalization>()?;
    m.add_function(wrap_pyfunction!(probe_asset, m)?)?;
    m.add_function(wrap_pyfunction!(decode_asset, m)?)?;

    // Configure FFmpeg to only log error messages to stderr
    // to avoid unwanted output when this is being used from Python.
    unsafe {
        av_log_set_level(ffi::AV_LOG_ERROR as i32);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //use crate::util::{generate_test_video_file, TestVideoParameters};

    // (rikheijdens): This test is disabled because pyo3::prepare_freethreaded_python() causes the test runner to exit with SIGSEGV on shutdown.
    // #[test]
    // fn test_decode_asset() -> anyhow::Result<()> {
    //     pyo3::prepare_freethreaded_python();

    //     let test_video = generate_test_video_file(TestVideoParameters::default())?;
    //     Python::with_gil(|py| {
    //         let request = MediaDecodeRequest::py_new(MediaInput::Uri(
    //             test_video.path().to_str().unwrap().into(),
    //         ));
    //         let result = decode_asset(py, request);
    //         assert!(result.is_ok());
    //     });

    //     Ok(())
    // }
}
