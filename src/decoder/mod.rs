use anyhow::{anyhow, bail, Context};
use rsmpeg::{
    avcodec::{AVCodec, AVCodecContext, AVPacket},
    avfilter::{AVFilter, AVFilterContextMut, AVFilterGraph, AVFilterInOut},
    avformat::AVFormatContextInput,
    avutil::{av_q2d, get_sample_fmt_name, AVChannelLayout, AVDictionary, AVFrame, AVRational},
    error::RsmpegError,
    ffi::{self, AVPixelFormat, AV_NOPTS_VALUE},
};
use std::fmt::{Debug, Display};
use std::{cmp::min, ffi::CString};
use tch::IndexOp;

use crate::{
    conversion::convert_audio_into_tensor,
    decoder::io::{cloud_storage_avio_reader, memory_avio_reader},
    util::{ffmpeg::RetUpgrade, pts_to_seconds, s3::S3Config},
};

use crate::conversion;

mod cuda;
mod io;

struct StreamContext {
    stream_type: StreamType,
    stream_index: usize,
    dec_ctx: AVCodecContext,
    filter_config: FilterConfig,
    frame_data: tch::Tensor,
    metadata: StreamMetadata,
}

struct FilteringContext<'graph> {
    stream_type: StreamType,
    stream_index: usize,
    metadata: StreamMetadata,
    dec_ctx: AVCodecContext,
    /// Filter graph endpoints; `None` when the direct swscale path is active
    /// (the graph is not even built in that case).
    buffersrc_ctx: Option<AVFilterContextMut<'graph>>,
    buffersink_ctx: Option<AVFilterContextMut<'graph>>,
    /// How decoded frames reach `frame_data`: through the filter graph, or
    /// directly when the graph would add nothing but copies.
    direct_path: DirectPath,
    /// A Tensor, allocated at the start of the media decoding process, to which
    /// filtered frames will be written.
    frame_data: tch::Tensor,
    /// The index to which any new frames should be written in `frame_data`.
    frame_data_ptr: usize,
    /// Presentation timestamp (seconds) of each frame written to
    /// `frame_data`, in write order. Video streams only.
    frame_pts: Vec<f64>,
    // Whether we should continue to route new packets through the FilteringContext.
    finished: bool,
}

struct FilterContext<'graph> {
    buffersrc_ctx: AVFilterContextMut<'graph>,
    buffersink_ctx: AVFilterContextMut<'graph>,
}

/// Direct (graph-bypassing) processing mode for a stream. When a stream's
/// only processing is something the decoder can do while writing into the
/// output tensor, the filter graph — and its per-decode slice-threading
/// worker pool — is not built at all.
enum DirectPath {
    /// Frames go through the filter graph.
    Disabled,
    /// Video frames are converted/written by libswscale directly into the
    /// output tensor.
    VideoScale(DirectVideoScaler),
    /// Audio frames are converted directly into the output tensor; used when
    /// the graph would be a no-op (`anull`).
    AudioPassthrough,
    /// NVDEC frames stay in CUDA memory and are converted NV12 -> RGB by NPP
    /// straight into the (CUDA) output tensor.
    CudaConvert(cuda::CudaNv12ToRgb),
}

pub struct DecodedStream {
    pub src_stream_index: usize,
    pub decoded_frames: Vec<Frame>,
    pub data: Option<tch::Tensor>,
    /// Presentation timestamp (seconds) for each row of `data`, in order.
    /// Populated for video streams; empty for audio.
    pub frame_pts: Vec<f64>,
    pub metadata: StreamMetadata,
}

impl DecodedStream {
    pub fn stream_type(&self) -> StreamType {
        match self.metadata {
            StreamMetadata::Video { .. } => StreamType::Video,
            StreamMetadata::Audio { .. } => StreamType::Audio,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamMetadata {
    Video { frame_rate: f64 },
    Audio { sample_rate: u32 },
}

struct DecodedFrames {
    /// Stream Index associated with the decoded frames.
    stream_index: usize,
    /// Decoded frames.
    frames: Vec<Frame>,
}

/// Element type of the decoded video tensor.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDtype {
    /// 8-bit RGB (`rgb24`), `uint8` tensors. The default.
    #[default]
    Uint8,
    /// Full-precision RGB: frames are converted to planar float
    /// (`gbrpf32le`) by FFmpeg and returned as `float32` in [0, 1],
    /// channels-first and contiguous. Preserves the extra bits of
    /// 10/12-bit sources that `Uint8` would quantize away.
    Float32,
}

#[derive(Default, Debug, Clone)]
pub struct VideoStreamRequest {
    pub index: Option<usize>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frame_rate: Option<f64>,
    /// Number of threads to use for video decoding.
    pub number_of_threads: Option<usize>,
    /// Decode on GPU hardware (NVDEC via FFmpeg's `*_cuvid` decoders).
    ///
    /// Bitstream decoding runs on the GPU's dedicated decode engine; frames
    /// are returned in system memory (unless `device` is set), so the
    /// rest of the pipeline (filters, tensor conversion) is unchanged. Fails
    /// if the FFmpeg build has no hardware decoder for the stream's codec or
    /// no GPU is available.
    pub hardware_acceleration: Option<bool>,
    /// CUDA ordinal to keep decoded frames on. Implies
    /// `hardware_acceleration` (an explicit `false` is an error); the output
    /// tensor is allocated on this device and NV12 -> RGB conversion runs on
    /// the GPU (NPP), so frames are never copied back to system memory. Only NVDEC's own downscaling
    /// is available (`width`/`height` must both be set, even, and a strict
    /// downscale — or both unset); fps resampling and HDR tone mapping are
    /// not supported on this path.
    pub device: Option<i32>,
    /// Element type of the decoded video tensor. `Float32` decodes via
    /// 16-bit RGB, preserving the full depth of >8-bit sources.
    pub dtype: OutputDtype,
}

#[derive(Default, Debug, Clone)]
pub struct AudioStreamRequest {
    pub index: Option<usize>,
    pub sample_rate: Option<u32>,
    pub loudness_normalization: Option<LoudnessNormalization>,
}

/// The media asset to decode: a URI (local path, `gs://`/`s3://`, or any URL
/// FFmpeg's protocol layer supports) or an in-memory byte buffer.
pub enum MediaSource {
    Uri(String),
    Bytes(Vec<u8>),
}

impl MediaSource {
    /// Opens an [`AVFormatContextInput`] for this source.
    ///
    /// `s3_config` is the explicit S3 client configuration for `s3://` URIs;
    /// it is ignored for other sources.
    fn open(self, s3_config: Option<S3Config>) -> Result<AVFormatContextInput, anyhow::Error> {
        match self {
            MediaSource::Uri(uri) => {
                let file_path =
                    CString::new(uri).context("Failed to create CStr from file path")?;
                cloud_storage_avio_reader(&file_path, s3_config)
            }
            MediaSource::Bytes(bytes) => memory_avio_reader(bytes),
        }
    }
}

/// A request to decode media.
pub struct MediaDecodeRequest {
    pub source: MediaSource,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub video_stream: Option<VideoStreamRequest>,
    pub audio_streams: Option<Vec<AudioStreamRequest>>,
}

struct Seek {
    start_time: Option<f64>,
    end_time: Option<f64>,
}

/// Decodes media given a [`MediaDecodeRequest`].
pub fn decode_media(
    request: MediaDecodeRequest,
    s3_config: Option<S3Config>,
) -> Result<Vec<DecodedStream>, anyhow::Error> {
    let MediaDecodeRequest {
        source,
        start_time,
        end_time,
        video_stream: video_stream_request,
        audio_streams,
    } = request;

    let mut input_format_context = source.open(s3_config)?;

    // Stream selection
    let video_stream = select_video_stream(&input_format_context, video_stream_request)?;
    let audio_streams = select_audio_streams(&input_format_context, audio_streams)?;

    // Count the number of streams that we'll be decoding.
    let num_streams_to_decode = if video_stream.is_some() { 1 } else { 0 }
        + audio_streams
            .as_ref()
            .map(|streams| streams.len())
            .unwrap_or(0);
    let mut stream_ctx = Vec::with_capacity(num_streams_to_decode);

    let seek = Seek {
        start_time,
        end_time,
    };

    // Configure video decoder
    if let Some(video_stream) = &video_stream {
        stream_ctx.push(init_video_stream_context(
            &input_format_context,
            video_stream,
            &seek,
        )?);
    }

    // Configure audio decoder(s)
    if let Some(audio_streams) = audio_streams {
        for audio_stream in audio_streams.iter() {
            stream_ctx.push(init_audio_stream_context(
                &input_format_context,
                audio_stream,
                &seek,
            )?);
        }
    }

    // Plain conversions bypass these graphs entirely (see `DirectPath`); the
    // graph handles resize, frame-rate resampling, HDR tone mapping, and
    // widths libswscale can't vectorize.
    log::debug!("Setting up filter graphs for the streams.");
    let mut filter_graphs: Vec<_> = (0..stream_ctx.len())
        .map(|_| AVFilterGraph::new())
        .collect();
    let mut filter_ctx = init_filters(&mut filter_graphs, stream_ctx)?;

    // Seek to the desired start time using `avformat_seek_file`.
    if let Seek {
        start_time: Some(start_time),
        ..
    } = &seek
    {
        let min_ts = ((start_time - 0.2) / av_q2d(ffi::AV_TIME_BASE_Q)) as i64;
        let ts = (start_time / av_q2d(ffi::AV_TIME_BASE_Q)) as i64;
        let max_ts = ts;
        log::debug!(
            "Attempting to seek to start_time={}, min_ts={}, ts={}, max_ts={}, av_q2d(ffi::AV_TIME_BASE_Q)={}",
            start_time,
            min_ts,
            ts,
            max_ts,
            av_q2d(ffi::AV_TIME_BASE_Q)
        );
        unsafe {
            ffi::avformat_seek_file(
                input_format_context.as_mut_ptr(),
                -1,
                min_ts,
                ts,
                max_ts,
                ffi::AVSEEK_FLAG_FRAME as i32,
            )
        }
        .upgrade()
        .map_err(|e| anyhow!("Failed to seek to start time: {e}"))?;
    }

    log::debug!("Starting decoding and filtering process.");
    let mut decoded_streams: Vec<DecodedStream> = filter_ctx
        .iter()
        .map(|s| DecodedStream {
            src_stream_index: s.stream_index,
            decoded_frames: Vec::new(),
            metadata: s.metadata.clone(),
            data: None,
            frame_pts: Vec::new(),
        })
        .collect();

    // Read packets from the demuxer and route them to the filter graph.
    loop {
        let packet = match input_format_context.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break, // End of file reached
            Err(e) => bail!("Failed to read packet: {e:?}"),
        };

        let in_stream_index = packet.stream_index as usize;

        if let Some(filter) = filter_ctx
            .iter_mut()
            .find(|f| f.stream_index == in_stream_index && !f.finished)
        {
            let pkt_timebase = filter.dec_ctx.pkt_timebase;
            let pts_seconds = pts_to_seconds(packet.pts, pkt_timebase);
            log::trace!(
                stream = in_stream_index,
                pts = packet.pts,
                pts_seconds = pts_seconds;
                "Received packet"
            );

            if let Some(start_time) = start_time {
                // TODO (rikheijdens): Do we risk dropping packets here that we actually need?
                if pts_seconds < start_time {
                    log::debug!(
                        "Packet pts={} for stream #{} is before start time={}.",
                        pts_seconds,
                        in_stream_index,
                        start_time
                    );
                    //continue; // Skip packets before the start time.
                }
            }

            if let Some(end_time) = end_time {
                if pts_seconds > end_time {
                    log::debug!(
                        "Packet pts {} exceeds end time {}, stopping decoding.",
                        pts_seconds,
                        end_time
                    );
                    filter.finished = true; // Stop decoding for this stream once we exceed the end time.
                }
            }

            let DecodedFrames {
                stream_index,
                frames,
            } = decode_packet(Some(&packet), filter, &seek).context("Failed to decode packet")?;

            // Drop any frames returned by the decoder past the end time.
            let frames = filter_frames_seek(frames, &seek);

            // TODO (rikheijdens): add another check to stop decoding if we're starting to drop frames?

            // Track the decoded frames for the stream.
            let decoded_stream = decoded_streams
                .iter_mut()
                .find(|s| s.src_stream_index == stream_index)
                .context("Decoded stream not found for the given stream index")?;
            decoded_stream.decoded_frames.extend(frames);
        }

        if filter_ctx.iter().all(|f| f.finished) {
            log::debug!("All filter contexts finished processing, breaking out of the loop.");
            break; // All streams have been processed, exit the loop.
        }
    }

    // Flush the decoders and filter graph.
    for filter_ctx in filter_ctx.iter_mut() {
        let FilteringContext {
            stream_type,
            stream_index,
            dec_ctx: decode_context,
            buffersrc_ctx,
            buffersink_ctx,
            direct_path,
            frame_data,
            frame_data_ptr,
            frame_pts,
            ..
        } = filter_ctx;

        let decoded_stream = decoded_streams
            .iter_mut()
            .find(|s| s.src_stream_index == *stream_index)
            .context("Decoded stream not found for the given stream index")?;

        // Flush the decoder
        decode_context.send_packet(None)?;

        // Route any frames through the filter graph that surface as a result of flushing the decoder.
        let frames = receive_and_filter_frames(
            decode_context,
            stream_type,
            buffersrc_ctx,
            buffersink_ctx,
            direct_path,
            frame_data,
            frame_data_ptr,
            frame_pts,
            &seek,
        )?;
        log::debug!(
            "Received {} frames after flushing the decoder",
            frames.len()
        );
        // Drop any frames returned by the decoder past the seek range.
        let frames = filter_frames_seek(frames, &seek);
        decoded_stream.decoded_frames.extend(frames);

        if let DirectPath::CudaConvert(converter) = &*direct_path {
            // NPP conversions are asynchronous; make sure they are complete
            // before the tensors are handed out.
            converter.synchronize()?;
        }
        if !matches!(direct_path, DirectPath::Disabled) {
            // The direct paths are stateless per frame; there is no graph to flush.
            continue;
        }

        // Flush the filter graph.
        let frames = match filter_frame(
            *stream_type,
            None,
            buffersrc_ctx
                .as_mut()
                .context("filter graph missing for graph-path stream")?,
            buffersink_ctx
                .as_mut()
                .context("filter graph missing for graph-path stream")?,
            frame_data,
            frame_data_ptr,
            frame_pts,
            &seek,
        ) {
            Ok(frames) => frames,
            Err(FilterFrameError::NoStorageAvailable(_e)) => {
                // This is alright - we can just drop the frames as they may be extraneous.
                vec![]
            }
            Err(FilterFrameError::Unexpected(e)) => {
                return Err(e.context("flushing the filter graph."));
            }
        };
        log::debug!(
            "Received {} frames after flushing the filter graph",
            frames.len(),
        );
        // Drop any frames returned by the decoder outside of the seek range.
        let frames = filter_frames_seek(frames, &seek);
        decoded_stream.decoded_frames.extend(frames);
    }

    // Move data buffer from FilteringContext -> DecodedStream.
    // TODO (rikheijdens): Should we have a different variant of the DecodedStream for which
    // the `data` field is not optional such that the type system can guarantee its presence?
    for FilteringContext {
        stream_index,
        frame_data,
        frame_data_ptr,
        mut frame_pts,
        stream_type,
        ..
    } in filter_ctx.into_iter()
    {
        let trimmed_data = match stream_type {
            StreamType::Video => {
                let rows = min(frame_data_ptr as i64, frame_data.size()[0]);
                frame_pts.truncate(rows as usize);
                frame_data.f_narrow(0, 0, rows).context(
                "Failed to trim video frame data Tensor to the correct number of decoded frames",
            )?
            }
            StreamType::Audio => frame_data
                .f_narrow(1, 0, min(frame_data_ptr as i64, frame_data.size()[1]))
                .context(
                "Failed to trim audio frame data Tensor to the correct number of decoded samples",
            )?,
        };
        let decoded_stream = decoded_streams
            .iter_mut()
            .find(|s| s.src_stream_index == stream_index)
            .context("Decoded stream not found for the given stream index")?;
        decoded_stream.data = Some(trimmed_data);
        decoded_stream.frame_pts = frame_pts;
    }

    Ok(decoded_streams)
}

/// Metadata for a probed video stream.
#[derive(Debug, Clone)]
pub struct ProbedVideoStream {
    pub index: usize,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
}

/// Metadata for a probed audio stream.
#[derive(Debug, Clone)]
pub struct ProbedAudioStream {
    pub index: usize,
    pub sample_rate: i32,
}

/// Container-level metadata returned by [`probe_media`].
#[derive(Debug, Clone, Default)]
pub struct ProbedMedia {
    pub video_streams: Vec<ProbedVideoStream>,
    pub audio_streams: Vec<ProbedAudioStream>,
}

/// Probes a media asset's stream layout without decoding it.
pub fn probe_media(
    source: MediaSource,
    s3_config: Option<S3Config>,
) -> Result<ProbedMedia, anyhow::Error> {
    let input_format_context = source.open(s3_config)?;
    let mut probed = ProbedMedia::default();
    for (index, stream) in input_format_context.streams().iter().enumerate() {
        let codecpar = stream.codecpar();
        match codecpar.codec_type {
            ffi::AVMEDIA_TYPE_VIDEO => {
                let fps = stream
                    .guess_framerate()
                    .map(av_q2d)
                    .unwrap_or_else(|| av_q2d(stream.avg_frame_rate));
                probed.video_streams.push(ProbedVideoStream {
                    index,
                    width: codecpar.width,
                    height: codecpar.height,
                    fps,
                });
            }
            ffi::AVMEDIA_TYPE_AUDIO => {
                probed.audio_streams.push(ProbedAudioStream {
                    index,
                    sample_rate: codecpar.sample_rate,
                });
            }
            _ => {}
        }
    }
    Ok(probed)
}

/// Selects video streams to decode
///
/// Arguments:
/// * `input_format_context`: The input format context containing stream metadata.
/// * `video_stream_request`: The video stream request from the caller.
fn select_video_stream(
    input_format_context: &AVFormatContextInput,
    video_stream_request: Option<VideoStreamRequest>,
) -> Result<Option<VideoStreamRequest>, anyhow::Error> {
    Ok(if let Some(video_stream) = video_stream_request {
        log::debug!("Video stream requested: {:?}", video_stream);
        match video_stream.index {
            Some(_idx) => Some(video_stream),
            None => {
                log::debug!(
                    "No video stream specified, attempting to find the best stream to return."
                );
                let index = input_format_context
                    .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)
                    .context("Failed to find best video stream!")?
                    .map(|(video_index, _codecref)| video_index)
                    .ok_or(anyhow::anyhow!("No video stream found"))?;
                Some(VideoStreamRequest {
                    index: Some(index),
                    ..video_stream
                })
            }
        }
    } else {
        None
    })
}

/// Selects audio streams to decode
///
/// Arguments:
/// * `input_format_context`: The input format context containing stream metadata
/// * `audio_stream_requests`: The requests for audio streams to decode.
fn select_audio_streams(
    input_format_context: &AVFormatContextInput,
    audio_stream_requests: Option<Vec<AudioStreamRequest>>,
) -> Result<Option<Vec<AudioStreamRequest>>, anyhow::Error> {
    Ok(match audio_stream_requests {
        Some(audio_streams) => {
            let mut streams_to_decode = Vec::new();
            for audio_stream in audio_streams {
                log::debug!("Audio stream requested: {:?}", audio_stream);
                let index = match audio_stream.index {
                    Some(idx) => idx,
                    None => {
                        log::debug!("No audio stream specified, attempting to find the best stream to return.");
                        input_format_context
                            .find_best_stream(ffi::AVMEDIA_TYPE_AUDIO)
                            .context("Failed to find best audio stream!")?
                            .map(|(audio_index, _codecref)| audio_index)
                            .ok_or(anyhow::anyhow!("No audio stream found"))?
                    }
                };

                let audio_stream = AudioStreamRequest {
                    index: Some(index),
                    ..audio_stream
                };

                // Ensure we decode streams only once.
                if !streams_to_decode
                    .iter()
                    .any(|s: &AudioStreamRequest| s.index == Some(index))
                {
                    streams_to_decode.push(audio_stream);
                }
            }
            Some(streams_to_decode)
        }
        None => None, // No audio requested.
    })
}

/// Initializes the video stream processor.
fn init_video_stream_context(
    input_format_context: &AVFormatContextInput,
    request: &VideoStreamRequest,
    seek: &Seek,
) -> Result<StreamContext, anyhow::Error> {
    let video_index = request
        .index
        .ok_or(anyhow!("No video stream index specified"))?;
    let video_stream = input_format_context
        .streams()
        .get(video_index)
        .context("Failed to get video stream")?;
    let codecpar = video_stream.codecpar();
    let default_decoder = AVCodec::find_decoder(codecpar.codec_id)
        .with_context(|| anyhow!("Failed to find decoder for video stream #{}", video_index))?;

    // device="cuda" decodes on NVDEC, so an unset hardware_acceleration is
    // implied; an explicit false contradicts it.
    let hardware_acceleration = match (request.hardware_acceleration, request.device) {
        (Some(false), Some(_)) => {
            return Err(anyhow!(
                "device=\"cuda\" decodes on NVDEC and cannot be combined with \
                 hardware_acceleration=false"
            ));
        }
        (hw, device) => hw == Some(true) || device.is_some(),
    };

    // When hardware acceleration is requested, decode on NVDEC via the
    // codec's cuvid wrapper (e.g. h264_cuvid). Without a hardware device
    // context these decoders return frames in system memory (NV12), so the
    // downstream filter graph and tensor conversion are unchanged.
    let decoder = if hardware_acceleration {
        let hw_name = format!("{}_cuvid", default_decoder.name().to_string_lossy());
        let hw_name_c = CString::new(hw_name.clone()).context("building decoder name")?;
        let hw_decoder = AVCodec::find_decoder_by_name(&hw_name_c).with_context(|| {
            anyhow!(
                "Hardware acceleration was requested, but decoder '{}' is not available in this FFmpeg build",
                hw_name
            )
        })?;
        log::debug!("Using hardware decoder {}", hw_name);
        hw_decoder
    } else {
        default_decoder
    };

    let mut decode_context = AVCodecContext::new(&decoder);
    decode_context
        .apply_codecpar(&codecpar)
        .context("Failed to apply codec parameters to video decode context")?;

    if let Some(device) = request.device {
        // GPU-resident output is a constrained pipeline: frames never reach
        // system memory, so only NVDEC's own capabilities are available.
        if request.frame_rate.is_some() {
            return Err(anyhow!(
                "device=\"cuda\" does not support fps resampling (frames never \
                 reach the CPU filter graph)"
            ));
        }
        if request.dtype != OutputDtype::Uint8 {
            return Err(anyhow!(
                "device=\"cuda\" currently supports only the uint8 dtype"
            ));
        }
        match (request.width, request.height) {
            (None, None) => {}
            (Some(w), Some(h)) => {
                // The output tensor is allocated with the requested
                // dimensions and NVDEC's scaler must produce exactly them.
                if w % 2 != 0 || h % 2 != 0 {
                    return Err(anyhow!(
                        "device=\"cuda\" resizing requires even dimensions, got {w}x{h}"
                    ));
                }
                if compute_hw_resize(codecpar.width, codecpar.height, request)
                    != Some((w as i32, h as i32))
                {
                    return Err(anyhow!(
                        "device=\"cuda\" only supports strict downscaling on NVDEC \
                         ({}x{} -> {w}x{h} is not)",
                        codecpar.width,
                        codecpar.height
                    ));
                }
            }
            _ => {
                return Err(anyhow!(
                    "device=\"cuda\" requires both width and height (or neither)"
                ));
            }
        }
        let _ = device;
    }

    if hardware_acceleration {
        // Attach a CUDA device context to the decoder. The ffmpeg CLI creates
        // one implicitly for cuvid decoders; in library use the decoder needs
        // it to reach the GPU.
        //
        // With device, bind the context to the requested ordinal's
        // CUDA *primary* context: torch owns the output tensor and NPP runs
        // the color conversion, and both use the primary context, so all
        // three see the same address space bookkeeping (torchcodec does the
        // same for device="cuda").
        let device_cstr = request
            .device
            .map(|d| CString::new(d.to_string()).context("building device ordinal"))
            .transpose()?;
        let mut hw_opts: *mut ffi::AVDictionary = std::ptr::null_mut();
        if request.device.is_some() {
            let ret = unsafe {
                ffi::av_dict_set(&mut hw_opts, c"primary_ctx".as_ptr(), c"1".as_ptr(), 0)
            };
            if ret < 0 {
                return Err(anyhow!("Failed to build CUDA device options: {ret}"));
            }
        }
        let mut hw_device_ctx: *mut ffi::AVBufferRef = std::ptr::null_mut();
        let ret = unsafe {
            ffi::av_hwdevice_ctx_create(
                &mut hw_device_ctx,
                ffi::AV_HWDEVICE_TYPE_CUDA,
                device_cstr
                    .as_ref()
                    .map_or(std::ptr::null(), |c| c.as_ptr()),
                hw_opts,
                0,
            )
        };
        unsafe { ffi::av_dict_free(&mut hw_opts) };
        if ret < 0 {
            return Err(anyhow!(
                "Failed to create a CUDA device context for hardware-accelerated \
                 decoding (is an NVIDIA GPU available?): FFmpeg error {ret}"
            ));
        }
        // The codec context takes ownership of this reference and unrefs it
        // when the context is freed.
        unsafe { (*decode_context.as_mut_ptr()).hw_device_ctx = hw_device_ctx };
    }

    decode_context.set_pkt_timebase(video_stream.time_base);
    let frame_rate = if let Some(frame_rate) = video_stream.guess_framerate() {
        decode_context.set_framerate(frame_rate);
        frame_rate
    } else {
        video_stream.avg_frame_rate
    };

    let num_threads = request.number_of_threads.unwrap_or(1);
    let mut opts = AVDictionary::new(
        c"threads",
        &CString::new(format!("{}", num_threads)).context("creating AVDictionary")?,
        0,
    );

    // GPU-side scaling: when decoding on hardware and a smaller output size
    // is requested, let NVDEC's scaler resize frames on the GPU before they
    // are transferred to system memory. This shrinks the GPU->CPU transfer
    // by the scale factor and removes most of the CPU scaling cost; the
    // filter graph still runs afterwards and guarantees the exact requested
    // dimensions.
    let hw_resize = if hardware_acceleration {
        compute_hw_resize(codecpar.width, codecpar.height, request)
    } else {
        None
    };
    if let Some((w, h)) = hw_resize {
        log::debug!("Resizing to {}x{} on the GPU during decode", w, h);
        opts = opts.set(
            c"resize",
            &CString::new(format!("{w}x{h}")).context("creating resize option")?,
            0,
        );
    }

    decode_context
        .open(Some(opts))
        .context("Failed to open video codec")?;

    if let Some((w, h)) = hw_resize {
        // The decoder only reports the resized dimensions once the stream has
        // been parsed; reflect them on the context now so the filter graph is
        // built with the correct input size.
        decode_context.set_width(w);
        decode_context.set_height(h);
    }

    let mut video_filter = VideoFilterConfig::try_from(request)?;
    video_filter.source_color = SourceColorInfo::from_codec_context(&decode_context);
    log::debug!("Source color info: {:?}", video_filter.source_color);

    let (num_frames, frame_rate) = match video_stream.duration {
        d if d == AV_NOPTS_VALUE => {
            // If the duration of the stream is not known, assume we are dealing with an image.
            (1, 1.0)
        }
        duration => {
            // Estimate the duration of media we'll be decoding, this will be the minimum of the specified end time (if any), or duration of the stream.
            let stream_duration = video_stream.duration as f64 * av_q2d(video_stream.time_base);
            let duration_seconds = asset_duration(stream_duration, seek)?;

            // In order to allocate storage for the decoded frames, we need to know the frame rate.
            let frame_rate = if let Some(frame_rate) = video_filter.frame_rate {
                frame_rate
            } else {
                // Convert `frame_rate` to double.
                match frame_rate {
                    AVRational { num: 0, den: 1 } => {
                        // If the frame rate is 0/1, we cannot determine the frame rate.
                        // Try to calculate from nb_frames / duration instead.
                        log::debug!(
                            "Could not guess frame rate, falling back to nb_frames / duration"
                        );
                        let num_frames = video_stream.nb_frames;
                        if num_frames > 0 && duration > 0 {
                            num_frames as f64 / (duration as f64 * av_q2d(video_stream.time_base))
                        } else {
                            // Err: could not determine framerate.
                            return Err(anyhow!("Could not determine frame rate."));
                        }
                    }
                    AVRational { .. } => av_q2d(frame_rate),
                }
            };
            let epsilon = (1.0 / frame_rate) * 0.5; // Small epsilon to avoid floating point precision errors causing trailing black frames because we've allocated too many frames.
            let num_frames = ((duration_seconds - epsilon) * frame_rate).ceil() as i64;
            (num_frames, frame_rate)
        }
    };

    let out_height = request
        .height
        .map(|h| h as i64)
        .unwrap_or(codecpar.height as i64);
    let out_width = request
        .width
        .map(|w| w as i64)
        .unwrap_or(codecpar.width as i64);
    // uint8 decodes to packed rgb24 ([T, H, W, C]); float32 decodes to
    // planar float ([T, C, H, W], contiguous channels-first).
    let storage_size = match request.dtype {
        OutputDtype::Uint8 => vec![num_frames, out_height, out_width, 3],
        OutputDtype::Float32 => vec![num_frames, 3, out_height, out_width],
    };
    log::debug!(
        "Allocating Tensor with shape {:?} to decode video to.",
        storage_size
    );
    let (kind, element_size) = match request.dtype {
        OutputDtype::Uint8 => (tch::Kind::Uint8, std::mem::size_of::<u8>()),
        OutputDtype::Float32 => (tch::Kind::Float, std::mem::size_of::<f32>()),
    };
    check_output_tensor_size(&storage_size, element_size)?;
    let device = request
        .device
        .map_or(tch::Device::Cpu, |d| tch::Device::Cuda(d as usize));
    let dest =
        tch::Tensor::f_empty(storage_size, (kind, device)).context("allocating output Tensor")?;

    Ok(StreamContext {
        stream_type: StreamType::Video,
        stream_index: video_index,
        dec_ctx: decode_context,
        filter_config: FilterConfig::Video(video_filter),
        frame_data: dest,
        metadata: StreamMetadata::Video { frame_rate },
    })
}

fn init_audio_stream_context(
    input_format_context: &AVFormatContextInput,
    request: &AudioStreamRequest,
    seek: &Seek,
) -> Result<StreamContext, anyhow::Error> {
    let audio_index = request
        .index
        .ok_or(anyhow!("No audio stream index specified"))?;
    let audio_stream = input_format_context
        .streams()
        .get(audio_index)
        .context("Failed to get audio stream")?;
    let codecpar = audio_stream.codecpar();
    let decoder = AVCodec::find_decoder(codecpar.codec_id)
        .with_context(|| anyhow!("Failed to find decoder for audio stream #{}", audio_index))?;
    let mut decode_context = AVCodecContext::new(&decoder);

    decode_context
        .apply_codecpar(&codecpar)
        .context("Failed to apply codec parameters to audio decode context")?;
    decode_context.set_pkt_timebase(audio_stream.time_base);

    decode_context
        .open(None) // TODO (rikheijdens): here we need to pass options such as number of threads
        .context("Failed to open audio codec")?;

    // A stream without a reported duration (AV_NOPTS_VALUE) yields a large
    // negative `stream_duration`, which would produce a negative sample count
    // and panic when allocating the output Tensor. The container does not tell
    // us how much to preallocate, so fail with a clear error instead.
    if audio_stream.duration == AV_NOPTS_VALUE {
        return Err(anyhow!(
            "Audio stream #{audio_index} does not report a duration; cannot preallocate the output buffer"
        ));
    }

    // Estimate the duration of media we'll be decoding, this will be the minimum of the specified end time (if any), or duration of the stream.
    let stream_duration = audio_stream.duration as f64 * av_q2d(audio_stream.time_base);
    let duration_seconds = asset_duration(stream_duration, seek)?;

    // The sample rate will either be the source sample rate, or the sample rate that we're targeting.
    let filter_config = AudioFilterConfig::try_from(request)?;
    let sample_rate = if let Some(sample_rate) = filter_config.sample_rate {
        sample_rate as i32
    } else {
        codecpar.sample_rate
    };

    // The number of channels should be equal to the source.
    let num_channels = codecpar.ch_layout().nb_channels;

    let num_samples = (duration_seconds * (sample_rate as f64)).ceil();
    if !num_samples.is_finite() || num_samples < 0.0 {
        return Err(anyhow!(
            "Computed an invalid sample count ({num_samples}) for audio stream #{audio_index}"
        ));
    }
    let shape = vec![num_channels as i64, num_samples as i64];
    log::debug!(
        "Allocating Tensor with shape {:?} to decode audio to for stream #{}.",
        shape,
        audio_index
    );
    check_output_tensor_size(&shape, std::mem::size_of::<f32>())?;
    let dest = tch::Tensor::f_empty(shape, (tch::Kind::Float, tch::Device::Cpu))
        .context("allocating audio output Tensor")?;

    Ok(StreamContext {
        stream_type: StreamType::Audio,
        stream_index: audio_index,
        dec_ctx: decode_context,
        filter_config: FilterConfig::Audio(filter_config),
        frame_data: dest,
        metadata: StreamMetadata::Audio {
            sample_rate: sample_rate as u32,
        },
    })
}

/// Maximum number of bytes avtensor will preallocate for a single decoded
/// stream. Duration, frame rate, dimensions, and sample rate all come from
/// file metadata and are attacker-controllable, so this bounds the up-front
/// allocation rather than trusting those values.
const MAX_OUTPUT_TENSOR_BYTES: u128 = 16 * 1024 * 1024 * 1024; // 16 GiB

/// Rejects output shapes that contain a negative dimension or whose total
/// allocation would exceed [`MAX_OUTPUT_TENSOR_BYTES`].
fn check_output_tensor_size(shape: &[i64], element_size: usize) -> Result<(), anyhow::Error> {
    let mut total = element_size as u128;
    for &dim in shape {
        let dim = u128::try_from(dim)
            .map_err(|_| anyhow!("output tensor has a negative dimension: {shape:?}"))?;
        total = total
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("output tensor size overflow for shape {shape:?}"))?;
    }
    if total > MAX_OUTPUT_TENSOR_BYTES {
        return Err(anyhow!(
            "output tensor for shape {shape:?} would require {total} bytes, exceeding the maximum of {MAX_OUTPUT_TENSOR_BYTES} bytes"
        ));
    }
    Ok(())
}

fn asset_duration(stream_duration: f64, seek: &Seek) -> Result<f64, anyhow::Error> {
    Ok(match seek {
        Seek {
            start_time: Some(start_time),
            end_time: Some(end_time),
        } => {
            if end_time <= start_time {
                return Err(anyhow!(
                    "End time must be greater than start time, got start_time: {}, end_time: {}",
                    start_time,
                    end_time
                ));
            }
            let duration = *end_time - *start_time;
            stream_duration.min(duration)
        }
        Seek {
            start_time: None,
            end_time: Some(end_time),
        } => stream_duration.min(*end_time),
        Seek {
            start_time: None,
            end_time: None,
        } => stream_duration,
        Seek {
            start_time: Some(start_time),
            end_time: None,
        } => {
            if stream_duration <= *start_time {
                return Err(anyhow!(
                    "Start time {} exceeds stream duration {}",
                    start_time,
                    stream_duration
                ));
            }
            stream_duration - *start_time
        }
    })
}

/// Utility function to filter `frames` with a presentation timestamp post the provided `end_time`.
///
/// Arguments:
/// - `frames`: The frames to filter
/// - `end_time`: A presentation timestamp (in seconds) after which frames should be filtered out
fn filter_frames_seek(frames: Vec<Frame>, seek: &Seek) -> Vec<Frame> {
    match seek {
        Seek {
            start_time: None,
            end_time: Some(end_time),
        } => frames
            .into_iter()
            .filter(|f| f.pts_seconds() < *end_time)
            .collect(),
        Seek {
            start_time: Some(start_time),
            end_time: None,
        } => frames
            .into_iter()
            .filter(|f| f.pts_seconds() >= *start_time)
            .collect(),
        Seek {
            start_time: Some(start_time),
            end_time: Some(end_time),
        } => frames
            .into_iter()
            .filter(|f| {
                let pts = f.pts_seconds();
                pts >= *start_time && pts < *end_time
            })
            .collect(),
        Seek {
            start_time: None,
            end_time: None,
        } => frames,
    }
}

/// Decodes a single [`AVPacket`] and routes the returned frames through the filter graph.
fn decode_packet(
    packet: Option<&AVPacket>,
    filter_ctx: &mut FilteringContext<'_>,
    seek: &Seek,
) -> Result<DecodedFrames, anyhow::Error> {
    let FilteringContext {
        stream_type,
        stream_index,
        dec_ctx: decode_context,
        buffersrc_ctx,
        buffersink_ctx,
        direct_path,
        frame_data,
        frame_data_ptr,
        frame_pts,
        finished,
        ..
    } = filter_ctx;

    let mut all_frames = Vec::new();

    // Try to send packet to the decoder, and handle EAGAIN in case the decoder is not ready
    // to accept new input (e.g. because its buffers are full).
    loop {
        match decode_context.send_packet(packet) {
            Ok(()) => break,
            Err(RsmpegError::DecoderFullError) => {
                // Decoder buffer full - drain frames first, then retry
                let frames = receive_and_filter_frames(
                    decode_context,
                    stream_type,
                    buffersrc_ctx,
                    buffersink_ctx,
                    direct_path,
                    frame_data,
                    frame_data_ptr,
                    frame_pts,
                    seek,
                )
                .context("Failed to receive frames from the decoder when draining")?;

                let received_frames = !frames.is_empty();
                all_frames.extend(frames);

                if received_frames {
                    // Re-submit packet.
                    continue;
                }

                // We did not receive frames, check if we've finished decoding already.
                let has_finished_decoding =
                    has_finished_decoding(stream_type, frame_data, frame_data_ptr);

                if has_finished_decoding || *finished {
                    log::debug!("We've finished decoding already, skipping packet.");
                    break;
                } else {
                    return Err(anyhow!(
                        "Decoder is full but no frames could be drained, cannot make progress"
                    ));
                }
            }
            Err(e) => {
                let hw_hint = if is_hw_decoder(decode_context) {
                    " (the stream's format may not be supported by the GPU decoder — \
                     e.g. NVDEC only supports 4:2:0 chroma subsampling for H.264)"
                } else {
                    ""
                };
                return Err(e).context(format!("Failed to send packet to decoder{hw_hint}"));
            }
        }
    }

    all_frames.extend(
        receive_and_filter_frames(
            decode_context,
            stream_type,
            buffersrc_ctx,
            buffersink_ctx,
            direct_path,
            frame_data,
            frame_data_ptr,
            frame_pts,
            seek,
        )
        .context("Failed to receive frames from the decoder")?,
    );

    Ok(DecodedFrames {
        stream_index: *stream_index,
        frames: all_frames,
    })
}

fn has_finished_decoding(
    stream_type: &StreamType,
    frame_data: &tch::Tensor,
    frame_data_ptr: &usize,
) -> bool {
    match stream_type {
        StreamType::Video => {
            *frame_data_ptr >= frame_data.size().first().copied().unwrap_or(0) as usize
        }
        StreamType::Audio => {
            *frame_data_ptr >= frame_data.size().get(1).copied().unwrap_or(0) as usize
        }
    }
}

/// Computes the output dimensions for NVDEC's on-GPU resizer.
///
/// Preserves the source aspect ratio when only one dimension is requested,
/// rounds to even dimensions (an NVDEC requirement), and returns None unless
/// the result is a strict downscale — upscaling stays in the filter graph.
fn compute_hw_resize(src_w: i32, src_h: i32, request: &VideoStreamRequest) -> Option<(i32, i32)> {
    if src_w <= 0 || src_h <= 0 {
        return None;
    }
    let (src_w, src_h) = (src_w as i64, src_h as i64);
    let (w, h) = match (request.width, request.height) {
        (None, None) => return None,
        (Some(w), Some(h)) => (w as i64, h as i64),
        (Some(w), None) => (w as i64, (w as i64 * src_h + src_w / 2) / src_w),
        (None, Some(h)) => ((h as i64 * src_w + src_h / 2) / src_h, h as i64),
    };
    let w = (w & !1).max(2);
    let h = (h & !1).max(2);
    if w >= src_w || h >= src_h {
        return None;
    }
    Some((w as i32, h as i32))
}

/// Returns true when the codec context decodes on GPU hardware (cuvid).
fn is_hw_decoder(ctx: &AVCodecContext) -> bool {
    unsafe {
        let codec = (*ctx.as_ptr()).codec;
        if codec.is_null() {
            return false;
        }
        std::ffi::CStr::from_ptr((*codec).name)
            .to_string_lossy()
            .ends_with("_cuvid")
    }
}

/// Transfers a frame residing in GPU memory to a new frame in system memory,
/// preserving frame properties (timestamps, color metadata, ...).
fn download_hw_frame(hw_frame: &AVFrame) -> Result<AVFrame, anyhow::Error> {
    let mut sw_frame = AVFrame::new();
    let ret = unsafe { ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), hw_frame.as_ptr(), 0) };
    if ret < 0 {
        return Err(anyhow!(
            "Failed to transfer frame from GPU to system memory: FFmpeg error {ret}"
        ));
    }
    let ret = unsafe { ffi::av_frame_copy_props(sw_frame.as_mut_ptr(), hw_frame.as_ptr()) };
    if ret < 0 {
        return Err(anyhow!(
            "Failed to copy frame properties from hardware frame: FFmpeg error {ret}"
        ));
    }
    Ok(sw_frame)
}

/// Consumes frames from the decoder and processes them through the filter
/// graph, or through the direct swscale path when one is configured.
#[allow(clippy::too_many_arguments)]
fn receive_and_filter_frames(
    decode_context: &mut AVCodecContext,
    stream_type: &StreamType,
    buffersrc_ctx: &mut Option<AVFilterContextMut<'_>>,
    buffersink_ctx: &mut Option<AVFilterContextMut<'_>>,
    direct_path: &mut DirectPath,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    seek: &Seek,
) -> Result<Vec<Frame>, anyhow::Error> {
    let mut frames: Vec<Frame> = Vec::new();

    loop {
        // Check if we need to stop decoding.
        match stream_type {
            StreamType::Video => {
                if has_finished_decoding(stream_type, frame_data, frame_data_ptr) {
                    log::debug!(
                        "No more storage available to write video frames to, exiting early"
                    );
                    break;
                }
            }
            StreamType::Audio => {
                if has_finished_decoding(stream_type, frame_data, frame_data_ptr) {
                    log::debug!(
                        "No more storage available to write audio frames to, exiting early"
                    );
                    break;
                }
            }
        }

        let mut frame = match decode_context.receive_frame() {
            Ok(frame) => frame,
            Err(RsmpegError::DecoderDrainError) | Err(RsmpegError::DecoderFlushedError) => break,
            Err(e) => Err(e).context("Error during decoding")?,
        };

        // Hardware decoders return frames in GPU memory; download them to
        // system memory so the software filter graph can process them —
        // unless the output is GPU-resident, in which case they stay put.
        if frame.format == ffi::AV_PIX_FMT_CUDA
            && !matches!(direct_path, DirectPath::CudaConvert(_))
        {
            frame = download_hw_frame(&frame).context("Downloading hardware frame")?;
        }

        frame.set_pts(frame.best_effort_timestamp);

        // Direct paths: convert straight into the output tensor.
        match direct_path {
            DirectPath::VideoScale(scaler) => {
                let time_base = decode_context.pkt_timebase;
                if let Some(converted) = direct_convert_frame(
                    scaler,
                    &frame,
                    frame_data,
                    frame_data_ptr,
                    frame_pts,
                    time_base,
                    seek,
                )
                .context("Error during direct frame conversion")?
                {
                    frames.push(converted);
                }
                continue;
            }
            DirectPath::AudioPassthrough => {
                let time_base = decode_context.pkt_timebase;
                if let Some(converted) =
                    direct_convert_audio_frame(&frame, frame_data, frame_data_ptr, time_base, seek)
                        .context("Error during direct audio frame conversion")?
                {
                    frames.push(converted);
                }
                continue;
            }
            DirectPath::CudaConvert(converter) => {
                let time_base = decode_context.pkt_timebase;
                if let Some(converted) = direct_convert_cuda_frame(
                    converter,
                    &frame,
                    frame_data,
                    frame_data_ptr,
                    frame_pts,
                    time_base,
                    seek,
                )
                .context("Error during CUDA frame conversion")?
                {
                    frames.push(converted);
                }
                continue;
            }
            DirectPath::Disabled => {}
        }

        // Send the frame to the filter source.
        let filtered_frames = match filter_frame(
            *stream_type,
            Some(frame),
            buffersrc_ctx
                .as_mut()
                .context("filter graph missing for graph-path stream")?,
            buffersink_ctx
                .as_mut()
                .context("filter graph missing for graph-path stream")?,
            frame_data,
            frame_data_ptr,
            frame_pts,
            seek,
        ) {
            Ok(frames) => frames,
            Err(FilterFrameError::NoStorageAvailable(e)) => {
                // We did not allocate sufficient (output) storage for the frames yielded by the filter.
                //
                // When we allocate output storage we use an (estimate) of the average frame rate and duration of the video. If
                // this estimate is off by more than one frame, we will run out of storage near the end of decoding. In this case we return early
                // and log a warning.
                //
                // However, if it is the case that `frame_data_ptr == 0`, then it may be the case that we failed to wholly allocate output buffers
                // in this case we treat it as an error and bubble it up.
                if *frame_data_ptr > 0 {
                    log::warn!("Insufficient storage available for filtered frames: {}", e);
                    continue;
                }
                return Err(
                    anyhow!(e).context("Insufficient storage allocated for filtered frames")
                );
            }
            Err(e) => return Err(anyhow!(e).context("Error during filtering"))?,
        };
        frames.extend(filtered_frames);
    }

    Ok(frames)
}

#[derive(thiserror::Error, Debug)]
enum FilterFrameError {
    #[error("No storage available: {0:?}")]
    NoStorageAvailable(anyhow::Error),
    #[error("Unexpected error occurred during filtering: {0:?}")]
    Unexpected(anyhow::Error),
}

/// Apply FFmpeg filter graph to a decoded frame.
#[allow(clippy::too_many_arguments)]
fn filter_frame(
    stream_type: StreamType,
    frame: Option<AVFrame>,
    buffersrc_ctx: &mut AVFilterContextMut,
    buffersink_ctx: &mut AVFilterContextMut,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    seek: &Seek,
) -> Result<Vec<Frame>, FilterFrameError> {
    // Send the frame to the filter source.
    buffersrc_ctx
        .buffersrc_add_frame(frame, None)
        .map_err(|e| {
            FilterFrameError::Unexpected(
                anyhow!(e).context("Error submitting the frame to the filtergraph"),
            )
        })?;

    let mut filtered_frames: Vec<Frame> = Vec::new();
    loop {
        // Get the filtered frames from the buffersink.
        let mut filtered_frame = match buffersink_ctx.buffersink_get_frame(None) {
            Ok(frame) => frame,
            Err(RsmpegError::BufferSinkDrainError) | Err(RsmpegError::BufferSinkEofError) => break,
            Err(_) => {
                return Err(FilterFrameError::Unexpected(anyhow!(
                    "Get frame from buffersink failed"
                )))
            }
        };

        filtered_frame.set_time_base(buffersink_ctx.get_time_base());
        filtered_frame.set_pict_type(ffi::AV_PICTURE_TYPE_NONE);

        if let Seek {
            start_time: Some(start_time),
            ..
        } = seek
        {
            let frame_pts = pts_to_seconds(filtered_frame.pts, filtered_frame.time_base);
            if frame_pts < *start_time {
                log::debug!(
                    "Filtered frame PTS {} is less than start time {}, dropping...",
                    frame_pts,
                    start_time
                );
                continue;
            }
        }

        // Fetch a slice of the destination Tensor to copy the decoded frames to.
        let (dest, step) = match stream_type {
            StreamType::Video => {
                let step = 1;
                let slice = frame_data.f_i(*frame_data_ptr as i64..(*frame_data_ptr as i64) + step);
                (slice, step)
            }
            StreamType::Audio => {
                let max_idx = frame_data.size().get(1).copied().unwrap_or(0) as usize;
                let max_step = max_idx - *frame_data_ptr;

                let step = min(filtered_frame.nb_samples as usize, max_step);
                let end = *frame_data_ptr as i64 + step as i64;

                // Because audio frames typically contain multiple samples, it may be the case we haven't allocated sufficient
                // storage. In this case we truncate the yielded audio frame to the size of the remaining storage.
                if step < filtered_frame.nb_samples as usize {
                    log::debug!(
                        "Not enough storage allocated for audio frame, truncating to fit. \
                         frame_data_ptr: {}, step: {}, frame_data size: {:?}",
                        *frame_data_ptr,
                        step,
                        frame_data.size()
                    );
                }

                let slice = frame_data.f_i((.., *frame_data_ptr as i64..end));
                (slice, step as i64)
            }
        };

        let dest = dest.map_err(|e| FilterFrameError::NoStorageAvailable(anyhow!(e)))?;
        *frame_data_ptr += step as usize;
        if stream_type == StreamType::Video {
            frame_pts.push(pts_to_seconds(filtered_frame.pts, filtered_frame.time_base));
        }

        let frame = convert_frame_to_tensor(stream_type, &filtered_frame, dest)
            .map_err(FilterFrameError::Unexpected)?;
        filtered_frames.push(frame);
    }
    Ok(filtered_frames)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamType {
    Video,
    Audio,
}

/// A video or audio frame.
pub enum Frame {
    Video {
        // Width of the video frame in pixels.
        width: i32,
        // Height of the video frame in pixels.
        height: i32,
        // Presentation timestamp in time_base units (time when frame should be shown to user).
        pts: i64,
        // frame timestamp estimated using various heuristics, in stream time base
        best_effort_timestamp: i64,
        // Frame data as a Tensor [T, C, H, W].
        data: tch::Tensor,
        // Time base for the timestamps in this frame. In the future, this field may be set on frames output by decoders or filters, but its value will be by default ignored on input to encoders or filters.
        time_base: AVRational,
    },
    Audio {
        // Sample rate of the audio data
        sample_rate: i32,
        // Number of audio samples (per channel) described by this frame.
        nb_samples: i32,
        // Presentation timestamp in time_base units (time when frame should be shown to user).
        pts: i64,
        // frame timestamp estimated using various heuristics, in stream time base
        best_effort_timestamp: i64,
        // Time base for the timestamps in this frame. In the future, this field may be set on frames output by decoders or filters, but its value will be by default ignored on input to encoders or filters.
        time_base: AVRational,
        // Frame data as Tensor [C, T].
        data: tch::Tensor,
    },
}

impl PartialEq for Frame {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Video {
                    width: l_width,
                    height: l_height,
                    pts: l_pts,
                    best_effort_timestamp: l_best_effort_timestamp,
                    data: l_data,
                    time_base: l_time_base,
                },
                Self::Video {
                    width: r_width,
                    height: r_height,
                    pts: r_pts,
                    best_effort_timestamp: r_best_effort_timestamp,
                    data: r_data,
                    time_base: r_time_base,
                },
            ) => {
                l_width == r_width
                    && l_height == r_height
                    && l_pts == r_pts
                    && l_best_effort_timestamp == r_best_effort_timestamp
                    && l_data == r_data
                    && l_time_base.num == r_time_base.num
                    && l_time_base.den == r_time_base.den
            }
            (
                Self::Audio {
                    sample_rate: l_sample_rate,
                    nb_samples: l_nb_samples,
                    pts: l_pts,
                    best_effort_timestamp: l_best_effort_timestamp,
                    time_base: l_time_base,
                    data: l_data,
                },
                Self::Audio {
                    sample_rate: r_sample_rate,
                    nb_samples: r_nb_samples,
                    pts: r_pts,
                    best_effort_timestamp: r_best_effort_timestamp,
                    time_base: r_time_base,
                    data: r_data,
                },
            ) => {
                l_sample_rate == r_sample_rate
                    && l_nb_samples == r_nb_samples
                    && l_pts == r_pts
                    && l_best_effort_timestamp == r_best_effort_timestamp
                    && l_data == r_data
                    && l_time_base.num == r_time_base.num
                    && l_time_base.den == r_time_base.den
            }
            _ => false,
        }
    }
}

impl Frame {
    /// Presentation timestamp for the frame in seconds.
    pub fn pts_seconds(&self) -> f64 {
        let (pts, time_base) = match self {
            Frame::Video { pts, time_base, .. } => (*pts, *time_base),
            Frame::Audio { pts, time_base, .. } => (*pts, *time_base),
        };

        pts_to_seconds(pts, time_base)
    }

    /// Best effort timestamp in seconds
    ///
    /// Unclear when this differs from the pts_seconds -- likely media container / codec specific.
    pub fn best_effort_timestamp_seconds(&self) -> f64 {
        let (best_effort_timestamp, time_base) = match self {
            Frame::Video {
                best_effort_timestamp,
                time_base,
                ..
            } => (*best_effort_timestamp, *time_base),
            Frame::Audio {
                best_effort_timestamp,
                time_base,
                ..
            } => (*best_effort_timestamp, *time_base),
        };

        pts_to_seconds(best_effort_timestamp, time_base)
    }

    #[allow(dead_code)]
    pub fn data(&self) -> &tch::Tensor {
        match self {
            Frame::Video { data, .. } => data,
            Frame::Audio { data, .. } => data,
        }
    }
}

impl Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frame::Video {
                width,
                height,
                pts,
                best_effort_timestamp,
                data,
                time_base,
            } => f
                .debug_struct("VideoFrame")
                .field("width", width)
                .field("height", height)
                .field("pts", pts)
                .field("pts_seconds", &self.pts_seconds())
                .field("best_effort_timestamp", best_effort_timestamp)
                .field(
                    "best_effort_timestamp_seconds",
                    &self.best_effort_timestamp_seconds(),
                )
                .field("data_shape", &data.size())
                .field("time_base", time_base)
                .finish(),
            Frame::Audio {
                sample_rate,
                nb_samples,
                pts,
                best_effort_timestamp,
                time_base,
                data,
            } => f
                .debug_struct("AudioFrame")
                .field("sample_rate", sample_rate)
                .field("nb_samples", nb_samples)
                .field("pts", pts)
                .field("pts_seconds", &self.pts_seconds())
                .field("best_effort_timestamp", best_effort_timestamp)
                .field(
                    "best_effort_timestamp_seconds",
                    &self.best_effort_timestamp_seconds(),
                )
                .field("data_shape", &data.size())
                .field("time_base", time_base)
                .finish(),
        }
    }
}

/// Converts (and optionally scales) decoded video frames straight into the
/// output tensor with libswscale, bypassing the filter graph.
///
/// The filter-graph path materializes its own RGB output `AVFrame`, which
/// then has to be copied into the tensor — a full extra copy of every frame.
/// libswscale can write directly into caller-provided memory, so this path
/// is used whenever it produces output identical to the graph (see
/// [`DirectVideoScaler::is_eligible`]).
struct DirectVideoScaler {
    /// Cached swscale context, rebuilt by `sws_getCachedContext` whenever the
    /// source frame parameters change. Null until the first frame.
    sws_ctx: *mut ffi::SwsContext,
    dst_width: i32,
    dst_height: i32,
}

// SAFETY: the context is owned exclusively by this scaler and all use happens
// on the decoding thread.
unsafe impl Send for DirectVideoScaler {}

impl Drop for DirectVideoScaler {
    fn drop(&mut self) {
        unsafe { ffi::sws_freeContext(self.sws_ctx) };
    }
}

impl DirectVideoScaler {
    fn new(dst_width: i32, dst_height: i32) -> Self {
        Self {
            sws_ctx: std::ptr::null_mut(),
            dst_width,
            dst_height,
        }
    }

    /// Returns true when the direct path should handle this configuration:
    /// RGB24 output, no resize, no frame-rate resampling, no HDR tone
    /// mapping, and a 32-multiple output width (libswscale's vectorized
    /// RGB24 output requires this on the tightly packed rows of the output
    /// tensor). Its output is identical to the filter graph's.
    ///
    /// Resizes deliberately stay in the filter graph: the graph's `scale`
    /// filter downscales with slice threading (and in subsampled YUV before
    /// the RGB conversion), which measures faster than a single-threaded
    /// one-pass `sws_scale`, and with a small output the extra frame copy
    /// the graph costs is small anyway.
    fn is_eligible(config: &VideoFilterConfig, dst_width: i64) -> bool {
        config.frame_rate.is_none()
            && config.width.is_none()
            && config.height.is_none()
            && !config.source_color.is_hdr()
            && config.pixel_format == "rgb24"
            && dst_width % 32 == 0
            && direct_paths_enabled()
    }

    /// Scales/converts `frame` into `dest`, a `[1, dst_height, dst_width, 3]`
    /// uint8 tensor view.
    fn scale_into(&mut self, frame: &AVFrame, dest: &tch::Tensor) -> Result<(), anyhow::Error> {
        let ctx = unsafe {
            ffi::sws_getCachedContext(
                self.sws_ctx,
                frame.width,
                frame.height,
                frame.format,
                self.dst_width,
                self.dst_height,
                ffi::AV_PIX_FMT_RGB24,
                // The scale filter's default; keeps output identical to the
                // filter-graph path.
                ffi::SWS_BICUBIC as i32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        if ctx.is_null() {
            self.sws_ctx = std::ptr::null_mut();
            bail!(
                "sws_getCachedContext failed ({}x{} pix_fmt {} -> {}x{} rgb24)",
                frame.width,
                frame.height,
                frame.format,
                self.dst_width,
                self.dst_height
            );
        }
        self.sws_ctx = ctx;

        // Mirror the filter graph's color handling: it forwards the stream's
        // colorspace and range to buffersrc so the YUV->RGB conversion uses
        // the tagged metadata rather than a guess. When both are untagged,
        // leave swscale's defaults in place (same as the graph).
        let colorspace = frame.colorspace;
        let range = frame.color_range;
        if colorspace != ffi::AVCOL_SPC_UNSPECIFIED || range != ffi::AVCOL_RANGE_UNSPECIFIED {
            let sws_cs = match colorspace {
                ffi::AVCOL_SPC_BT709 => ffi::SWS_CS_ITU709,
                ffi::AVCOL_SPC_SMPTE170M | ffi::AVCOL_SPC_BT470BG => ffi::SWS_CS_ITU601,
                ffi::AVCOL_SPC_SMPTE240M => ffi::SWS_CS_SMPTE240M,
                ffi::AVCOL_SPC_BT2020_NCL => ffi::SWS_CS_BT2020,
                _ => ffi::SWS_CS_DEFAULT,
            };
            let src_range = (range == ffi::AVCOL_RANGE_JPEG) as i32;
            unsafe {
                let inv_table = ffi::sws_getCoefficients(sws_cs as i32);
                let table = ffi::sws_getCoefficients(ffi::SWS_CS_DEFAULT as i32);
                // Brightness/contrast/saturation at their neutral values; RGB
                // output is always full range.
                ffi::sws_setColorspaceDetails(
                    ctx,
                    inv_table,
                    src_range,
                    table,
                    1,
                    0,
                    1 << 16,
                    1 << 16,
                );
            }
        }

        let dst_ptr = dest.data_ptr() as *mut u8;
        let dst_data = [
            dst_ptr,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        let dst_stride = [self.dst_width * 3, 0, 0, 0];
        let scaled_rows = unsafe {
            ffi::sws_scale(
                ctx,
                frame.data.as_ptr() as *const *const u8,
                frame.linesize.as_ptr(),
                0,
                frame.height,
                dst_data.as_ptr(),
                dst_stride.as_ptr(),
            )
        };
        if scaled_rows != self.dst_height {
            bail!(
                "sws_scale produced {} rows, expected {}",
                scaled_rows,
                self.dst_height
            );
        }
        Ok(())
    }
}

/// Returns false when the `AVTENSOR_DISABLE_DIRECT_PATH` debug escape hatch
/// forces all streams through the filter graph.
fn direct_paths_enabled() -> bool {
    std::env::var_os("AVTENSOR_DISABLE_DIRECT_PATH").is_none()
}

/// Converts a decoded video frame straight into the output tensor via the
/// direct swscale path. Returns `None` when the frame is dropped (before the
/// seek start, or no storage left).
fn direct_convert_frame(
    scaler: &mut DirectVideoScaler,
    frame: &AVFrame,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    time_base: AVRational,
    seek: &Seek,
) -> Result<Option<Frame>, anyhow::Error> {
    let pts_seconds = pts_to_seconds(frame.pts, time_base);
    if let Seek {
        start_time: Some(start_time),
        ..
    } = seek
    {
        if pts_seconds < *start_time {
            log::debug!(
                "Frame PTS {} is less than start time {}, dropping...",
                pts_seconds,
                start_time
            );
            return Ok(None);
        }
    }

    let capacity = frame_data.size().first().copied().unwrap_or(0);
    if *frame_data_ptr as i64 >= capacity {
        // Mirrors the filter-graph path: the storage estimate can be off by a
        // frame near the end of the stream.
        log::warn!("Insufficient storage available for decoded frames, dropping frame");
        return Ok(None);
    }

    let dest = frame_data
        .f_i(*frame_data_ptr as i64..*frame_data_ptr as i64 + 1)
        .context("Failed to slice destination tensor")?;
    scaler.scale_into(frame, &dest)?;
    *frame_data_ptr += 1;
    frame_pts.push(pts_seconds);

    Ok(Some(Frame::Video {
        width: scaler.dst_width,
        height: scaler.dst_height,
        pts: frame.pts,
        best_effort_timestamp: frame.best_effort_timestamp,
        data: dest,
        time_base,
    }))
}

/// Converts a decoded audio frame straight into the output tensor via the
/// direct passthrough path, mirroring the filter-graph path's slicing and
/// truncation behavior. Returns `None` when the frame is dropped (before the
/// seek start, or no storage left).
fn direct_convert_audio_frame(
    frame: &AVFrame,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    time_base: AVRational,
    seek: &Seek,
) -> Result<Option<Frame>, anyhow::Error> {
    let pts_seconds = pts_to_seconds(frame.pts, time_base);
    if let Seek {
        start_time: Some(start_time),
        ..
    } = seek
    {
        if pts_seconds < *start_time {
            log::debug!(
                "Frame PTS {} is less than start time {}, dropping...",
                pts_seconds,
                start_time
            );
            return Ok(None);
        }
    }

    let max_idx = frame_data.size().get(1).copied().unwrap_or(0) as usize;
    let max_step = max_idx.saturating_sub(*frame_data_ptr);
    let step = min(frame.nb_samples as usize, max_step);
    if step == 0 {
        log::warn!("Insufficient storage available for decoded audio, dropping frame");
        return Ok(None);
    }
    if step < frame.nb_samples as usize {
        // Storage is sized from a duration estimate; mirror the graph path
        // and truncate the final frame to the remaining space.
        log::debug!(
            "Not enough storage allocated for audio frame, truncating to fit. \
             frame_data_ptr: {}, step: {}, frame_data size: {:?}",
            *frame_data_ptr,
            step,
            frame_data.size()
        );
    }
    let end = *frame_data_ptr as i64 + step as i64;
    let mut dest = frame_data
        .f_i((.., *frame_data_ptr as i64..end))
        .context("Failed to slice destination tensor")?;
    convert_audio_into_tensor(frame, &mut dest)
        .context("Failed to convert audio frame to tensor")?;
    *frame_data_ptr += step;

    Ok(Some(Frame::Audio {
        sample_rate: frame.sample_rate,
        nb_samples: frame.nb_samples,
        pts: frame.pts,
        best_effort_timestamp: frame.best_effort_timestamp,
        time_base,
        data: dest,
    }))
}

/// Converts one NVDEC CUDA frame into the CUDA output tensor via NPP.
/// Returns `None` when the frame is dropped (before the seek start, or no
/// storage left).
fn direct_convert_cuda_frame(
    converter: &cuda::CudaNv12ToRgb,
    frame: &AVFrame,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    time_base: AVRational,
    seek: &Seek,
) -> Result<Option<Frame>, anyhow::Error> {
    if frame.format != ffi::AV_PIX_FMT_CUDA {
        return Err(anyhow!(
            "expected a CUDA frame on the GPU-resident path, got pixel format {}",
            frame.format
        ));
    }
    if frame.width != converter.dst_width || frame.height != converter.dst_height {
        return Err(anyhow!(
            "CUDA frame is {}x{} but the output tensor expects {}x{}",
            frame.width,
            frame.height,
            converter.dst_width,
            converter.dst_height
        ));
    }
    if frame.linesize[0] != frame.linesize[1] {
        return Err(anyhow!(
            "NV12 frame has unequal Y/UV pitches ({} vs {})",
            frame.linesize[0],
            frame.linesize[1]
        ));
    }

    let pts_seconds = pts_to_seconds(frame.pts, time_base);
    if let Seek {
        start_time: Some(start_time),
        ..
    } = seek
    {
        if pts_seconds < *start_time {
            log::debug!(
                "Frame PTS {} is less than start time {}, dropping...",
                pts_seconds,
                start_time
            );
            return Ok(None);
        }
    }

    let capacity = frame_data.size().first().copied().unwrap_or(0);
    if *frame_data_ptr as i64 >= capacity {
        // Mirrors the other direct paths: the storage estimate can be off by
        // a frame near the end of the stream.
        log::warn!("Insufficient storage available for decoded frames, dropping frame");
        return Ok(None);
    }

    let dest = frame_data
        .f_i(*frame_data_ptr as i64..*frame_data_ptr as i64 + 1)
        .context("Failed to slice destination tensor")?;
    converter
        .convert_frame(
            frame.data[0] as *const u8,
            frame.data[1] as *const u8,
            frame.linesize[0],
        )
        .into_dst(dest.data_ptr() as *mut u8)?;
    *frame_data_ptr += 1;
    frame_pts.push(pts_seconds);

    Ok(Some(Frame::Video {
        width: converter.dst_width,
        height: converter.dst_height,
        pts: frame.pts,
        best_effort_timestamp: frame.best_effort_timestamp,
        data: dest,
        time_base,
    }))
}

fn convert_frame_to_tensor(
    stream_type: StreamType,
    frame: &AVFrame,
    mut destination: tch::Tensor,
) -> Result<Frame, anyhow::Error> {
    match stream_type {
        StreamType::Video => {
            let width = frame.width;
            let height = frame.height;
            let pts = frame.pts;
            let best_effort_timestamp = frame.best_effort_timestamp;
            let time_base = frame.time_base;

            // Convert AVFrame to Tensor.
            match frame.format {
                f if f == ffi::AV_PIX_FMT_RGB24 => {
                    log::trace!("Decoding RGB24 frame");
                    conversion::convert_rgb24_frame_to_tensor(frame, &mut destination)?
                }
                f if f == ffi::AV_PIX_FMT_GBRPF32LE => {
                    log::trace!("Decoding planar float frame");
                    conversion::convert_gbrpf32_frame_to_tensor(frame, &mut destination)?
                }
                other => {
                    bail!(
                        "Unsupported pixel format: {other}. Only AV_PIX_FMT_RGB24 ({}) and AV_PIX_FMT_GBRPF32LE ({}) are currently supported.",
                        ffi::AV_PIX_FMT_RGB24,
                        ffi::AV_PIX_FMT_GBRPF32LE
                    );
                }
            };

            Ok(Frame::Video {
                width,
                height,
                pts,
                best_effort_timestamp,
                data: destination,
                time_base,
            })
        }
        StreamType::Audio => {
            let sample_rate = frame.sample_rate;
            let nb_samples = frame.nb_samples;
            let pts = frame.pts;
            let best_effort_timestamp = frame.best_effort_timestamp;
            let time_base = frame.time_base;

            convert_audio_into_tensor(frame, &mut destination)
                .context("Failed to convert audio frame to tensor")?;

            Ok(Frame::Audio {
                sample_rate,
                nb_samples,
                pts,
                best_effort_timestamp,
                time_base,
                data: destination,
            })
        }
    }
}

impl TryFrom<(StreamType, &AVFrame)> for Frame {
    type Error = anyhow::Error;

    fn try_from((stream_type, frame): (StreamType, &AVFrame)) -> Result<Self, Self::Error> {
        match stream_type {
            StreamType::Video => {
                let dest = tch::Tensor::empty(
                    [1, frame.height as i64, frame.width as i64, 3],
                    (tch::Kind::Uint8, tch::Device::Cpu),
                );
                convert_frame_to_tensor(stream_type, frame, dest)
            }
            StreamType::Audio => {
                let nb_samples = frame.nb_samples;
                let num_channels = frame.ch_layout().nb_channels;
                let dest = tch::Tensor::empty(
                    [nb_samples as i64, num_channels as i64],
                    (tch::Kind::Float, tch::Device::Cpu),
                );
                convert_frame_to_tensor(stream_type, frame, dest)
            }
        }
    }
}

/// Returns the FFmpeg string name for a color metadata enum value,
/// or `None` if the value is unspecified/unknown or the FFI function returns NULL.
fn ffi_color_name(ptr: *const std::os::raw::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    if name == "unknown" || name == "unspecified" {
        return None;
    }
    Some(name)
}

/// Color properties of the source video, extracted from the decoder context.
/// Used to configure the buffersrc filter and determine whether HDR→SDR
/// conversion is needed in the filter chain.
#[derive(Debug, Clone, Default)]
struct SourceColorInfo {
    /// YUV colorspace matrix (e.g. "bt709", "bt2020nc", "smpte170m").
    /// Surfaced via the Debug log; the buffersrc filter reads it from the
    /// decoder context directly.
    #[allow(dead_code)]
    colorspace: Option<String>,
    /// Color range: "tv"/"mpeg" (limited) or "pc"/"jpeg" (full).
    #[allow(dead_code)]
    color_range: Option<String>,
    /// Color primaries (e.g. "bt709", "bt2020").
    color_primaries: Option<String>,
    /// Transfer characteristics (e.g. "bt709", "smpte2084", "arib-std-b67").
    color_trc: Option<String>,
}

impl SourceColorInfo {
    /// Extract color metadata from the decoder context.
    fn from_codec_context(dec_ctx: &AVCodecContext) -> Self {
        Self {
            colorspace: ffi_color_name(unsafe { ffi::av_color_space_name(dec_ctx.colorspace) }),
            color_range: ffi_color_name(unsafe { ffi::av_color_range_name(dec_ctx.color_range) }),
            color_primaries: ffi_color_name(unsafe {
                ffi::av_color_primaries_name(dec_ctx.color_primaries)
            }),
            color_trc: ffi_color_name(unsafe { ffi::av_color_transfer_name(dec_ctx.color_trc) }),
        }
    }

    /// Returns true if the source is HDR content requiring tone mapping and
    /// gamut conversion to produce correct sRGB output.
    fn is_hdr(&self) -> bool {
        let hdr_transfer = matches!(
            self.color_trc.as_deref(),
            Some("smpte2084") | Some("arib-std-b67")
        );
        let wide_gamut = matches!(self.color_primaries.as_deref(), Some("bt2020"));
        hdr_transfer || wide_gamut
    }
}

#[derive(Debug)]
pub struct VideoFilterConfig {
    /// Desired frame rate for the video.
    frame_rate: Option<f64>,
    /// Desired width of the video frame.
    width: Option<usize>,
    /// Desired height of the video frame.
    height: Option<usize>,
    /// Desired pixel format for the video (defaults to `rgb24`).
    pixel_format: String,
    /// Source color properties, used to decide whether HDR→SDR conversion is needed.
    source_color: SourceColorInfo,
    /// CUDA ordinal for GPU-resident output (frames stay on the GPU).
    device: Option<i32>,
}

impl Display for VideoFilterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut filters = Vec::new();
        if let Some(frame_rate) = self.frame_rate {
            filters.push(format!("fps={}", frame_rate));
        }

        let mut scales = Vec::new();
        if let Some(width) = self.width {
            scales.push(format!("width={}", width));
        }
        if let Some(height) = self.height {
            scales.push(format!("height={}", height));
        }
        if !scales.is_empty() {
            filters.push(format!("scale={}", scales.join(":")));
        }

        // For HDR/wide-gamut content, insert tone mapping and gamut conversion
        // to produce correct sRGB output instead of a naive YUV→RGB conversion.
        if self.source_color.is_hdr() {
            log::debug!(
                "HDR source detected (trc={:?}, primaries={:?}), inserting tone mapping pipeline",
                self.source_color.color_trc,
                self.source_color.color_primaries
            );
            // 1. Linearize the transfer function (npl=100 for HLG nominal peak luminance)
            filters.push("zscale=t=linear:npl=100".to_string());
            // 2. Convert to float for precision during tone mapping
            filters.push("format=gbrpf32le".to_string());
            // 3. Tone map HDR → SDR dynamic range
            filters.push("tonemap=hable:desat=0".to_string());
            // 4. Convert primaries, transfer, and matrix to BT.709 (sRGB)
            filters.push("zscale=p=bt709:t=bt709:m=bt709:range=tv".to_string());
        }

        filters.push(format!("format=pix_fmts={}", self.pixel_format));
        if !filters.is_empty() {
            write!(f, "{}", filters.join(","))
        } else {
            Ok(())
        }
    }
}

impl Default for VideoFilterConfig {
    fn default() -> Self {
        Self {
            pixel_format: "rgb24".to_string(),
            frame_rate: Default::default(),
            width: Default::default(),
            height: Default::default(),
            source_color: Default::default(),
            device: Default::default(),
        }
    }
}

impl TryFrom<&VideoStreamRequest> for VideoFilterConfig {
    type Error = anyhow::Error;

    fn try_from(req: &VideoStreamRequest) -> Result<Self, Self::Error> {
        Ok(VideoFilterConfig {
            frame_rate: req.frame_rate,
            width: req.width.map(|w| w as usize),
            height: req.height.map(|h| h as usize),
            device: req.device,
            pixel_format: match req.dtype {
                OutputDtype::Uint8 => "rgb24".to_string(),
                OutputDtype::Float32 => "gbrpf32le".to_string(),
            },
            ..Default::default()
        })
    }
}

#[derive(Debug, Default)]
pub struct AudioFilterConfig {
    sample_rate: Option<usize>,
    //num_channels: Option<usize>,
    loudness_normalization: Option<LoudnessNormalization>,
}

impl Display for AudioFilterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut filters = Vec::new();
        if let Some(loudness_normalization) = &self.loudness_normalization {
            filters.push(format!("{}", loudness_normalization));
        }
        if let Some(sample_rate) = self.sample_rate {
            filters.push(format!("aresample={}", sample_rate));
        }
        // TODO (rikheijdens): add support for num_channels.
        if !filters.is_empty() {
            write!(f, "{}", filters.join(","))
        } else {
            write!(f, "anull") // Default filter if no options are provided
        }
    }
}

impl TryFrom<&AudioStreamRequest> for AudioFilterConfig {
    type Error = anyhow::Error;

    fn try_from(req: &AudioStreamRequest) -> Result<Self, Self::Error> {
        Ok(AudioFilterConfig {
            sample_rate: req.sample_rate.map(|r| r as usize),
            loudness_normalization: req.loudness_normalization.clone(),
        })
    }
}

/// FFmpeg `loudnorm` filter configuration
#[derive(Debug, Clone)]
pub struct LoudnessNormalization {
    /// Integrated loudness target. Range is -70.0 - -5.0. Default value is -24.0.
    pub integrated_loudness_target: Option<f32>,
    /// loudness range target. Range is 1.0 - 50.0. Default value is 7.0.
    pub loudness_range_target: Option<f32>,
    /// true peak level. Default value is -2.0.
    pub true_peak_level_target: Option<f32>,
    /// Measured integrated loudness of the input audio. Range is -99.0 - +0.0.
    pub measured_integrated_loudness: Option<f32>,
    /// Measured loudness range of input file. Range is 0.0 - 99.0.
    pub measured_loudness_range: Option<f32>,
    /// Measured true peak level of input file. Range is -99.0 - +99.0
    pub measured_true_peak_level: Option<f32>,
    /// Measured threshold of input file. Range is -99.0 - +0.
    pub measured_threshold: Option<f32>,
    /// Offset gain to apply to input audio. Gain is applied before the true-peak limiter. Range is -99.0 - +99.0. Default is +0.0.
    pub offset_gain: Option<f32>,
    /// Normalize by linearly scaling the source audio. measured_integrated_loudness, measured_loudness_range, measured_true_peak_level, and measured_threshold must all be specified. `loudness_range_target` shouldn’t be lower than source LRA and the change in integrated loudness shouldn’t result in a true peak which exceeds the target TP. If any of these conditions aren’t met, normalization mode will revert to dynamic. Options are true or false. Default is true.
    pub linear: Option<bool>,
    /// Treat mono input files as "dual-mono". If a mono file is intended for playback on a stereo system, its EBU R128 measurement will be perceptually incorrect. If set to true, this option will compensate for this effect. Multi-channel input files are not affected by this option. Options are true or false. Default is false.
    pub dual_mono: Option<bool>,
}

impl Display for LoudnessNormalization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut options = Vec::new();
        if let Some(integrated_loudness_target) = self.integrated_loudness_target {
            options.push(format!("I={}", integrated_loudness_target));
        }
        if let Some(loudness_range_target) = self.loudness_range_target {
            options.push(format!("LRA={}", loudness_range_target));
        }
        if let Some(true_peak_level) = self.true_peak_level_target {
            options.push(format!("TP={}", true_peak_level));
        }
        if let Some(measured_integrated_loudness) = self.measured_integrated_loudness {
            options.push(format!("measured_I={}", measured_integrated_loudness));
        }
        if let Some(measured_loudness_range) = self.measured_loudness_range {
            options.push(format!("measured_LRA={}", measured_loudness_range));
        }
        if let Some(measured_true_peak_level) = self.measured_true_peak_level {
            options.push(format!("measured_TP={}", measured_true_peak_level));
        }
        if let Some(measured_threshold) = self.measured_threshold {
            options.push(format!("measured_thresh={}", measured_threshold));
        }
        if let Some(offset_gain) = self.offset_gain {
            options.push(format!("offset={}", offset_gain));
        }
        if let Some(linear) = self.linear {
            options.push(format!("linear={}", if linear { "true" } else { "false" }));
        }
        if let Some(dual_mono) = self.dual_mono {
            options.push(format!(
                "dual_mono={}",
                if dual_mono { "true" } else { "false" }
            ));
        }

        write!(f, "loudnorm={}", options.join(":"))
    }
}

impl Default for LoudnessNormalization {
    fn default() -> Self {
        Self {
            integrated_loudness_target: Some(-24.0),
            loudness_range_target: Some(7.0),
            true_peak_level_target: Some(-2.0),
            measured_integrated_loudness: None,
            measured_loudness_range: None,
            measured_true_peak_level: None,
            measured_threshold: None,
            offset_gain: Some(0.0),
            linear: Some(true),
            dual_mono: Some(false),
        }
    }
}

/// Create transcoding context corresponding to the given `stream_contexts`, the
/// added filter contexts is mutable reference to objects stored in
/// `filter_graphs`.
fn init_filters(
    filter_graphs: &mut [AVFilterGraph],
    stream_contexts: Vec<StreamContext>,
) -> Result<Vec<FilteringContext<'_>>, anyhow::Error> {
    let mut filter_ctx = Vec::with_capacity(stream_contexts.len());

    for (filter_graph, stream_context) in filter_graphs.iter_mut().zip(stream_contexts) {
        let StreamContext {
            mut dec_ctx,
            stream_index,
            metadata,
            filter_config,
            stream_type,
            frame_data,
        } = stream_context;

        let direct_path = match &filter_config {
            // Software-decoded video streams whose only processing is format
            // conversion skip the filter graph and convert straight into the
            // output tensor. Hardware-decoded streams stay in the graph: its
            // slice-threaded conversion beats a single sws_scale call when
            // the CPU work is conversion only.
            FilterConfig::Video(config) => {
                let sizes = frame_data.size();
                let (dst_height, dst_width) = (sizes[1], sizes[2]);
                if let Some(device) = config.device {
                    // GPU-resident output: NVDEC frames stay in CUDA memory
                    // and NPP converts them into the CUDA output tensor. No
                    // fallback exists (frames never reach the CPU), so this
                    // path is not subject to AVTENSOR_DISABLE_DIRECT_PATH.
                    let use_bt709 = dec_ctx.colorspace == ffi::AVCOL_SPC_BT709;
                    log::debug!(
                        "Using the CUDA NPP path for stream {stream_index} on device {device}"
                    );
                    DirectPath::CudaConvert(cuda::CudaNv12ToRgb::new(
                        device,
                        use_bt709,
                        dst_width as i32,
                        dst_height as i32,
                    )?)
                } else if !is_hw_decoder(&dec_ctx)
                    && DirectVideoScaler::is_eligible(config, dst_width)
                {
                    log::debug!("Using the direct swscale path for stream {stream_index}");
                    DirectPath::VideoScale(DirectVideoScaler::new(
                        dst_width as i32,
                        dst_height as i32,
                    ))
                } else {
                    DirectPath::Disabled
                }
            }
            // Audio streams with no resampling or loudness normalization
            // would run an `anull` (no-op) graph; the sample-format
            // conversion happens in convert_audio_into_tensor either way.
            FilterConfig::Audio(config) => {
                if config.sample_rate.is_none()
                    && config.loudness_normalization.is_none()
                    && direct_paths_enabled()
                {
                    log::debug!("Using the direct audio path for stream {stream_index}");
                    DirectPath::AudioPassthrough
                } else {
                    DirectPath::Disabled
                }
            }
        };

        // Only build the filter graph when it will actually process frames:
        // configuring a graph spins up its slice-threading worker pool, which
        // is pure overhead for direct-path streams.
        let (buffersrc_ctx, buffersink_ctx) = if matches!(direct_path, DirectPath::Disabled) {
            let FilterContext {
                buffersrc_ctx,
                buffersink_ctx,
            } = init_filter(filter_graph, &mut dec_ctx, &filter_config)
                .context("Failed to initialize filter")?;
            log::debug!("Initialized filter for stream {stream_index} of type {stream_type:?}");
            (Some(buffersrc_ctx), Some(buffersink_ctx))
        } else {
            (None, None)
        };

        filter_ctx.push(FilteringContext {
            stream_type,
            stream_index,
            metadata,
            dec_ctx,
            buffersrc_ctx,
            buffersink_ctx,
            direct_path,
            frame_data,
            frame_data_ptr: 0,
            frame_pts: Vec::new(),
            finished: false,
        })
    }

    Ok(filter_ctx)
}

fn init_filter<'graph>(
    filter_graph: &'graph mut AVFilterGraph,
    dec_ctx: &mut AVCodecContext,
    filter_spec: &FilterConfig,
) -> Result<FilterContext<'graph>, anyhow::Error> {
    // Cap the graph's slice-threading pool. FFmpeg's default (0 = one worker
    // per core) builds a large thread pool for every decode on many-core
    // machines, which oversubscribes badly when many decodes run
    // concurrently. Slice threading cannot use more workers than there are
    // output rows/slices anyway, so a small pool preserves single-decode
    // filter performance.
    let graph_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(16)
        .min(16) as i32;
    unsafe { (*filter_graph.as_mut_ptr()).nb_threads = graph_threads };

    let (mut buffersrc_ctx, mut buffersink_ctx, filter_spec) = {
        match (dec_ctx.codec_type, &filter_spec) {
            (ffi::AVMEDIA_TYPE_VIDEO, FilterConfig::Video(filter_config)) => {
                let (buffersrc_ctx, buffersink_ctx) =
                    init_video_filter(filter_graph, dec_ctx, filter_config)?;
                (
                    buffersrc_ctx,
                    buffersink_ctx,
                    filter_config.to_filter_spec()?,
                )
            }
            (ffi::AVMEDIA_TYPE_AUDIO, FilterConfig::Audio(filter_config)) => {
                let (buffersrc_ctx, buffersink_ctx) =
                    init_audio_filter(filter_graph, dec_ctx, filter_config)?;
                (
                    buffersrc_ctx,
                    buffersink_ctx,
                    filter_config.to_filter_spec()?,
                )
            }
            _ => {
                bail!("Only video and audio needs filter initialization")
            }
        }
    };

    // Endpoints for the filter graph
    //
    // Yes the outputs' name is `in`, this is in alignment with upstream samples in ffmpeg / rsmpeg which may confuse you when debugging this.
    let outputs = AVFilterInOut::new(c"in", &mut buffersrc_ctx, 0);
    let inputs = AVFilterInOut::new(c"out", &mut buffersink_ctx, 0);

    let (_inputs, _outputs) = filter_graph.parse_ptr(&filter_spec, Some(inputs), Some(outputs))?;
    filter_graph.config()?;

    Ok(FilterContext {
        buffersrc_ctx,
        buffersink_ctx,
    })
}

/// Initializes the video filter graph according to the provided `filter_spec`.
///
/// Arguments:
/// - `filter_graph`: The [`AVFilterGraph`] to initialize. This should be an empty, newly allocated filter graph.
/// - `dec_ctx`: The [`AVCodecContext`] for the video stream whose output will be fed into the graph.
/// - `filter_spec`: The spec for the filter graph.
fn init_video_filter<'graph, F: VideoFilterSpec>(
    filter_graph: &'graph AVFilterGraph,
    dec_ctx: &mut AVCodecContext,
    filter_spec: &F,
) -> Result<(AVFilterContextMut<'graph>, AVFilterContextMut<'graph>), anyhow::Error> {
    log::debug!("Creating buffer/buffersink filters for video stream.");
    let buffersrc =
        AVFilter::get_by_name(c"buffer").context("filtering source element not found")?;
    let buffersink =
        AVFilter::get_by_name(c"buffersink").context("filtering sink element not found")?;

    // Hardware decoders report a GPU pixel format (e.g. AV_PIX_FMT_CUDA), but
    // frames are downloaded to system memory before filtering — declare the
    // underlying software format to the filter graph instead.
    let buffersrc_pix_fmt = if dec_ctx.pix_fmt == ffi::AV_PIX_FMT_CUDA {
        if dec_ctx.sw_pix_fmt != ffi::AV_PIX_FMT_NONE {
            dec_ctx.sw_pix_fmt
        } else {
            // NVDEC emits NV12 for 8-bit content.
            ffi::AV_PIX_FMT_NV12
        }
    } else {
        dec_ctx.pix_fmt
    };

    let mut args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
        dec_ctx.width,
        dec_ctx.height,
        buffersrc_pix_fmt,
        dec_ctx.pkt_timebase.num,
        dec_ctx.pkt_timebase.den,
        dec_ctx.sample_aspect_ratio.num,
        dec_ctx.sample_aspect_ratio.den,
    );

    // Pass color metadata from the decoder context to the buffersrc filter so that
    // FFmpeg uses the correct color matrix for YUV→RGB conversion instead of
    // guessing based on resolution.
    if let Some(name) = ffi_color_name(unsafe { ffi::av_color_space_name(dec_ctx.colorspace) }) {
        args.push_str(&format!(":colorspace={name}"));
    }
    if let Some(name) = ffi_color_name(unsafe { ffi::av_color_range_name(dec_ctx.color_range) }) {
        args.push_str(&format!(":range={name}"));
    }

    let args = &CString::new(args).unwrap();

    let buffer_src_context = filter_graph
        .create_filter_context(&buffersrc, c"in", Some(args))
        .context("Cannot create buffer source")?;

    let mut buffer_sink_context = filter_graph
        .alloc_filter_context(&buffersink, c"out")
        .context("Cannot create buffer sink")?;

    let pix_fmt = filter_spec
        .pix_fmt()
        .ok_or_else(|| anyhow!("Pixel format must be specified in video filter config"))?;
    buffer_sink_context
        .opt_set_bin(c"pix_fmts", &pix_fmt)
        .context("Cannot set output pixel format")?;

    buffer_sink_context
        .init_dict(&mut None)
        .context("Cannot initialize buffer sink")?;

    log::debug!("Buffer source and sink initialized with args: {args:?}");
    Ok((buffer_src_context, buffer_sink_context))
}

/// Initializes the audio filter graph according to the provided `filter_spec`.
///
/// Arguments
/// - `filter_graph`: The [`AVFilterGraph`] to initialize. This should be an emptpy, newly allocated filter graph.
/// - `dec_ctx`: The [`AVCodecContext`] for the audio stream whose output will be fed into the filter graph.
/// - `filter_spec`: The spec for the filter graph.
fn init_audio_filter<'graph, F: AudioFilterSpec>(
    filter_graph: &'graph AVFilterGraph,
    dec_ctx: &mut AVCodecContext,
    filter_spec: &F,
) -> Result<(AVFilterContextMut<'graph>, AVFilterContextMut<'graph>), anyhow::Error> {
    log::debug!("Creating buffer/buffersink filters for audio stream.");
    let buffersrc = AVFilter::get_by_name(c"abuffer").unwrap();
    let buffersink = AVFilter::get_by_name(c"abuffersink").unwrap();

    if dec_ctx.ch_layout.order == ffi::AV_CHANNEL_ORDER_UNSPEC {
        dec_ctx.set_ch_layout(
            AVChannelLayout::from_nb_channels(dec_ctx.ch_layout.nb_channels).into_inner(),
        );
    }

    let args = format!(
        "time_base={}/{}:sample_rate={}:sample_fmt={}:channel_layout={}",
        dec_ctx.pkt_timebase.num,
        dec_ctx.pkt_timebase.den,
        dec_ctx.sample_rate,
        // We can unwrap here, because we are sure that the given
        // sample_fmt is valid.
        get_sample_fmt_name(dec_ctx.sample_fmt)
            .unwrap()
            .to_string_lossy(),
        dec_ctx.ch_layout().describe().unwrap().to_string_lossy(),
    );
    let args = &CString::new(args).unwrap();

    let buffersrc_ctx = filter_graph
        .create_filter_context(&buffersrc, c"in", Some(args))
        .context("Cannot create audio buffer source")?;

    let mut buffersink_ctx = filter_graph
        .alloc_filter_context(&buffersink, c"out")
        .context("Cannot create audio buffer sink")?;
    buffersink_ctx
        .opt_set_bin(c"sample_fmts", &dec_ctx.sample_fmt) // Copy from decoder
        .context("Cannot set output sample format")?;
    buffersink_ctx
        .opt_set(
            c"ch_layouts",
            &dec_ctx
                .ch_layout()
                .describe()
                .context("Failed to describe channel layout")?,
        ) // Copy from decoder, TODO (rikheijdens): allow on-the-fly downmixing
        .context("Cannot set output channel layout")?;
    let sample_rate = filter_spec
        .sample_rate()
        .unwrap_or(dec_ctx.sample_rate as usize) as i32;
    buffersink_ctx
        .opt_set_bin(c"sample_rates", &sample_rate)
        .context("Cannot set output sample rate")?;

    // `av_buffersink_set_frame_size` will SIGSEGV even on FFmpeg 7.1, problem persists until
    // https://github.com/FFmpeg/FFmpeg/commit/6b402cdbf46e4398b3285277f3ff7c3654d57ce6.
    // Waiting for FFmpeg 7.2 release.
    /*
    if enc_ctx.frame_size > 0 {
        buffersink_ctx.buffersink_set_frame_size(enc_ctx.frame_size as u32);
    }
     */

    buffersink_ctx
        .init_dict(&mut None)
        .context("Cannot initialize audio buffer sink")?;

    log::debug!("Audio buffer source and sink initialized with args: {args:?}");
    Ok((buffersrc_ctx, buffersink_ctx))
}

trait FilterSpec {
    /// Convert to FFmpeg filter spec
    fn to_filter_spec(&self) -> Result<CString, anyhow::Error>;
}

trait VideoFilterSpec: FilterSpec {
    /// Output pixel format (used for video)
    fn pix_fmt(&self) -> Option<AVPixelFormat>;
}

trait AudioFilterSpec: FilterSpec {
    /// Output sample rate (used for audio)
    fn sample_rate(&self) -> Option<usize>;
}

impl FilterSpec for VideoFilterConfig {
    fn to_filter_spec(&self) -> Result<CString, anyhow::Error> {
        let spec = self.to_string();
        CString::new(spec).context("Failed to create CStr from video filter spec")
    }
}

impl VideoFilterSpec for VideoFilterConfig {
    fn pix_fmt(&self) -> Option<AVPixelFormat> {
        match self.pixel_format.as_str() {
            "rgb24" => Some(ffi::AV_PIX_FMT_RGB24),
            "gbrpf32le" => Some(ffi::AV_PIX_FMT_GBRPF32LE),
            "yuv420p" => Some(ffi::AV_PIX_FMT_YUV420P),
            _ => None, // Default to None if no pixel format is specified
        }
    }
}

impl FilterSpec for AudioFilterConfig {
    fn to_filter_spec(&self) -> Result<CString, anyhow::Error> {
        let spec = self.to_string();
        CString::new(spec).context("Failed to create CStr from audio filter spec")
    }
}

impl AudioFilterSpec for AudioFilterConfig {
    fn sample_rate(&self) -> Option<usize> {
        self.sample_rate
    }
}

enum FilterConfig {
    Video(VideoFilterConfig),
    Audio(AudioFilterConfig),
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::util::{
        mel::{create_mel_spectrogram, MelSpectrogramConfig},
        test_utils::{generate_test_video_file, init_logger, ChannelLayout, TestVideoParameters},
    };

    use super::*;
    use test_case::test_case;

    #[test]
    fn test_decode_media_defaults() -> anyhow::Result<()> {
        init_logger();

        let params = TestVideoParameters::default();
        let test_video = generate_test_video_file(&params)?;
        let start = Instant::now();
        let result = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        );
        match result {
            Ok(decoded_streams) => {
                let elapsed = start.elapsed();
                log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
                for stream in decoded_streams {
                    log::info!(
                        "Stream index: {}, type: {:?}, frames: {}",
                        stream.src_stream_index,
                        stream.stream_type(),
                        stream.decoded_frames.len()
                    );
                    assert!(!stream.decoded_frames.is_empty());

                    let first_frame = &stream.decoded_frames.first().unwrap();
                    log::info!("First frame: {:?}", first_frame);
                    let last_frame = &stream.decoded_frames.last().unwrap();
                    log::info!("Last frame: {:?}", last_frame);

                    match stream.metadata {
                        StreamMetadata::Video { frame_rate } => {
                            log::info!("Video stream - Frame rate: {}", frame_rate);
                            assert_eq!(frame_rate, params.frame_rate);
                        }
                        StreamMetadata::Audio { sample_rate } => {
                            log::info!("Audio stream - Sample rate: {}", sample_rate);
                            assert_eq!(sample_rate as usize, params.sample_rate);
                        }
                    }
                }
            }
            Err(e) => {
                panic!("Decoding failed: {}", e); // Fail the test if decoding fails
            }
        }

        // Do not specify which stream to decode -> should not yield decoded streams.
        match decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: None,
            },
            None,
        ) {
            Ok(decoded_streams) => assert!(
                decoded_streams.is_empty(),
                "Expected no streams to be decoded when no video or audio streams are requested."
            ),
            Err(e) => panic!("Decoding failed: {}", e),
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_downscale() -> anyhow::Result<()> {
        init_logger();

        let width = 320;
        let height = 180;

        let test_video = generate_test_video_file(&TestVideoParameters {
            width: width * 2,
            height: height * 2,
            ..Default::default()
        })?;
        let start = Instant::now();
        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    width: Some(width as u32),
                    height: Some(height as u32),
                    ..Default::default()
                }),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        )
        .unwrap();

        let elapsed = start.elapsed();
        log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
        for stream in decoded_streams {
            log::info!(
                "Stream index: {}, type: {:?}, frames: {}",
                stream.src_stream_index,
                stream.stream_type(),
                stream.decoded_frames.len()
            );
            assert!(!stream.decoded_frames.is_empty());

            let first_frame = &stream.decoded_frames.first().unwrap();
            log::info!("First frame: {:?}", first_frame);

            // Validate dimensions
            if let StreamType::Video = stream.stream_type() {
                if let Frame::Video {
                    width: w,
                    height: h,
                    ..
                } = first_frame
                {
                    assert_eq!(
                        *w, width as i32,
                        "Expected downscaled width to match request"
                    );
                    assert_eq!(
                        *h, height as i32,
                        "Expected downscaled height to match request"
                    );
                } else {
                    panic!("Expected video frame");
                }

                let shape = stream.data.unwrap().size();
                assert_eq!(
                    shape[1], height as i64,
                    "Expected downscaled height to match request"
                );
                assert_eq!(
                    shape[2], width as i64,
                    "Expected downscaled width to match request"
                );
            }
            let last_frame = &stream.decoded_frames.last().unwrap();
            log::info!("Last frame: {:?}", last_frame);
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_subsample_frames() -> anyhow::Result<()> {
        init_logger();

        let source_frame_rate = 30.0;
        let target_frame_rate = 24.0;
        let duration = Duration::from_secs(2);

        let test_video = generate_test_video_file(&TestVideoParameters {
            frame_rate: source_frame_rate,
            duration,
            ..Default::default()
        })?;
        let start = Instant::now();
        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    frame_rate: Some(target_frame_rate),
                    ..Default::default()
                }),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        )
        .unwrap();

        let elapsed = start.elapsed();
        log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
        for stream in decoded_streams {
            log::info!(
                "Stream index: {}, type: {:?}, frames: {}",
                stream.src_stream_index,
                stream.stream_type(),
                stream.decoded_frames.len()
            );
            assert!(!stream.decoded_frames.is_empty());

            match stream.metadata {
                StreamMetadata::Video { frame_rate } => {
                    log::info!("Video stream - Frame rate: {}", frame_rate);
                    assert_eq!(frame_rate, target_frame_rate);

                    let shape = stream.data.unwrap().size();
                    assert_eq!(
                        shape[0],
                        (duration.as_secs() * (target_frame_rate as u64)) as i64
                    )
                }
                StreamMetadata::Audio { .. } => {}
            }
            // Validate dimensions
            let first_frame = &stream.decoded_frames.first().unwrap();
            log::info!("First frame: {:?}", first_frame);
            let last_frame = &stream.decoded_frames.last().unwrap();
            log::info!("Last frame: {:?}", last_frame);
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_resample_audio() -> anyhow::Result<()> {
        init_logger();

        let source_sample_rate = 48000;
        let target_sample_rate = 16000;
        let duration = Duration::from_secs(2);

        let test_video = generate_test_video_file(&TestVideoParameters {
            sample_rate: source_sample_rate,
            duration,
            ..Default::default()
        })?;
        let start = Instant::now();
        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: Some(vec![AudioStreamRequest {
                    sample_rate: Some(target_sample_rate as u32),
                    ..Default::default()
                }]),
            },
            None,
        )
        .unwrap();

        let elapsed = start.elapsed();
        log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
        for stream in decoded_streams {
            log::info!(
                "Stream index: {}, type: {:?}, frames: {}",
                stream.src_stream_index,
                stream.stream_type(),
                stream.decoded_frames.len()
            );
            assert!(!stream.decoded_frames.is_empty());

            match stream.metadata {
                StreamMetadata::Video { .. } => {}
                StreamMetadata::Audio { sample_rate } => {
                    assert_eq!(sample_rate as usize, target_sample_rate);
                    assert_eq!(
                        stream.data.unwrap().size()[1],
                        (duration.as_secs() * (target_sample_rate as u64)) as i64
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_resample_audio_loudness_normalization() -> anyhow::Result<()> {
        init_logger();

        let source_sample_rate = 48000;
        let target_sample_rate = 16000;
        let duration = Duration::from_secs(2);

        let test_video = generate_test_video_file(&TestVideoParameters {
            sample_rate: source_sample_rate,
            duration,
            ..Default::default()
        })?;
        let start = Instant::now();
        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: Some(vec![AudioStreamRequest {
                    sample_rate: Some(target_sample_rate as u32),
                    loudness_normalization: Some(LoudnessNormalization {
                        integrated_loudness_target: Some(-18.0),
                        loudness_range_target: Some(7.0),
                        true_peak_level_target: Some(-2.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                }]),
            },
            None,
        )
        .unwrap();

        let elapsed = start.elapsed();
        log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
        for stream in decoded_streams {
            log::info!(
                "Stream index: {}, type: {:?}, frames: {}",
                stream.src_stream_index,
                stream.stream_type(),
                stream.decoded_frames.len()
            );
            assert!(!stream.decoded_frames.is_empty());

            match stream.metadata {
                StreamMetadata::Video { .. } => {}
                StreamMetadata::Audio { sample_rate } => {
                    assert_eq!(sample_rate as usize, target_sample_rate);
                    assert_eq!(
                        stream.data.unwrap().size()[1],
                        (duration.as_secs() * (target_sample_rate as u64)) as i64
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_buffer_full_libopenh264() -> anyhow::Result<()> {
        init_logger();

        let start = Instant::now();
        let params = TestVideoParameters {
            width: 240,
            height: 224,
            frame_rate: 60.0,
            duration: Duration::from_secs(10),
            video_codec: "libopenh264".to_string(), // N.b. need to use libopenh264 to repro
            audio_codec: "aac".to_string(),
            sample_rate: 44100,
            channel_layout: ChannelLayout::Mono,
            ..Default::default()
        };
        let test_video = generate_test_video_file(&params)?;
        let gcs_decoded = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: Some(7.319999999999999), // (rikheijdens): The end_time here is significant and necessary for reproducing.
                video_stream: Some(VideoStreamRequest {
                    frame_rate: Some(10.0),
                    width: Some(256),
                    height: Some(256),
                    ..Default::default()
                }),
                audio_streams: None,
            },
            None,
        )
        .unwrap();

        let elapsed = start.elapsed();
        log::info!("Decoded {} streams in {:?}", gcs_decoded.len(), elapsed);

        for gcs_stream in gcs_decoded.into_iter() {
            log::info!(
                "Stream index: {}, type: {:?}, frames: {}",
                gcs_stream.src_stream_index,
                gcs_stream.stream_type(),
                gcs_stream.decoded_frames.len()
            );
            assert!(!gcs_stream.decoded_frames.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_seek() -> anyhow::Result<()> {
        init_logger();

        let start_time = 40.;
        let end_time = 45.;

        let test_video = generate_test_video_file(&TestVideoParameters {
            frame_rate: 30.0,
            sample_rate: 16000,
            width: 1920,
            height: 1080,
            duration: Duration::from_secs(60),
            ..Default::default()
        })?;
        let start = Instant::now();
        let result = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: Some(start_time),
                end_time: Some(end_time),
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        );
        match result {
            Ok(decoded_streams) => {
                let elapsed = start.elapsed();
                log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
                for stream in decoded_streams {
                    log::info!(
                        "Stream index: {}, type: {:?}, frames: {}",
                        stream.src_stream_index,
                        stream.stream_type(),
                        stream.decoded_frames.len()
                    );
                    assert!(!stream.decoded_frames.is_empty());

                    let first_frame = &stream.decoded_frames.first().unwrap();
                    assert!(
                        first_frame.pts_seconds() >= start_time,
                        "First frame PTS is less than start time"
                    );

                    log::info!("First frame: {:?}", first_frame);
                    let last_frame = &stream.decoded_frames.last().unwrap();
                    assert!(
                        last_frame.pts_seconds() <= end_time,
                        "Last frame PTS is greater than end time"
                    );
                    log::info!("Last frame: {:?}", last_frame);
                }
            }
            Err(e) => {
                panic!("Decoding failed: {}", e); // Fail the test if decoding fails
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_video_only() -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_test_video_file(&TestVideoParameters::default())?;

        // Specify to decode video only, should only yield the video stream.
        match decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
            },
            None,
        ) {
            Ok(decoded_streams) => {
                assert_eq!(decoded_streams.len(), 1);
                assert_eq!(decoded_streams[0].stream_type(), StreamType::Video);
            }
            Err(e) => panic!("Decoding failed: {}", e),
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_audio_only() -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_test_video_file(&TestVideoParameters::default())?;

        // Requesting audio only should only yield the audio stream.
        match decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        ) {
            Ok(decoded_streams) => {
                assert_eq!(decoded_streams.len(), 1);
                assert_eq!(decoded_streams[0].stream_type(), StreamType::Audio);
            }
            Err(e) => panic!("Decoding failed: {}", e),
        }

        Ok(())
    }

    #[test_case(ChannelLayout::Stereo, vec![5.5]; "stereo")]
    // TODO (rikheijdens): It is not entirely clear to me why we need more lenient thresholds
    // for the center and LFE channel. Is there any chance we need to normalize the energy/loudness of
    // individual channels?
    #[test_case(ChannelLayout::Surround5_1, vec![15., 150., 9500., 15., 15.]; "5.1 surround")]
    fn test_decode_multi_channel_audio(
        channel_layout: ChannelLayout,
        distance_thresholds: Vec<f64>,
    ) -> anyhow::Result<()> {
        init_logger();

        let params = TestVideoParameters {
            channel_layout,
            sample_rate: 44100,
            ..Default::default()
        };
        let test_video = generate_test_video_file(&params)?;

        let num_channels = channel_layout.num_channels();
        let expected_num_samples = params.sample_rate as u64 * params.duration.as_secs();

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 1);
        assert_eq!(decoded_streams[0].stream_type(), StreamType::Audio);

        let audio_stream = &decoded_streams[0];
        // Verify shape of the audio tensor is correct.
        let data = audio_stream.data.as_ref().unwrap();
        assert_eq!(
            data.size(),
            vec![num_channels as i64, expected_num_samples as i64]
        );

        let mel_config = MelSpectrogramConfig {
            sample_rate: params.sample_rate,
            fft_size: 2048,
            hop_size: 512,
            num_mels: 128,
        };

        let first_channel_data = data.get(0);
        let first_channel_samples = first_channel_data.i(0..100);
        let first_channel_mel_spec = create_mel_spectrogram(&first_channel_data, &mel_config)?;
        log::debug!(
            "Channel #0 Mel Spectrogram dimensions: {:?}",
            first_channel_mel_spec.size()
        );

        // For the test video, audio across all channels should be duplicated, verify this is the case.
        // To check whether audio is similar we compute mel spectrograms, we then require the euclidean distance
        // between the spectrograms to be below a certain threshold.
        for channel in 1..num_channels {
            let channel_data = data.get(channel as i64);
            let mel_spec = create_mel_spectrogram(&channel_data, &mel_config)?;
            let channel_samples = channel_data.i(0..100);
            log::debug!(
                "Channel #0 samples: {}, Channel #{} samples: {}",
                first_channel_samples,
                channel,
                channel_samples
            );

            log::debug!(
                "Channel #0 mel: {}, Channel #{} mel: {}",
                first_channel_mel_spec,
                channel,
                mel_spec
            );

            let dist = (&first_channel_mel_spec - mel_spec).abs().sum(None);
            let dist = dist.double_value(&[]);
            log::debug!(
                "Channel #{}, Sum of absolute differences: {}",
                channel,
                dist
            );
            assert!(dist < distance_thresholds[channel - 1], "Sum of absolute differences between mel spectrograms of channel #0 and channel #{channel} is too high: {dist}")
        }
        Ok(())
    }

    #[test]
    fn test_decode_video_end_time() -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters::default())?;
        let frame_rate = 30;
        let sample_rate = 48000;
        let end_time = 2.0;
        let start = Instant::now();
        let result = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: Some(end_time),
                video_stream: Some(VideoStreamRequest {
                    frame_rate: Some(frame_rate as f64),
                    ..Default::default()
                }),
                audio_streams: Some(vec![AudioStreamRequest {
                    sample_rate: Some(sample_rate),
                    ..Default::default()
                }]),
            },
            None,
        );
        match result {
            Ok(decoded_streams) => {
                let elapsed = start.elapsed();
                log::info!("Decoded {} streams in {:?}", decoded_streams.len(), elapsed);
                for stream in decoded_streams {
                    log::info!(
                        "Stream index: {}, type: {:?}, frames: {}",
                        stream.src_stream_index,
                        stream.stream_type(),
                        stream.decoded_frames.len()
                    );

                    match stream.stream_type() {
                        StreamType::Video => {
                            let num_frames: usize = stream
                                .decoded_frames
                                .iter()
                                .map(|f| *f.data().size().first().unwrap_or(&0) as usize)
                                .sum();
                            let num_expected_frames = (frame_rate as f64 * end_time) as usize; // TODO (rikheijdens): Should this be 60 or 59 for 2 seconds of video at 30 FPS
                            log::info!(
                                "Total video frames: {}, stream.decoded_frames.len(): {}",
                                num_frames,
                                stream.decoded_frames.len()
                            );
                            assert_eq!(num_frames, num_expected_frames);
                        }
                        StreamType::Audio => {
                            // For audio, we expect the number of frames to be less than or equal to the sample rate times the end time.
                            let expected_num_samples = (sample_rate as f64 * end_time) as usize;
                            let num_audio_samples: usize = stream
                                .decoded_frames
                                .iter()
                                .map(|f| *f.data().size().last().unwrap_or(&0) as usize)
                                .sum();
                            let num_samples_in_last_frame = stream
                                .decoded_frames
                                .last()
                                .map(|f| f.data().size().last().copied().unwrap_or(0))
                                .unwrap_or(0);

                            log::info!(
                                "Total num audio samples: {}, stream.decoded_frames.len(): {}, last frame shape: {:?}",
                                num_audio_samples,
                                stream.decoded_frames.len(),
                                stream.decoded_frames.last().map(|f| f.data().size())
                            );

                            // TODO (rikheijdens): Need to trim to the exact right size but we can also do this at a higher level.
                            assert!(num_audio_samples >= expected_num_samples);
                            assert!(
                                num_audio_samples
                                    < expected_num_samples + num_samples_in_last_frame as usize
                            );
                        }
                    }
                }
            }
            Err(e) => {
                panic!("Decoding failed: {}", e); // Fail the test if decoding fails
            }
        }

        Ok(())
    }

    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(30.0),
            width: Some(640),
            height: Some(480),
            pixel_format: "rgb24".to_string(),
            device: None,
            ..Default::default()
        },
        "fps=30,scale=width=640:height=480,format=pix_fmts=rgb24";
        "all options provided"
    )]
    #[test_case(
        VideoFilterConfig {
            pixel_format: "rgb24".to_string(),
            device: None,
            ..Default::default()
        },
        "format=pix_fmts=rgb24"; // Default is to convert to RGB24.
        "no options provided"
    )]
    #[test_case(
        VideoFilterConfig {
            pixel_format: "rgb24".to_string(),
            device: None,
            source_color: SourceColorInfo {
                color_trc: Some("bt709".to_string()),
                color_primaries: Some("bt709".to_string()),
                colorspace: Some("bt709".to_string()),
                color_range: Some("tv".to_string()),
            },
            ..Default::default()
        },
        "format=pix_fmts=rgb24";
        "SDR BT.709 source produces no extra filters"
    )]
    #[test_case(
        VideoFilterConfig {
            pixel_format: "rgb24".to_string(),
            device: None,
            source_color: SourceColorInfo {
                color_trc: Some("arib-std-b67".to_string()),
                color_primaries: Some("bt2020".to_string()),
                colorspace: Some("bt2020nc".to_string()),
                color_range: Some("tv".to_string()),
            },
            ..Default::default()
        },
        "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,format=pix_fmts=rgb24";
        "HLG BT.2020 source triggers HDR tone mapping pipeline"
    )]
    #[test_case(
        VideoFilterConfig {
            pixel_format: "rgb24".to_string(),
            device: None,
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                colorspace: Some("bt2020nc".to_string()),
                color_range: Some("tv".to_string()),
            },
            ..Default::default()
        },
        "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,format=pix_fmts=rgb24";
        "PQ BT.2020 source triggers HDR tone mapping pipeline"
    )]
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            width: Some(1920),
            height: Some(1080),
            pixel_format: "rgb24".to_string(),
            device: None,
            source_color: SourceColorInfo {
                color_trc: Some("arib-std-b67".to_string()),
                color_primaries: Some("bt2020".to_string()),
                ..Default::default()
            },
        },
        "fps=24,scale=width=1920:height=1080,zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,format=pix_fmts=rgb24";
        "HDR with fps and scale options"
    )]
    fn test_video_filter_display(filter: VideoFilterConfig, expected: &str) {
        assert_eq!(filter.to_string(), expected);
        assert_eq!(filter.pix_fmt(), Some(ffi::AV_PIX_FMT_RGB24));
        assert_eq!(
            filter.to_filter_spec().unwrap(),
            CString::new(expected).unwrap()
        );
    }

    #[test_case(
        AudioFilterConfig {
            sample_rate: Some(44100),
            loudness_normalization: None
        },
        "aresample=44100";
        "sample rate provided"
    )]
    #[test_case(
        AudioFilterConfig {
            sample_rate: Some(44100),
            loudness_normalization: Some(LoudnessNormalization {
                integrated_loudness_target: Some(-18.0),
                loudness_range_target: Some(7.0),
                true_peak_level_target: Some(-1.0),
                measured_integrated_loudness: None,
                measured_loudness_range: None,
                measured_true_peak_level: None,
                measured_threshold: None,
                offset_gain: None,
                linear: None,
                dual_mono: None,
            })
        },
        "loudnorm=I=-18:LRA=7:TP=-1,aresample=44100";
        "sample rate and loudness normalization"
    )]
    #[test_case(
        AudioFilterConfig::default(),
        "anull"; // Default is a no-op filter.
        "no options provided"
    )]
    fn test_audio_filter_display(filter: AudioFilterConfig, expected: &str) {
        assert_eq!(filter.to_string(), expected);
        assert_eq!(
            filter.to_filter_spec().unwrap(),
            CString::new(expected).unwrap()
        );
    }

    #[test]
    fn test_tch_slice_error_handling() -> anyhow::Result<()> {
        let tensor = tch::Tensor::randn([2, 2], (tch::Kind::Float, tch::Device::Cpu));
        let result = tensor.f_i((0, 3..4));
        match result {
            Ok(_) => panic!("Expected error when slicing out of bounds"),
            Err(tch::TchError::Torch(e)) => {
                assert!(e.contains("start out of range"));
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
        Ok(())
    }

    #[test_case(10.0, Seek { start_time: None, end_time: None}, 10.0; "no seeking")]
    #[test_case(10.0, Seek { start_time: None, end_time: Some(5.0)}, 5.0; "end time")]
    #[test_case(10.0, Seek { start_time: Some(2.0), end_time: Some(5.0)}, 3.0; "start and end time")]
    #[test_case(10.0, Seek { start_time: Some(2.0), end_time: None}, 8.0; "start time")]
    fn test_asset_duration(stream_duration: f64, seek: Seek, expected: f64) -> anyhow::Result<()> {
        let asset_duration = asset_duration(stream_duration, &seek)?;
        let delta = (asset_duration - expected).abs();
        assert!(
            delta < 0.001,
            "Expected duration: {}, got: {}",
            expected,
            asset_duration
        );
        Ok(())
    }

    #[test]
    fn test_asset_duration_edge_cases() {
        assert!(asset_duration(
            10.0,
            &Seek {
                start_time: Some(15.0),
                end_time: None
            }
        )
        .is_err());

        assert!(asset_duration(
            10.0,
            &Seek {
                start_time: Some(15.0),
                end_time: Some(5.0)
            }
        )
        .is_err());
    }

    #[test]
    fn test_partial_eq_frame() {
        let video_frame = Frame::Video {
            width: 320,
            height: 240,
            pts: 1,
            best_effort_timestamp: 1,
            data: tch::Tensor::zeros([320, 240, 3], (tch::Kind::Uint8, tch::Device::Cpu)),
            time_base: AVRational { num: 0, den: 1 },
        };

        let identical_video_frame = Frame::Video {
            width: 320,
            height: 240,
            pts: 1,
            best_effort_timestamp: 1,
            data: tch::Tensor::zeros([320, 240, 3], (tch::Kind::Uint8, tch::Device::Cpu)),
            time_base: AVRational { num: 0, den: 1 },
        };

        let different_frame = Frame::Video {
            width: 320,
            height: 240,
            pts: 2,
            best_effort_timestamp: 2,
            data: tch::Tensor::zeros([320, 240, 3], (tch::Kind::Uint8, tch::Device::Cpu)),
            time_base: AVRational { num: 0, den: 1 },
        };

        assert_eq!(video_frame, identical_video_frame);
        assert_ne!(video_frame, different_frame);
    }

    #[test]
    fn test_loudness_normalization() {
        let norm = LoudnessNormalization {
            integrated_loudness_target: Some(-23.0),
            loudness_range_target: Some(5.0),
            true_peak_level_target: Some(-2.0),
            measured_integrated_loudness: Some(-30.0),
            measured_loudness_range: Some(10.0),
            measured_true_peak_level: Some(-2.0),
            measured_threshold: Some(-40.0),
            offset_gain: Some(0.0),
            linear: Some(true),
            dual_mono: Some(true),
        };

        assert_eq!(norm.to_string(), "loudnorm=I=-23:LRA=5:TP=-2:measured_I=-30:measured_LRA=10:measured_TP=-2:measured_thresh=-40:offset=0:linear=true:dual_mono=true");
    }

    // ---------------------------------------------------------------------
    // Synthetic edge-case media: FLAC audio, fractional frame rates,
    // variable frame rates, misaligned A/V start times. All generated
    // locally with FFmpeg.
    // ---------------------------------------------------------------------

    #[test]
    fn test_decode_flac_audio() -> anyhow::Result<()> {
        init_logger();

        let sample_rate = 32000;
        let flac =
            crate::util::test_utils::generate_test_flac_file(sample_rate, Duration::from_secs(2))?;

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(flac.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 1, "Expected a single audio stream");
        for stream in decoded_streams {
            match stream.metadata {
                StreamMetadata::Video { .. } => panic!("Expected no video stream"),
                StreamMetadata::Audio { sample_rate: sr } => {
                    assert_eq!(sr as usize, sample_rate);
                    let data = stream.data.unwrap();
                    // ~2s of audio at the source sample rate.
                    let num_samples = *data.size().last().unwrap();
                    let expected = 2 * sample_rate as i64;
                    assert!(
                        (num_samples - expected).abs() <= sample_rate as i64 / 10,
                        "Expected ~{expected} samples, got {num_samples}"
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_no_phantom_trailing_frame_fractional_fps() -> anyhow::Result<()> {
        init_logger();

        // 50 frames at 24000/1001 fps: duration * frame_rate floats to just
        // above 50, which must not produce a 51st (black) frame.
        let video = crate::util::test_utils::generate_fractional_fps_video(50)?;

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
            },
            None,
        )?;

        for stream in decoded_streams {
            let num_tensor_frames = stream.data.as_ref().unwrap().size()[0] as usize;
            assert_eq!(
                stream.decoded_frames.len(),
                num_tensor_frames,
                "Output tensor frame count must match the number of decoded frames"
            );
            assert_eq!(num_tensor_frames, 50, "Expected exactly 50 frames");
        }

        Ok(())
    }

    #[test]
    fn test_decode_vfr_video_pts_monotonic() -> anyhow::Result<()> {
        init_logger();

        let source = generate_test_video_file(&TestVideoParameters::default())?;
        let vfr = crate::util::test_utils::make_vfr_video(source.path())?;

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(vfr.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
            },
            None,
        )?;

        for stream in decoded_streams {
            assert!(!stream.decoded_frames.is_empty());
            assert_eq!(
                stream.decoded_frames.len(),
                stream.data.as_ref().unwrap().size()[0] as usize,
                "Output tensor frame count must match the number of decoded frames"
            );
            let mut prev = f64::NEG_INFINITY;
            for frame in &stream.decoded_frames {
                let pts = frame.pts_seconds();
                assert!(pts > prev, "Frame PTS must be strictly increasing");
                prev = pts;
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_av_start_time_offset() -> anyhow::Result<()> {
        init_logger();

        let offset = 1.5;
        let source = generate_test_video_file(&TestVideoParameters::default())?;
        let offset_video = crate::util::test_utils::make_av_offset_video(source.path(), offset)?;

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(offset_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 2, "Expected video and audio streams");
        for stream in decoded_streams {
            assert!(!stream.decoded_frames.is_empty());
            let first_pts = stream.decoded_frames.first().unwrap().pts_seconds();
            match stream.metadata {
                StreamMetadata::Video { .. } => {
                    assert!(
                        first_pts < 0.5,
                        "Video should start near 0, first PTS = {first_pts}"
                    );
                }
                StreamMetadata::Audio { .. } => {
                    // The container quantizes the offset to the codec frame
                    // size, so allow some slack below the requested offset.
                    assert!(
                        first_pts >= offset - 0.5,
                        "Audio should start near the {offset}s offset, first PTS = {first_pts}"
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_media_http() -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters::default())?;
        let url = crate::util::test_utils::serve_file_over_http(test_video.path())?;

        // Non-gs:// URLs are handed to FFmpeg's own protocol layer; http(s)
        // is how presigned cloud-storage URLs (S3, GCS, Azure, ...) decode.
        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(url),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 2, "Expected video and audio streams");
        for stream in decoded_streams {
            assert!(!stream.decoded_frames.is_empty());
            assert!(stream.data.is_some(), "Expected decoded data");
        }

        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU with NVDEC"]
    fn test_decode_media_hardware_acceleration() -> anyhow::Result<()> {
        init_logger();

        // NVDEC only supports 4:2:0 chroma for H.264; the generator's default
        // (testsrc RGB input) makes x264 produce 4:4:4.
        let test_video = generate_test_video_file(&TestVideoParameters {
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();

        let sw = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path.clone()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
            },
            None,
        )?;
        let hw = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    hardware_acceleration: Some(true),
                    ..Default::default()
                }),
                audio_streams: None,
            },
            None,
        )?;

        let sw_data = sw[0].data.as_ref().unwrap().to_kind(tch::Kind::Float);
        let hw_data = hw[0].data.as_ref().unwrap().to_kind(tch::Kind::Float);
        assert_eq!(sw_data.size(), hw_data.size());

        // NVDEC H.264 decoding is spec-exact; NV12 and yuv420p carry the same
        // values, so after the same RGB conversion the outputs should agree
        // up to tiny rounding differences.
        let diff = sw_data.f_sub(&hw_data)?.abs();
        let mean_diff: f64 = diff.mean(tch::Kind::Float).try_into()?;
        assert!(
            mean_diff < 1.0,
            "hardware and software decode diverge: mean |diff| = {mean_diff}"
        );

        Ok(())
    }

    #[test]
    fn test_compute_hw_resize() {
        let req = |w: Option<u32>, h: Option<u32>| VideoStreamRequest {
            width: w,
            height: h,
            ..Default::default()
        };
        // No size requested -> no GPU resize.
        assert_eq!(compute_hw_resize(1920, 1080, &req(None, None)), None);
        // Both dimensions given.
        assert_eq!(
            compute_hw_resize(1920, 1080, &req(Some(512), Some(288))),
            Some((512, 288))
        );
        // Aspect-preserving when only one dimension is given.
        assert_eq!(
            compute_hw_resize(1920, 1080, &req(Some(640), None)),
            Some((640, 360))
        );
        assert_eq!(
            compute_hw_resize(1920, 1080, &req(None, Some(360))),
            Some((640, 360))
        );
        // Odd dimensions round down to even (NVDEC requirement); the filter
        // graph still produces the exact requested size afterwards.
        assert_eq!(
            compute_hw_resize(1920, 1080, &req(Some(511), Some(287))),
            Some((510, 286))
        );
        // Upscaling is left to the filter graph.
        assert_eq!(
            compute_hw_resize(640, 480, &req(Some(1280), Some(960))),
            None
        );
    }

    #[test]
    fn test_device_with_hardware_acceleration_false_is_rejected() -> anyhow::Result<()> {
        init_logger();
        // The contradiction is caught before any CUDA machinery is touched,
        // so this runs on CPU-only machines.
        let test_video = generate_test_video_file(&TestVideoParameters::default())?;
        let err = match decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    hardware_acceleration: Some(false),
                    device: Some(0),
                    ..Default::default()
                }),
                audio_streams: None,
            },
            None,
        ) {
            Ok(_) => anyhow::bail!("expected the contradiction to be rejected"),
            Err(err) => err,
        };
        assert!(
            format!("{err:#}").contains("cannot be combined with hardware_acceleration=false"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU with NVDEC"]
    fn test_decode_media_gpu_resident_output() -> anyhow::Result<()> {
        init_logger();
        // The test binary must load libtorch_cuda (which registers the CUDA
        // kernels) explicitly; in Python this happens via `import torch`.
        let libtorch_cuda = unsafe {
            libloading::os::unix::Library::open(
                Some("libtorch_cuda.so"),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
            )
        }
        .context("loading libtorch_cuda.so (is this a CUDA torch build?)")?;
        std::mem::forget(libtorch_cuda);

        let test_video = generate_test_video_file(&TestVideoParameters {
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();

        let cpu = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path.clone()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    hardware_acceleration: Some(true),
                    ..Default::default()
                }),
                audio_streams: None,
            },
            None,
        )?;
        // device alone implies hardware acceleration.
        let gpu = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    device: Some(0),
                    ..Default::default()
                }),
                audio_streams: None,
            },
            None,
        )?;

        let cpu_data = cpu[0].data.as_ref().unwrap();
        let gpu_data = gpu[0].data.as_ref().unwrap();
        assert_eq!(gpu_data.device(), tch::Device::Cuda(0));
        assert_eq!(cpu_data.size(), gpu_data.size());
        assert_eq!(cpu[0].frame_pts, gpu[0].frame_pts);

        // NPP and swscale round differently; require close, not identical.
        let diff = gpu_data
            .to_device(tch::Device::Cpu)
            .to_kind(tch::Kind::Float)
            .f_sub(&cpu_data.to_kind(tch::Kind::Float))?
            .abs();
        let mean_diff: f64 = diff.mean(tch::Kind::Float).try_into()?;
        let max_diff: f64 = diff.max().try_into()?;
        log::info!("GPU vs CPU decode: mean |diff| {mean_diff:.3}, max {max_diff}");
        assert!(
            mean_diff < 1.5,
            "GPU output diverges from CPU decode (mean |diff| {mean_diff})"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU with NVDEC"]
    fn test_decode_media_hardware_acceleration_gpu_resize() -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters {
            width: 1280,
            height: 720,
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();

        let request = |hw: bool| MediaDecodeRequest {
            source: MediaSource::Uri(path.clone()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                width: Some(320),
                height: Some(180),
                hardware_acceleration: Some(hw),
                ..Default::default()
            }),
            audio_streams: None,
        };

        let sw = decode_media(request(false), None)?;
        let hw = decode_media(request(true), None)?;

        let sw_data = sw[0].data.as_ref().unwrap().to_kind(tch::Kind::Float);
        let hw_data = hw[0].data.as_ref().unwrap().to_kind(tch::Kind::Float);
        assert_eq!(sw_data.size(), hw_data.size());
        assert_eq!(&sw_data.size()[1..], &[180, 320, 3]);

        // The GPU resizer and swscale use different interpolation kernels, so
        // outputs are similar but not identical.
        let diff = sw_data.f_sub(&hw_data)?.abs();
        let mean_diff: f64 = diff.mean(tch::Kind::Float).try_into()?;
        log::info!("GPU-resize vs software-scale mean |diff| = {mean_diff:.2}");
        assert!(
            mean_diff < 8.0,
            "GPU-resized and software-scaled decodes diverge: mean |diff| = {mean_diff}"
        );

        Ok(())
    }

    #[test]
    fn test_source_color_info_is_hdr() {
        // HLG + BT.2020 → HDR
        assert!(SourceColorInfo {
            color_trc: Some("arib-std-b67".to_string()),
            color_primaries: Some("bt2020".to_string()),
            ..Default::default()
        }
        .is_hdr());

        // PQ + BT.2020 → HDR
        assert!(SourceColorInfo {
            color_trc: Some("smpte2084".to_string()),
            color_primaries: Some("bt2020".to_string()),
            ..Default::default()
        }
        .is_hdr());

        // Wide gamut alone → HDR
        assert!(SourceColorInfo {
            color_primaries: Some("bt2020".to_string()),
            ..Default::default()
        }
        .is_hdr());

        // BT.709 → not HDR
        assert!(!SourceColorInfo {
            color_trc: Some("bt709".to_string()),
            color_primaries: Some("bt709".to_string()),
            ..Default::default()
        }
        .is_hdr());

        // Unspecified → not HDR
        assert!(!SourceColorInfo::default().is_hdr());
    }

    #[test]
    fn test_decode_video_with_color_metadata() -> anyhow::Result<()> {
        init_logger();

        // Generate a test video tagged with BT.2020 HLG color metadata.
        let hdr_file = generate_test_video_file(&TestVideoParameters {
            width: 320,
            height: 240,
            duration: Duration::from_secs(1),
            colorspace: Some("bt2020nc".to_string()),
            color_primaries: Some("bt2020".to_string()),
            color_trc: Some("arib-std-b67".to_string()),
            color_range: Some("tv".to_string()),
            ..Default::default()
        })?;

        // Generate same content tagged as BT.709.
        let sdr_file = generate_test_video_file(&TestVideoParameters {
            width: 320,
            height: 240,
            duration: Duration::from_secs(1),
            colorspace: Some("bt709".to_string()),
            color_primaries: Some("bt709".to_string()),
            color_trc: Some("bt709".to_string()),
            color_range: Some("tv".to_string()),
            ..Default::default()
        })?;

        let hdr_request = MediaDecodeRequest {
            source: MediaSource::Uri(hdr_file.path().to_str().unwrap().into()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                frame_rate: Some(30.0),
                ..Default::default()
            }),
            audio_streams: None,
        };

        let sdr_request = MediaDecodeRequest {
            source: MediaSource::Uri(sdr_file.path().to_str().unwrap().into()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                frame_rate: Some(30.0),
                ..Default::default()
            }),
            audio_streams: None,
        };

        let hdr_result = decode_media(hdr_request, None)?;
        let sdr_result = decode_media(sdr_request, None)?;

        // Both should decode successfully with video streams.
        assert!(!hdr_result.is_empty(), "HDR decode produced no output");
        assert!(!sdr_result.is_empty(), "SDR decode produced no output");

        // Get video data tensors.
        let hdr_data = hdr_result[0]
            .data
            .as_ref()
            .expect("HDR video should have data");
        let sdr_data = sdr_result[0]
            .data
            .as_ref()
            .expect("SDR video should have data");
        assert_eq!(hdr_data.size(), sdr_data.size());

        // The decoded RGB pixel values should differ because:
        // - HDR decode goes through zscale→tonemap→zscale pipeline
        // - SDR decode goes through plain format=rgb24
        let diff = hdr_data.f_sub(sdr_data)?.abs();
        let mean_diff: f64 = diff.mean(tch::Kind::Float).try_into()?;
        log::info!(
            "Mean absolute difference between HDR and SDR decode: {:.2}",
            mean_diff
        );
        assert!(
            mean_diff > 1.0,
            "HDR and SDR decodes should produce different pixel values (mean diff: {mean_diff})"
        );

        Ok(())
    }

    /// Decodes only the video stream of `path` and returns its output tensor
    /// and per-frame pts.
    fn decode_video_tensor(source: MediaSource) -> anyhow::Result<(tch::Tensor, Vec<f64>)> {
        let mut streams = decode_media(
            MediaDecodeRequest {
                source,
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
            },
            None,
        )?;
        let stream = streams.remove(0);
        Ok((
            stream.data.context("video stream should have data")?,
            stream.frame_pts,
        ))
    }

    /// The direct swscale path must produce output identical to the filter
    /// graph, both for untagged and BT.709-tagged sources.
    #[test_case(None, None; "untagged")]
    #[test_case(Some("bt709"), Some("tv"); "bt709_tv")]
    fn test_direct_scale_matches_filter_graph(
        colorspace: Option<&str>,
        color_range: Option<&str>,
    ) -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters {
            colorspace: colorspace.map(String::from),
            color_range: color_range.map(String::from),
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();

        // Default parameters (640 px wide) are eligible for the direct path.
        let (direct, direct_pts) = decode_video_tensor(MediaSource::Uri(path.clone()))?;

        // Force the filter-graph path. The env var may briefly affect tests
        // decoding in parallel, which is benign: both paths are correct.
        std::env::set_var("AVTENSOR_DISABLE_DIRECT_PATH", "1");
        let graph = decode_video_tensor(MediaSource::Uri(path));
        std::env::remove_var("AVTENSOR_DISABLE_DIRECT_PATH");
        let (graph, graph_pts) = graph?;

        assert_eq!(direct.size(), graph.size());
        assert_eq!(direct_pts, graph_pts);
        let max_diff: f64 = direct
            .f_sub(&graph)?
            .abs()
            .max()
            .to_kind(tch::Kind::Float)
            .try_into()?;
        assert_eq!(
            max_diff, 0.0,
            "direct swscale output must be identical to the filter graph"
        );
        Ok(())
    }

    /// The direct audio path must produce output identical to the `anull`
    /// filter graph it replaces.
    #[test_case(ChannelLayout::Mono; "mono")]
    #[test_case(ChannelLayout::Stereo; "stereo")]
    fn test_direct_audio_matches_filter_graph(channel_layout: ChannelLayout) -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters {
            channel_layout,
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();

        let decode_audio = |path: String| -> anyhow::Result<tch::Tensor> {
            let mut streams = decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(path),
                    start_time: None,
                    end_time: None,
                    video_stream: None,
                    audio_streams: Some(vec![AudioStreamRequest::default()]),
                },
                None,
            )?;
            streams
                .remove(0)
                .data
                .context("audio stream should have data")
        };

        let direct = decode_audio(path.clone())?;

        // Force the filter-graph path. The env var may briefly affect tests
        // decoding in parallel, which is benign: both paths are correct.
        std::env::set_var("AVTENSOR_DISABLE_DIRECT_PATH", "1");
        let graph = decode_audio(path);
        std::env::remove_var("AVTENSOR_DISABLE_DIRECT_PATH");
        let graph = graph?;

        assert_eq!(direct.size(), graph.size());
        let max_diff: f64 = direct
            .f_sub(&graph)?
            .abs()
            .max()
            .to_kind(tch::Kind::Float)
            .try_into()?;
        assert_eq!(
            max_diff, 0.0,
            "direct audio output must be identical to the filter graph"
        );
        Ok(())
    }

    /// Float32 output must preserve the sub-8-bit precision of a 10-bit
    /// source that the uint8 path quantizes away.
    #[test]
    fn test_decode_media_float32_output() -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters {
            pixel_format: Some("yuv420p10le".to_string()),
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();

        let decode = |dtype: OutputDtype| -> anyhow::Result<tch::Tensor> {
            let mut streams = decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(path.clone()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        dtype,
                        ..Default::default()
                    }),
                    audio_streams: None,
                },
                None,
            )?;
            streams
                .remove(0)
                .data
                .context("video stream should have data")
        };

        let f32_data = decode(OutputDtype::Float32)?;
        let u8_data = decode(OutputDtype::Uint8)?;

        assert_eq!(f32_data.kind(), tch::Kind::Float);
        // float32 is planar ([T, C, H, W]); uint8 is packed ([T, H, W, C]).
        let f32_data = f32_data.permute([0, 2, 3, 1]);
        assert_eq!(f32_data.size(), u8_data.size());
        let min: f64 = f32_data.min().try_into()?;
        let max: f64 = f32_data.max().try_into()?;
        assert!((0.0..=1.0).contains(&min) && (0.0..=1.0).contains(&max));

        // Consistent with the uint8 decode up to quantization.
        let diff = f32_data
            .f_mul_scalar(255.0)?
            .f_sub(&u8_data.to_kind(tch::Kind::Float))?
            .abs();
        let mean_diff: f64 = diff.mean(tch::Kind::Float).try_into()?;
        // swscale's float pipeline computes at full precision while the
        // uint8 pipeline rounds/dithers to 8 bits, so the two legitimately
        // differ by ~1/255 on average; anything much larger would indicate a
        // range or matrix mishandling (those show up as ~18/255).
        assert!(
            mean_diff < 2.0,
            "float32 decode diverges from uint8 (mean |diff| {mean_diff})"
        );

        // The proof of extra depth: swscale expands 8-bit values v to
        // v * 257 in 16-bit, so a source with only 8 bits of information
        // would produce exclusively multiples of 257. A 10-bit source must
        // not.
        let sixteen_bit = f32_data.f_mul_scalar(65535.0)?.round();
        let sub_8bit = sixteen_bit
            .f_fmod(257.0)?
            .f_ne(0.0)?
            .to_kind(tch::Kind::Float);
        let frac: f64 = sub_8bit.mean(tch::Kind::Float).try_into()?;
        log::info!("fraction of samples with sub-8-bit precision: {frac:.3}");
        assert!(
            frac > 0.05,
            "float32 output carries no more depth than uint8 (sub-8-bit fraction {frac:.4})"
        );
        Ok(())
    }

    #[test]
    fn test_decode_media_from_bytes() -> anyhow::Result<()> {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters::default())?;
        let bytes = std::fs::read(test_video.path())?;
        let (from_bytes, _) = decode_video_tensor(MediaSource::Bytes(bytes))?;
        let (from_file, _) =
            decode_video_tensor(MediaSource::Uri(test_video.path().to_str().unwrap().into()))?;

        assert_eq!(from_bytes.size(), from_file.size());
        let max_diff: f64 = from_bytes
            .f_sub(&from_file)?
            .abs()
            .max()
            .to_kind(tch::Kind::Float)
            .try_into()?;
        assert_eq!(max_diff, 0.0);
        Ok(())
    }

    #[test]
    fn test_decode_media_frame_pts() -> anyhow::Result<()> {
        init_logger();

        let params = TestVideoParameters::default();
        let test_video = generate_test_video_file(&params)?;
        let (data, pts) =
            decode_video_tensor(MediaSource::Uri(test_video.path().to_str().unwrap().into()))?;

        assert_eq!(
            pts.len(),
            data.size()[0] as usize,
            "one pts entry per decoded frame"
        );
        assert!(
            pts.windows(2).all(|w| w[0] < w[1]),
            "pts must be strictly increasing"
        );
        let frame_interval = 1.0 / params.frame_rate;
        assert!(
            (pts[1] - pts[0] - frame_interval).abs() < 1e-3,
            "pts spacing should match the frame rate"
        );
        Ok(())
    }

    #[test]
    fn test_probe_media() -> anyhow::Result<()> {
        init_logger();

        let params = TestVideoParameters::default();
        let test_video = generate_test_video_file(&params)?;
        let probed = probe_media(
            MediaSource::Uri(test_video.path().to_str().unwrap().into()),
            None,
        )?;

        assert_eq!(probed.video_streams.len(), 1);
        let video = &probed.video_streams[0];
        assert_eq!(video.width as usize, params.width);
        assert_eq!(video.height as usize, params.height);
        assert!((video.fps - params.frame_rate).abs() < 1e-6);

        assert_eq!(probed.audio_streams.len(), 1);
        assert_eq!(
            probed.audio_streams[0].sample_rate as usize,
            params.sample_rate
        );

        // Probing from bytes goes through the in-memory reader.
        let bytes = std::fs::read(test_video.path())?;
        let probed_bytes = probe_media(MediaSource::Bytes(bytes), None)?;
        assert_eq!(probed_bytes.video_streams.len(), 1);
        assert_eq!(probed_bytes.audio_streams.len(), 1);
        Ok(())
    }
}
