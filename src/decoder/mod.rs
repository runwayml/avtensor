use anyhow::{anyhow, bail, Context};
use rsmpeg::{
    avcodec::{AVCodec, AVCodecContext, AVPacket},
    avfilter::{AVFilter, AVFilterContextMut, AVFilterGraph, AVFilterInOut},
    avformat::{AVFormatContextInput, AVStreamRef},
    avutil::{
        av_d2q, av_inv_q, av_q2d, get_sample_fmt_name, AVChannelLayout, AVDictionary, AVFrame,
        AVRational,
    },
    error::RsmpegError,
    ffi::{self, AVPixelFormat, AV_NOPTS_VALUE},
};
use std::fmt::{Debug, Display};
use std::{
    cmp::min,
    ffi::{CStr, CString},
};
use tch::IndexOp;

use crate::{
    conversion::convert_audio_into_tensor,
    decoder::io::{cloud_storage_avio_reader, memory_avio_reader},
    util::{ffmpeg::RetUpgrade, pts_to_seconds, s3::S3Config},
};

use crate::conversion;

mod cfr;
mod cuda;
mod io;
mod motion;
mod pairwise;
mod scenecut;

pub use pairwise::interpolate_rgb_pair;

struct StreamContext {
    stream_type: StreamType,
    stream_index: usize,
    dec_ctx: AVCodecContext,
    filter_config: FilterConfig,
    frame_data: tch::Tensor,
    metadata: StreamMetadata,
    cfr: Option<CliSync>,
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
    /// Motion-interpolating frame-rate conversion, fed by the filter graph.
    motion: Option<MotionStage<'graph>>,
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
    /// The CLI's frame-rate sync (`FpsMode::Cli` video streams only), and
    /// the last frame it emitted, for duplicates of it.
    cfr: Option<CliSync>,
}

/// [`cfr::CfrSync`] plus where the previous emitted frame is.
struct CliSync {
    sync: cfr::CfrSync,
    /// The CLI's fixes to packet timestamps, before decoding.
    demux: cfr::DemuxFixup,
    /// Row of the output tensor holding the previous emitted frame.
    previous_row: Option<i64>,
    /// The request's resolved output limit, which growth stays within.
    output_limit: OutputLimit,
    /// Whether the output outgrew its plan.
    grew: bool,
}

impl CliSync {
    /// Whether `rows` rows of `frame_data`'s shape fit the output limit.
    fn fits(&self, frame_data: &tch::Tensor, rows: i64) -> bool {
        rows <= max_cli_rows(frame_data, self.output_limit.bytes)
    }

    /// Grows the output to at least `needed` rows (see [`ensure_cli_capacity`]).
    fn ensure_capacity(
        &mut self,
        frame_data: &mut tch::Tensor,
        needed: i64,
        wanted: i64,
    ) -> anyhow::Result<()> {
        self.grew |= ensure_cli_capacity(frame_data, needed, wanted, &self.output_limit)?;
        Ok(())
    }
}

struct FilterContext<'graph> {
    buffersrc_ctx: AVFilterContextMut<'graph>,
    buffersink_ctx: AVFilterContextMut<'graph>,
}

/// Interpolating frame-rate conversion ([`FpsConversion::Blend`] and
/// [`FpsConversion::Motion`]). The stream's filter graph resizes/tone maps
/// into planar YUV (see [`motion_working_format`]); the resampler retimes,
/// detects scene cuts and
/// interpolates; a second small graph converts the result to the output
/// pixel format.
struct MotionStage<'graph> {
    resampler: motion::Resampler,
    output: FilterContext<'graph>,
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
    /// Number of threads to use for video decoding, and for the RGB
    /// conversion when it takes the direct swscale path. Defaults to 1; 0
    /// picks automatically.
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
    /// HDR handling: `Tonemap` (default, SDR BT.709 preview) or `Raw`
    /// (preserve PQ/HLG code values; matrix/range conversion only).
    pub hdr_mode: HdrMode,
    /// YUV->RGB conversion: `None` resolves to [`ColorConversion::Fast`] for
    /// `Uint8` (bit-identical to earlier releases) and
    /// [`ColorConversion::Accurate`] for `Float32`.
    pub color_conversion: Option<ColorConversion>,
    /// How `frame_rate` resampling builds each output frame. Ignored when
    /// `frame_rate` is unset.
    pub fps_conversion: FpsConversion,
    /// Scene-cut handling for the interpolating `fps_conversion` modes.
    pub scene_cuts: SceneCuts,
    /// Apply the stream's display-matrix rotation, as the ffmpeg CLI's
    /// `-autorotate` does: multiples of 90° transpose or flip the frame (and
    /// swap the output dimensions for 90°/270°), any other angle rotates it
    /// within the coded frame size. `width`/`height` then refer to the
    /// rotated frame. Off by default. Only the stream-level display matrix
    /// is applied: orientation carried per frame (JPEG EXIF Orientation,
    /// H.264/HEVC display-orientation SEI), which the CLI also applies, is
    /// not.
    pub autorotate: bool,
    /// How the output frame rate is reached (see [`FpsMode`]).
    pub fps_mode: FpsMode,
}

/// How decoded frames are mapped onto the output frame rate.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpsMode {
    /// The `fps` filter at `frame_rate` when one is requested; otherwise
    /// every decoded frame once. The default.
    #[default]
    Filter,
    /// The ffmpeg CLI's constant-frame-rate output (`-fps_mode cfr`) at the
    /// stream's guessed frame rate: each frame takes the next output slot and
    /// is dropped or duplicated by its timestamp drift, so the output matches
    /// what `ffmpeg -i <input> -f rawvideo -` produces, including the CLI's
    /// timestamp fixes before decoding (discontinuities in MPEG-TS/PS/HLS).
    /// `frame_rate`, `device`, `hardware_acceleration` and a seek window must
    /// be unset. Duplicates are copies of the frame in the output tensor. The
    /// plan is an estimate the output grows past, up to the output limit.
    /// Unlike the CLI, frames are autorotated only with `autorotate`.
    Cli,
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
    /// Largest output tensor (bytes) each stream may allocate; `None` uses
    /// `AVTENSOR_MAX_OUTPUT_TENSOR_BYTES`, else 16 GiB. Every stream is
    /// checked before any buffer is allocated.
    pub max_output_bytes: Option<u64>,
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
        max_output_bytes,
    } = request;
    let output_limit = OutputLimit::resolve(max_output_bytes);

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

    // Plan every stream before allocating any buffer, so a stream over the
    // output limit fails the request before another stream's buffer exists.
    let video_plan = video_stream
        .as_ref()
        .map(|video_stream| {
            plan_video_stream(&input_format_context, video_stream, &seek, &output_limit)
        })
        .transpose()?;
    let audio_plans = audio_streams
        .iter()
        .flatten()
        .map(|audio_stream| {
            plan_audio_stream(&input_format_context, audio_stream, &seek, &output_limit)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Configure video decoder
    if let (Some(video_stream), Some(plan)) = (&video_stream, video_plan) {
        stream_ctx.push(init_video_stream_context(
            &input_format_context,
            video_stream,
            plan,
            &output_limit,
        )?);
    }

    // Configure audio decoder(s)
    for (audio_stream, plan) in audio_streams.iter().flatten().zip(audio_plans) {
        stream_ctx.push(init_audio_stream_context(
            &input_format_context,
            audio_stream,
            plan,
        )?);
    }

    // Plain conversions bypass these graphs entirely (see `DirectPath`); the
    // graph handles resize, frame-rate resampling, HDR tone mapping, and
    // widths libswscale can't vectorize.
    log::debug!("Setting up filter graphs for the streams.");
    let mut filter_graphs: Vec<_> = (0..stream_ctx.len())
        .map(|_| (AVFilterGraph::new(), AVFilterGraph::new()))
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
        let mut packet = match input_format_context.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break, // End of file reached
            Err(e) => bail!("Failed to read packet: {e:?}"),
        };

        let in_stream_index = packet.stream_index as usize;

        if let Some(filter) = filter_ctx
            .iter_mut()
            .find(|f| f.stream_index == in_stream_index && !f.finished)
        {
            if let Some(cli) = &mut filter.cfr {
                let stream = input_format_context
                    .streams()
                    .get(in_stream_index)
                    .context("packet of an unknown stream")?;
                fix_cli_packet(&mut cli.demux, &mut packet, stream);
            }
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
                // Packets arrive in decode order, so with B-frames a packet
                // past the end can precede in-window frames; dts is
                // monotonic and never exceeds pts, so once it passes the end
                // every in-window frame has been read.
                let dts_seconds = if packet.dts != AV_NOPTS_VALUE {
                    pts_to_seconds(packet.dts, pkt_timebase)
                } else {
                    pts_seconds
                };
                if dts_seconds > end_time {
                    log::debug!(
                        "Packet dts {} exceeds end time {}, stopping decoding.",
                        dts_seconds,
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
            motion,
            frame_data,
            frame_data_ptr,
            frame_pts,
            cfr,
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
            motion,
            frame_data,
            frame_data_ptr,
            frame_pts,
            cfr,
            &seek,
        )?;
        log::debug!(
            "Received {} frames after flushing the decoder",
            frames.len()
        );
        if let Some(cli) = cfr {
            // The CLI repeats the last frame as often as recent frames were.
            let copies = cli.sync.finish();
            emit_previous_copies(cli, copies, frame_data, frame_data_ptr, frame_pts)?;
        }
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
            motion.as_mut(),
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
        mut frame_data,
        frame_data_ptr,
        mut frame_pts,
        stream_type,
        cfr,
        ..
    } in filter_ctx.into_iter()
    {
        let decoded_stream = decoded_streams
            .iter_mut()
            .find(|s| s.src_stream_index == stream_index)
            .context("Decoded stream not found for the given stream index")?;
        let trimmed_data = match stream_type {
            StreamType::Video => {
                let rows = min(frame_data_ptr as i64, frame_data.size()[0]);
                frame_pts.truncate(rows as usize);
                if cfr.is_some_and(|cli| cli.grew) {
                    // Growth leaves up to a quarter of the rows unused; keep
                    // them out of the returned tensor's storage.
                    trim_cli_output(&mut frame_data, rows, &decoded_stream.decoded_frames);
                }
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
        decoded_stream.data = Some(trimmed_data);
        decoded_stream.frame_pts = frame_pts;
    }

    Ok(decoded_streams)
}

/// Metadata for a probed video stream.
#[derive(Debug, Clone)]
pub struct ProbedVideoStream {
    pub index: usize,
    /// Coded width; the decoder outputs this unless a width is requested
    /// or `autorotate` turns the frame by 90°/270° (it then outputs
    /// `height`).
    pub width: i32,
    /// Coded height; the decoder outputs this unless a height is requested
    /// or `autorotate` turns the frame by 90°/270° (it then outputs
    /// `width`).
    pub height: i32,
    pub fps: f64,
    pub codec_name: String,
    pub duration_seconds: Option<f64>,
    pub duration_source: DurationSource,
    /// Frame count from the container header, if it stores one.
    pub nb_frames: Option<i64>,
    /// Display-matrix rotation in degrees, as reported by
    /// `av_display_rotation_get` (counterclockwise; ffprobe's `rotation`).
    /// The decoder applies it only with [`VideoStreamRequest::autorotate`].
    /// Stream-level matrix only: `None` when the orientation is carried per
    /// frame (JPEG EXIF Orientation, H.264/HEVC display-orientation SEI).
    pub rotation: Option<f64>,
    /// Sample (pixel) aspect ratio as `(num, den)`; `(0, 1)` when unknown.
    /// The decoder does not apply it.
    pub sample_aspect_ratio: (i32, i32),
}

/// Metadata for a probed audio stream.
#[derive(Debug, Clone)]
pub struct ProbedAudioStream {
    pub index: usize,
    pub sample_rate: i32,
    pub channels: i32,
    pub codec_name: String,
    pub duration_seconds: Option<f64>,
    pub duration_source: DurationSource,
}

/// Container-level metadata returned by [`probe_media`].
#[derive(Debug, Clone, Default)]
pub struct ProbedMedia {
    /// Demuxer name(s), e.g. `"mov,mp4,m4a,3gp,3g2,mj2"` or `"matroska,webm"`.
    pub format_name: String,
    pub duration_seconds: Option<f64>,
    pub video_streams: Vec<ProbedVideoStream>,
    pub audio_streams: Vec<ProbedAudioStream>,
}

/// Probes a media asset's stream layout without decoding it.
pub fn probe_media(
    source: MediaSource,
    s3_config: Option<S3Config>,
) -> Result<ProbedMedia, anyhow::Error> {
    let input_format_context = source.open(s3_config)?;
    let mut probed = ProbedMedia {
        format_name: format_name(&input_format_context),
        duration_seconds: container_duration_seconds(&input_format_context),
        ..Default::default()
    };
    for (index, stream) in input_format_context.streams().iter().enumerate() {
        let codecpar = stream.codecpar();
        let (duration_seconds, duration_source) =
            match stream_duration_seconds(&input_format_context, stream) {
                Some((duration, source)) => (Some(duration), source),
                None => (None, DurationSource::None),
            };
        let codec_name = codec_name(codecpar.codec_id);
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
                    codec_name,
                    duration_seconds,
                    duration_source,
                    nb_frames: (stream.nb_frames > 0).then_some(stream.nb_frames),
                    rotation: display_rotation(&codecpar),
                    sample_aspect_ratio: (
                        codecpar.sample_aspect_ratio.num,
                        codecpar.sample_aspect_ratio.den,
                    ),
                });
            }
            ffi::AVMEDIA_TYPE_AUDIO => {
                probed.audio_streams.push(ProbedAudioStream {
                    index,
                    sample_rate: codecpar.sample_rate,
                    channels: codecpar.ch_layout().nb_channels,
                    codec_name,
                    duration_seconds,
                    duration_source,
                });
            }
            _ => {}
        }
    }
    Ok(probed)
}

/// Plans a decode without decoding: opens the container, selects streams
/// exactly as [`decode_media`] does, and returns the buffer each stream would
/// preallocate (video first, then audio in request order). Decoders are not
/// opened, so codec and hardware-acceleration errors only surface at decode.
pub fn plan_media(
    request: MediaDecodeRequest,
    s3_config: Option<S3Config>,
) -> Result<Vec<StreamPlan>, anyhow::Error> {
    let MediaDecodeRequest {
        source,
        start_time,
        end_time,
        video_stream,
        audio_streams,
        max_output_bytes,
    } = request;
    let output_limit = OutputLimit::resolve(max_output_bytes);
    let input_format_context = source.open(s3_config)?;
    let video_stream = select_video_stream(&input_format_context, video_stream)?;
    let audio_streams = select_audio_streams(&input_format_context, audio_streams)?;
    let seek = Seek {
        start_time,
        end_time,
    };
    let mut plans = Vec::new();
    if let Some(video_stream) = &video_stream {
        plans.push(plan_video_stream(
            &input_format_context,
            video_stream,
            &seek,
            &output_limit,
        )?);
    }
    for audio_stream in audio_streams.iter().flatten() {
        plans.push(plan_audio_stream(
            &input_format_context,
            audio_stream,
            &seek,
            &output_limit,
        )?);
    }
    Ok(plans)
}

fn format_name(input_format_context: &AVFormatContextInput) -> String {
    let iformat = input_format_context.iformat;
    if iformat.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr((*iformat).name) }
        .to_string_lossy()
        .into_owned()
}

fn codec_name(codec_id: ffi::AVCodecID) -> String {
    let name = unsafe { ffi::avcodec_get_name(codec_id) };
    if name.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

/// The stream's display-matrix rotation (degrees, counterclockwise), if any.
fn display_rotation(codecpar: &ffi::AVCodecParameters) -> Option<f64> {
    let rotation = unsafe { ffi::av_display_rotation_get(display_matrix(codecpar)?.as_ptr()) };
    rotation.is_finite().then_some(rotation)
}

/// The stream's display matrix, if it has one.
///
/// This is the stream-level matrix only (e.g. an MP4 `tkhd` matrix). The
/// ffmpeg CLI takes the matrix from each decoded frame, so it also applies
/// orientation that only reaches the frames: JPEG EXIF Orientation and
/// H.264/HEVC display-orientation SEI. Those are neither reported nor
/// applied here; doing so needs the first decoded frame, which `plan_media`
/// does not decode.
fn display_matrix(codecpar: &ffi::AVCodecParameters) -> Option<[i32; 9]> {
    let side_data = unsafe {
        ffi::av_packet_side_data_get(
            codecpar.coded_side_data,
            codecpar.nb_coded_side_data,
            ffi::AV_PKT_DATA_DISPLAYMATRIX,
        )
    };
    if side_data.is_null() || unsafe { (*side_data).size } < 9 * 4 {
        return None;
    }
    Some(unsafe { *((*side_data).data as *const [i32; 9]) })
}

/// The filters ffmpeg's `-autorotate` inserts for a display matrix
/// (`get_rotation` and `configure_input_video_filter` in FFmpeg 7.1's
/// fftools), including the flips of mirrored matrices.
fn autorotate_filters(matrix: &[i32; 9]) -> Vec<String> {
    let rotation = unsafe { ffi::av_display_rotation_get(matrix.as_ptr()) };
    if !rotation.is_finite() {
        return vec![]; // a degenerate matrix
    }
    // Clockwise, in whole degrees normalized to 0..=359, as `get_rotation`.
    let mut theta = -rotation.round();
    theta -= 360.0 * (theta / 360.0 + 0.9 / 360.0).floor();
    if (theta - 90.0).abs() < 1.0 {
        vec![if matrix[3] > 0 {
            "transpose=cclock_flip"
        } else {
            "transpose=clock"
        }
        .to_string()]
    } else if (theta - 180.0).abs() < 1.0 {
        let mut filters = vec![];
        if matrix[0] < 0 {
            filters.push("hflip".to_string());
        }
        if matrix[4] < 0 {
            filters.push("vflip".to_string());
        }
        filters
    } else if (theta - 270.0).abs() < 1.0 {
        vec![if matrix[3] < 0 {
            "transpose=clock_flip"
        } else {
            "transpose=cclock"
        }
        .to_string()]
    } else if theta.abs() > 1.0 {
        vec![format!("rotate={theta:.6}*PI/180")]
    } else if theta.abs() < 1.0 && matrix[4] < 0 {
        // theta is whole, so this is theta == 0: at exactly 1° the CLI
        // applies nothing, not even a mirrored matrix's vflip.
        vec!["vflip".to_string()]
    } else {
        vec![]
    }
}

/// Whether the rotation `filters` swap the frame's width and height.
fn swaps_dimensions(filters: &[String]) -> bool {
    filters.first().is_some_and(|f| f.starts_with("transpose"))
}

/// The rotation filters `request` applies to the stream: empty unless
/// autorotate is on and the stream's display matrix rotates or mirrors.
/// Shared by decode and plan, so both refuse GPU-resident output (which
/// never reaches the filter graph) for a rotated stream.
fn requested_rotation(
    request: &VideoStreamRequest,
    codecpar: &ffi::AVCodecParameters,
) -> Result<Vec<String>, anyhow::Error> {
    if !request.autorotate {
        return Ok(vec![]);
    }
    let filters = display_matrix(codecpar)
        .map(|m| autorotate_filters(&m))
        .unwrap_or_default();
    if request.device.is_some() && !filters.is_empty() {
        return Err(anyhow!(
            "device=\"cuda\" does not support autorotate (frames never \
             reach the CPU filter graph)"
        ));
    }
    Ok(filters)
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

/// Initializes the video stream processor and allocates `plan`'s buffer,
/// which must come from [`plan_video_stream`] (which also validates
/// `fps_mode` before any decoder or CUDA device is set up). A
/// `FpsMode::Cli` output grows within `output_limit`.
fn init_video_stream_context(
    input_format_context: &AVFormatContextInput,
    request: &VideoStreamRequest,
    plan: StreamPlan,
    output_limit: &OutputLimit,
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
    if request.device.is_some() && request.color_conversion == Some(ColorConversion::Accurate) {
        return Err(anyhow!(
            "device=\"cuda\" converts on the GPU with NPP and does not \
             support color_conversion=\"accurate\""
        ));
    }

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

    // Rotation filters for autorotate (empty when off or not rotated).
    let rotate = requested_rotation(request, &codecpar)?;

    if let Some(device) = request.device {
        // GPU-resident output is a constrained pipeline: frames never reach
        // system memory, so only NVDEC's own capabilities are available
        // (autorotate is refused by requested_rotation above).
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
    if let Some(frame_rate) = video_stream.guess_framerate() {
        decode_context.set_framerate(frame_rate);
    }

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
    // The resize request refers to the rotated frame, so NVDEC (which scales
    // before rotation) only pre-resizes unrotated streams.
    let hw_resize = if hardware_acceleration && rotate.is_empty() {
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
    if request.width.is_some() || request.height.is_some() {
        // Give the graph both dimensions so it produces exactly the planned
        // shape (`scale` and `zscale` reject a lone width or height). With
        // autorotate they refer to the rotated frame, as in the plan.
        let (src_w, src_h) = if swaps_dimensions(&rotate) {
            (codecpar.height as i64, codecpar.width as i64)
        } else {
            (codecpar.width as i64, codecpar.height as i64)
        };
        let (w, h) = resolve_output_size(src_w, src_h, request);
        video_filter.width = Some(w as usize);
        video_filter.height = Some(h as usize);
    }
    video_filter.source_color = SourceColorInfo::from_codec_context(&decode_context);
    video_filter.rotate = rotate;
    log::debug!("Source color info: {:?}", video_filter.source_color);
    video_filter.source_format = SourceFormat::new(
        buffersrc_pixel_format(&decode_context),
        decode_context.width,
        decode_context.height,
    );
    if let (Some(target), Some(source)) = (request.frame_rate, video_stream.guess_framerate()) {
        // Interpolation adds nothing when every output frame lands on a
        // source frame (48 -> 24), and for NTSC rate pairs (23.976 -> 24) it
        // would re-synthesize nearly every frame to fix a 0.1% drift.
        if video_filter.fps_conversion != FpsConversion::Nearest
            && is_integer_downsample(av_q2d(source), target)
        {
            log::debug!(
                "Source frame rate {} is an integer multiple of {target}, \
                 using nearest-frame conversion",
                av_q2d(source)
            );
            video_filter.fps_conversion = FpsConversion::Nearest;
        }
    }

    log::debug!(
        "Allocating Tensor with shape {:?} to decode video to.",
        plan.shape
    );
    let kind = match request.dtype {
        OutputDtype::Uint8 => tch::Kind::Uint8,
        OutputDtype::Float32 => tch::Kind::Float,
    };
    let device = request
        .device
        .map_or(tch::Device::Cpu, |d| tch::Device::Cuda(d as usize));
    let dest =
        crate::alloc::empty(&plan.shape, kind, device).context("allocating output Tensor")?;

    let cfr = match request.fps_mode {
        FpsMode::Filter => None,
        // Validated above.
        FpsMode::Cli => {
            let start_time = cli_start_time(input_format_context, video_stream);
            Some(CliSync {
                sync: cfr::CfrSync::new(
                    cli_frame_rate(video_stream),
                    video_stream.time_base,
                    start_time,
                    input_format_context.nb_streams == 1,
                ),
                demux: cli_demux_fixup(input_format_context, video_stream, start_time),
                previous_row: None,
                output_limit: output_limit.clone(),
                grew: false,
            })
        }
    };

    Ok(StreamContext {
        stream_type: StreamType::Video,
        stream_index: video_index,
        dec_ctx: decode_context,
        filter_config: FilterConfig::Video(video_filter),
        frame_data: dest,
        metadata: plan.metadata,
        cfr,
    })
}

/// Where a stream's duration came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationSource {
    /// The stream's own duration.
    Stream,
    /// The container duration (e.g. Matroska/WebM, whose streams report none).
    Container,
    /// Neither reports one; video is then treated as a single-frame image.
    None,
}

/// The output buffer a decode preallocates for one stream, derived from
/// container metadata and the request alone. [`decode_media`] allocates
/// exactly `shape` and trims it to the frames/samples actually decoded, so
/// `shape` is an upper bound on the returned tensor, except for
/// `FpsMode::Cli` video, which grows past it when more frames are emitted.
#[derive(Debug, Clone)]
pub struct StreamPlan {
    pub stream_index: usize,
    /// Preallocated shape in storage order: uint8 video `[T, H, W, 3]`,
    /// float32 video `[T, 3, H, W]`, audio `[C, N]`.
    pub shape: Vec<i64>,
    /// Element type of the buffer (audio is always `Float32`).
    pub dtype: OutputDtype,
    pub metadata: StreamMetadata,
    /// Duration being decoded (after the seek window), if known.
    pub duration_seconds: Option<f64>,
    pub duration_source: DurationSource,
}

impl StreamPlan {
    pub fn stream_type(&self) -> StreamType {
        match self.metadata {
            StreamMetadata::Video { .. } => StreamType::Video,
            StreamMetadata::Audio { .. } => StreamType::Audio,
        }
    }
}

/// Plans the decode of the video stream selected by `request.index`, whose
/// buffer may not exceed `max_output_bytes`.
fn plan_video_stream(
    input_format_context: &AVFormatContextInput,
    request: &VideoStreamRequest,
    seek: &Seek,
    output_limit: &OutputLimit,
) -> Result<StreamPlan, anyhow::Error> {
    validate_fps_mode(request, seek)?;
    let video_index = request
        .index
        .ok_or(anyhow!("No video stream index specified"))?;
    let video_stream = input_format_context
        .streams()
        .get(video_index)
        .context("Failed to get video stream")?;
    let codecpar = video_stream.codecpar();
    let rotate = requested_rotation(request, &codecpar)?;
    let source_frame_rate = video_stream
        .guess_framerate()
        .unwrap_or(video_stream.avg_frame_rate);

    let (num_frames, frame_rate, duration_seconds, duration_source) =
        match stream_duration_seconds(input_format_context, video_stream) {
            None => {
                // If neither the stream nor the container reports a duration, assume we are dealing with an image.
                // CLI mode still outputs at the stream's rate (and grows).
                let frame_rate = match request.fps_mode {
                    FpsMode::Cli => av_q2d(cli_frame_rate(video_stream)),
                    FpsMode::Filter => 1.0,
                };
                (1, frame_rate, None, DurationSource::None)
            }
            Some((stream_duration, duration_source)) => {
                // Estimate the duration of media we'll be decoding, this will be the minimum of the specified end time (if any), or duration of the stream.
                let duration_seconds = asset_duration(stream_duration, seek)?;

                // In order to allocate storage for the decoded frames, we need to know the frame rate.
                let frame_rate = if request.fps_mode == FpsMode::Cli {
                    av_q2d(cli_frame_rate(video_stream))
                } else if let Some(frame_rate) = request.frame_rate {
                    frame_rate
                } else {
                    // Convert `frame_rate` to double.
                    match source_frame_rate {
                        AVRational { num: 0, den: 1 } => {
                            // If the frame rate is 0/1, we cannot determine the frame rate.
                            // Try to calculate from nb_frames / duration instead.
                            log::debug!(
                                "Could not guess frame rate, falling back to nb_frames / duration"
                            );
                            let num_frames = video_stream.nb_frames;
                            if num_frames > 0 && stream_duration > 0.0 {
                                num_frames as f64 / stream_duration
                            } else {
                                // Err: could not determine framerate.
                                return Err(anyhow!("Could not determine frame rate."));
                            }
                        }
                        AVRational { .. } => av_q2d(source_frame_rate),
                    }
                };
                let epsilon = (1.0 / frame_rate) * 0.5; // Small epsilon to avoid floating point precision errors causing trailing black frames because we've allocated too many frames.
                let num_frames = if request.fps_mode == FpsMode::Cli {
                    // The CLI's output runs from the file's start (which an
                    // audio track can put before the video's) to the
                    // stream's end, and its last frame may repeat; allow
                    // for both. The tensor is trimmed to what is emitted.
                    // A container-wide duration already spans the lead.
                    let lead = match duration_source {
                        DurationSource::Stream => (stream_start_seconds(video_stream)
                            - cli_start_time(input_format_context, video_stream) as f64 / 1e6)
                            .max(0.0),
                        _ => 0.0,
                    };
                    ((duration_seconds + lead) * frame_rate).ceil() as i64 + 2
                } else {
                    ((duration_seconds - epsilon) * frame_rate).ceil() as i64
                };
                (
                    num_frames,
                    frame_rate,
                    Some(duration_seconds),
                    duration_source,
                )
            }
        };

    // With autorotate, a 90°/270° rotation swaps the displayed dimensions;
    // `width`/`height` (and a lone one's aspect ratio) refer to that frame.
    let (display_width, display_height) = if swaps_dimensions(&rotate) {
        (codecpar.height, codecpar.width)
    } else {
        (codecpar.width, codecpar.height)
    };
    let (out_width, out_height) =
        resolve_output_size(display_width as i64, display_height as i64, request);
    // uint8 decodes to packed rgb24 ([T, H, W, C]); float32 decodes to
    // planar float ([T, C, H, W], contiguous channels-first).
    let (shape, element_size) = match request.dtype {
        OutputDtype::Uint8 => (
            vec![num_frames, out_height, out_width, 3],
            std::mem::size_of::<u8>(),
        ),
        OutputDtype::Float32 => (
            vec![num_frames, 3, out_height, out_width],
            std::mem::size_of::<f32>(),
        ),
    };
    check_output_tensor_size(&shape, element_size, output_limit)?;

    Ok(StreamPlan {
        stream_index: video_index,
        shape,
        dtype: request.dtype,
        metadata: StreamMetadata::Video { frame_rate },
        duration_seconds,
        duration_source,
    })
}

/// Plans the decode of the audio stream selected by `request.index`, whose
/// buffer may not exceed `max_output_bytes`.
fn plan_audio_stream(
    input_format_context: &AVFormatContextInput,
    request: &AudioStreamRequest,
    seek: &Seek,
    output_limit: &OutputLimit,
) -> Result<StreamPlan, anyhow::Error> {
    let audio_index = request
        .index
        .ok_or(anyhow!("No audio stream index specified"))?;
    let audio_stream = input_format_context
        .streams()
        .get(audio_index)
        .context("Failed to get audio stream")?;
    let codecpar = audio_stream.codecpar();

    // Without a duration from either the stream or the container we cannot
    // tell how much to preallocate, so fail with a clear error.
    let (stream_duration, duration_source) =
        stream_duration_seconds(input_format_context, audio_stream).ok_or_else(|| {
            anyhow!(
                "Audio stream #{audio_index} does not report a duration; cannot preallocate the output buffer"
            )
        })?;

    // Estimate the duration of media we'll be decoding, this will be the minimum of the specified end time (if any), or duration of the stream.
    let duration_seconds = asset_duration(stream_duration, seek)?;

    // The sample rate will either be the source sample rate, or the sample rate that we're targeting.
    let sample_rate = if let Some(sample_rate) = request.sample_rate {
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
    check_output_tensor_size(&shape, std::mem::size_of::<f32>(), output_limit)?;

    Ok(StreamPlan {
        stream_index: audio_index,
        shape,
        dtype: OutputDtype::Float32,
        metadata: StreamMetadata::Audio {
            sample_rate: sample_rate as u32,
        },
        duration_seconds: Some(duration_seconds),
        duration_source,
    })
}

/// Initializes an audio stream processor and allocates `plan`'s buffer, which
/// must come from [`plan_audio_stream`].
fn init_audio_stream_context(
    input_format_context: &AVFormatContextInput,
    request: &AudioStreamRequest,
    plan: StreamPlan,
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

    let filter_config = AudioFilterConfig::try_from(request)?;
    log::debug!(
        "Allocating Tensor with shape {:?} to decode audio to for stream #{}.",
        plan.shape,
        audio_index
    );
    let dest = crate::alloc::empty(&plan.shape, tch::Kind::Float, tch::Device::Cpu)
        .context("allocating audio output Tensor")?;

    Ok(StreamContext {
        stream_type: StreamType::Audio,
        stream_index: audio_index,
        dec_ctx: decode_context,
        filter_config: FilterConfig::Audio(filter_config),
        cfr: None,
        frame_data: dest,
        metadata: plan.metadata,
    })
}

/// Default ceiling on the buffer avtensor preallocates for a single decoded
/// stream. Duration, frame rate, dimensions, and sample rate all come from
/// file metadata and are attacker-controllable, so this bounds the up-front
/// allocation rather than trusting those values.
const DEFAULT_MAX_OUTPUT_TENSOR_BYTES: u128 = 16 * 1024 * 1024 * 1024; // 16 GiB

const MAX_OUTPUT_TENSOR_BYTES_VAR: &str = "AVTENSOR_MAX_OUTPUT_TENSOR_BYTES";

/// The per-stream output size limit of one request, and where it came from
/// (named in the error when a stream exceeds it).
#[derive(Debug, Clone, PartialEq, Eq)]
struct OutputLimit {
    bytes: u128,
    source: OutputLimitSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OutputLimitSource {
    /// `MediaDecodeRequest.max_output_bytes`.
    Request,
    /// `AVTENSOR_MAX_OUTPUT_TENSOR_BYTES`, with its raw value.
    Env(String),
    /// Neither is set.
    Default,
    /// `AVTENSOR_MAX_OUTPUT_TENSOR_BYTES` holds this invalid value, so the
    /// default applies.
    InvalidEnv(String),
}

impl OutputLimit {
    /// The request's own limit, else `AVTENSOR_MAX_OUTPUT_TENSOR_BYTES`, else
    /// [`DEFAULT_MAX_OUTPUT_TENSOR_BYTES`]. Some decodes legitimately exceed
    /// the default: 30 s of uncropped 4K rgb24 is about 22 GB.
    fn resolve(requested: Option<u64>) -> Self {
        match requested {
            Some(bytes) => OutputLimit {
                bytes: bytes.into(),
                source: OutputLimitSource::Request,
            },
            None => Self::from_env(match std::env::var(MAX_OUTPUT_TENSOR_BYTES_VAR) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(std::env::VarError::NotUnicode(value)) => {
                    Some(value.to_string_lossy().into_owned())
                }
            }),
        }
    }

    fn from_env(value: Option<String>) -> Self {
        let default = |source| OutputLimit {
            bytes: DEFAULT_MAX_OUTPUT_TENSOR_BYTES,
            source,
        };
        match value {
            None => default(OutputLimitSource::Default),
            Some(value) if value.trim().is_empty() => default(OutputLimitSource::Default),
            Some(value) => match parse_byte_size(&value) {
                Some(bytes) => OutputLimit {
                    bytes,
                    source: OutputLimitSource::Env(value),
                },
                None => {
                    log::warn!(
                        "Ignoring invalid {MAX_OUTPUT_TENSOR_BYTES_VAR} value {value:?}; \
                         using the default of {DEFAULT_MAX_OUTPUT_TENSOR_BYTES} bytes"
                    );
                    default(OutputLimitSource::InvalidEnv(value))
                }
            },
        }
    }
}

impl Display for OutputLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.bytes;
        let var = MAX_OUTPUT_TENSOR_BYTES_VAR;
        match &self.source {
            OutputLimitSource::Request => {
                write!(f, "the request's max_output_bytes of {bytes} bytes")
            }
            OutputLimitSource::Env(value) => write!(
                f,
                "the limit of {bytes} bytes set by {var}={value:?} (raise it, or \
                 per request with MediaDecodeRequest(max_output_bytes=...))"
            ),
            OutputLimitSource::Default => write!(
                f,
                "the default limit of {bytes} bytes (raise it per request with \
                 MediaDecodeRequest(max_output_bytes=...) or process-wide with {var})"
            ),
            OutputLimitSource::InvalidEnv(value) => write!(
                f,
                "the default limit of {bytes} bytes: {var}={value:?} was ignored because \
                 it is not a byte count such as 34359738368 or 32G (fix it, or raise the \
                 limit per request with MediaDecodeRequest(max_output_bytes=...))"
            ),
        }
    }
}

/// Parses a byte count with an optional binary suffix, case-insensitive:
/// `1024`, `512K`, `64M`, `32G`, `1T`, or `KiB`/`MiB`/`GiB`/`TiB`. `KB`-style
/// suffixes are rejected rather than guessed as decimal or binary. A leading
/// `+` is accepted, as `u128::from_str` does.
fn parse_byte_size(value: &str) -> Option<u128> {
    let value = value.trim();
    let value = value.strip_prefix('+').unwrap_or(value);
    let (number, suffix) = value.split_at(
        value
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(value.len()),
    );
    let number: u128 = number.parse().ok()?;
    let shift = match suffix.trim_start().to_ascii_lowercase().as_str() {
        "" | "b" => 0,
        "k" | "kib" => 10,
        "m" | "mib" => 20,
        "g" | "gib" => 30,
        "t" | "tib" => 40,
        _ => return None,
    };
    number.checked_mul(1 << shift)
}

/// Rejects output shapes that contain a negative dimension or whose total
/// allocation would exceed `limit`.
fn check_output_tensor_size(
    shape: &[i64],
    element_size: usize,
    limit: &OutputLimit,
) -> Result<(), anyhow::Error> {
    let mut total = element_size as u128;
    for &dim in shape {
        let dim = u128::try_from(dim)
            .map_err(|_| anyhow!("output tensor has a negative dimension: {shape:?}"))?;
        total = total
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("output tensor size overflow for shape {shape:?}"))?;
    }
    if total > limit.bytes {
        return Err(anyhow!(
            "output tensor for shape {shape:?} would require {total} bytes, exceeding {limit}"
        ));
    }
    Ok(())
}

/// Returns the duration of `stream` in seconds, falling back to the container
/// duration when the stream does not report one. Matroska/WebM only stores a
/// segment-level duration, so its streams report `AV_NOPTS_VALUE`. The
/// container duration spans every stream, so it never underestimates a
/// single stream; the output tensor is trimmed to what was actually decoded.
fn stream_duration_seconds(
    input_format_context: &AVFormatContextInput,
    stream: &AVStreamRef,
) -> Option<(f64, DurationSource)> {
    if stream.duration != AV_NOPTS_VALUE {
        return Some((
            stream.duration as f64 * av_q2d(stream.time_base),
            DurationSource::Stream,
        ));
    }
    let container_duration = container_duration_seconds(input_format_context)?;
    log::debug!(
        "Stream #{} does not report a duration, using the container duration",
        stream.index
    );
    Some((container_duration, DurationSource::Container))
}

/// Rejects requests `FpsMode::Cli` cannot serve, for planning and decoding
/// alike.
fn validate_fps_mode(request: &VideoStreamRequest, seek: &Seek) -> Result<(), anyhow::Error> {
    if request.fps_mode != FpsMode::Cli {
        return Ok(());
    }
    if request.frame_rate.is_some() {
        return Err(anyhow!(
            "fps_mode=\"cli\" outputs the stream's own frame rate and cannot be combined with fps"
        ));
    }
    if request.device.is_some() {
        return Err(anyhow!(
            "fps_mode=\"cli\" is not supported with device=\"cuda\""
        ));
    }
    // NVDEC frames carry no duration, which the sync needs to match the CLI.
    if request.hardware_acceleration == Some(true) {
        return Err(anyhow!(
            "fps_mode=\"cli\" is not supported with hardware_acceleration=True"
        ));
    }
    if seek.start_time.is_some() || seek.end_time.is_some() {
        return Err(anyhow!(
            "fps_mode=\"cli\" decodes whole files; start_time and end_time are not supported"
        ));
    }
    Ok(())
}

/// The stream's first timestamp in seconds (0 when unknown).
fn stream_start_seconds(stream: &AVStreamRef) -> f64 {
    if stream.start_time == AV_NOPTS_VALUE {
        0.0
    } else {
        stream.start_time as f64 * av_q2d(stream.time_base)
    }
}

/// The output frame rate the ffmpeg CLI uses for a stream: its guessed
/// frame rate (`tbr`), or 25 without one.
fn cli_frame_rate(stream: &AVStreamRef) -> AVRational {
    match stream.guess_framerate() {
        Some(rate) if rate.num > 0 && rate.den > 0 => rate,
        _ => AVRational { num: 25, den: 1 },
    }
}

/// Whether the format allows timestamp discontinuities (MPEG-TS/PS, HLS).
fn has_ts_discontinuities(input_format_context: &AVFormatContextInput) -> bool {
    input_format_context.iformat().flags & ffi::AVFMT_TS_DISCONT as i32 != 0
}

/// The start time (microseconds; 0 when unknown) the CLI shifts all
/// timestamps by so its output starts at 0: the file's, or in formats with
/// timestamp discontinuities, that of the streams it reads, which for raw
/// video output is just the video stream (`correct_input_start_times`).
fn cli_start_time(input_format_context: &AVFormatContextInput, stream: &AVStreamRef) -> i64 {
    let start = input_format_context.start_time;
    if start == AV_NOPTS_VALUE {
        0
    } else if has_ts_discontinuities(input_format_context) && stream.start_time != AV_NOPTS_VALUE {
        cfr::rescale(stream.start_time, stream.time_base, cfr::AV_TIME_BASE_Q)
    } else {
        start
    }
}

/// The CLI's packet timestamp fixes for `stream`.
fn cli_demux_fixup(
    input_format_context: &AVFormatContextInput,
    stream: &AVStreamRef,
    start_time: i64,
) -> cfr::DemuxFixup {
    let codecpar = stream.codecpar();
    // ist_dts_update's first-dts estimate: back by the decoder delay. C
    // computes it in double and truncates.
    let initial_dts = if stream.avg_frame_rate.num != 0 {
        ((-(codecpar.video_delay as i64) * 1_000_000) as f64 / av_q2d(stream.avg_frame_rate)) as i64
    } else {
        0
    };
    let rate = codecpar.framerate;
    let field_rate = (rate.num != 0)
        .then(|| rate.num.checked_mul(2))
        .flatten()
        .map(|num| AVRational { num, den: rate.den });
    cfr::DemuxFixup::new(
        stream.time_base,
        start_time,
        has_ts_discontinuities(input_format_context),
        initial_dts,
        field_rate,
    )
}

/// Applies the CLI's timestamp fixes to a packet just read from `stream`.
fn fix_cli_packet(demux: &mut cfr::DemuxFixup, packet: &mut AVPacket, stream: &AVStreamRef) {
    let ts = |ts: i64| (ts != AV_NOPTS_VALUE).then_some(ts);
    // Fields in the packet, which predict the next dts when it has no
    // duration: the parser's count for codecs that can code fields, else 2.
    let fields = if packet.duration != 0 {
        2
    } else {
        let descriptor = unsafe { ffi::avcodec_descriptor_get(stream.codecpar().codec_id) };
        let parser = unsafe { ffi::av_stream_get_parser(stream.as_ptr()) };
        if !descriptor.is_null()
            && unsafe { (*descriptor).props } & ffi::AV_CODEC_PROP_FIELDS as i32 != 0
            && !parser.is_null()
        {
            1 + i64::from(unsafe { (*parser).repeat_pict })
        } else {
            2
        }
    };
    let (pts, dts) = demux.fix(ts(packet.pts), ts(packet.dts), packet.duration, fields);
    packet.set_pts(pts.unwrap_or(AV_NOPTS_VALUE));
    packet.set_dts(dts.unwrap_or(AV_NOPTS_VALUE));
}

fn container_duration_seconds(input_format_context: &AVFormatContextInput) -> Option<f64> {
    let duration = input_format_context.duration;
    (duration != AV_NOPTS_VALUE && duration > 0)
        .then(|| duration as f64 * av_q2d(ffi::AV_TIME_BASE_Q))
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
        motion,
        frame_data,
        frame_data_ptr,
        frame_pts,
        finished,
        cfr,
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
                    motion,
                    frame_data,
                    frame_data_ptr,
                    frame_pts,
                    cfr,
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
            motion,
            frame_data,
            frame_data_ptr,
            frame_pts,
            cfr,
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
    if request.width.is_none() && request.height.is_none() {
        return None;
    }
    let (w, h) = resolve_output_size(src_w, src_h, request);
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
    motion: &mut Option<MotionStage<'_>>,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    cfr: &mut Option<CliSync>,
    seek: &Seek,
) -> Result<Vec<Frame>, anyhow::Error> {
    let mut frames: Vec<Frame> = Vec::new();

    loop {
        // Check if we need to stop decoding.
        match stream_type {
            StreamType::Video => {
                // CLI mode grows its buffer on demand rather than stopping.
                if cfr.is_none() && has_finished_decoding(stream_type, frame_data, frame_data_ptr) {
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

        frame.set_pts(frame.best_effort_timestamp);

        // CLI frame-rate sync: how many times this frame (and the previous
        // one, before it) is emitted. Decided from timestamps alone, before
        // any pixels move.
        let cli_decision = match cfr {
            Some(cli) => {
                let pts = (frame.pts != AV_NOPTS_VALUE).then_some(frame.pts);
                let mut decision = cli.sync.push(pts, frame.duration);
                if cli.previous_row.is_none() {
                    // No previous frame yet: the CLI pads with this one.
                    decision.current += decision.previous;
                    decision.previous = 0;
                }
                Some(decision)
            }
            None => None,
        };

        // Hardware decoders return frames in GPU memory; download them to
        // system memory so the software filter graph can process them —
        // unless the output is GPU-resident, in which case they stay put.
        if frame.format == ffi::AV_PIX_FMT_CUDA
            && !matches!(direct_path, DirectPath::CudaConvert(_))
        {
            frame = download_hw_frame(&frame).context("Downloading hardware frame")?;
        }

        if let Some(decision) = cli_decision {
            let cli = cfr.as_mut().expect("decision implies CLI sync");
            emit_previous_copies(
                cli,
                decision.previous,
                frame_data,
                frame_data_ptr,
                frame_pts,
            )?;
            // A dropped frame is not emitted, but it becomes the frame later
            // gaps repeat (the CLI keeps the last frame it processed), so it is
            // still converted, into the row after the output. Unless that row
            // is past the output limit: then any row emitted later fails the
            // limit anyway, so the frame can't be needed.
            let needed = *frame_data_ptr as i64 + 1;
            if decision.current == 0 && !cli.fits(frame_data, needed) {
                continue;
            }
            cli.ensure_capacity(frame_data, needed, needed)?;
        }
        let row_before = *frame_data_ptr;
        let frames_before = frames.len();

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
                after_cli_frame(
                    cfr,
                    cli_decision,
                    row_before,
                    frame_data,
                    frame_data_ptr,
                    frame_pts,
                )?;
                if cli_decision.is_some_and(|d| d.current == 0) {
                    frames.truncate(frames_before);
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
            motion.as_mut(),
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
        after_cli_frame(
            cfr,
            cli_decision,
            row_before,
            frame_data,
            frame_data_ptr,
            frame_pts,
        )?;
        if cli_decision.is_some_and(|d| d.current == 0) {
            frames.truncate(frames_before);
        }
    }

    Ok(frames)
}

/// Writes `copies` more copies of the previous emitted frame, as the CLI
/// duplicates a frame before the one that follows a gap.
fn emit_previous_copies(
    cli: &mut CliSync,
    copies: i64,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
) -> Result<(), anyhow::Error> {
    let Some(row) = cli.previous_row else {
        return Ok(());
    };
    if copies > 0 {
        // Room for one more row if the limit allows, so a frame that follows
        // (dropped or not) needs no growth of its own.
        let needed = *frame_data_ptr as i64 + copies;
        cli.ensure_capacity(frame_data, needed, needed + 1)?;
    }
    for _ in 0..copies {
        let pts = cli.sync.pts_seconds_of(*frame_data_ptr as i64);
        // A dropped frame's first copy is the row it was converted into.
        if row != *frame_data_ptr as i64 {
            frame_data
                .f_narrow(0, *frame_data_ptr as i64, 1)?
                .f_copy_(&frame_data.f_narrow(0, row, 1)?)?;
        }
        cli.previous_row = Some(*frame_data_ptr as i64);
        *frame_data_ptr += 1;
        frame_pts.push(pts);
    }
    Ok(())
}

/// After a CLI-synced frame was written (one row, at `row_before`): stamps it
/// with its output slot's timestamp and writes its own duplicates.
fn after_cli_frame(
    cfr: &mut Option<CliSync>,
    decision: Option<cfr::SyncDecision>,
    row_before: usize,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
) -> Result<(), anyhow::Error> {
    let (Some(cli), Some(decision)) = (cfr.as_mut(), decision) else {
        return Ok(());
    };
    if *frame_data_ptr == row_before {
        return Ok(()); // not written (e.g. before the seek start)
    }
    let row = row_before as i64;
    if decision.current == 0 {
        // Converted only to serve as the previous frame: it stays in the row
        // past the output, which its own copies (if any) start at; any other
        // frame written there replaces it as the previous frame.
        *frame_data_ptr = row_before;
        frame_pts.pop();
        cli.previous_row = Some(row);
        return Ok(());
    }
    if let Some(pts) = frame_pts.last_mut() {
        *pts = cli.sync.pts_seconds_of(row);
    }
    cli.previous_row = Some(row);
    emit_previous_copies(
        cli,
        decision.current - 1,
        frame_data,
        frame_data_ptr,
        frame_pts,
    )
}

/// Grows the CLI-mode output tensor to at least `needed` rows, and to
/// `wanted` or a quarter more than it holds as headroom, but never past
/// `output_limit`; returns whether it grew. The plan is an estimate;
/// duplicates past it are written rather than dropped, unless the rows
/// needed take more than `output_limit`, which is an error. Headroom is
/// only ever clamped, never an error.
fn ensure_cli_capacity(
    frame_data: &mut tch::Tensor,
    needed: i64,
    wanted: i64,
    output_limit: &OutputLimit,
) -> Result<bool, anyhow::Error> {
    let mut shape = frame_data.size();
    let held = shape[0];
    if held >= needed {
        return Ok(false);
    }
    let element_size = frame_data.kind().elt_size_in_bytes();
    let max_rows = max_cli_rows(frame_data, output_limit.bytes);
    // A quarter more than held, so growth stays rare, but the headroom stops
    // at the limit; the rows needed past it fail the check below.
    shape[0] = (held + held / 4 + 1).max(wanted).min(max_rows).max(needed);
    check_output_tensor_size(&shape, element_size, output_limit)
        .context("fps_mode=\"cli\" output outgrew its plan")?;
    // In place: the frame views already handed out move to the new buffer
    // instead of keeping each superseded one alive.
    crate::alloc::grow_rows(frame_data, shape[0]).context("growing the output tensor")?;
    Ok(true)
}

/// How many rows of `frame_data`'s shape fit in `max_output_bytes`.
fn max_cli_rows(frame_data: &tch::Tensor, max_output_bytes: u128) -> i64 {
    let shape = frame_data.size();
    let row_bytes = shape[1..].iter().map(|&dim| dim as u128).product::<u128>()
        * frame_data.kind().elt_size_in_bytes() as u128;
    (max_output_bytes / row_bytes.max(1)).min(i64::MAX as u128) as i64
}

/// Trims the grown CLI-mode output's storage to its `rows` emitted rows
/// (see [`crate::alloc::trim_rows`]), unless a frame view reaches past them.
/// Best effort: on failure the output keeps its unused rows.
fn trim_cli_output(frame_data: &mut tch::Tensor, rows: i64, frames: &[Frame]) {
    if frame_data.size()[0] <= rows {
        return;
    }
    let start = frame_data.data_ptr() as usize;
    let row_bytes =
        frame_data.numel() / frame_data.size()[0] as usize * frame_data.kind().elt_size_in_bytes();
    let end = start + rows as usize * row_bytes;
    let inside = |view: &tch::Tensor| {
        let at = view.data_ptr() as usize;
        at >= start && at + view.numel() * view.kind().elt_size_in_bytes() <= end
    };
    if !frames.iter().all(|frame| inside(frame.data())) {
        log::debug!("Not trimming the CLI-mode output: a frame view reaches past it");
        return;
    }
    // SAFETY: every view of the output handed out (the frames) lies in the
    // first `rows` rows, checked above.
    if let Err(e) = unsafe { crate::alloc::trim_rows(frame_data, rows) } {
        log::warn!("Not trimming the CLI-mode output: {e:#}");
    }
}

#[derive(thiserror::Error, Debug)]
enum FilterFrameError {
    #[error("No storage available: {0:?}")]
    NoStorageAvailable(anyhow::Error),
    #[error("Unexpected error occurred during filtering: {0:?}")]
    Unexpected(anyhow::Error),
}

/// Apply FFmpeg filter graph to a decoded frame (`None` flushes it), then
/// the stream's [`MotionStage`] if it has one.
#[allow(clippy::too_many_arguments)]
fn filter_frame(
    stream_type: StreamType,
    frame: Option<AVFrame>,
    buffersrc_ctx: &mut AVFilterContextMut,
    buffersink_ctx: &mut AVFilterContextMut,
    motion: Option<&mut MotionStage>,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    seek: &Seek,
) -> Result<Vec<Frame>, FilterFrameError> {
    let flush = frame.is_none();
    submit_frame(buffersrc_ctx, frame)?;
    let Some(stage) = motion else {
        return drain_sink(
            stream_type,
            buffersink_ctx,
            frame_data,
            frame_data_ptr,
            frame_pts,
            seek,
        );
    };

    let MotionStage {
        resampler,
        output:
            FilterContext {
                buffersrc_ctx: output_src,
                buffersink_ctx: output_sink,
            },
    } = stage;
    let mut filtered_frames = Vec::new();
    let mut emit = |frames: Vec<AVFrame>, filtered_frames: &mut Vec<Frame>| {
        for frame in frames {
            submit_frame(output_src, Some(frame))?;
            filtered_frames.extend(drain_sink(
                stream_type,
                output_sink,
                frame_data,
                frame_data_ptr,
                frame_pts,
                seek,
            )?);
        }
        Ok::<_, FilterFrameError>(())
    };
    while let Some(mut frame) = receive_filtered(buffersink_ctx)? {
        frame.set_time_base(buffersink_ctx.get_time_base());
        let frames = resampler
            .push(frame)
            .map_err(|e| FilterFrameError::Unexpected(e.context("Frame interpolation failed")))?;
        emit(frames, &mut filtered_frames)?;
    }
    if flush {
        emit(resampler.finish(), &mut filtered_frames)?;
        submit_frame(output_src, None)?;
        filtered_frames.extend(drain_sink(
            stream_type,
            output_sink,
            frame_data,
            frame_data_ptr,
            frame_pts,
            seek,
        )?);
    }
    Ok(filtered_frames)
}

/// Sends a frame (or, with `None`, end of stream) to a graph's source.
fn submit_frame(
    buffersrc_ctx: &mut AVFilterContextMut,
    frame: Option<AVFrame>,
) -> Result<(), FilterFrameError> {
    buffersrc_ctx.buffersrc_add_frame(frame, None).map_err(|e| {
        FilterFrameError::Unexpected(
            anyhow!(e).context("Error submitting the frame to the filtergraph"),
        )
    })
}

/// The next frame waiting in a graph's sink, if any.
fn receive_filtered(
    buffersink_ctx: &mut AVFilterContextMut,
) -> Result<Option<AVFrame>, FilterFrameError> {
    match buffersink_ctx.buffersink_get_frame(None) {
        Ok(frame) => Ok(Some(frame)),
        Err(RsmpegError::BufferSinkDrainError) | Err(RsmpegError::BufferSinkEofError) => Ok(None),
        Err(_) => Err(FilterFrameError::Unexpected(anyhow!(
            "Get frame from buffersink failed"
        ))),
    }
}

/// Writes every frame waiting in `buffersink_ctx` to `frame_data`.
fn drain_sink(
    stream_type: StreamType,
    buffersink_ctx: &mut AVFilterContextMut,
    frame_data: &mut tch::Tensor,
    frame_data_ptr: &mut usize,
    frame_pts: &mut Vec<f64>,
    seek: &Seek,
) -> Result<Vec<Frame>, FilterFrameError> {
    let mut filtered_frames: Vec<Frame> = Vec::new();
    while let Some(mut filtered_frame) = receive_filtered(buffersink_ctx)? {
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
    /// swscale context, rebuilt whenever the source frame parameters change.
    /// Null until the first frame.
    sws_ctx: *mut ffi::SwsContext,
    /// Source parameters `sws_ctx` was built for.
    src: (i32, i32, i32),
    dst_width: i32,
    dst_height: i32,
    /// swscale slice threads, from the request's `number_of_threads` (see
    /// [`direct_scaler_threads`]). The default of 1 converts on one thread,
    /// like the single-threaded `sws_scale` this path used before, so loaders
    /// running many decodes in parallel spend no extra CPU per decode;
    /// `number_of_threads=0` opts into the pool for a faster single decode.
    threads: i32,
    /// Fail on frames whose YUV matrix the `scale` filter rejects instead of
    /// converting them as BT.601 (see [`direct_scale_supports_colorspace`]).
    /// Set for raw-mode HDR streams, which went through the graph (and failed
    /// there on such matrices) before this path handled them.
    check_matrix: bool,
    /// Matrix for frames with none tagged. Untagged HDR is BT.2020 (NCL),
    /// the HDR standard, as the filter graph's `setparams` tags it (see
    /// [`SourceColorInfo::hdr_defaults`]); otherwise swscale's default.
    untagged_colorspace: ffi::AVColorSpace,
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
    fn new(config: &VideoFilterConfig, dst_width: i32, dst_height: i32) -> Self {
        Self {
            sws_ctx: std::ptr::null_mut(),
            src: (0, 0, ffi::AV_PIX_FMT_NONE),
            dst_width,
            dst_height,
            threads: direct_scaler_threads(config.number_of_threads),
            check_matrix: config.source_color.is_hdr(),
            untagged_colorspace: if config.source_color.is_hdr() {
                ffi::AVCOL_SPC_BT2020_NCL
            } else {
                ffi::AVCOL_SPC_UNSPECIFIED
            },
        }
    }

    /// Returns a swscale context for `frame`, (re)building it when the source
    /// parameters change.
    fn context_for(&mut self, frame: &AVFrame) -> Result<*mut ffi::SwsContext, anyhow::Error> {
        let src = (frame.width, frame.height, frame.format);
        if !self.sws_ctx.is_null() && self.src == src {
            return Ok(self.sws_ctx);
        }
        unsafe { ffi::sws_freeContext(self.sws_ctx) };
        self.sws_ctx = std::ptr::null_mut();
        let ctx = unsafe { ffi::sws_alloc_context() };
        if ctx.is_null() {
            bail!("sws_alloc_context failed");
        }
        let options: [(&CStr, i64); 8] = [
            (c"srcw", frame.width as i64),
            (c"srch", frame.height as i64),
            (c"src_format", frame.format as i64),
            (c"dstw", self.dst_width as i64),
            (c"dsth", self.dst_height as i64),
            (c"dst_format", ffi::AV_PIX_FMT_RGB24 as i64),
            // The scale filter's default; keeps output identical to the
            // filter-graph path.
            (c"sws_flags", ffi::SWS_BICUBIC as i64),
            (
                c"threads",
                scaler_threads(frame.format, self.threads) as i64,
            ),
        ];
        for (name, value) in options {
            let ret = unsafe { ffi::av_opt_set_int(ctx as *mut _, name.as_ptr(), value, 0) };
            if ret < 0 {
                unsafe { ffi::sws_freeContext(ctx) };
                bail!("setting swscale option {name:?} failed: {ret}");
            }
        }
        if unsafe { ffi::sws_init_context(ctx, std::ptr::null_mut(), std::ptr::null_mut()) } < 0 {
            unsafe { ffi::sws_freeContext(ctx) };
            bail!(
                "sws_init_context failed ({}x{} pix_fmt {} -> {}x{} rgb24)",
                frame.width,
                frame.height,
                frame.format,
                self.dst_width,
                self.dst_height
            );
        }
        self.sws_ctx = ctx;
        self.src = src;
        Ok(ctx)
    }

    /// Returns true when the direct path should handle this configuration:
    /// RGB24 output, the fast (swscale) color conversion, no resize, no
    /// frame-rate resampling, no HDR tone
    /// mapping (an HDR source with `HdrMode::Raw` is a plain conversion, as
    /// long as the `scale` filter accepts its matrix), and a 32-multiple
    /// output width (libswscale's vectorized RGB24 output requires this on
    /// the tightly packed rows of the output tensor). Its output is
    /// identical to the filter graph's.
    ///
    /// Resizes deliberately stay in the filter graph: the graph's `scale`
    /// filter downscales with slice threading (and in subsampled YUV before
    /// the RGB conversion), which measures faster than a single-threaded
    /// one-pass `sws_scale`, and with a small output the extra frame copy
    /// the graph costs is small anyway.
    fn is_eligible(
        config: &VideoFilterConfig,
        dst_width: i64,
        colorspace: ffi::AVColorSpace,
    ) -> bool {
        config.frame_rate.is_none()
            && config.rotate.is_empty()
            && config.width.is_none()
            && config.height.is_none()
            && (!config.source_color.is_hdr()
                || (config.hdr_mode == HdrMode::Raw
                    && direct_scale_supports_colorspace(colorspace)))
            && config.color_conversion == ColorConversion::Fast
            && config.pixel_format == "rgb24"
            && dst_width % 32 == 0
            && direct_paths_enabled()
    }

    /// Scales/converts `frame` into `dest`, a `[1, dst_height, dst_width, 3]`
    /// uint8 tensor view.
    fn scale_into(&mut self, frame: &AVFrame, dest: &tch::Tensor) -> Result<(), anyhow::Error> {
        if self.check_matrix && !direct_scale_supports_colorspace(frame.colorspace) {
            bail!(
                "unsupported YUV matrix {} (AVColorSpace {})",
                ffi_color_name(unsafe { ffi::av_color_space_name(frame.colorspace) })
                    .unwrap_or_default(),
                frame.colorspace
            );
        }
        let ctx = self.context_for(frame)?;

        // Mirror the filter graph's color handling: it forwards the stream's
        // colorspace and range to buffersrc so the YUV->RGB conversion uses
        // the tagged metadata rather than a guess. When both are untagged,
        // leave swscale's defaults in place (same as the graph).
        let colorspace = if frame.colorspace == ffi::AVCOL_SPC_UNSPECIFIED {
            self.untagged_colorspace
        } else {
            frame.colorspace
        };
        let range = frame.color_range;
        if colorspace != ffi::AVCOL_SPC_UNSPECIFIED || range != ffi::AVCOL_RANGE_UNSPECIFIED {
            let src_range = (range == ffi::AVCOL_RANGE_JPEG) as i32;
            unsafe {
                // swscale's coefficient table is indexed by AVColorSpace, and
                // the scale filter looks the frame's colorspace up the same
                // way. swscale maps matrices it has no entry for to BT.601,
                // but the filter rejects those during format negotiation;
                // see `direct_scale_supports_colorspace`.
                let inv_table = ffi::sws_getCoefficients(colorspace as i32);
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

        // sws_scale_frame runs the context's slice threads (sws_scale does
        // not). It needs a refcounted destination, so the tensor's row block
        // is wrapped in a buffer that does not own it.
        let dst_ptr = dest.data_ptr() as *mut u8;
        let dst_stride = self.dst_width * 3;
        let dst_size = dst_stride as usize * self.dst_height as usize;
        let mut dst = unsafe { ffi::av_frame_alloc() };
        if dst.is_null() {
            bail!("av_frame_alloc failed");
        }
        let result = (|| {
            let buf = unsafe {
                ffi::av_buffer_create(
                    dst_ptr,
                    dst_size,
                    Some(borrowed_buffer_free),
                    std::ptr::null_mut(),
                    0,
                )
            };
            if buf.is_null() {
                bail!("av_buffer_create failed");
            }
            unsafe {
                (*dst).buf[0] = buf;
                (*dst).data[0] = dst_ptr;
                (*dst).linesize[0] = dst_stride;
                (*dst).width = self.dst_width;
                (*dst).height = self.dst_height;
                (*dst).format = ffi::AV_PIX_FMT_RGB24;
            }
            let ret = unsafe { ffi::sws_scale_frame(ctx, dst, frame.as_ptr()) };
            if ret < 0 {
                bail!("sws_scale_frame failed: {ret}");
            }
            Ok(())
        })();
        unsafe { ffi::av_frame_free(&mut dst) };
        #[cfg(test)]
        if result.is_ok() {
            let mut threads = 0;
            unsafe { ffi::av_opt_get_int(ctx as *mut _, c"threads".as_ptr(), 0, &mut threads) };
            direct_scale_stats::record(threads);
        }
        result
    }
}

/// Test-only record of the direct swscale conversions on the current thread.
/// Decoding runs on the caller's thread, so a test can tell which path its
/// own decode took, even with other tests decoding in parallel.
#[cfg(test)]
mod direct_scale_stats {
    use std::cell::Cell;

    #[derive(Debug, Clone, Copy, Default)]
    pub(super) struct Stats {
        /// Frames converted on the direct path.
        pub frames: usize,
        /// swscale threads of the last conversion's context.
        pub threads: i64,
    }

    thread_local! {
        static STATS: Cell<Stats> = Cell::new(Stats::default());
    }

    pub(super) fn record(threads: i64) {
        STATS.with(|s| {
            s.set(Stats {
                frames: s.get().frames + 1,
                threads,
            })
        });
    }

    /// Returns the conversions recorded since the last call.
    pub(super) fn take() -> Stats {
        STATS.with(|s| s.take())
    }
}

/// swscale slice threads for the direct path, from the request's
/// `number_of_threads` the way the decoder reads it: the default of 1 is one
/// thread, `0` (auto) is the filter graph's cap, and an explicit N is N,
/// capped the same way since slice threading gains little beyond it.
fn direct_scaler_threads(number_of_threads: Option<usize>) -> i32 {
    // The cap reads the CPU affinity and cgroup quota (~15 us), which is
    // measurable on small images, so skip it for the default.
    match number_of_threads.unwrap_or(1) {
        1 => 1,
        0 => filter_graph_threads(),
        n => n.min(filter_graph_threads() as usize) as i32,
    }
}

/// Returns true for the YUV matrices the direct path converts like the
/// filter graph: those the `scale` filter accepts (`sws_colorspaces` in
/// FFmpeg 7.1's vf_scale.c), plus BT.2020 constant luminance, which the
/// filter rejects but swscale has coefficients for. swscale converts any
/// other matrix (YCgCo, chroma-derived, ICtCp, ...) as BT.601 without an
/// error, while the graph fails on it.
fn direct_scale_supports_colorspace(colorspace: ffi::AVColorSpace) -> bool {
    matches!(
        colorspace,
        ffi::AVCOL_SPC_UNSPECIFIED
            | ffi::AVCOL_SPC_RGB
            | ffi::AVCOL_SPC_BT709
            | ffi::AVCOL_SPC_BT470BG
            | ffi::AVCOL_SPC_SMPTE170M
            | ffi::AVCOL_SPC_FCC
            | ffi::AVCOL_SPC_SMPTE240M
            | ffi::AVCOL_SPC_BT2020_NCL
            | ffi::AVCOL_SPC_BT2020_CL
    )
}

/// swscale slice threads for a source pixel format. XYZ sources get one:
/// every slice worker converts the whole frame from XYZ before scaling its own
/// rows, so each extra worker repeats that work and keeps another full-frame
/// scratch buffer (about 50 MiB at 4K).
fn scaler_threads(src_format: i32, threads: i32) -> i32 {
    let desc = unsafe { ffi::av_pix_fmt_desc_get(src_format) };
    if !desc.is_null() && unsafe { (*desc).flags } & ffi::AV_PIX_FMT_FLAG_XYZ as u64 != 0 {
        1
    } else {
        threads
    }
}

/// `av_buffer_create` free callback for memory the buffer only borrows (the
/// output tensor owns it).
unsafe extern "C" fn borrowed_buffer_free(_opaque: *mut std::ffi::c_void, _data: *mut u8) {}

/// Slice threads for a filter graph and its scalers, and the cap on the
/// direct path's swscale pool. FFmpeg's default (0 = one worker per core)
/// builds a large thread pool for every decode on many-core machines, which
/// oversubscribes badly when many decodes run concurrently. Slice threading
/// cannot use more workers than there are output rows/slices anyway, so a
/// small pool preserves single-decode performance.
fn filter_graph_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(16)
        .min(16) as i32
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
    /// The buffersrc filter reads it from the decoder context directly; the
    /// accurate conversion path checks whether it is tagged at all.
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

    /// `setparams` options that tag HDR frames missing a matrix or primaries
    /// tag as BT.2020 (NCL), the HDR standard. Without a matrix zimg cannot
    /// linearize ("no path between colorspaces") and swscale guesses BT.601;
    /// without primaries the conversion to BT.709 has nothing to convert from.
    fn hdr_defaults(&self) -> Vec<&'static str> {
        let mut defaults = Vec::new();
        if self.is_hdr() {
            if self.colorspace.is_none() {
                defaults.push("colorspace=bt2020nc");
            }
            if self.color_primaries.is_none() {
                defaults.push("color_primaries=bt2020");
            }
        }
        defaults
    }
}

/// How HDR/wide-gamut sources are handled during decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HdrMode {
    /// Tone map to SDR BT.709 (the historical default): linearize, hable
    /// tone map, convert primaries/transfer/matrix to BT.709.
    #[default]
    Tonemap,
    /// Preserve the source's raw code values: YUV→RGB uses the stream's
    /// tagged matrix and range only; the transfer function (e.g. PQ/HLG)
    /// is NOT linearized, tone mapped, or converted. Required when the
    /// consumer needs the actual HDR signal (training on PQ masters,
    /// colorimetric measurement) rather than an SDR preview.
    Raw,
}

impl TryFrom<Option<&str>> for HdrMode {
    type Error = anyhow::Error;

    fn try_from(value: Option<&str>) -> Result<Self, Self::Error> {
        match value {
            None | Some("tonemap") => Ok(Self::Tonemap),
            Some("raw") => Ok(Self::Raw),
            Some(other) => Err(anyhow::anyhow!(
                "hdr_mode must be \"tonemap\" or \"raw\", got {other:?}"
            )),
        }
    }
}

/// How decoded YUV frames are converted to RGB.
///
/// libswscale's default YUV->RGB path (FFmpeg 7.1) has two measurable errors:
/// its 8-bit packed `rgb24` output is biased low by ~0.5-1.4 code values per
/// channel on 4:2:0 sources (chroma handling and truncation without
/// `SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT`), and its float `gbrpf32le` output
/// is scaled by 255/256 (white decodes to 0.99615, not 1.0). `Accurate`
/// converts with zimg (`zscale`), which is exact in float and rounds to
/// nearest for 8-bit output, at a comparable cost to swscale's fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorConversion {
    /// libswscale's default conversion (historical behaviour).
    #[default]
    Fast,
    /// zimg conversion: exact float output and round-to-nearest 8-bit output.
    Accurate,
}

impl ColorConversion {
    /// Resolves the requested conversion. Unset means `Fast` for `Uint8`
    /// (bit-identical to earlier releases) and `Accurate` for `Float32`,
    /// falling back to `Fast` (with a warning) when FFmpeg lacks `zscale`;
    /// an explicit `Accurate` without `zscale` is an error.
    fn resolve(requested: Option<Self>, dtype: OutputDtype) -> anyhow::Result<Self> {
        match (requested, dtype) {
            (Some(Self::Accurate), _) if !zscale_available() => Err(anyhow!(
                "color_conversion=\"accurate\" needs FFmpeg's zscale filter \
                 (libzimg), which this FFmpeg build lacks; use a build with \
                 libzimg or pass color_conversion=\"fast\""
            )),
            (Some(conversion), _) => Ok(conversion),
            (None, OutputDtype::Uint8) => Ok(Self::Fast),
            (None, OutputDtype::Float32) if zscale_available() => Ok(Self::Accurate),
            (None, OutputDtype::Float32) => {
                static WARN: std::sync::Once = std::sync::Once::new();
                WARN.call_once(|| {
                    log::warn!(
                        "FFmpeg lacks the zscale filter (libzimg): float32 video \
                         falls back to color_conversion=\"fast\" (swscale; \
                         scaled by 255/256)"
                    )
                });
                Ok(Self::Fast)
            }
        }
    }
}

/// Whether the linked FFmpeg has the `zscale` filter (built with libzimg).
fn zscale_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| AVFilter::get_by_name(c"zscale").is_some())
}

/// Pixel layout of the frames entering the filter graph, as far as the
/// zscale conversions need it.
#[derive(Debug, Clone, Default)]
struct SourceFormat {
    /// Frame size entering the graph.
    width: usize,
    height: usize,
    /// For chroma-subsampled YUV: the 4:4:4 format of the same depth and
    /// range, and the subsampling shifts (log2) along each axis.
    yuv444: Option<&'static str>,
    log2_chroma: (u8, u8),
    /// The format has no chroma planes (gray / gray + alpha).
    gray: bool,
}

impl SourceFormat {
    fn new(pix_fmt: AVPixelFormat, width: i32, height: i32) -> Self {
        let mut format = Self {
            width: width.max(0) as usize,
            height: height.max(0) as usize,
            ..Default::default()
        };
        let desc = unsafe { ffi::av_pix_fmt_desc_get(pix_fmt) };
        if desc.is_null() {
            return format;
        }
        let desc = unsafe { &*desc };
        let flags = (ffi::AV_PIX_FMT_FLAG_RGB | ffi::AV_PIX_FMT_FLAG_PAL) as u64;
        if desc.flags & flags != 0 {
            return format;
        }
        format.gray = desc.nb_components < 3;
        let subsampled = desc.log2_chroma_w > 0 || desc.log2_chroma_h > 0;
        if !format.gray && subsampled {
            // Keep JPEG-style full-range formats full range.
            let full_range = unsafe { std::ffi::CStr::from_ptr(desc.name) }
                .to_bytes()
                .starts_with(b"yuvj");
            format.yuv444 = Some(match desc.comp[0].depth {
                _ if full_range => "yuvj444p",
                ..=8 => "yuv444p",
                9..=10 => "yuv444p10le",
                11..=12 => "yuv444p12le",
                _ => "yuv444p16le",
            });
            format.log2_chroma = (desc.log2_chroma_w, desc.log2_chroma_h);
        }
        format
    }

    /// The 4:4:4 format to convert to before a frame of `width` x `height`
    /// (in this subsampled format) reaches zscale: zimg rejects subsampled
    /// frames whose dimensions are not divisible by the subsampling factor
    /// (swscale does not).
    fn upsample_for(&self, width: usize, height: usize) -> Option<&'static str> {
        let (sw, sh) = self.log2_chroma;
        let misaligned = !width.is_multiple_of(1 << sw) || !height.is_multiple_of(1 << sh);
        self.yuv444.filter(|_| misaligned)
    }
}

/// Resolves the requested output size against the source size. A lone
/// `width` or `height` keeps the source aspect ratio (rounded to the nearest
/// pixel); neither keeps the source size.
fn resolve_output_size(src_w: i64, src_h: i64, req: &VideoStreamRequest) -> (i64, i64) {
    let scaled = |num: i64, a: i64, b: i64| {
        if b > 0 {
            ((num * a + b / 2) / b).max(1)
        } else {
            a
        }
    };
    match (req.width.map(i64::from), req.height.map(i64::from)) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, scaled(w, src_h, src_w)),
        (None, Some(h)) => (scaled(h, src_w, src_h), h),
        (None, None) => (src_w, src_h),
    }
}

impl TryFrom<&str> for ColorConversion {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "fast" => Ok(Self::Fast),
            "accurate" => Ok(Self::Accurate),
            other => Err(anyhow::anyhow!(
                "color_conversion must be \"fast\" or \"accurate\", got {other:?}"
            )),
        }
    }
}

/// Whether `source` fps reaches `target` by keeping every n-th frame,
/// allowing for NTSC's 1000/1001 rate offset (23.976 vs 24).
fn is_integer_downsample(source: f64, target: f64) -> bool {
    if !(source > 0.0 && target > 0.0) {
        return false;
    }
    let ratio = source / target;
    let n = ratio.round();
    n >= 1.0 && (ratio / n - 1.0).abs() < 2e-3
}

/// How frame-rate resampling builds each output frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FpsConversion {
    /// Keep the source frame nearest each output timestamp (FFmpeg `fps`),
    /// dropping or repeating whole frames. Fast, but a non-integer ratio
    /// gives uneven frame spacing (30 -> 24 fps keeps 4 of every 5 frames,
    /// so motion steps go 1x, 1x, 1x, 2x).
    #[default]
    Nearest,
    /// Cross-fade the two source frames around each output timestamp. Even
    /// timing at little more than `Nearest`'s cost, but moving edges show as
    /// double images.
    Blend,
    /// Motion-compensated interpolation (block matching with overlapped
    /// block motion compensation; the model of FFmpeg's `minterpolate`
    /// defaults, implemented in [`motion`] for speed). Even timing without
    /// double images, at about 1.2-1.6x `Nearest`'s cost.
    Motion,
    // `Blend` and `Motion` both run in the [`MotionStage`] and take the
    // nearer source frame across detected scene cuts (see [`SceneCuts`]).
    // `Blend` keeps sources deeper than 8 bits at 16 bits; `Motion` works on
    // 8-bit YUV, so it reduces them to 8 bits before interpolation.
    // `Blend` and `Motion` fall back to `Nearest` when the source rate is an
    // integer multiple of the target (see `is_integer_downsample`).
}

/// Scene-cut handling for [`FpsConversion::Blend`] and
/// [`FpsConversion::Motion`]. Output frames between two source frames on
/// opposite sides of a cut would mix two unrelated shots, so with
/// detection on they repeat the nearer source frame instead (as `Nearest`
/// would). Frame count and timestamps are the same either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SceneCuts {
    /// Detect cuts (see [`scenecut`]) and never interpolate across them.
    #[default]
    Detect,
    /// Interpolate every output frame, even across cuts.
    Off,
}

impl TryFrom<Option<&str>> for SceneCuts {
    type Error = anyhow::Error;

    fn try_from(value: Option<&str>) -> Result<Self, Self::Error> {
        match value {
            None | Some("detect") => Ok(Self::Detect),
            Some("off") => Ok(Self::Off),
            Some(other) => Err(anyhow::anyhow!(
                "scene_cuts must be \"detect\" or \"off\", got {other:?}"
            )),
        }
    }
}

impl TryFrom<Option<&str>> for FpsConversion {
    type Error = anyhow::Error;

    fn try_from(value: Option<&str>) -> Result<Self, Self::Error> {
        match value {
            None | Some("nearest") => Ok(Self::Nearest),
            Some("blend") => Ok(Self::Blend),
            Some("motion") => Ok(Self::Motion),
            Some(other) => Err(anyhow::anyhow!(
                "fps_conversion must be \"nearest\", \"blend\" or \"motion\", got {other:?}"
            )),
        }
    }
}

#[derive(Debug, Clone)]
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
    /// How HDR sources are converted (tone-mapped SDR vs raw code values).
    hdr_mode: HdrMode,
    /// YUV->RGB conversion backend (resolved; see [`ColorConversion`]).
    color_conversion: ColorConversion,
    /// Pixel layout of the frames entering the graph (accurate path only).
    source_format: SourceFormat,
    /// How `frame_rate` resampling builds each output frame.
    fps_conversion: FpsConversion,
    /// Scene-cut handling for the interpolating `fps_conversion` modes.
    scene_cuts: SceneCuts,
    /// The request's `number_of_threads`; sizes the direct path's swscale
    /// pool (see [`direct_scaler_threads`]).
    number_of_threads: Option<usize>,
    /// Rotation filters to apply (empty unless autorotate is requested and
    /// the stream has a non-zero display rotation).
    rotate: Vec<String>,
}

impl Display for VideoFilterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut filters = Vec::new();
        let hdr_defaults = self.source_color.hdr_defaults();
        if !hdr_defaults.is_empty() {
            filters.push(format!("setparams={}", hdr_defaults.join(":")));
        }
        // Nearest-frame selection runs first so dropped frames are never
        // scaled. The interpolating modes need every source frame and their
        // cost grows with pixel count, so they run after the resize and tone
        // mapping, outside the graph (see `MotionStage`).
        if let (Some(frame_rate), FpsConversion::Nearest) = (self.frame_rate, self.fps_conversion) {
            filters.push(format!("fps={}", frame_rate));
        }
        // Rotate before scaling, so `width`/`height` refer to the displayed
        // (rotated) frame, as with the ffmpeg CLI.
        filters.extend(self.rotate.iter().cloned());

        // For HDR/wide-gamut content, insert tone mapping and gamut conversion
        // to produce correct sRGB output instead of a naive YUV→RGB conversion.
        // With `HdrMode::Raw` the block is skipped entirely: the plain
        // format conversion below honors the tagged matrix/range but leaves
        // the transfer function (PQ/HLG code values) untouched.
        let tonemap = self.source_color.is_hdr() && self.hdr_mode == HdrMode::Tonemap;
        let accurate = self.color_conversion == ColorConversion::Accurate;
        // The accurate path resizes in zscale, inside the first zscale of the
        // chain: swscale would resize in YUV first, which reintroduces its
        // bias and can hand zimg a subsampled frame it rejects (e.g. 4:2:0 at
        // 853x480).
        let zscale_size = || {
            let mut size = String::new();
            if let Some(width) = self.width {
                size.push_str(&format!(":w={width}"));
            }
            if let Some(height) = self.height {
                size.push_str(&format!(":h={height}"));
            }
            size
        };
        let source = &self.source_format;

        let mut scales = Vec::new();
        if let Some(width) = self.width {
            scales.push(format!("width={}", width));
        }
        if let Some(height) = self.height {
            scales.push(format!("height={}", height));
        }
        if !scales.is_empty() && !accurate {
            filters.push(format!("scale={}", scales.join(":")));
        }

        if tonemap {
            log::debug!(
                "HDR source detected (trc={:?}, primaries={:?}), inserting tone mapping pipeline",
                self.source_color.color_trc,
                self.source_color.color_primaries
            );
            // The first zscale reads the YUV frame, so it must be 4:4:4 when
            // its size is not divisible by the chroma subsampling: the
            // source size in accurate mode (the resize happens in this
            // zscale), the output size otherwise.
            let upsample = if accurate {
                source.upsample_for(source.width, source.height)
            } else {
                source.upsample_for(
                    self.width.unwrap_or(source.width),
                    self.height.unwrap_or(source.height),
                )
            };
            if let Some(yuv444) = upsample {
                filters.push(format!("format={yuv444}"));
            }
            // 1. Linearize the transfer function (npl=100 for HLG nominal
            // peak luminance). Untagged matrix/primaries were set by the
            // `setparams` above.
            let mut linearize = "zscale=t=linear:npl=100".to_string();
            if accurate {
                linearize.push_str(&format!(":f=bicubic{}", zscale_size()));
            }
            filters.push(linearize);
            // 2. Convert to float for precision during tone mapping
            filters.push("format=gbrpf32le".to_string());
            // 3. Tone map HDR → SDR dynamic range
            filters.push("tonemap=hable:desat=0".to_string());
            // 4. Convert primaries, transfer, and matrix to BT.709 (sRGB)
            filters.push("zscale=p=bt709:t=bt709:m=bt709:range=tv".to_string());
        }

        if accurate {
            // Tone-mapped frames are already RGB at the output size here.
            if !tonemap {
                if let Some(yuv444) = source.upsample_for(source.width, source.height) {
                    filters.push(format!("format={yuv444}"));
                }
            }
            // Convert to RGB with zscale instead of letting the final
            // `format` auto-insert swscale. zimg refuses a YUV source with an
            // unspecified matrix, so default it to BT.601 like swscale does
            // (tone-mapped frames are tagged BT.709 by the chain above).
            // Gray sources need it too, even when tagged: FFmpeg feeds them
            // to zscale through an auto-inserted yuvj444p conversion that
            // carries no matrix. Gray has neutral chroma, so the choice of
            // matrix does not change the result.
            let mut zscale = "zscale=f=bicubic".to_string();
            if !tonemap {
                zscale.push_str(&zscale_size());
            }
            // (`min` overrides frame tags, so it must not undo the HDR
            // default set by `setparams`.)
            let untagged = self.source_color.colorspace.is_none() && !self.source_color.is_hdr();
            if !tonemap && (untagged || source.gray) {
                zscale.push_str(":min=170m");
            }
            filters.push(zscale);
            // zscale cannot emit packed rgb24; have it quantize to planar
            // gbrp (round to nearest) so the final step is a lossless repack.
            if self.pixel_format == "rgb24" {
                filters.push("format=gbrp".to_string());
            }
        }

        filters.push(format!("format=pix_fmts={}", self.pixel_format));
        if !filters.is_empty() {
            write!(f, "{}", filters.join(","))
        } else {
            Ok(())
        }
    }
}

impl VideoFilterConfig {
    /// Whether frame-rate conversion runs in a [`MotionStage`] after the
    /// filter graph rather than inside it (the interpolating modes).
    fn uses_motion_stage(&self) -> bool {
        self.frame_rate.is_some() && self.fps_conversion != FpsConversion::Nearest
    }

    /// Graph feeding a [`MotionStage`]: this config's resize and tone
    /// mapping, without frame-rate conversion, ending in the planar YUV
    /// format `working_format` that interpolation works in.
    fn motion_input(&self, working_format: &str) -> Self {
        let tonemap = self.source_color.is_hdr() && self.hdr_mode == HdrMode::Tonemap;
        Self {
            frame_rate: None,
            pixel_format: working_format.to_string(),
            // This graph ends in YUV, not RGB: the RGB conversion belongs to
            // the stage's output graph, which applies `color_conversion`.
            // Tone mapping runs here, though, so it keeps the requested
            // conversion: `Accurate` resizes and linearizes in zscale, as the
            // same request without frame-rate conversion does.
            color_conversion: if tonemap {
                self.color_conversion
            } else {
                ColorConversion::Fast
            },
            ..self.clone()
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
            hdr_mode: Default::default(),
            color_conversion: Default::default(),
            source_format: Default::default(),
            fps_conversion: Default::default(),
            scene_cuts: Default::default(),
            number_of_threads: Default::default(),
            rotate: Default::default(),
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
            hdr_mode: req.hdr_mode,
            color_conversion: ColorConversion::resolve(req.color_conversion, req.dtype)?,
            fps_conversion: req.fps_conversion,
            scene_cuts: req.scene_cuts,
            number_of_threads: req.number_of_threads,
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
/// Builds each stream's processing. Every stream gets a pair of graphs: the
/// first is its filter graph, the second is only used by a [`MotionStage`].
fn init_filters(
    filter_graphs: &mut [(AVFilterGraph, AVFilterGraph)],
    stream_contexts: Vec<StreamContext>,
) -> Result<Vec<FilteringContext<'_>>, anyhow::Error> {
    let mut filter_ctx = Vec::with_capacity(stream_contexts.len());

    for ((filter_graph, motion_graph), stream_context) in
        filter_graphs.iter_mut().zip(stream_contexts)
    {
        let StreamContext {
            mut dec_ctx,
            stream_index,
            metadata,
            filter_config,
            stream_type,
            frame_data,
            cfr,
        } = stream_context;

        let direct_path = match &filter_config {
            // Software-decoded video streams whose only processing is format
            // conversion skip the filter graph and convert straight into the
            // output tensor. Hardware-decoded streams stay in the graph:
            // their CPU work is conversion only, which the graph
            // slice-threads at its usual cap, while the direct path follows
            // `number_of_threads` (one thread by default).
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
                    && DirectVideoScaler::is_eligible(config, dst_width, dec_ctx.colorspace)
                {
                    log::debug!("Using the direct swscale path for stream {stream_index}");
                    DirectPath::VideoScale(DirectVideoScaler::new(
                        config,
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
        let mut motion = None;
        let (buffersrc_ctx, buffersink_ctx) = match &filter_config {
            _ if !matches!(direct_path, DirectPath::Disabled) => (None, None),
            FilterConfig::Video(config) if config.uses_motion_stage() => {
                let working_format = motion_working_format(config, &dec_ctx);
                let input_config = FilterConfig::Video(config.motion_input(working_format));
                let FilterContext {
                    buffersrc_ctx,
                    buffersink_ctx,
                } = init_filter(filter_graph, &mut dec_ctx, &input_config)
                    .context("Failed to initialize filter")?;
                motion = Some(
                    init_motion_stage(motion_graph, &buffersink_ctx, config)
                        .context("Failed to initialize frame interpolation")?,
                );
                log::debug!("Initialized frame interpolation for stream {stream_index}");
                (Some(buffersrc_ctx), Some(buffersink_ctx))
            }
            _ => {
                let FilterContext {
                    buffersrc_ctx,
                    buffersink_ctx,
                } = init_filter(filter_graph, &mut dec_ctx, &filter_config)
                    .context("Failed to initialize filter")?;
                log::debug!("Initialized filter for stream {stream_index} of type {stream_type:?}");
                (Some(buffersrc_ctx), Some(buffersink_ctx))
            }
        };

        filter_ctx.push(FilteringContext {
            stream_type,
            stream_index,
            metadata,
            dec_ctx,
            buffersrc_ctx,
            buffersink_ctx,
            direct_path,
            motion,
            frame_data,
            frame_data_ptr: 0,
            frame_pts: Vec::new(),
            finished: false,
            cfr,
        })
    }

    Ok(filter_ctx)
}

/// Caps a graph's slice-threading pool. FFmpeg's default (0 = one worker per
/// core) builds a large thread pool for every decode on many-core machines,
/// which oversubscribes badly when many decodes run concurrently. Slice
/// threading cannot use more workers than there are output rows/slices
/// anyway, so a small pool preserves single-decode filter performance.
fn cap_graph_threads(filter_graph: &mut AVFilterGraph) {
    unsafe { (*filter_graph.as_mut_ptr()).nb_threads = filter_graph_threads() };
}

/// The planar YUV format a [`MotionStage`] interpolates in. Keeps the
/// source's chroma subsampling when frames keep their size, so nothing is
/// resampled. A resize or HDR tone mapping already resamples chroma, and
/// subsampling at the output size would leave less chroma detail than the
/// other conversions (which upsample the source chroma straight to the
/// output size), so those use 4:4:4.
///
/// Samples are 8-bit, except that [`FpsConversion::Blend`] keeps sources
/// deeper than 8 bits (10-bit, HDR) at 16 bits, as the in-graph `framerate`
/// filter did: their passthrough and blended frames keep the source's
/// precision. That holds for `uint8` output too: an 8-bit working format
/// would round the source to 8-bit YUV and the RGB conversion would round
/// again, while 16-bit YUV rounds once, as a decode without frame-rate
/// conversion does. [`FpsConversion::Motion`] interpolates 8-bit frames.
fn motion_working_format(config: &VideoFilterConfig, dec_ctx: &AVCodecContext) -> &'static str {
    let resized = config.width.is_some_and(|w| w != dec_ctx.width as usize)
        || config.height.is_some_and(|h| h != dec_ctx.height as usize);
    let tonemapped = config.source_color.is_hdr() && config.hdr_mode == HdrMode::Tonemap;
    let source = buffersrc_pixel_format(dec_ctx);
    let format = match motion::working_pixel_format(source) {
        "yuvj420p" | "yuvj422p" | "yuvj444p" if resized || tonemapped => "yuvj444p",
        _ if resized || tonemapped => "yuv444p",
        format => format,
    };
    if config.fps_conversion == FpsConversion::Blend && pixel_format_depth(source) > 8 {
        deep_working_format(format)
    } else {
        format
    }
}

/// Largest component depth (bits) of `pix_fmt`; 8 when unknown.
fn pixel_format_depth(pix_fmt: AVPixelFormat) -> i32 {
    let desc = unsafe { ffi::av_pix_fmt_desc_get(pix_fmt) };
    if desc.is_null() {
        return 8;
    }
    let desc = unsafe { &*desc };
    desc.comp[..desc.nb_components as usize]
        .iter()
        .map(|c| c.depth)
        .max()
        .unwrap_or(8)
}

/// The 16-bit counterpart of an 8-bit working format. Deeper sources carry
/// their range in the frame properties (there are no 16-bit `yuvj`
/// formats), so a full-range `yuvj` format maps to the limited-range name.
fn deep_working_format(format: &'static str) -> &'static str {
    match format {
        "yuv422p" | "yuvj422p" => "yuv422p16le",
        "yuv444p" | "yuvj444p" => "yuv444p16le",
        _ => "yuv420p16le",
    }
}

/// Builds the [`MotionStage`] that resamples the frames leaving `input_sink`
/// (the stream's filter graph, configured by [`VideoFilterConfig::motion_input`])
/// to `config`'s frame rate and converts them to its pixel format.
fn init_motion_stage<'graph>(
    filter_graph: &'graph mut AVFilterGraph,
    input_sink: &AVFilterContextMut,
    config: &VideoFilterConfig,
) -> Result<MotionStage<'graph>, anyhow::Error> {
    let frame_rate = config
        .frame_rate
        .context("frame interpolation needs a frame rate")?;
    // The `fps` filter's rate and time base for the same value, so output
    // timestamps match `Nearest`'s.
    let out_tb = av_inv_q(av_d2q(frame_rate, i32::MAX));
    let (width, height, format) = (
        input_sink.get_w(),
        input_sink.get_h(),
        input_sink.get_format(),
    );
    let synthesis = match config.fps_conversion {
        FpsConversion::Blend => motion::Synthesis::Blend,
        FpsConversion::Motion => motion::Synthesis::Motion,
        FpsConversion::Nearest => bail!("nearest-frame conversion runs in the filter graph"),
    };
    let resampler = motion::Resampler::new(
        motion::Layout::new(width, height, format)?,
        input_sink.get_time_base(),
        out_tb,
        synthesis,
        config.scene_cuts == SceneCuts::Detect,
    )?;

    let sar = input_sink.get_sample_aspect_ratio();
    let mut args = format!(
        "video_size={width}x{height}:pix_fmt={format}:time_base={}/{}:pixel_aspect={}/{}",
        out_tb.num, out_tb.den, sar.num, sar.den,
    );
    let (colorspace, range) = unsafe {
        (
            ffi::av_buffersink_get_colorspace(input_sink.as_ptr()),
            ffi::av_buffersink_get_color_range(input_sink.as_ptr()),
        )
    };
    if let Some(name) = ffi_color_name(unsafe { ffi::av_color_space_name(colorspace) }) {
        args.push_str(&format!(":colorspace={name}"));
    }
    if let Some(name) = ffi_color_name(unsafe { ffi::av_color_range_name(range) }) {
        args.push_str(&format!(":range={name}"));
    }
    // The RGB conversion of the interpolated frames. They are YUV at the
    // output size (the input graph already resized and tone mapped), so only
    // the conversion settings and the frames' own layout carry over: the
    // HDR tags must not, or the output graph would tone map a second time.
    let output_config = VideoFilterConfig {
        pixel_format: config.pixel_format.clone(),
        color_conversion: config.color_conversion,
        source_color: SourceColorInfo {
            colorspace: ffi_color_name(unsafe { ffi::av_color_space_name(colorspace) }),
            ..Default::default()
        },
        source_format: SourceFormat::new(format, width, height),
        ..Default::default()
    };

    cap_graph_threads(filter_graph);
    let (mut buffersrc_ctx, mut buffersink_ctx) =
        create_video_endpoints(filter_graph, &args, &output_config)?;
    link_graph(
        filter_graph,
        &mut buffersrc_ctx,
        &mut buffersink_ctx,
        &output_config.to_filter_spec()?,
    )?;
    Ok(MotionStage {
        resampler,
        output: FilterContext {
            buffersrc_ctx,
            buffersink_ctx,
        },
    })
}

fn init_filter<'graph>(
    filter_graph: &'graph mut AVFilterGraph,
    dec_ctx: &mut AVCodecContext,
    filter_spec: &FilterConfig,
) -> Result<FilterContext<'graph>, anyhow::Error> {
    cap_graph_threads(filter_graph);

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

    link_graph(
        filter_graph,
        &mut buffersrc_ctx,
        &mut buffersink_ctx,
        &filter_spec,
    )?;

    Ok(FilterContext {
        buffersrc_ctx,
        buffersink_ctx,
    })
}

/// Parses `filter_spec` between the graph's source and sink and configures
/// the graph.
fn link_graph(
    filter_graph: &AVFilterGraph,
    buffersrc_ctx: &mut AVFilterContextMut,
    buffersink_ctx: &mut AVFilterContextMut,
    filter_spec: &std::ffi::CStr,
) -> Result<(), anyhow::Error> {
    // Endpoints for the filter graph
    //
    // Yes the outputs' name is `in`, this is in alignment with upstream samples in ffmpeg / rsmpeg which may confuse you when debugging this.
    let outputs = AVFilterInOut::new(c"in", buffersrc_ctx, 0);
    let inputs = AVFilterInOut::new(c"out", buffersink_ctx, 0);

    let (_inputs, _outputs) = filter_graph.parse_ptr(filter_spec, Some(inputs), Some(outputs))?;
    filter_graph.config()?;
    Ok(())
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
    let buffersrc_pix_fmt = buffersrc_pixel_format(dec_ctx);
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
    // Filters that synthesize frames (e.g. `tpad`) size them from the link's
    // frame rate; without it they compute a 1/0 duration.
    if dec_ctx.framerate.num > 0 && dec_ctx.framerate.den > 0 {
        args.push_str(&format!(
            ":frame_rate={}/{}",
            dec_ctx.framerate.num, dec_ctx.framerate.den
        ));
    }

    create_video_endpoints(filter_graph, &args, filter_spec)
}

/// The pixel format decoded frames enter the filter graph in. Hardware
/// decoders report a GPU pixel format (e.g. AV_PIX_FMT_CUDA), but frames are
/// downloaded to system memory before filtering, so this is the underlying
/// software format.
fn buffersrc_pixel_format(dec_ctx: &AVCodecContext) -> AVPixelFormat {
    if dec_ctx.pix_fmt == ffi::AV_PIX_FMT_CUDA {
        if dec_ctx.sw_pix_fmt != ffi::AV_PIX_FMT_NONE {
            dec_ctx.sw_pix_fmt
        } else {
            // NVDEC emits NV12 for 8-bit content.
            ffi::AV_PIX_FMT_NV12
        }
    } else {
        dec_ctx.pix_fmt
    }
}

/// Creates a video graph's `buffer` source (with `args`) and a `buffersink`
/// accepting `filter_spec`'s pixel format.
fn create_video_endpoints<'graph, F: VideoFilterSpec>(
    filter_graph: &'graph AVFilterGraph,
    args: &str,
    filter_spec: &F,
) -> Result<(AVFilterContextMut<'graph>, AVFilterContextMut<'graph>), anyhow::Error> {
    log::debug!("Creating buffer/buffersink filters for video stream.");
    let buffersrc =
        AVFilter::get_by_name(c"buffer").context("filtering source element not found")?;
    let buffersink =
        AVFilter::get_by_name(c"buffersink").context("filtering sink element not found")?;
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
            "yuv422p" => Some(ffi::AV_PIX_FMT_YUV422P),
            "yuv444p" => Some(ffi::AV_PIX_FMT_YUV444P),
            "yuvj420p" => Some(ffi::AV_PIX_FMT_YUVJ420P),
            "yuvj422p" => Some(ffi::AV_PIX_FMT_YUVJ422P),
            "yuvj444p" => Some(ffi::AV_PIX_FMT_YUVJ444P),
            "yuv420p16le" => Some(ffi::AV_PIX_FMT_YUV420P16LE),
            "yuv422p16le" => Some(ffi::AV_PIX_FMT_YUV422P16LE),
            "yuv444p16le" => Some(ffi::AV_PIX_FMT_YUV444P16LE),
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
        test_utils::{
            generate_bt709_ffv1_video, generate_gray_png, generate_hard_cut_video,
            generate_limited_range_luma_ramp, generate_moving_bar_video, generate_pq_ffv1_video,
            generate_test_video_file, generate_xyz12_video_file, init_logger,
            retag_matrix_coefficients, ChannelLayout, TestVideoParameters, JITTER_AND_GAPS,
        },
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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

    /// Horizontal centroid of the white bar in each decoded frame.
    fn bar_centroids(frames: &tch::Tensor) -> Vec<f64> {
        // [T, H, W, C] uint8 -> mean brightness per column, [T, W].
        let columns = frames.to_kind(tch::Kind::Float).mean_dim(
            [1i64, 3].as_slice(),
            false,
            tch::Kind::Float,
        );
        let x = tch::Tensor::arange(columns.size()[1], (tch::Kind::Float, tch::Device::Cpu));
        let centroids = (&columns * &x).sum_dim_intlist(1, false, tch::Kind::Float)
            / columns.sum_dim_intlist(1, false, tch::Kind::Float);
        Vec::<f64>::try_from(centroids.to_kind(tch::Kind::Double)).unwrap()
    }

    #[test_case(FpsConversion::Nearest ; "nearest")]
    #[test_case(FpsConversion::Blend ; "blend")]
    #[test_case(FpsConversion::Motion ; "motion")]
    fn test_fps_conversion_motion_spacing(fps_conversion: FpsConversion) -> anyhow::Result<()> {
        init_logger();
        // 120 px/s at 30 fps is 4 px per source frame; an evenly spaced 24
        // fps output moves 5 px per frame. Nearest keeps 4 of every 5 source
        // frames, so its steps go 4, 8, 4, 4, 4, 8, ...
        let video = generate_moving_bar_video(30, 120, Duration::from_secs(2))?;
        let decoded = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    frame_rate: Some(24.0),
                    fps_conversion,
                    ..Default::default()
                }),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        )?;
        let stream = &decoded[0];
        let frames = stream.data.as_ref().unwrap();
        // Every mode fills the planned buffer exactly.
        assert_eq!(frames.size()[0], 48);
        assert_eq!(stream.frame_pts.len(), 48);
        for (i, pts) in stream.frame_pts.iter().enumerate() {
            assert!((pts - i as f64 / 24.0).abs() < 1e-6, "frame {i} pts {pts}");
        }

        let centroids = bar_centroids(frames);
        let steps: Vec<f64> = centroids.windows(2).map(|w| w[1] - w[0]).collect();
        let max_step = steps.iter().cloned().fold(f64::MIN, f64::max);
        let min_step = steps.iter().cloned().fold(f64::MAX, f64::min);
        log::info!("{fps_conversion:?}: steps {min_step:.2}..{max_step:.2}");
        match fps_conversion {
            FpsConversion::Nearest => {
                assert!(min_step < 4.5 && max_step > 7.5, "{steps:?}");
            }
            FpsConversion::Blend | FpsConversion::Motion => {
                assert!(min_step > 4.5 && max_step < 5.5, "{steps:?}");
            }
        }
        Ok(())
    }

    /// The output frame that falls between two shots (source time 22.92
    /// frames, shots meet between 22 and 23) is the nearer source frame with
    /// scene-cut detection on, which is also what `Nearest` picks, and a mix
    /// of the shots with it off. Frame count and timestamps match `Nearest`.
    #[test_case(FpsConversion::Blend ; "blend")]
    #[test_case(FpsConversion::Motion ; "motion")]
    fn test_fps_conversion_scene_cut(fps_conversion: FpsConversion) -> anyhow::Result<()> {
        init_logger();
        let video = generate_hard_cut_video()?;
        let decode = |fps_conversion, scene_cuts| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate: Some(24.0),
                        fps_conversion,
                        scene_cuts,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0))
        };
        let nearest = decode(FpsConversion::Nearest, SceneCuts::Detect)?;
        let detect = decode(fps_conversion, SceneCuts::Detect)?;
        let off = decode(fps_conversion, SceneCuts::Off)?;
        let frame =
            |s: &DecodedStream, i: i64| s.data.as_ref().unwrap().get(i).to_kind(tch::Kind::Float);
        let mad = |a: &tch::Tensor, b: &tch::Tensor| {
            (a - b).abs().mean(tch::Kind::Float).double_value(&[])
        };
        for s in [&detect, &off] {
            assert_eq!(
                s.data.as_ref().unwrap().size(),
                nearest.data.as_ref().unwrap().size()
            );
            assert_eq!(s.frame_pts, nearest.frame_pts);
        }
        // Output 11 is source 23 (first frame of shot B) for `Nearest`.
        let (shot_a, shot_b) = (frame(&nearest, 10), frame(&nearest, 11));
        assert!(mad(&shot_a, &shot_b) > 30.0, "test clip has no cut");
        let at_cut = |s: &DecodedStream| mad(&frame(s, 11), &shot_b);
        assert!(at_cut(&detect) < 1.0, "detect: {}", at_cut(&detect));
        assert!(at_cut(&off) > 5.0, "off: {}", at_cut(&off));
        Ok(())
    }

    /// Motion and blend interpolation work for every source chroma layout
    /// and HDR tone mapping, with any output size and dtype, and keep
    /// `Nearest`'s frame count and timestamps.
    #[test_case("yuv420p", None, OutputDtype::Uint8 ; "yuv420p")]
    #[test_case("yuv422p10le", None, OutputDtype::Uint8 ; "yuv422p10le")]
    #[test_case("yuv444p", Some((150, 98)), OutputDtype::Uint8 ; "yuv444p resized odd width")]
    #[test_case("yuvj420p", Some((96, 64)), OutputDtype::Float32 ; "yuvj420p float32")]
    #[test_case("yuv420p10le", Some((96, 64)), OutputDtype::Uint8 ; "hdr tone mapped")]
    #[test_case("yuv420p", Some((101, 67)), OutputDtype::Uint8 ; "yuv420p resized odd width and height")]
    #[test_case("yuv422p", Some((77, 33)), OutputDtype::Uint8 ; "yuv422p resized odd width and height")]
    #[test_case("yuv420p", Some((17, 15)), OutputDtype::Uint8 ; "frames smaller than one block")]
    fn test_fps_conversion_motion_formats(
        pixel_format: &str,
        size: Option<(u32, u32)>,
        dtype: OutputDtype,
    ) -> anyhow::Result<()> {
        init_logger();
        let hdr = pixel_format == "yuv420p10le";
        let params = TestVideoParameters {
            width: 160,
            height: 120,
            frame_rate: 25.0,
            duration: Duration::from_secs(2),
            video_codec: if pixel_format == "yuvj420p" {
                "mjpeg".to_string()
            } else {
                "libx264".to_string()
            },
            pixel_format: Some(pixel_format.to_string()),
            color_primaries: hdr.then(|| "bt2020".to_string()),
            color_trc: hdr.then(|| "smpte2084".to_string()),
            colorspace: hdr.then(|| "bt2020nc".to_string()),
            ..Default::default()
        };
        let video = generate_test_video_file(&params)?;
        let decode = |fps_conversion| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate: Some(24.0),
                        width: size.map(|s| s.0),
                        height: size.map(|s| s.1),
                        dtype,
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0))
        };
        let nearest = decode(FpsConversion::Nearest)?;
        let n = nearest.data.unwrap();
        // Blend runs through the same stage as motion.
        for fps_conversion in [FpsConversion::Motion, FpsConversion::Blend] {
            let converted = decode(fps_conversion)?;
            let m = converted.data.unwrap();
            assert_eq!(m.size(), n.size());
            assert_eq!(converted.frame_pts, nearest.frame_pts);
            // The first frame lands on a source frame and passes through;
            // later ones are interpolated but stay close to the nearest
            // source frame (testsrc moves only a little per frame).
            let diff = |i: i64| -> f64 {
                (m.get(i).to_kind(tch::Kind::Float) - n.get(i).to_kind(tch::Kind::Float))
                    .abs()
                    .mean(tch::Kind::Float)
                    .double_value(&[])
            };
            let scale = if dtype == OutputDtype::Float32 {
                1.0 / 255.0
            } else {
                1.0
            };
            // Motion reduces 10-bit sources to 8 bits, so allow a quantization
            // step.
            assert!(
                diff(0) < 1.0 * scale,
                "{fps_conversion:?}: first frame differs by {}",
                diff(0)
            );
            for i in 1..m.size()[0] {
                assert!(
                    diff(i) < 20.0 * scale,
                    "{fps_conversion:?}: frame {i} differs by {}",
                    diff(i)
                );
            }
        }
        Ok(())
    }

    /// `blend` keeps a 10-bit source's depth (its working format is 16-bit).
    /// With float32 output, output frames that land on a source frame equal
    /// that frame decoded without frame-rate conversion, and blended frames
    /// keep far more than 8 bits of levels. Each ramp row holds 877 distinct
    /// 10-bit codes; through an 8-bit working format only ~220 survive.
    #[test_case(false, HdrMode::Tonemap ; "10-bit sdr")]
    #[test_case(true, HdrMode::Raw ; "pq raw")]
    #[test_case(true, HdrMode::Tonemap ; "pq tone mapped")]
    fn test_fps_conversion_blend_keeps_10bit_depth(
        pq: bool,
        hdr_mode: HdrMode,
    ) -> anyhow::Result<()> {
        init_logger();
        let video = crate::util::test_utils::generate_10bit_ramp_video(pq)?;
        let decode = |frame_rate, fps_conversion| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate,
                        dtype: OutputDtype::Float32,
                        hdr_mode,
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0))
        };
        let source = decode(None, FpsConversion::Nearest)?.data.unwrap();
        let nearest = decode(Some(24.0), FpsConversion::Nearest)?;
        let blend = decode(Some(24.0), FpsConversion::Blend)?;
        let b = blend.data.unwrap();
        assert_eq!(b.size(), nearest.data.unwrap().size());
        assert_eq!(blend.frame_pts, nearest.frame_pts);
        // Distinct G values (planar [T, C, H, W]) in a frame's first row, in
        // 16-bit steps.
        let levels = |frame: tch::Tensor| -> usize {
            let row = frame.get(1).get(0).f_mul_scalar(65535.0).unwrap().round();
            let mut v = Vec::<f32>::try_from(row.to_kind(tch::Kind::Float)).unwrap();
            v.sort_by(f32::total_cmp);
            v.dedup();
            v.len()
        };
        let max_diff = |a: tch::Tensor, b: tch::Tensor| (a - b).abs().max().double_value(&[]);
        let source_levels = levels(source.get(0));
        assert!(source_levels > 600, "fixture has {source_levels} levels");
        // 50 -> 24 fps: output frames 0 and 12 land exactly on source frames
        // 0 and 25 and pass through.
        for (out, src) in [(0, 0), (12, 25)] {
            let diff = max_diff(b.get(out), source.get(src));
            assert!(diff < 1e-3, "passthrough frame {out} differs by {diff}");
            assert_eq!(levels(b.get(out)), levels(source.get(src)), "frame {out}");
        }
        // Output frame 6 (source time 12.5 frames) is an even blend of
        // source frames 12 and 13.
        let blended = levels(b.get(6));
        log::info!("blend levels: source {source_levels}, blended {blended}");
        assert!(
            max_diff(b.get(6), source.get(12)) > 1e-3,
            "frame 6 was not blended"
        );
        assert!(
            blended > 2 * 256,
            "blended frame has {blended} levels, no more than 8 bits"
        );
        Ok(())
    }

    /// `blend` keeps a 10-bit source at 16 bits for `uint8` output too, so
    /// the YUV -> RGB conversion rounds once: output frames that land on a
    /// source frame are bit-identical to that frame decoded without
    /// frame-rate conversion. Through an 8-bit working format the source is
    /// rounded to 8-bit YUV first, and about a quarter of the ramp's pixels
    /// come out more than one level off.
    #[test_case(false, HdrMode::Tonemap ; "10-bit sdr")]
    #[test_case(true, HdrMode::Raw ; "pq raw")]
    fn test_fps_conversion_blend_uint8_rounds_10bit_once(
        pq: bool,
        hdr_mode: HdrMode,
    ) -> anyhow::Result<()> {
        init_logger();
        let video = crate::util::test_utils::generate_10bit_ramp_video(pq)?;
        let decode = |frame_rate, fps_conversion| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate,
                        hdr_mode,
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0).data.unwrap())
        };
        let source = decode(None, FpsConversion::Nearest)?;
        let blend = decode(Some(24.0), FpsConversion::Blend)?;
        // 50 -> 24 fps: output frames 0 and 12 pass source frames 0 and 25
        // through.
        for (out, src) in [(0, 0), (12, 25)] {
            assert!(
                blend.get(out).equal(&source.get(src)),
                "passthrough frame {out} differs from source frame {src}"
            );
        }
        Ok(())
    }

    /// With HDR tone mapping, the stage's input graph runs the tone-mapping
    /// chain, which keeps `color_conversion`: float32 output defaults to
    /// `accurate` (bicubic resize and linearization in zscale), so `blend`
    /// passthrough frames match the same request without frame-rate
    /// conversion. With the chain in its `fast` form (swscale resizes in YUV
    /// before linearizing), edges of a textured frame are off by up to the
    /// full range; a smooth ramp would not show it. What remains (~0.045) is
    /// the tone-mapped frames' round trip through YUV, which `framerate` in
    /// the graph had too.
    #[test_case(None ; "native size")]
    #[test_case(Some((160, 90)) ; "resized")]
    fn test_fps_conversion_blend_tone_mapping_keeps_color_conversion(
        size: Option<(u32, u32)>,
    ) -> anyhow::Result<()> {
        init_logger();
        let video = crate::util::test_utils::generate_textured_pq_video()?;
        let decode = |frame_rate, fps_conversion| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate,
                        width: size.map(|(w, _)| w),
                        height: size.map(|(_, h)| h),
                        dtype: OutputDtype::Float32,
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0).data.unwrap())
        };
        let source = decode(None, FpsConversion::Nearest)?;
        let blend = decode(Some(24.0), FpsConversion::Blend)?;
        for (out, src) in [(0, 0), (12, 25)] {
            let diff = (blend.get(out) - source.get(src))
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                diff < 0.1,
                "passthrough frame {out} differs from source frame {src} by {diff}"
            );
        }
        Ok(())
    }

    /// `fps_conversion="motion"` with `color_conversion="accurate"`: the
    /// stage's input graph stays in YUV, and its output graph converts with
    /// zscale using the frames' own layout. Odd-size and untagged sources
    /// would otherwise hit zimg's subsampling or missing-matrix errors.
    #[test_case(Some("bt709"), (162, 120), None ; "tagged even source")]
    #[test_case(None, (162, 120), None ; "untagged source")]
    #[test_case(Some("bt709"), (162, 120), Some((101, 67)) ; "odd output size")]
    #[test_case(Some("bt709"), (161, 121), None ; "odd source size")]
    #[test_case(None, (161, 121), Some((101, 67)) ; "odd untagged source and output")]
    fn test_fps_conversion_motion_with_accurate_color(
        colorspace: Option<&str>,
        source_size: (usize, usize),
        size: Option<(u32, u32)>,
    ) -> anyhow::Result<()> {
        init_logger();
        // FFV1 accepts odd 4:2:0 sizes (libx264 does not).
        let video = crate::util::test_utils::generate_ffv1_video(
            source_size.0,
            source_size.1,
            25,
            "yuv420p",
            colorspace,
        )?;
        let decode = |fps_conversion, dtype| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate: Some(24.0),
                        width: size.map(|s| s.0),
                        height: size.map(|s| s.1),
                        dtype,
                        color_conversion: Some(ColorConversion::Accurate),
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0).data.unwrap())
        };
        for dtype in [OutputDtype::Uint8, OutputDtype::Float32] {
            let nearest = decode(FpsConversion::Nearest, dtype)?;
            let motion = decode(FpsConversion::Motion, dtype)?;
            assert_eq!(motion.size(), nearest.size());
            // The first frame lands on a source frame. Motion converts it via
            // an 8-bit 4:4:4 hop and (with a resize) swscale's YUV scaler in
            // the stage's input graph, where nearest resizes in zscale, so
            // allow a few code values on testsrc's hard edges.
            let scale = if dtype == OutputDtype::Float32 {
                1.0 / 255.0
            } else {
                1.0
            };
            let first = (motion.get(0).to_kind(tch::Kind::Float)
                - nearest.get(0).to_kind(tch::Kind::Float))
            .abs()
            .mean(tch::Kind::Float)
            .double_value(&[]);
            assert!(
                first < 5.0 * scale,
                "{dtype:?}: first frame differs by {first}"
            );
        }
        Ok(())
    }

    #[test_case(48.0, 24.0, true ; "exact 2x")]
    #[test_case(24000.0 / 1001.0, 24.0, true ; "ntsc 23.976")]
    #[test_case(120000.0 / 1001.0, 24.0, true ; "ntsc 119.88")]
    #[test_case(24.0, 24.0, true ; "same rate")]
    #[test_case(30.0, 24.0, false ; "30 to 24")]
    #[test_case(25.0, 24.0, false ; "25 to 24")]
    #[test_case(60000.0 / 1001.0, 24.0, false ; "59.94 to 24")]
    #[test_case(12.0, 24.0, false ; "upsample")]
    fn test_is_integer_downsample(source: f64, target: f64, expected: bool) {
        assert_eq!(is_integer_downsample(source, target), expected);
    }

    #[test]
    fn test_fps_conversion_ntsc_source_matches_nearest() -> anyhow::Result<()> {
        init_logger();
        let video = crate::util::test_utils::generate_fractional_fps_video(48)?;
        let decode = |fps_conversion| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        frame_rate: Some(24.0),
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0).data.unwrap())
        };
        let nearest = decode(FpsConversion::Nearest)?;
        assert!(nearest.equal(&decode(FpsConversion::Motion)?));
        assert!(nearest.equal(&decode(FpsConversion::Blend)?));
        Ok(())
    }

    #[test_case(FpsConversion::Blend ; "blend")]
    #[test_case(FpsConversion::Motion ; "motion")]
    fn test_fps_conversion_seek_window_matches_nearest_timing(
        fps_conversion: FpsConversion,
    ) -> anyhow::Result<()> {
        init_logger();
        let video = generate_moving_bar_video(30, 60, Duration::from_secs(4))?;
        let decode = |fps_conversion| {
            decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(video.path().to_str().unwrap().into()),
                    start_time: Some(1.1),
                    end_time: Some(3.3),
                    video_stream: Some(VideoStreamRequest {
                        frame_rate: Some(24.0),
                        fps_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
            .map(|mut streams| streams.remove(0))
        };
        let nearest = decode(FpsConversion::Nearest)?;
        let converted = decode(fps_conversion)?;
        assert_eq!(converted.data.unwrap().size(), nearest.data.unwrap().size());
        // Same timestamps (allowing for float rounding).
        assert_eq!(converted.frame_pts.len(), nearest.frame_pts.len());
        for (a, b) in converted.frame_pts.iter().zip(&nearest.frame_pts) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn test_fps_conversion_parse() {
        assert_eq!(
            FpsConversion::try_from(None).unwrap(),
            FpsConversion::Nearest
        );
        assert_eq!(
            FpsConversion::try_from(Some("motion")).unwrap(),
            FpsConversion::Motion
        );
        assert!(FpsConversion::try_from(Some("minterpolate")).is_err());
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
            pixel_format: "rgb24".to_string(),
            device: None,
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                colorspace: Some("bt2020nc".to_string()),
                color_range: Some("tv".to_string()),
            },
            hdr_mode: HdrMode::Raw,
            ..Default::default()
        },
        "format=pix_fmts=rgb24";
        "PQ BT.2020 source with raw hdr_mode skips tone mapping and keeps code values"
    )]
    #[test_case(
        VideoFilterConfig {
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                ..Default::default()
            },
            hdr_mode: HdrMode::Raw,
            ..Default::default()
        },
        "setparams=colorspace=bt2020nc:color_primaries=bt2020,format=pix_fmts=rgb24";
        "untagged PQ source with raw hdr_mode is tagged BT.2020"
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
            hdr_mode: HdrMode::Tonemap,
            color_conversion: ColorConversion::Fast,
            source_format: SourceFormat::default(),
            fps_conversion: FpsConversion::Nearest,
            scene_cuts: SceneCuts::Detect,
            number_of_threads: None,
            rotate: vec![],
        },
        "setparams=colorspace=bt2020nc,fps=24,scale=width=1920:height=1080,zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,format=pix_fmts=rgb24";
        "HDR with fps and scale options"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            source_color: SourceColorInfo {
                colorspace: Some("bt709".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        "zscale=f=bicubic,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion of a tagged source quantizes in zscale"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            width: Some(320),
            ..Default::default()
        },
        "zscale=f=bicubic:w=320:min=170m,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion of an untagged source defaults the matrix to BT.601"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            width: Some(853),
            height: Some(480),
            source_color: SourceColorInfo {
                colorspace: Some("bt709".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        "zscale=f=bicubic:w=853:h=480,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion resizes in zscale after the conversion to RGB"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            source_format: SourceFormat {
                width: 321,
                height: 241,
                yuv444: Some("yuv444p10le"),
                log2_chroma: (1, 1),
                gray: false,
            },
            ..Default::default()
        },
        "format=yuv444p10le,zscale=f=bicubic:min=170m,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion upsamples odd-size subsampled frames to 4:4:4"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            source_color: SourceColorInfo {
                colorspace: Some("gbr".to_string()),
                ..Default::default()
            },
            source_format: SourceFormat {
                width: 64,
                height: 64,
                gray: true,
                ..Default::default()
            },
            ..Default::default()
        },
        "zscale=f=bicubic:min=170m,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion of a tagged gray source still sets a matrix"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            width: Some(853),
            height: Some(480),
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        "setparams=colorspace=bt2020nc,zscale=t=linear:npl=100:f=bicubic:w=853:h=480,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,zscale=f=bicubic,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion of a tone-mapped source resizes in the linearizing zscale"
    )]
    #[test_case(
        VideoFilterConfig {
            width: Some(853),
            height: Some(480),
            source_color: SourceColorInfo {
                colorspace: Some("bt2020nc".to_string()),
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                ..Default::default()
            },
            source_format: SourceFormat {
                width: 640,
                height: 360,
                yuv444: Some("yuv444p10le"),
                log2_chroma: (1, 1),
                gray: false,
            },
            ..Default::default()
        },
        "scale=width=853:height=480,format=yuv444p10le,zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,format=pix_fmts=rgb24";
        "fast tone mapping upsamples to 4:4:4 when the output size is odd"
    )]
    #[test_case(
        VideoFilterConfig {
            color_conversion: ColorConversion::Accurate,
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        "setparams=colorspace=bt2020nc,zscale=t=linear:npl=100:f=bicubic,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,zscale=f=bicubic,format=gbrp,format=pix_fmts=rgb24";
        "accurate conversion after tone mapping"
    )]
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            width: Some(640),
            height: Some(360),
            fps_conversion: FpsConversion::Blend,
            ..Default::default()
        },
        "scale=width=640:height=360,format=pix_fmts=rgb24";
        "blend fps conversion runs outside the graph"
    )]
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            width: Some(640),
            height: Some(360),
            fps_conversion: FpsConversion::Motion,
            ..Default::default()
        },
        "scale=width=640:height=360,format=pix_fmts=rgb24";
        "motion fps conversion runs outside the graph"
    )]
    #[test_case(
        VideoFilterConfig {
            fps_conversion: FpsConversion::Motion,
            ..Default::default()
        },
        "format=pix_fmts=rgb24";
        "fps conversion without a frame rate adds no filters"
    )]
    fn test_video_filter_display(filter: VideoFilterConfig, expected: &str) {
        assert_eq!(filter.to_string(), expected);
        assert_eq!(filter.pix_fmt(), Some(ffi::AV_PIX_FMT_RGB24));
        assert_eq!(
            filter.to_filter_spec().unwrap(),
            CString::new(expected).unwrap()
        );
    }

    /// The graph feeding motion interpolation keeps the resize and tone
    /// mapping, drops the frame-rate conversion, and ends in the working
    /// YUV format.
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            width: Some(640),
            height: Some(360),
            fps_conversion: FpsConversion::Motion,
            ..Default::default()
        },
        "yuv420p",
        "scale=width=640:height=360,format=pix_fmts=yuv420p";
        "resize"
    )]
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                ..Default::default()
            },
            fps_conversion: FpsConversion::Motion,
            ..Default::default()
        },
        "yuv422p",
        "setparams=colorspace=bt2020nc,zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,format=pix_fmts=yuv422p";
        "after HDR tone mapping"
    )]
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            width: Some(640),
            height: Some(360),
            color_conversion: ColorConversion::Accurate,
            source_color: SourceColorInfo {
                color_trc: Some("smpte2084".to_string()),
                color_primaries: Some("bt2020".to_string()),
                ..Default::default()
            },
            fps_conversion: FpsConversion::Blend,
            ..Default::default()
        },
        "yuv444p16le",
        "setparams=colorspace=bt2020nc,zscale=t=linear:npl=100:f=bicubic:w=640:h=360,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:range=tv,zscale=f=bicubic,format=pix_fmts=yuv444p16le";
        "accurate tone mapping keeps its zscale resize"
    )]
    #[test_case(
        VideoFilterConfig {
            frame_rate: Some(24.0),
            width: Some(640),
            height: Some(360),
            color_conversion: ColorConversion::Accurate,
            fps_conversion: FpsConversion::Blend,
            ..Default::default()
        },
        "yuv444p16le",
        "scale=width=640:height=360,format=pix_fmts=yuv444p16le";
        "accurate sdr conversion is left to the output graph"
    )]
    fn test_motion_input_filter(filter: VideoFilterConfig, working_format: &str, expected: &str) {
        assert!(filter.uses_motion_stage());
        let input = filter.motion_input(working_format);
        assert!(!input.uses_motion_stage());
        assert_eq!(input.to_string(), expected);
        assert!(input.pix_fmt().is_some());
    }

    #[test_case(ffi::AV_PIX_FMT_YUV420P, "yuv420p")]
    #[test_case(ffi::AV_PIX_FMT_YUV420P10LE, "yuv420p")]
    #[test_case(ffi::AV_PIX_FMT_NV12, "yuv420p")]
    #[test_case(ffi::AV_PIX_FMT_YUVJ420P, "yuvj420p")]
    #[test_case(ffi::AV_PIX_FMT_YUV422P10LE, "yuv422p")]
    #[test_case(ffi::AV_PIX_FMT_YUVJ422P, "yuvj422p")]
    #[test_case(ffi::AV_PIX_FMT_YUV444P, "yuv444p")]
    #[test_case(ffi::AV_PIX_FMT_RGB24, "yuv444p")]
    #[test_case(ffi::AV_PIX_FMT_YUV411P, "yuv420p")]
    fn test_motion_working_pixel_format(source: AVPixelFormat, expected: &str) {
        assert_eq!(motion::working_pixel_format(source), expected);
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

    /// `fps_mode="cli"` returns the frames the ffmpeg CLI writes for the
    /// same input, for variable-frame-rate sources with and without
    /// B-frames, a late-starting video track beside audio, and MOV.
    #[test_case(".mp4", JITTER_AND_GAPS, &[], &["-bf", "0"], None; "vfr")]
    #[test_case(".mp4", JITTER_AND_GAPS, &[], &["-bf", "2"], None; "vfr_bframes")]
    #[test_case(".mov", JITTER_AND_GAPS, &[], &["-bf", "0", "-video_track_timescale", "600"], None; "vfr_mov_600")]
    // A burst of frames in one slot: the CLI drops the last of them, then
    // repeats it (not the last emitted frame) across the gap that follows.
    #[test_case(".mkv", "if(lt(N,20),N,if(lt(N,25),20+(N-20)*0.03,N+6))", &[], &["-bf", "0"], None; "dropped_frame_fills_gap")]
    // Gaps right before the end: the end-of-stream repeats follow them.
    #[test_case(".mp4", "N+6*gt(N,80)+6*gt(N,84)+6*gt(N,87)", &[], &["-bf", "0"], None; "repeats_at_the_end")]
    #[test_case(".mp4", JITTER_AND_GAPS, &[], &["-bf", "0"], Some((".mp4", true)); "late_video_beside_audio")]
    #[test_case(".mp4", JITTER_AND_GAPS, &[], &["-bf", "0"], Some((".mp4", false)); "late_video_alone")]
    // MPEG-TS starts at the video's own start, not the earlier audio's.
    #[test_case(".mp4", JITTER_AND_GAPS, &[], &["-bf", "2"], Some((".ts", true)); "late_video_beside_audio_ts")]
    fn test_fps_mode_cli_matches_ffmpeg_cli(
        suffix: &str,
        setpts: &str,
        input: &[&str],
        extra: &[&str],
        late: Option<(&str, bool)>,
    ) -> anyhow::Result<()> {
        init_logger();
        let (w, h) = (64, 48);
        let mut video =
            crate::util::test_utils::generate_vfr_video_file(suffix, w, h, setpts, input, extra)?;
        if let Some((container, with_audio)) = late {
            // A late video track is padded with its first frame when audio
            // starts earlier, and not padded when it is the only stream.
            video = crate::util::test_utils::make_late_video(
                video.path(),
                container,
                "0.5",
                with_audio,
            )?;
        }
        let path = video.path().to_str().unwrap();
        let expected = crate::util::test_utils::ffmpeg_rgb24_frames(&["-i", path], w, h)?;
        let make_request = || MediaDecodeRequest {
            source: MediaSource::Uri(path.into()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                fps_mode: FpsMode::Cli,
                ..Default::default()
            }),
            audio_streams: None,
            max_output_bytes: None,
        };
        assert_plan_bounds_decode(make_request, false)?;
        let mut streams = decode_media(make_request(), None)?;
        let stream = streams.remove(0);
        let data = stream.data.context("video data")?;
        let got = data.size()[0] as usize;
        assert_eq!(got, expected.len(), "frame count vs the ffmpeg CLI");
        for (i, want) in expected.iter().enumerate() {
            let frame = Vec::<u8>::try_from(data.get(i as i64).flatten(0, -1))?;
            assert!(frame == *want, "frame {i} differs from the ffmpeg CLI's");
        }
        // Timestamps are the output slots: from 0 (the CLI shifts the file
        // to start there), evenly spaced at the stream rate.
        let pts = &stream.frame_pts;
        assert_eq!(pts.len(), got);
        assert_eq!(pts[0], 0.0);
        for pair in pts.windows(2) {
            assert!((pair[1] - pair[0] - 1.0 / 30.0).abs() < 1e-9, "{pair:?}");
        }
        // The gaps and collisions in these clips mean some frames repeat and
        // some are dropped; the plain decode yields a different sequence.
        let mut plain = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path.into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        )?;
        let plain = plain.remove(0).data.context("video data")?;
        assert_ne!(plain.size()[0] as usize, got);
        Ok(())
    }

    /// Matroska stores no last-frame duration, so end-of-stream repeats after
    /// late gaps outrun the plan and the output grows. The frames still match
    /// the CLI, and every frame view points into the final buffer instead of
    /// keeping a superseded one alive.
    #[test]
    fn test_fps_mode_cli_grows_past_the_plan() -> anyhow::Result<()> {
        init_logger();
        let (w, h) = (64, 48);
        let video = crate::util::test_utils::generate_vfr_video_file(
            ".mkv",
            w,
            h,
            "N+6*gt(N,86)+6*gt(N,87)+6*gt(N,88)",
            &[],
            &["-bf", "0"],
        )?;
        let path = video.path().to_str().unwrap();
        let expected = crate::util::test_utils::ffmpeg_rgb24_frames(&["-i", path], w, h)?;
        let make_request = || MediaDecodeRequest {
            source: MediaSource::Uri(path.into()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                fps_mode: FpsMode::Cli,
                ..Default::default()
            }),
            audio_streams: None,
            max_output_bytes: None,
        };
        let planned = plan_media(make_request(), None)?[0].shape[0];
        let stream = decode_media(make_request(), None)?.remove(0);
        let data = stream.data.context("video data")?;
        assert!(data.size()[0] > planned, "the clip must outgrow its plan");
        assert_eq!(data.size()[0] as usize, expected.len());
        for (i, want) in expected.iter().enumerate() {
            let frame = Vec::<u8>::try_from(data.get(i as i64).flatten(0, -1))?;
            assert!(frame == *want, "frame {i} differs from the ffmpeg CLI's");
        }
        let output = data.data_ptr() as usize..data.data_ptr() as usize + data.numel();
        assert!(!stream.decoded_frames.is_empty());
        for frame in &stream.decoded_frames {
            let view = frame.data().data_ptr() as usize;
            assert!(output.contains(&view), "frame view outside the output");
        }
        Ok(())
    }

    /// Decodes `path` with `fps_mode="cli"` and checks every frame against
    /// the ffmpeg CLI's output for it.
    fn assert_cli_frames_match(path: &str, width: usize, height: usize) -> anyhow::Result<()> {
        let expected = crate::util::test_utils::ffmpeg_rgb24_frames(&["-i", path], width, height)?;
        let stream = decode_media(cli_request(path), None)?.remove(0);
        let data = stream.data.context("video data")?;
        assert_eq!(
            data.size()[0] as usize,
            expected.len(),
            "frame count vs the ffmpeg CLI"
        );
        for (i, want) in expected.iter().enumerate() {
            let frame = Vec::<u8>::try_from(data.get(i as i64).flatten(0, -1))?;
            assert!(frame == *want, "frame {i} differs from the ffmpeg CLI's");
        }
        Ok(())
    }

    fn cli_request(path: &str) -> MediaDecodeRequest {
        MediaDecodeRequest {
            source: MediaSource::Uri(path.into()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                fps_mode: FpsMode::Cli,
                ..Default::default()
            }),
            audio_streams: None,
            max_output_bytes: None,
        }
    }

    /// MPEG-TS/PS segments joined with a timestamp jump: the CLI shifts the
    /// timestamps after a jump of more than 10 s (or any backwards one) to
    /// continue the stream, rather than filling or dropping the gap. Without
    /// the correction, a +60 s jump gives 1,860 frames instead of 120.
    #[test_case(".ts", "60", false, false; "ts_jump_forward")]
    #[test_case(".ts", "60", true, false; "ts_jump_backward")]
    #[test_case(".ts", "60", false, true; "ts_jump_forward_with_audio")]
    #[test_case(".mpg", "60", false, false; "ps_jump_forward")]
    // A 0.5 s step back is a discontinuity too; 5 s forward is not.
    #[test_case(".ts", "1.5", false, false; "ts_step_back")]
    #[test_case(".ts", "7", false, false; "ts_jump_below_threshold")]
    fn test_fps_mode_cli_corrects_timestamp_discontinuities(
        suffix: &str,
        offset: &str,
        reverse: bool,
        with_audio: bool,
    ) -> anyhow::Result<()> {
        init_logger();
        let (w, h) = (64, 48);
        let audio: &[&str] = if with_audio {
            &["-f", "lavfi", "-i", "sine=sample_rate=48000", "-c:a", "aac"]
        } else {
            &[]
        };
        let first = crate::util::test_utils::generate_segment(suffix, w, h, "0", audio)?;
        let second = crate::util::test_utils::generate_segment(suffix, w, h, offset, audio)?;
        let mut parts = [first.path(), second.path()];
        if reverse {
            parts.reverse();
        }
        let joined = crate::util::test_utils::concat_files(suffix, &parts)?;
        assert_cli_frames_match(joined.path().to_str().unwrap(), w, h)
    }

    /// Outside MPEG-TS/PS, the CLI drops a timestamp more than 30 h off the
    /// predicted one, so a bogus jump does not turn into duplicates: the
    /// frames after it take extrapolated timestamps instead.
    #[test]
    fn test_fps_mode_cli_drops_timestamps_30h_off() -> anyhow::Result<()> {
        init_logger();
        let (w, h) = (64, 48);
        // No stored duration, so the plan doesn't already refuse the jump.
        let video =
            crate::util::test_utils::generate_streamed_vfr_mkv(w, h, "N+31*3600*30*gt(N,4)")?;
        assert_cli_frames_match(video.path().to_str().unwrap(), w, h)
    }

    /// A tiny file without a stored duration plans one frame; a 60 s
    /// timestamp gap then grows the output to ~1,800 frames. Growth stops at
    /// the output limit with the usual size error instead of allocating
    /// past it, and a grown output's storage is trimmed to its rows.
    #[test]
    fn test_fps_mode_cli_growth_stops_at_the_output_limit() -> anyhow::Result<()> {
        init_logger();
        let (w, h) = (64, 48);
        let video = crate::util::test_utils::generate_streamed_vfr_mkv(w, h, "N+1800*gt(N,1)")?;
        let path = video.path().to_str().unwrap();
        let plan = plan_media(cli_request(path), None)?.remove(0);
        assert_eq!(plan.shape[0], 1);
        assert_eq!(plan.duration_source, DurationSource::None);
        // The plan reports the stream's rate, which the output has, not the
        // image fallback of 1 fps.
        let probed = probe_media(MediaSource::Uri(path.into()), None)?;
        assert_eq!(
            plan.metadata,
            StreamMetadata::Video {
                frame_rate: probed.video_streams[0].fps
            }
        );
        assert!(probed.video_streams[0].fps > 25.0);

        let row_bytes = (w * h * 3) as u64;
        let limited = MediaDecodeRequest {
            max_output_bytes: Some(100 * row_bytes),
            ..cli_request(path)
        };
        let err = decode_media(limited, None)
            .err()
            .context("decoding past the limit must fail")?;
        assert!(
            format!("{err:#}").contains(&format!(
                "exceeding the request's max_output_bytes of {} bytes",
                100 * row_bytes
            )),
            "{err:#}"
        );

        let stream = decode_media(cli_request(path), None)?.remove(0);
        let data = stream.data.context("video data")?;
        assert_eq!(stream.metadata, plan.metadata);
        let rows = data.size()[0];
        assert!(rows > 1800, "{rows}");
        // The storage is trimmed to exactly the returned rows: one more is
        // out of bounds.
        let mut shape = data.size();
        shape[0] += 1;
        assert!(
            data.f_as_strided(&shape, data.stride(), 0).is_err(),
            "the grown output keeps unused rows"
        );
        drop(data);
        assert_cli_frames_match(path, w, h)
    }

    /// The output may fill the limit exactly: only the rows emitted count
    /// against it, not the headroom growth asks for, nor the row a dropped
    /// frame is converted into. Without a stored duration the plan is 1
    /// frame, so every row is grown.
    // Gaps before each of the last three frames: the CLI repeats the last.
    #[test_case("N+6*gt(N,86)+6*gt(N,87)+6*gt(N,88)"; "end_repeats")]
    // A burst in the last slots: the CLI drops the last frames.
    #[test_case("if(lt(N,85),N,85+(N-85)*0.03)"; "dropped_last_frames")]
    fn test_fps_mode_cli_output_can_fill_the_limit(setpts: &str) -> anyhow::Result<()> {
        init_logger();
        let (w, h) = (64, 48);
        let video = crate::util::test_utils::generate_streamed_vfr_mkv(w, h, setpts)?;
        let path = video.path().to_str().unwrap();
        let expected = crate::util::test_utils::ffmpeg_rgb24_frames(&["-i", path], w, h)?;
        let row_bytes = (w * h * 3) as u64;
        let limit = expected.len() as u64 * row_bytes;
        let limited = |bytes| MediaDecodeRequest {
            max_output_bytes: Some(bytes),
            ..cli_request(path)
        };
        let stream = decode_media(limited(limit), None)?.remove(0);
        let data = stream.data.context("video data")?;
        assert_eq!(data.size()[0] as usize, expected.len());
        for (i, want) in expected.iter().enumerate() {
            let frame = Vec::<u8>::try_from(data.get(i as i64).flatten(0, -1))?;
            assert!(frame == *want, "frame {i} differs from the ffmpeg CLI's");
        }
        // One row less does not fit.
        let err = decode_media(limited(limit - row_bytes), None)
            .err()
            .context("decoding past the limit must fail")?;
        assert!(
            format!("{err:#}").contains("exceeding the request's max_output_bytes"),
            "{err:#}"
        );
        Ok(())
    }

    /// Growth clamps its headroom to the limit, and fails without growing
    /// once the rows needed exceed it.
    #[test]
    fn test_ensure_cli_capacity_respects_the_limit() -> anyhow::Result<()> {
        let mut data = tch::Tensor::f_zeros([8, 2, 2, 3], (tch::Kind::Uint8, tch::Device::Cpu))?;
        let row_bytes = 2 * 2 * 3;
        let limit = |bytes: u64| OutputLimit::resolve(Some(bytes));
        // Headroom would take 11 rows; the limit allows 9.
        assert!(ensure_cli_capacity(&mut data, 9, 9, &limit(9 * row_bytes))?);
        assert_eq!(data.size()[0], 9);
        assert!(!ensure_cli_capacity(
            &mut data,
            9,
            9,
            &limit(9 * row_bytes)
        )?);
        // Rows wanted past the limit are only headroom: no error.
        assert!(!ensure_cli_capacity(
            &mut data,
            9,
            10,
            &limit(9 * row_bytes)
        )?);
        let mut exact = tch::Tensor::f_zeros([8, 2, 2, 3], (tch::Kind::Uint8, tch::Device::Cpu))?;
        assert!(ensure_cli_capacity(
            &mut exact,
            9,
            10,
            &limit(9 * row_bytes)
        )?);
        assert_eq!(exact.size()[0], 9);
        let err = ensure_cli_capacity(&mut data, 10, 10, &limit(9 * row_bytes))
            .err()
            .context("growing past the limit must fail")?;
        assert!(
            format!("{err:#}").contains(
                "would require 120 bytes, exceeding the request's max_output_bytes of 108"
            ),
            "{err:#}"
        );
        assert_eq!(data.size()[0], 9);
        // Below the limit, growth adds a quarter, or what is wanted.
        assert!(ensure_cli_capacity(&mut data, 10, 10, &limit(1 << 20))?);
        assert_eq!(data.size()[0], 12);
        assert!(ensure_cli_capacity(&mut data, 13, 20, &limit(1 << 20))?);
        assert_eq!(data.size()[0], 20);
        Ok(())
    }

    #[test]
    fn test_fps_mode_cli_rejects_fps_and_windows() -> anyhow::Result<()> {
        init_logger();
        let video = crate::util::test_utils::generate_vfr_video_file(
            ".mp4",
            64,
            48,
            JITTER_AND_GAPS,
            &[],
            &[],
        )?;
        let path: String = video.path().to_str().unwrap().into();
        let request = |frame_rate, end_time| MediaDecodeRequest {
            source: MediaSource::Uri(path.clone()),
            start_time: None,
            end_time,
            video_stream: Some(VideoStreamRequest {
                fps_mode: FpsMode::Cli,
                frame_rate,
                ..Default::default()
            }),
            audio_streams: None,
            max_output_bytes: None,
        };
        let err = decode_media(request(Some(24.0), None), None)
            .err()
            .context("fps")?;
        assert!(
            format!("{err:#}").contains("cannot be combined with fps"),
            "{err:#}"
        );
        let err = decode_media(request(None, Some(1.0)), None)
            .err()
            .context("window")?;
        assert!(
            format!("{err:#}").contains("start_time and end_time"),
            "{err:#}"
        );
        // plan_asset refuses what decode_asset refuses.
        let err = plan_media(request(Some(24.0), None), None)
            .err()
            .context("plan fps")?;
        assert!(
            format!("{err:#}").contains("cannot be combined with fps"),
            "{err:#}"
        );
        let err = plan_media(request(None, Some(1.0)), None)
            .err()
            .context("plan window")?;
        assert!(
            format!("{err:#}").contains("start_time and end_time"),
            "{err:#}"
        );
        // NVDEC frames have no duration, so the CLI's output can't be
        // matched; both entry points refuse before touching a GPU.
        let hardware = || MediaDecodeRequest {
            video_stream: Some(VideoStreamRequest {
                fps_mode: FpsMode::Cli,
                hardware_acceleration: Some(true),
                ..Default::default()
            }),
            ..request(None, None)
        };
        for err in [
            plan_media(hardware(), None)
                .err()
                .context("plan hardware")?,
            decode_media(hardware(), None)
                .err()
                .context("decode hardware")?,
        ] {
            assert!(
                format!("{err:#}").contains("not supported with hardware_acceleration=True"),
                "{err:#}"
            );
        }
        // device is refused before a CUDA device is created, so an ordinal
        // no GPU has gives the same error as one that exists.
        let on_device = MediaDecodeRequest {
            video_stream: Some(VideoStreamRequest {
                fps_mode: FpsMode::Cli,
                device: Some(999),
                ..Default::default()
            }),
            ..request(None, None)
        };
        let err = decode_media(on_device, None).err().context("device")?;
        assert!(
            format!("{err:#}").contains("not supported with device"),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn test_parse_byte_size() {
        assert_eq!(parse_byte_size("0"), Some(0));
        assert_eq!(parse_byte_size(" 1024 "), Some(1024));
        assert_eq!(parse_byte_size("1024B"), Some(1024));
        assert_eq!(parse_byte_size("512K"), Some(512 << 10));
        assert_eq!(parse_byte_size("64m"), Some(64 << 20));
        assert_eq!(parse_byte_size("32G"), Some(32 << 30));
        assert_eq!(parse_byte_size("32 GiB"), Some(32 << 30));
        assert_eq!(parse_byte_size("2tib"), Some(2 << 40));
        assert_eq!(parse_byte_size("+1048576"), Some(1 << 20));
        assert_eq!(parse_byte_size("+32G"), Some(32 << 30));
        let big = 64u128 << 30;
        assert_eq!(parse_byte_size(&big.to_string()), Some(big));
        for invalid in [
            "",
            "G",
            "+",
            "++1",
            "-1",
            "1.5G",
            "32GB",
            "32 G B",
            "1e9",
            "16 GiB of RAM",
            "0x10",
        ] {
            assert_eq!(parse_byte_size(invalid), None, "{invalid:?}");
        }
        // Overflow is invalid rather than wrapping.
        assert_eq!(parse_byte_size(&format!("{}T", u128::MAX >> 39)), None);
    }

    #[test]
    fn test_output_limit_resolution_and_errors() {
        let default = DEFAULT_MAX_OUTPUT_TENSOR_BYTES;
        assert_eq!(OutputLimit::from_env(None).bytes, default);
        assert_eq!(OutputLimit::from_env(Some(" ".into())).bytes, default);
        assert_eq!(OutputLimit::from_env(Some("64G".into())).bytes, 64 << 30);
        // A request's own limit wins over the environment's.
        assert_eq!(OutputLimit::resolve(Some(1024)).bytes, 1024);

        // 900 frames of 4K rgb24, about 22 GB: over the default, within 64 GiB.
        let shape = [900, 2160, 3840, 3];
        let error = |limit: &OutputLimit| {
            check_output_tensor_size(&shape, 1, limit)
                .unwrap_err()
                .to_string()
        };
        let err = error(&OutputLimit::from_env(None));
        assert!(err.contains("default limit of 17179869184 bytes"), "{err}");
        assert!(err.contains("max_output_bytes"), "{err}");
        assert!(err.contains("AVTENSOR_MAX_OUTPUT_TENSOR_BYTES"), "{err}");
        check_output_tensor_size(&shape, 1, &OutputLimit::from_env(Some("64G".into()))).unwrap();
        assert!(check_output_tensor_size(&[-1, 2], 1, &OutputLimit::resolve(Some(1))).is_err());

        // A value that is set but too small is named, so the error does not
        // just say to set it.
        let err = error(&OutputLimit::from_env(Some("8G".into())));
        assert!(
            err.contains("set by AVTENSOR_MAX_OUTPUT_TENSOR_BYTES=\"8G\""),
            "{err}"
        );

        // An invalid value falls back to the default, and the error names it.
        for invalid in ["32GB", "lots", "-1"] {
            let limit = OutputLimit::from_env(Some(invalid.into()));
            assert_eq!(limit.bytes, default, "{invalid:?}");
            let err = error(&limit);
            assert!(
                err.contains(&format!(
                    "AVTENSOR_MAX_OUTPUT_TENSOR_BYTES={invalid:?} was ignored"
                )),
                "{err}"
            );
        }

        let err = error(&OutputLimit::resolve(Some(1024)));
        assert!(
            err.contains("the request's max_output_bytes of 1024 bytes"),
            "{err}"
        );
    }

    /// `max_output_bytes` on the request bounds both planning and decoding.
    #[test]
    fn test_request_max_output_bytes() -> anyhow::Result<()> {
        init_logger();
        let video = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        })?;
        let request = |max_output_bytes| MediaDecodeRequest {
            source: MediaSource::Uri(video.path().to_str().unwrap().into()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest::default()),
            audio_streams: None,
            max_output_bytes,
        };
        // 30 frames of 640x480 rgb24 is ~27.6 MB.
        let err = decode_media(request(Some(1 << 20)), None)
            .err()
            .context("1 MiB limit")?;
        assert!(format!("{err:#}").contains("max_output_bytes"), "{err:#}");
        assert!(plan_media(request(Some(1 << 20)), None).is_err());
        decode_media(request(Some(64 << 20)), None)?;
        Ok(())
    }

    /// The limit applies to audio streams too, in planning and decoding.
    #[test]
    fn test_request_max_output_bytes_audio() -> anyhow::Result<()> {
        init_logger();
        let video = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        })?;
        let request = |max_output_bytes| MediaDecodeRequest {
            source: uri(&video),
            start_time: None,
            end_time: None,
            video_stream: None,
            audio_streams: Some(vec![AudioStreamRequest::default()]),
            max_output_bytes: Some(max_output_bytes),
        };
        // ~1 s of 44.1 kHz mono float32 is ~176 KB.
        let plan_err = plan_media(request(1000), None).unwrap_err();
        let decode_err = decode_media(request(1000), None).err().unwrap();
        for err in [plan_err, decode_err] {
            assert!(
                format!("{err:#}").contains("the request's max_output_bytes of 1000 bytes"),
                "{err:#}"
            );
        }
        let plans = plan_media(request(1 << 20), None)?;
        assert_eq!(plans[0].stream_type(), StreamType::Audio);
        decode_media(request(1 << 20), None)?;
        Ok(())
    }

    /// Every stream's planned size is checked before any buffer is allocated:
    /// an audio stream over the limit fails the request without allocating
    /// the (within-limit) video buffer first.
    #[test]
    fn test_output_limit_checked_before_any_allocation() -> anyhow::Result<()> {
        init_logger();
        let video = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        })?;
        let request = |max_output_bytes| MediaDecodeRequest {
            source: uri(&video),
            start_time: None,
            end_time: None,
            // 30 frames of 16x16 rgb24 is ~23 KB, under the limit below.
            video_stream: Some(VideoStreamRequest {
                width: Some(16),
                height: Some(16),
                ..Default::default()
            }),
            // ~1 s of 44.1 kHz mono float32 is ~176 KB, over it.
            audio_streams: Some(vec![AudioStreamRequest::default()]),
            max_output_bytes: Some(max_output_bytes),
        };
        let allocations = crate::alloc::allocations_on_this_thread;

        let before = allocations();
        let err = decode_media(request(100_000), None).err().unwrap();
        assert!(format!("{err:#}").contains("max_output_bytes"), "{err:#}");
        assert!(format!("{err:#}").contains("shape [1, "), "{err:#}");
        assert_eq!(allocations(), before, "no output buffer may be allocated");

        // With room for both, both buffers are allocated.
        let before = allocations();
        let streams = decode_media(request(1 << 20), None)?;
        assert_eq!(streams.len(), 2);
        assert_eq!(allocations(), before + 2);
        Ok(())
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
                max_output_bytes: None,
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

    /// Decodes the sole audio stream of `path`, asserting it holds ~`seconds`
    /// of audio at `expected_sample_rate`.
    fn assert_decodes_audio_only(
        path: &std::path::Path,
        request: AudioStreamRequest,
        expected_sample_rate: u32,
        seconds: f64,
    ) -> anyhow::Result<()> {
        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path.to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: None,
                audio_streams: Some(vec![request]),
                max_output_bytes: None,
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 1, "Expected a single audio stream");
        let stream = &decoded_streams[0];
        let StreamMetadata::Audio { sample_rate } = stream.metadata else {
            panic!("Expected an audio stream");
        };
        assert_eq!(sample_rate, expected_sample_rate);
        let num_samples = *stream.data.as_ref().unwrap().size().last().unwrap();
        let expected = (seconds * expected_sample_rate as f64) as i64;
        assert!(
            (num_samples - expected).abs() <= expected_sample_rate as i64 / 10,
            "Expected ~{expected} samples, got {num_samples}"
        );
        Ok(())
    }

    #[test]
    fn test_decode_ogg_opus_audio() -> anyhow::Result<()> {
        init_logger();

        let opus = crate::util::test_utils::generate_test_audio_file(
            48000,
            Duration::from_secs(2),
            "libopus",
            ".opus",
        )?;
        // Opus always decodes at 48 kHz.
        assert_decodes_audio_only(opus.path(), AudioStreamRequest::default(), 48000, 2.0)?;
        assert_decodes_audio_only(
            opus.path(),
            AudioStreamRequest {
                sample_rate: Some(16000),
                ..Default::default()
            },
            16000,
            2.0,
        )
    }

    #[test]
    fn test_decode_webm_opus_audio() -> anyhow::Result<()> {
        init_logger();

        // Matroska/WebM streams carry no per-stream duration, so the output
        // buffer must be sized from the container duration.
        let webm = crate::util::test_utils::generate_test_audio_file(
            48000,
            Duration::from_secs(2),
            "libopus",
            ".webm",
        )?;
        assert_decodes_audio_only(webm.path(), AudioStreamRequest::default(), 48000, 2.0)
    }

    #[test]
    fn test_decode_webm_video_and_audio() -> anyhow::Result<()> {
        init_logger();

        let webm = crate::util::test_utils::generate_test_webm_file(Duration::from_secs(2))?;

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(webm.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
                max_output_bytes: None,
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 2, "Expected video and audio streams");
        for stream in decoded_streams {
            let data = stream.data.as_ref().unwrap();
            match stream.metadata {
                StreamMetadata::Video { frame_rate } => {
                    assert_eq!(frame_rate, 30.0);
                    // Previously a missing stream duration was mistaken for
                    // an image and only a single frame was decoded.
                    assert_eq!(data.size(), vec![60, 240, 320, 3]);
                    assert_eq!(stream.decoded_frames.len(), 60);
                }
                StreamMetadata::Audio { sample_rate } => {
                    assert_eq!(sample_rate, 48000);
                    assert_eq!(data.size()[0], 2);
                    let num_samples = data.size()[1];
                    assert!(
                        (num_samples - 96000).abs() <= 4800,
                        "Expected ~96000 samples, got {num_samples}"
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_decode_webm_seek() -> anyhow::Result<()> {
        init_logger();

        let webm = crate::util::test_utils::generate_test_webm_file(Duration::from_secs(4))?;

        let decoded_streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(webm.path().to_str().unwrap().into()),
                start_time: Some(1.0),
                end_time: Some(3.0),
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: Some(vec![AudioStreamRequest::default()]),
                max_output_bytes: None,
            },
            None,
        )?;

        assert_eq!(decoded_streams.len(), 2, "Expected video and audio streams");
        for stream in decoded_streams {
            let first_pts = stream.decoded_frames.first().unwrap().pts_seconds();
            assert!(
                first_pts >= 1.0 - 0.05,
                "First PTS {first_pts} before start"
            );
            let data = stream.data.as_ref().unwrap();
            match stream.metadata {
                StreamMetadata::Video { .. } => assert_eq!(data.size()[0], 60),
                StreamMetadata::Audio { .. } => {
                    let num_samples = data.size()[1];
                    assert!(
                        (num_samples - 96000).abs() <= 4800,
                        "Expected ~96000 samples, got {num_samples}"
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_webm_preallocation_is_bounded_by_seek_window() -> anyhow::Result<()> {
        init_logger();

        // WebM streams fall back to the container duration; the preallocated
        // buffers must still be clamped to the requested window rather than
        // sized for the whole file. Checked on the stream contexts directly
        // because the decoded output is trimmed afterwards.
        let webm = crate::util::test_utils::generate_test_webm_file(Duration::from_secs(10))?;
        let input_format_context =
            MediaSource::Uri(webm.path().to_str().unwrap().into()).open(None)?;
        let video_request =
            select_video_stream(&input_format_context, Some(VideoStreamRequest::default()))?
                .unwrap();
        let audio_request = select_audio_streams(
            &input_format_context,
            Some(vec![AudioStreamRequest::default()]),
        )?
        .unwrap()
        .remove(0);

        let limit = OutputLimit::from_env(None);
        let video_context = |seek: &Seek| -> anyhow::Result<StreamContext> {
            let plan = plan_video_stream(&input_format_context, &video_request, seek, &limit)?;
            init_video_stream_context(&input_format_context, &video_request, plan, &limit)
        };
        let audio_context = |seek: &Seek| -> anyhow::Result<StreamContext> {
            let plan = plan_audio_stream(&input_format_context, &audio_request, seek, &limit)?;
            init_audio_stream_context(&input_format_context, &audio_request, plan)
        };

        let window = Seek {
            start_time: Some(2.0),
            end_time: Some(3.0),
        };
        let video = video_context(&window)?;
        assert_eq!(video.frame_data.size(), vec![30, 240, 320, 3]);
        let audio = audio_context(&window)?;
        assert_eq!(audio.frame_data.size(), vec![2, 48000]);

        // Without a window the buffers cover the whole file.
        let full = Seek {
            start_time: None,
            end_time: None,
        };
        let video = video_context(&full)?;
        assert!(video.frame_data.size()[0] >= 300);
        let audio = audio_context(&full)?;
        assert!(audio.frame_data.size()[1] >= 480000);

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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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

    /// GPU-resident output cannot run the rotation filters, so autorotate
    /// on a rotated stream is refused rather than returned unrotated.
    #[test]
    #[ignore = "requires an NVIDIA GPU with NVDEC"]
    fn test_autorotate_with_cuda_output_is_rejected() -> anyhow::Result<()> {
        init_logger();
        let source = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let rotated = crate::util::test_utils::make_rotated_video(source.path(), 90)?;
        let err = decode_video_tensor(
            MediaSource::Uri(rotated.path().to_str().unwrap().into()),
            VideoStreamRequest {
                device: Some(0),
                autorotate: true,
                ..Default::default()
            },
        )
        .err()
        .context("expected an error")?;
        assert!(
            format!("{err:#}").contains("does not support autorotate"),
            "{err:#}"
        );
        Ok(())
    }

    /// `plan_media` refuses GPU-resident output with autorotate on a rotated
    /// stream, as decode does, and accepts it on an unrotated one (nothing to
    /// rotate). Planning opens no decoder, so this needs no GPU.
    #[test]
    fn test_plan_autorotate_with_cuda_output_is_rejected() -> anyhow::Result<()> {
        init_logger();
        let source = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let rotated = crate::util::test_utils::make_rotated_video(source.path(), 90)?;
        let plan = |file: &tempfile::NamedTempFile| {
            plan_media(
                MediaDecodeRequest {
                    source: uri(file),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        device: Some(0),
                        autorotate: true,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )
        };
        let err = plan(&rotated).err().context("expected an error")?;
        assert!(
            format!("{err:#}").contains("does not support autorotate"),
            "{err:#}"
        );
        assert_eq!(plan(&source)?[0].shape, vec![30, 480, 640, 3]);
        Ok(())
    }

    /// With NVDEC decoding and autorotate, a resize is done after the
    /// rotation (NVDEC's pre-resize would scale the unrotated axes), so the
    /// output matches the software decoder's.
    #[test]
    #[ignore = "requires an NVIDIA GPU with NVDEC"]
    fn test_autorotate_resize_with_nvdec_matches_software() -> anyhow::Result<()> {
        init_logger();
        let source = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let rotated = crate::util::test_utils::make_rotated_video(source.path(), 90)?;
        let path: String = rotated.path().to_str().unwrap().into();
        let decode = |hardware_acceleration| {
            decode_video_tensor(
                MediaSource::Uri(path.clone()),
                VideoStreamRequest {
                    hardware_acceleration: Some(hardware_acceleration),
                    autorotate: true,
                    width: Some(240),
                    height: Some(320),
                    ..Default::default()
                },
            )
        };
        let (nvdec, _) = decode(true)?;
        let (software, _) = decode(false)?;
        assert_eq!(nvdec.size(), software.size());
        assert_eq!(nvdec.size()[1..3], [320, 240]);
        // NVDEC and libavcodec decode H.264 bit-exactly; the same filter
        // graph then rotates and scales both.
        let max_diff: f64 = nvdec
            .to_kind(tch::Kind::Int16)
            .f_sub(&software.to_kind(tch::Kind::Int16))?
            .abs()
            .max()
            .to_kind(tch::Kind::Float)
            .try_into()?;
        assert!(max_diff <= 1.0, "max diff {max_diff}");
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
                max_output_bytes: None,
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
            max_output_bytes: None,
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
            max_output_bytes: None,
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
            max_output_bytes: None,
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

    /// Decodes only the video stream of `source` with `request` and returns
    /// its output tensor and per-frame pts.
    fn decode_video_tensor(
        source: MediaSource,
        request: VideoStreamRequest,
    ) -> anyhow::Result<(tch::Tensor, Vec<f64>)> {
        let mut streams = decode_media(
            MediaDecodeRequest {
                source,
                start_time: None,
                end_time: None,
                video_stream: Some(request),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        )?;
        let stream = streams.remove(0);
        Ok((
            stream.data.context("video stream should have data")?,
            stream.frame_pts,
        ))
    }

    /// Guards `AVTENSOR_DISABLE_DIRECT_PATH`, which is process-wide. Tests
    /// that set it hold the write lock; tests whose decode must take the
    /// direct path (because only it works, or because they check that it
    /// ran) hold the read lock. Other tests decoding in parallel may briefly
    /// take the graph path, which is benign: both paths are correct.
    static DIRECT_PATH_ENV: std::sync::RwLock<()> = std::sync::RwLock::new(());

    /// Runs `f` with the direct paths disabled, forcing the filter graph.
    /// The variable is cleared even if `f` panics.
    fn with_direct_path_disabled<T>(f: impl FnOnce() -> T) -> T {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                std::env::remove_var("AVTENSOR_DISABLE_DIRECT_PATH");
            }
        }
        // Declared before `_restore`, so it is released after the variable
        // is cleared.
        let _guard = DIRECT_PATH_ENV.write().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AVTENSOR_DISABLE_DIRECT_PATH", "1");
        let _restore = Restore;
        f()
    }

    #[test]
    fn test_with_direct_path_disabled_clears_on_panic() {
        let result = std::panic::catch_unwind(|| {
            with_direct_path_disabled(|| {
                assert!(!direct_paths_enabled());
                panic!("decode failed");
            })
        });
        assert!(result.is_err());
        let _guard = DIRECT_PATH_ENV.read().unwrap_or_else(|e| e.into_inner());
        assert!(direct_paths_enabled());
    }

    /// The direct swscale path must produce output identical to the filter
    /// graph: for untagged and BT.709-tagged sources, for 10-bit sources, and
    /// for HDR sources decoded with `HdrMode::Raw`, which take the direct
    /// path too (including the less common FCC matrix, and an HDR source with
    /// no matrix tag, which both paths decode as BT.2020). Each source is
    /// converted on one swscale thread (the default) and slice-threaded
    /// (`number_of_threads=0`), and the test checks that the direct path ran.
    #[test_case(None, None, None, HdrMode::Tonemap; "untagged")]
    #[test_case(Some("bt709"), Some("tv"), None, HdrMode::Tonemap; "bt709_tv")]
    #[test_case(Some("bt709"), Some("tv"), Some("yuv420p10le"), HdrMode::Tonemap; "bt709_10bit")]
    #[test_case(Some("bt2020nc"), Some("tv"), Some("yuv420p10le"), HdrMode::Raw; "hlg_raw")]
    #[test_case(Some("fcc"), Some("tv"), Some("yuv420p10le"), HdrMode::Raw; "fcc_raw")]
    #[test_case(None, Some("tv"), Some("yuv420p10le"), HdrMode::Raw; "untagged_matrix_hlg_raw")]
    fn test_direct_scale_matches_filter_graph(
        colorspace: Option<&str>,
        color_range: Option<&str>,
        pixel_format: Option<&str>,
        hdr_mode: HdrMode,
    ) -> anyhow::Result<()> {
        init_logger();

        let hdr = hdr_mode == HdrMode::Raw;
        let test_video = generate_test_video_file(&TestVideoParameters {
            colorspace: colorspace.map(String::from),
            color_range: color_range.map(String::from),
            pixel_format: pixel_format.map(String::from),
            color_primaries: hdr.then(|| "bt2020".to_string()),
            color_trc: hdr.then(|| "arib-std-b67".to_string()),
            ..Default::default()
        })?;
        let path: String = test_video.path().to_str().unwrap().into();
        let request = |number_of_threads| VideoStreamRequest {
            hdr_mode,
            number_of_threads,
            ..Default::default()
        };

        direct_scale_stats::take();
        let (graph, graph_pts) = with_direct_path_disabled(|| {
            decode_video_tensor(MediaSource::Uri(path.clone()), request(None))
        })?;
        assert_eq!(direct_scale_stats::take().frames, 0);

        for (number_of_threads, threads) in [(None, 1), (Some(0), filter_graph_threads())] {
            // Default parameters (640 px wide) are eligible for the direct
            // path; hold the lock so no other test disables it meanwhile.
            let (direct, direct_pts) = {
                let _guard = DIRECT_PATH_ENV.read().unwrap_or_else(|e| e.into_inner());
                decode_video_tensor(MediaSource::Uri(path.clone()), request(number_of_threads))?
            };
            let stats = direct_scale_stats::take();
            assert!(stats.frames > 0, "the decode did not take the direct path");
            assert_eq!(stats.threads, threads as i64);

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
                "direct swscale output must be identical to the filter graph \
                 (number_of_threads={number_of_threads:?})"
            );
        }
        Ok(())
    }

    /// The direct path's swscale pool follows `number_of_threads`: one
    /// thread by default (like the single-threaded conversion before the
    /// pool), the filter graph's cap for 0 (auto), N for an explicit N
    /// (capped the same way), and always one for XYZ sources.
    #[test]
    fn test_direct_scaler_threads_follow_request() -> anyhow::Result<()> {
        init_logger();

        let cap = filter_graph_threads();
        assert_eq!(direct_scaler_threads(None), 1);
        assert_eq!(direct_scaler_threads(Some(1)), 1);
        assert_eq!(direct_scaler_threads(Some(0)), cap);
        assert_eq!(direct_scaler_threads(Some(2)), 2.min(cap));
        assert_eq!(direct_scaler_threads(Some(10_000)), cap);

        let yuv = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        })?;
        let xyz = generate_xyz12_video_file()?;
        let _guard = DIRECT_PATH_ENV.read().unwrap_or_else(|e| e.into_inner());
        for (video, number_of_threads, threads) in [
            (&yuv, None, 1),
            (&yuv, Some(1), 1),
            (&yuv, Some(0), cap),
            (&yuv, Some(2), 2.min(cap)),
            (&xyz, Some(0), 1),
        ] {
            let path = video.path().to_str().unwrap().into();
            let request = VideoStreamRequest {
                number_of_threads,
                ..Default::default()
            };
            direct_scale_stats::take();
            let _ = decode_video_tensor(MediaSource::Uri(path), request)?;
            let stats = direct_scale_stats::take();
            assert!(stats.frames > 0, "the decode did not take the direct path");
            assert_eq!(
                stats.threads,
                threads as i64,
                "number_of_threads={number_of_threads:?}, {:?}",
                video.path()
            );
        }
        Ok(())
    }

    /// Raw-mode HDR sources whose matrix the scale filter rejects (YCgCo,
    /// ICtCp, ...) must fail as they do in the filter graph, not decode as
    /// BT.601 on the direct path.
    #[test_case(8; "ycgco")]
    #[test_case(12; "chroma_derived_nc")]
    #[test_case(14; "ictcp")]
    fn test_raw_hdr_unsupported_matrix_fails(matrix_coefficients: u8) -> anyhow::Result<()> {
        init_logger();

        let hlg = generate_test_video_file(&TestVideoParameters {
            duration: Duration::from_secs(1),
            colorspace: Some("bt2020nc".to_string()),
            color_range: Some("tv".to_string()),
            pixel_format: Some("yuv420p10le".to_string()),
            color_primaries: Some("bt2020".to_string()),
            color_trc: Some("arib-std-b67".to_string()),
            ..Default::default()
        })?;
        let retagged = retag_matrix_coefficients(hlg.path(), matrix_coefficients)?;
        let path: String = retagged.path().to_str().unwrap().into();
        let request = |hdr_mode| VideoStreamRequest {
            hdr_mode,
            ..Default::default()
        };
        // Tone mapping converts the matrix in zscale, which accepts these.
        let _ = decode_video_tensor(MediaSource::Uri(path.clone()), request(HdrMode::Tonemap))?;
        // The stream stays in the graph, which fails to set up, as on main.
        let err = decode_video_tensor(MediaSource::Uri(path), request(HdrMode::Raw))
            .expect_err("raw decode of an unsupported matrix must fail");
        assert!(
            format!("{err:#}").contains("Failed to initialize filter"),
            "{err:#}"
        );
        Ok(())
    }

    /// Decodes the video stream of `path` with the given dtype and
    /// conversion, returning the tensor as float64 in [T, C, H, W] layout and
    /// [0, 1] scale.
    fn decode_rgb_unit(
        path: &str,
        dtype: OutputDtype,
        color_conversion: Option<ColorConversion>,
    ) -> anyhow::Result<tch::Tensor> {
        let mut streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path.to_string()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    dtype,
                    color_conversion,
                    ..Default::default()
                }),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        )?;
        let data = streams
            .remove(0)
            .data
            .context("video stream should have data")?;
        Ok(match dtype {
            OutputDtype::Uint8 => data
                .permute([0, 3, 1, 2])
                .to_kind(tch::Kind::Double)
                .f_div_scalar(255.0)?,
            OutputDtype::Float32 => data.to_kind(tch::Kind::Double),
        })
    }

    /// A limited-range luma ramp has an exact RGB answer (column x ->
    /// x / 219). The accurate path must hit it: exactly in float32 (swscale's
    /// fast path scales by 255/256 here: white decodes to 0.99615), and to
    /// the nearest code value in uint8.
    #[test_case(false; "8bit")]
    #[test_case(true; "10bit")]
    fn test_accurate_conversion_matches_exact_ramp(ten_bit: bool) -> anyhow::Result<()> {
        init_logger();
        let ramp = generate_limited_range_luma_ramp(ten_bit)?;
        let path = ramp.path().to_str().unwrap();

        let width = 224;
        let expected = tch::Tensor::arange(width, (tch::Kind::Double, tch::Device::Cpu))
            .clamp_max(219)
            .f_div_scalar(219.0)?;

        let float = decode_rgb_unit(path, OutputDtype::Float32, None)?;
        let float_err: f64 = float.f_sub(&expected)?.abs().max().try_into()?;
        assert!(
            float_err < 1e-5,
            "float32 ramp decode is off by up to {float_err} (white = {})",
            f64::try_from(float.max())?
        );

        let uint8 = decode_rgb_unit(path, OutputDtype::Uint8, Some(ColorConversion::Accurate))?;
        let rounded = expected.f_mul_scalar(255.0)?.round().f_div_scalar(255.0)?;
        let uint8_err: f64 = uint8.f_sub(&rounded)?.abs().max().try_into()?;
        assert_eq!(uint8_err, 0.0, "accurate uint8 must round to nearest");
        Ok(())
    }

    /// On 4:2:0 content with chroma, accurate uint8 output must be the
    /// float32 output rounded to the nearest code value (no bias), while the
    /// default fast path stays bit-identical to earlier releases.
    #[test]
    fn test_accurate_uint8_is_unbiased_on_420() -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_test_video_file(&TestVideoParameters {
            pixel_format: Some("yuv420p".to_string()),
            colorspace: Some("bt709".to_string()),
            color_range: Some("tv".to_string()),
            duration: Duration::from_secs(1),
            ..Default::default()
        })?;
        let path = test_video.path().to_str().unwrap();

        let reference = decode_rgb_unit(path, OutputDtype::Float32, None)?;
        let accurate = decode_rgb_unit(path, OutputDtype::Uint8, Some(ColorConversion::Accurate))?;
        assert_eq!(accurate.size(), reference.size());
        let diff = accurate.f_sub(&reference)?.f_mul_scalar(255.0)?;
        let max_err: f64 = diff.abs().max().try_into()?;
        assert!(
            max_err <= 0.5 + 1e-3,
            "accurate uint8 max error {max_err} code values"
        );
        for c in 0..3 {
            let bias: f64 = diff.select(1, c).mean(tch::Kind::Double).try_into()?;
            assert!(
                bias.abs() < 0.05,
                "accurate uint8 channel {c} bias {bias} code values"
            );
        }

        // The default stays on the fast path, unchanged.
        let default = decode_rgb_unit(path, OutputDtype::Uint8, None)?;
        let fast = decode_rgb_unit(path, OutputDtype::Uint8, Some(ColorConversion::Fast))?;
        assert!(default.equal(&fast));
        Ok(())
    }

    /// Decodes `path` as float32 at an optional size with the given
    /// conversion, returning [T, C, H, W] float64.
    fn decode_float_sized(
        path: &str,
        size: Option<(u32, u32)>,
        color_conversion: Option<ColorConversion>,
    ) -> anyhow::Result<tch::Tensor> {
        let mut streams = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(path.to_string()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    dtype: OutputDtype::Float32,
                    width: size.map(|(w, _)| w),
                    height: size.map(|(_, h)| h),
                    color_conversion,
                    ..Default::default()
                }),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        )?;
        Ok(streams
            .remove(0)
            .data
            .context("video stream should have data")?
            .to_kind(tch::Kind::Double))
    }

    /// zimg rejects subsampled frames whose dimensions are not a multiple of
    /// the subsampling factor. The accurate (default float32) path must still
    /// decode odd-size 4:2:0 / 4:2:2 sources, and resizes to sizes like
    /// 853x480, and agree with the fast path up to its 255/256 scale.
    #[test_case(640, 360, "yuv420p", Some((853, 480)); "420 resized to 853x480")]
    #[test_case(640, 360, "yuv420p", Some((427, 240)); "420 resized to 427x240")]
    #[test_case(321, 241, "yuv420p", None; "odd 420 native")]
    #[test_case(321, 240, "yuv422p", None; "odd 422 native")]
    #[test_case(321, 241, "yuv420p10le", None; "odd 420 10bit native")]
    fn test_accurate_float32_odd_sizes(
        width: usize,
        height: usize,
        pixel_format: &str,
        size: Option<(u32, u32)>,
    ) -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_bt709_ffv1_video(width, height, pixel_format)?;
        let path = test_video.path().to_str().unwrap();
        let accurate = decode_float_sized(path, size, None)?;
        let (w, h) = size.unwrap_or((width as u32, height as u32));
        assert_eq!(&accurate.size()[1..], &[3, h as i64, w as i64]);
        let fast = decode_float_sized(path, size, Some(ColorConversion::Fast))?;
        let mean_diff: f64 = accurate
            .f_sub(&fast.f_mul_scalar(256.0 / 255.0)?)?
            .abs()
            .mean(tch::Kind::Double)
            .try_into()?;
        assert!(
            mean_diff * 255.0 < 1.0,
            "accurate and fast disagree by {} code values on average",
            mean_diff * 255.0
        );
        Ok(())
    }

    /// HDR sources whose size (native, or after the resize) is not
    /// divisible by the chroma subsampling must tone map in both modes.
    #[test_case(640, 360, Some((853, 480)), None; "resized 853x480 default")]
    #[test_case(640, 360, Some((853, 480)), Some(ColorConversion::Fast); "resized 853x480 fast")]
    #[test_case(640, 360, Some((427, 241)), None; "resized 427x241 default")]
    #[test_case(640, 360, Some((427, 241)), Some(ColorConversion::Fast); "resized 427x241 fast")]
    #[test_case(321, 241, None, None; "odd native default")]
    #[test_case(321, 241, None, Some(ColorConversion::Fast); "odd native fast")]
    #[test_case(321, 241, Some((160, 120)), None; "odd source resized default")]
    fn test_hdr_odd_sizes(
        width: usize,
        height: usize,
        size: Option<(u32, u32)>,
        color_conversion: Option<ColorConversion>,
    ) -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_pq_ffv1_video(width, height, true, true)?;
        let float =
            decode_float_sized(test_video.path().to_str().unwrap(), size, color_conversion)?;
        let (w, h) = size.unwrap_or((width as u32, height as u32));
        assert_eq!(&float.size()[1..], &[3, h as i64, w as i64]);
        Ok(())
    }

    /// A PQ stream missing its matrix and/or primaries tag must decode as
    /// BT.2020 (NCL), the same as the fully tagged stream, in both
    /// conversion modes and in both HDR modes.
    #[test_case(false, true, HdrMode::Tonemap, None; "no matrix tonemap default")]
    #[test_case(false, true, HdrMode::Tonemap, Some(ColorConversion::Fast); "no matrix tonemap fast")]
    #[test_case(false, false, HdrMode::Tonemap, None; "transfer only tonemap default")]
    #[test_case(false, false, HdrMode::Tonemap, Some(ColorConversion::Fast); "transfer only tonemap fast")]
    #[test_case(false, true, HdrMode::Raw, None; "no matrix raw default")]
    #[test_case(false, true, HdrMode::Raw, Some(ColorConversion::Fast); "no matrix raw fast")]
    fn test_hdr_untagged_defaults_to_bt2020(
        tag_matrix: bool,
        tag_primaries: bool,
        hdr_mode: HdrMode,
        color_conversion: Option<ColorConversion>,
    ) -> anyhow::Result<()> {
        init_logger();
        let decode = |path: &str| -> anyhow::Result<tch::Tensor> {
            let mut streams = decode_media(
                MediaDecodeRequest {
                    source: MediaSource::Uri(path.to_string()),
                    start_time: None,
                    end_time: None,
                    video_stream: Some(VideoStreamRequest {
                        dtype: OutputDtype::Float32,
                        hdr_mode,
                        color_conversion,
                        ..Default::default()
                    }),
                    audio_streams: None,
                    max_output_bytes: None,
                },
                None,
            )?;
            streams
                .remove(0)
                .data
                .context("video stream should have data")
        };
        let untagged = generate_pq_ffv1_video(64, 48, tag_matrix, tag_primaries)?;
        let tagged = generate_pq_ffv1_video(64, 48, true, true)?;
        let output = decode(untagged.path().to_str().unwrap())?;
        let reference = decode(tagged.path().to_str().unwrap())?;
        let max_diff: f64 = output.f_sub(&reference)?.abs().max().try_into()?;
        assert!(
            max_diff < 1e-6,
            "untagged differs from bt2020 by {max_diff}"
        );
        Ok(())
    }

    /// A lone `width` or `height` keeps the source aspect ratio, and the
    /// decoded shape matches `plan_media`.
    #[test_case(Some(320), None, OutputDtype::Uint8, None; "width only uint8")]
    #[test_case(None, Some(180), OutputDtype::Uint8, None; "height only uint8")]
    #[test_case(Some(321), None, OutputDtype::Float32, None; "odd width only float32")]
    #[test_case(Some(320), None, OutputDtype::Uint8, Some(ColorConversion::Accurate); "width only accurate uint8")]
    fn test_single_dimension_resize(
        width: Option<u32>,
        height: Option<u32>,
        dtype: OutputDtype,
        color_conversion: Option<ColorConversion>,
    ) -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_test_video_file(&TestVideoParameters {
            width: 640,
            height: 360,
            duration: Duration::from_secs(1),
            pixel_format: Some("yuv420p".to_string()),
            ..Default::default()
        })?;
        let make_request = || MediaDecodeRequest {
            source: MediaSource::Uri(test_video.path().to_str().unwrap().to_string()),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest {
                width,
                height,
                dtype,
                color_conversion,
                ..Default::default()
            }),
            audio_streams: None,
            max_output_bytes: None,
        };
        let plan = plan_media(make_request(), None)?;
        let data = decode_media(make_request(), None)?
            .remove(0)
            .data
            .context("video stream should have data")?;
        let expected_hw = match (width, height) {
            (Some(w), None) => ((w as i64 * 360 + 320) / 640, w as i64),
            (None, Some(h)) => (h as i64, (h as i64 * 640 + 180) / 360),
            _ => unreachable!(),
        };
        let hw = match dtype {
            OutputDtype::Uint8 => (data.size()[1], data.size()[2]),
            OutputDtype::Float32 => (data.size()[2], data.size()[3]),
        };
        assert_eq!(hw, expected_hw);
        assert_eq!(&plan[0].shape[1..], &data.size()[1..]);
        Ok(())
    }

    /// Gray sources come tagged (`gbr` for PNG); zscale receives them through
    /// an untagged yuvj444p conversion, so the accurate path must still set a
    /// matrix. A uniform gray must decode to the same value on all channels.
    #[test]
    fn test_accurate_float32_tagged_gray() -> anyhow::Result<()> {
        init_logger();
        let png = generate_gray_png(123)?;
        let float = decode_float_sized(png.path().to_str().unwrap(), None, None)?;
        for c in 0..3 {
            let mean: f64 = float.select(1, c).mean(tch::Kind::Double).try_into()?;
            assert!(
                (mean - 123.0 / 255.0).abs() < 1e-5,
                "channel {c} mean {mean}, expected {}",
                123.0 / 255.0
            );
        }
        Ok(())
    }

    /// A raw-mode HDR stream that takes the direct path and then switches to
    /// a matrix the scale filter rejects must fail on those frames rather
    /// than convert them as BT.601.
    #[test]
    fn test_direct_scale_rejects_unsupported_matrix_mid_stream() -> anyhow::Result<()> {
        let config = VideoFilterConfig {
            hdr_mode: HdrMode::Raw,
            source_color: SourceColorInfo {
                color_primaries: Some("bt2020".to_string()),
                color_trc: Some("arib-std-b67".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let (width, height) = (64, 48);
        let mut scaler = DirectVideoScaler::new(&config, width, height);
        let dest = tch::Tensor::zeros(
            [1, height as i64, width as i64, 3],
            (tch::Kind::Uint8, tch::Device::Cpu),
        );
        let frame = |colorspace| -> anyhow::Result<AVFrame> {
            let mut frame = AVFrame::new();
            frame.set_width(width);
            frame.set_height(height);
            frame.set_format(ffi::AV_PIX_FMT_YUV420P10LE);
            frame.alloc_buffer()?;
            unsafe {
                (*frame.as_mut_ptr()).colorspace = colorspace;
                (*frame.as_mut_ptr()).color_range = ffi::AVCOL_RANGE_MPEG;
            }
            Ok(frame)
        };
        scaler.scale_into(&frame(ffi::AVCOL_SPC_BT2020_NCL)?, &dest)?;
        for colorspace in [ffi::AVCOL_SPC_YCGCO, ffi::AVCOL_SPC_ICTCP] {
            let err = scaler
                .scale_into(&frame(colorspace)?, &dest)
                .expect_err("unsupported matrix must not convert");
            assert!(err.to_string().contains("unsupported YUV matrix"), "{err}");
        }
        Ok(())
    }

    /// GPU-resident output converts with NPP, so an explicit accurate
    /// conversion is rejected before any CUDA machinery is touched.
    #[test]
    fn test_device_with_accurate_conversion_is_rejected() -> anyhow::Result<()> {
        init_logger();
        let test_video = generate_test_video_file(&TestVideoParameters::default())?;
        let err = match decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(test_video.path().to_str().unwrap().into()),
                start_time: None,
                end_time: None,
                video_stream: Some(VideoStreamRequest {
                    device: Some(0),
                    color_conversion: Some(ColorConversion::Accurate),
                    ..Default::default()
                }),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        ) {
            Ok(_) => anyhow::bail!("expected accurate conversion on cuda to be rejected"),
            Err(err) => err,
        };
        assert!(
            format!("{err:#}").contains("does not support color_conversion"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// The scale filter rejects BT.2020 constant-luminance input, so the
    /// direct path is the only one that converts it. swscale has the same
    /// coefficients for both BT.2020 matrices, so a `bt2020c` source must
    /// convert exactly like its `bt2020nc` twin.
    #[test]
    fn test_direct_scale_bt2020c_uses_bt2020_coefficients() -> anyhow::Result<()> {
        init_logger();

        let non_constant_luminance = generate_test_video_file(&TestVideoParameters {
            colorspace: Some("bt2020nc".to_string()),
            color_range: Some("tv".to_string()),
            pixel_format: Some("yuv420p10le".to_string()),
            color_primaries: Some("bt2020".to_string()),
            color_trc: Some("arib-std-b67".to_string()),
            ..Default::default()
        })?;
        // H.273 matrix coefficients 10: BT.2020 constant luminance.
        let constant_luminance = retag_matrix_coefficients(non_constant_luminance.path(), 10)?;
        // The graph cannot convert bt2020c, so keep the direct path enabled.
        let _guard = DIRECT_PATH_ENV.read().unwrap_or_else(|e| e.into_inner());
        let decode = |video: &tempfile::NamedTempFile| -> anyhow::Result<tch::Tensor> {
            let path = video.path().to_str().unwrap().into();
            let request = VideoStreamRequest {
                hdr_mode: HdrMode::Raw,
                ..Default::default()
            };
            Ok(decode_video_tensor(MediaSource::Uri(path), request)?.0)
        };
        assert!(decode(&constant_luminance)?.equal(&decode(&non_constant_luminance)?));
        Ok(())
    }

    /// XYZ sources scale on one swscale thread, since every slice worker
    /// would convert the whole frame from XYZ; other formats use the pool.
    #[test]
    fn test_scaler_threads() {
        assert_eq!(scaler_threads(ffi::AV_PIX_FMT_XYZ12LE, 16), 1);
        assert_eq!(scaler_threads(ffi::AV_PIX_FMT_XYZ12BE, 16), 1);
        assert_eq!(scaler_threads(ffi::AV_PIX_FMT_YUV420P10LE, 16), 16);
        assert_eq!(scaler_threads(ffi::AV_PIX_FMT_YUV420P, 16), 16);
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
                    max_output_bytes: None,
                },
                None,
            )?;
            streams
                .remove(0)
                .data
                .context("audio stream should have data")
        };

        let direct = decode_audio(path.clone())?;

        let graph = with_direct_path_disabled(|| decode_audio(path))?;

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
                    max_output_bytes: None,
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
        let (from_bytes, _) = decode_video_tensor(MediaSource::Bytes(bytes), Default::default())?;
        let (from_file, _) = decode_video_tensor(
            MediaSource::Uri(test_video.path().to_str().unwrap().into()),
            Default::default(),
        )?;

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
        let (data, pts) = decode_video_tensor(
            MediaSource::Uri(test_video.path().to_str().unwrap().into()),
            Default::default(),
        )?;

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

    fn uri(file: &tempfile::NamedTempFile) -> MediaSource {
        MediaSource::Uri(file.path().to_str().unwrap().into())
    }

    /// Decodes `request` and checks every stream against `plan_media`: same
    /// streams in the same order, same non-time dims, and the decoded length
    /// within the planned (preallocated) length, equal when `exact`.
    fn assert_plan_bounds_decode(
        make_request: impl Fn() -> MediaDecodeRequest,
        exact: bool,
    ) -> anyhow::Result<Vec<StreamPlan>> {
        let plans = plan_media(make_request(), None)?;
        let decoded = decode_media(make_request(), None)?;
        assert_eq!(plans.len(), decoded.len());
        for (plan, stream) in plans.iter().zip(&decoded) {
            assert_eq!(plan.stream_index, stream.src_stream_index);
            assert_eq!(plan.metadata, stream.metadata);
            let data = stream.data.as_ref().unwrap();
            let size = data.size();
            let time_dim = match plan.stream_type() {
                StreamType::Video => 0,
                StreamType::Audio => 1,
            };
            for (dim, (&planned, &got)) in plan.shape.iter().zip(&size).enumerate() {
                if dim == time_dim {
                    assert!(
                        got <= planned,
                        "decoded {size:?} exceeds plan {:?}",
                        plan.shape
                    );
                    if exact {
                        assert_eq!(got, planned, "decoded {size:?} != plan {:?}", plan.shape);
                    }
                } else {
                    assert_eq!(got, planned, "decoded {size:?} != plan {:?}", plan.shape);
                }
            }
        }
        Ok(plans)
    }

    #[test_case(None, None, None, None, OutputDtype::Uint8 ; "defaults")]
    #[test_case(Some(1.0), Some(3.5), None, None, OutputDtype::Uint8 ; "seek window")]
    #[test_case(Some(2.0), None, None, None, OutputDtype::Uint8 ; "start only")]
    #[test_case(None, Some(9.0), None, None, OutputDtype::Uint8 ; "end past duration")]
    #[test_case(None, Some(3.0), Some(24.0), None, OutputDtype::Uint8 ; "fps resample")]
    #[test_case(None, Some(2.0), None, Some((97, 65)), OutputDtype::Uint8 ; "odd resize")]
    #[test_case(None, Some(2.0), None, Some((96, 64)), OutputDtype::Float32 ; "float32")]
    fn test_plan_media_matches_decode(
        start_time: Option<f64>,
        end_time: Option<f64>,
        frame_rate: Option<f64>,
        size: Option<(u32, u32)>,
        dtype: OutputDtype,
    ) -> anyhow::Result<()> {
        init_logger();
        let file = generate_test_video_file(&TestVideoParameters::default())?;
        let make_request = || MediaDecodeRequest {
            source: uri(&file),
            start_time,
            end_time,
            video_stream: Some(VideoStreamRequest {
                frame_rate,
                width: size.map(|(w, _)| w),
                height: size.map(|(_, h)| h),
                dtype,
                ..Default::default()
            }),
            audio_streams: Some(vec![AudioStreamRequest {
                sample_rate: Some(16000),
                ..Default::default()
            }]),
            max_output_bytes: None,
        };
        // CFR video lands exactly on the plan; AAC frame padding makes the
        // audio tail slightly short of it.
        let plans = assert_plan_bounds_decode(make_request, false)?;
        let video = &plans[0];
        let decoded = decode_media(make_request(), None)?;
        assert_eq!(decoded[0].data.as_ref().unwrap().size()[0], video.shape[0]);
        assert_eq!(video.duration_source, DurationSource::Stream);
        let audio_planned = plans[1].shape[1];
        let audio_decoded = decoded[1].data.as_ref().unwrap().size()[1];
        assert!(
            audio_planned - audio_decoded <= 1600,
            "{audio_planned} vs {audio_decoded}"
        );
        Ok(())
    }

    #[test]
    fn test_plan_media_webm_uses_container_duration() -> anyhow::Result<()> {
        init_logger();
        let webm = crate::util::test_utils::generate_test_webm_file(Duration::from_secs(4))?;
        let make_request = || MediaDecodeRequest {
            source: uri(&webm),
            start_time: Some(1.0),
            end_time: Some(3.0),
            video_stream: Some(VideoStreamRequest::default()),
            audio_streams: Some(vec![AudioStreamRequest::default()]),
            max_output_bytes: None,
        };
        let plans = assert_plan_bounds_decode(make_request, false)?;
        for plan in &plans {
            assert_eq!(plan.duration_source, DurationSource::Container);
            assert_eq!(plan.duration_seconds, Some(2.0));
        }
        assert_eq!(plans[0].shape, vec![60, 240, 320, 3]);
        Ok(())
    }

    #[test]
    fn test_plan_media_audio_only() -> anyhow::Result<()> {
        init_logger();
        let flac = crate::util::test_utils::generate_test_flac_file(48000, Duration::from_secs(3))?;
        let make_request = || MediaDecodeRequest {
            source: uri(&flac),
            start_time: None,
            end_time: Some(2.0),
            video_stream: None,
            audio_streams: Some(vec![AudioStreamRequest::default()]),
            max_output_bytes: None,
        };
        let plans = assert_plan_bounds_decode(make_request, true)?;
        assert_eq!(plans[0].shape, vec![1, 96000]);
        Ok(())
    }

    #[test]
    fn test_plan_media_audio_shorter_than_video() -> anyhow::Result<()> {
        init_logger();
        let source = generate_test_video_file(&TestVideoParameters::default())?;
        let offset = crate::util::test_utils::make_av_offset_video(source.path(), 1.0)?;
        let make_request = || MediaDecodeRequest {
            source: uri(&offset),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest::default()),
            audio_streams: Some(vec![AudioStreamRequest::default()]),
            max_output_bytes: None,
        };
        assert_plan_bounds_decode(make_request, false)?;
        Ok(())
    }

    #[test]
    fn test_plan_media_errors_match_decode() -> anyhow::Result<()> {
        init_logger();
        let file = generate_test_video_file(&TestVideoParameters::default())?;
        let make_request = || MediaDecodeRequest {
            source: uri(&file),
            start_time: Some(3.0),
            end_time: Some(2.0),
            video_stream: Some(VideoStreamRequest::default()),
            audio_streams: None,
            max_output_bytes: None,
        };
        let plan_err = plan_media(make_request(), None).unwrap_err();
        let decode_err = decode_media(make_request(), None).err().unwrap();
        assert!(format!("{plan_err:?}").contains("End time must be greater"));
        assert!(format!("{decode_err:?}").contains("End time must be greater"));
        Ok(())
    }

    #[test]
    fn test_probe_media_durations_and_names() -> anyhow::Result<()> {
        init_logger();
        let mp4 = generate_test_video_file(&TestVideoParameters::default())?;
        let probed = probe_media(uri(&mp4), None)?;
        assert!(probed.format_name.contains("mp4"), "{}", probed.format_name);
        assert!((probed.duration_seconds.unwrap() - 5.0).abs() < 0.1);
        let video = &probed.video_streams[0];
        assert_eq!(video.codec_name, "h264");
        assert_eq!(video.duration_source, DurationSource::Stream);
        assert!((video.duration_seconds.unwrap() - 5.0).abs() < 0.05);
        assert_eq!(video.nb_frames, Some(150));
        assert_eq!(video.rotation, None);
        let audio = &probed.audio_streams[0];
        assert_eq!(audio.codec_name, "aac");
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.duration_source, DurationSource::Stream);

        let webm = crate::util::test_utils::generate_test_webm_file(Duration::from_secs(4))?;
        let probed = probe_media(uri(&webm), None)?;
        assert_eq!(probed.format_name, "matroska,webm");
        let video = &probed.video_streams[0];
        assert_eq!(video.codec_name, "vp9");
        assert_eq!(video.duration_source, DurationSource::Container);
        assert_eq!(video.nb_frames, None);
        let audio = &probed.audio_streams[0];
        assert_eq!(audio.codec_name, "opus");
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.duration_source, DurationSource::Container);
        Ok(())
    }

    #[test_case(90 ; "90")]
    #[test_case(180 ; "180")]
    #[test_case(-90 ; "minus 90")]
    fn test_probe_media_rotation_is_reported_not_applied(degrees: i32) -> anyhow::Result<()> {
        init_logger();
        let params = TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        };
        let source = generate_test_video_file(&params)?;
        let rotated = crate::util::test_utils::make_rotated_video(source.path(), degrees)?;
        let probed = probe_media(uri(&rotated), None)?;
        let video = &probed.video_streams[0];
        let rotation = video.rotation.expect("display matrix");
        let diff = (rotation - degrees as f64).rem_euclid(360.0);
        assert!(
            diff < 1e-6 || (360.0 - diff) < 1e-6,
            "{rotation} vs {degrees}"
        );
        // Coded dims, and the decoder does not rotate.
        assert_eq!((video.width, video.height), (640, 480));
        let make_request = || MediaDecodeRequest {
            source: uri(&rotated),
            start_time: None,
            end_time: None,
            video_stream: Some(VideoStreamRequest::default()),
            audio_streams: None,
            max_output_bytes: None,
        };
        let plans = assert_plan_bounds_decode(make_request, true)?;
        assert_eq!(plans[0].shape[1..], [480, 640, 3]);
        Ok(())
    }

    /// With `autorotate`, frames are rotated as ffmpeg's `-autorotate` does;
    /// compared here against the unrotated decode rotated in torch.
    #[test_case(90, 1, true ; "90")]
    #[test_case(-90, -1, true ; "minus 90")]
    #[test_case(180, 2, false ; "180")]
    fn test_decode_autorotate(degrees: i32, quarter_turns: i64, swaps: bool) -> anyhow::Result<()> {
        init_logger();
        let params = TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        };
        let source = generate_test_video_file(&params)?;
        let rotated = crate::util::test_utils::make_rotated_video(source.path(), degrees)?;
        let path: String = rotated.path().to_str().unwrap().into();
        let decode = |autorotate: bool| {
            decode_video_tensor(
                MediaSource::Uri(path.clone()),
                VideoStreamRequest {
                    autorotate,
                    ..Default::default()
                },
            )
        };
        let (coded, coded_pts) = decode(false)?;
        let (shown, shown_pts) = decode(true)?;
        // Exact equality with the rotated coded frames relies on this
        // fixture (even-sized and 8-bit; testsrc encodes as 4:4:4, and
        // even-sized 4:2:0 also works): only then does the RGB conversion
        // commute with the rotation. With odd-sized 4:2:0, >8-bit sources
        // or float32 output, autorotated frames still match the CLI's
        // `-autorotate` but not `rot90`, so a fixture change here needs a
        // comparison against the CLI instead.
        assert_eq!(coded.size()[1..], [480, 640, 3]);
        let expected_hw = if swaps { [640, 480] } else { [480, 640] };
        assert_eq!(shown.size()[1..3], expected_hw);
        assert_eq!(shown_pts, coded_pts);
        // Counterclockwise display rotation; torch.rot90 turns from the
        // first dim towards the second, i.e. counterclockwise on [H, W].
        let expected = coded.rot90(quarter_turns, [1, 2]);
        let max_diff: f64 = shown
            .to_kind(tch::Kind::Int16)
            .f_sub(&expected.to_kind(tch::Kind::Int16))?
            .abs()
            .max()
            .to_kind(tch::Kind::Float)
            .try_into()?;
        assert_eq!(
            max_diff, 0.0,
            "autorotated frames must be the coded frames rotated"
        );

        // width/height refer to the rotated frame.
        let (resized, _) = decode_video_tensor(
            MediaSource::Uri(path.clone()),
            VideoStreamRequest {
                autorotate: true,
                width: Some(expected_hw[1] as u32 / 2),
                height: Some(expected_hw[0] as u32 / 2),
                ..Default::default()
            },
        )?;
        assert_eq!(
            resized.size()[1..3],
            [expected_hw[0] / 2, expected_hw[1] / 2]
        );
        Ok(())
    }

    /// The display matrix `av_display_rotation_set(degrees)` gives (a
    /// clockwise rotation), then mirrored by `av_display_matrix_flip`.
    fn matrix(degrees: f64, hflip: bool, vflip: bool) -> [i32; 9] {
        let mut m = [0i32; 9];
        unsafe {
            ffi::av_display_rotation_set(m.as_mut_ptr(), degrees);
            ffi::av_display_matrix_flip(m.as_mut_ptr(), hflip as i32, vflip as i32);
        }
        m
    }

    /// The filters FFmpeg 7.1's CLI inserts for each matrix (checked against
    /// `configure_input_video_filter` with the same matrices).
    #[test]
    fn test_autorotate_filters() {
        let f = |d, h, v| autorotate_filters(&matrix(d, h, v));
        assert_eq!(f(0.0, false, false), Vec::<String>::new());
        assert_eq!(f(90.0, false, false), ["transpose=clock"]);
        assert_eq!(f(-90.0, false, false), ["transpose=cclock"]);
        assert_eq!(f(180.0, false, false), ["hflip", "vflip"]);
        assert_eq!(f(45.0, false, false), ["rotate=45.000000*PI/180"]);
        // Mirrored matrices.
        assert_eq!(f(0.0, true, false), ["hflip"]);
        assert_eq!(f(0.0, false, true), ["vflip"]);
        assert_eq!(f(90.0, true, false), ["transpose=cclock_flip"]);
        assert_eq!(f(90.0, false, true), ["transpose=clock_flip"]);
        assert_eq!(f(180.0, true, false), ["vflip"]);
        // Mirrored near-zero angles (the CLI's `-display_rotation r
        // -display_vflip` sets matrix(-r, false, true)). The angle is rounded
        // first: 0.4° rounds to 0 and gets the vflip; 0.6° and 1.4° round to
        // exactly 1°, which FFmpeg 7.1 neither rotates (`> 1`) nor flips
        // (`< 1`), so those come out unchanged, as `ffmpeg -autorotate` does.
        assert_eq!(f(-0.4, false, true), ["vflip"]);
        assert_eq!(f(-0.6, false, true), Vec::<String>::new());
        assert_eq!(f(-1.0, false, true), Vec::<String>::new());
        assert_eq!(f(-1.4, false, true), Vec::<String>::new());
        assert_eq!(f(1.0, false, true), ["rotate=359.000000*PI/180"]);
        let swaps = |d, h, v| swaps_dimensions(&f(d, h, v));
        assert!(swaps(90.0, false, false));
        assert!(swaps(-90.0, true, false));
        assert!(!swaps(180.0, false, false));
        assert!(!swaps(45.0, false, false));
        assert!(autorotate_filters(&[0; 9]).is_empty());
    }

    #[test]
    fn test_probe_media_sample_aspect_ratio() -> anyhow::Result<()> {
        init_logger();
        let params = TestVideoParameters {
            duration: Duration::from_secs(1),
            ..Default::default()
        };
        let source = generate_test_video_file(&params)?;
        let anamorphic = crate::util::test_utils::make_anamorphic_video(source.path())?;
        let probed = probe_media(uri(&anamorphic), None)?;
        assert_eq!(probed.video_streams[0].sample_aspect_ratio, (4, 3));
        Ok(())
    }

    /// With B-frames, packets past the end of the window precede in-window
    /// frames in decode order. The window must hold exactly the frames with
    /// start <= pts < end, contiguous.
    #[test_case(None, 1.7, 51 ; "end only")]
    #[test_case(Some(0.5), 1.7, 36 ; "start and end")]
    fn test_decode_end_time_with_b_frames(
        start_time: Option<f64>,
        end_time: f64,
        expected_frames: usize,
    ) -> anyhow::Result<()> {
        init_logger();
        let params = TestVideoParameters {
            b_frames: Some(3),
            ..Default::default()
        };
        let file = generate_test_video_file(&params)?;
        let decoded = decode_media(
            MediaDecodeRequest {
                source: MediaSource::Uri(file.path().to_str().unwrap().into()),
                start_time,
                end_time: Some(end_time),
                video_stream: Some(VideoStreamRequest::default()),
                audio_streams: None,
                max_output_bytes: None,
            },
            None,
        )?;
        let stream = &decoded[0];
        assert_eq!(
            stream.data.as_ref().unwrap().size()[0] as usize,
            expected_frames
        );
        let frame_indices: Vec<i64> = stream
            .frame_pts
            .iter()
            .map(|pts| (pts * params.frame_rate).round() as i64)
            .collect();
        let first = (start_time.unwrap_or(0.0) * params.frame_rate).round() as i64;
        let expected: Vec<i64> = (first..first + expected_frames as i64).collect();
        assert_eq!(frame_indices, expected);
        Ok(())
    }
}
