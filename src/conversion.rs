use anyhow::{bail, Context};
use rsmpeg::avutil::{get_sample_fmt_name, AVFrame};
use rsmpeg::ffi::{self, AVChannelLayout, AVSampleFormat};
use rsmpeg::swresample::SwrContext;
use std::cmp::min;
use std::num::NonZeroI32;
use tch::IndexOp;

/// Converts the rgb24 frame into the provided destination Tensor.
///
/// Arguments:
/// * frame: The frame to convert.
/// * dest: The destination tensor to convert the result to. Shape: [H, W, C] or [1, H, W, C].
pub fn convert_rgb24_frame_to_tensor(
    frame: &AVFrame,
    dest: &mut tch::Tensor,
) -> Result<(), anyhow::Error> {
    if frame.format != ffi::AV_PIX_FMT_RGB24 {
        return Err(anyhow::anyhow!(
            "Expected AV_PIX_FMT_RGB24 ({}) format, got {}",
            ffi::AV_PIX_FMT_RGB24,
            frame.format
        ));
    }

    if dest.kind() != tch::Kind::Uint8 {
        return Err(anyhow::anyhow!(
            "Expected destination Tensor to be of kind Uint8, got {:?}",
            dest.kind()
        ));
    }

    let shape = dest.size();
    if shape != vec![frame.height as i64, frame.width as i64, 3]
        && shape != vec![1, frame.height as i64, frame.width as i64, 3]
    {
        return Err(anyhow::anyhow!(
            "Expected destination Tensor to have shape [{}, {}, 3] or [1, {}, {}, 3], got {:?}",
            frame.height,
            frame.width,
            frame.height,
            frame.width,
            shape
        ));
    }

    let height = frame.height as usize;
    let width = frame.width as usize;
    let num_channels = 3; // RGB
    let stride = width * num_channels;

    // Copy data from the buffer backing the AVFrame to buffer backing the Tensor.
    let mut p_dst = dest.data_ptr() as *mut u8;
    let mut p_src = frame.data[0];
    for _h in 0..height {
        unsafe {
            std::ptr::copy_nonoverlapping(p_src, p_dst, stride);
            p_src = p_src.add(frame.linesize[0] as usize);
            p_dst = p_dst.add(stride);
        }
    }

    Ok(())
}

/// Copies a planar float frame (`AV_PIX_FMT_GBRPF32LE`) into the provided
/// float32 destination Tensor in channels-first RGB order. This is the
/// high-bit-depth decode path: FFmpeg converts the source to planar float in
/// [0, 1], preserving the full precision of 10/12-bit sources.
///
/// Arguments:
/// * frame: The frame to convert (`AV_PIX_FMT_GBRPF32LE`; planes are G, B, R).
/// * dest: The destination tensor. Kind Float, shape [3, H, W] or [1, 3, H, W].
pub fn convert_gbrpf32_frame_to_tensor(
    frame: &AVFrame,
    dest: &mut tch::Tensor,
) -> Result<(), anyhow::Error> {
    if frame.format != ffi::AV_PIX_FMT_GBRPF32LE {
        return Err(anyhow::anyhow!(
            "Expected AV_PIX_FMT_GBRPF32LE ({}) format, got {}",
            ffi::AV_PIX_FMT_GBRPF32LE,
            frame.format
        ));
    }

    if dest.kind() != tch::Kind::Float {
        return Err(anyhow::anyhow!(
            "Expected destination Tensor to be of kind Float, got {:?}",
            dest.kind()
        ));
    }

    let shape = dest.size();
    if shape != vec![3, frame.height as i64, frame.width as i64]
        && shape != vec![1, 3, frame.height as i64, frame.width as i64]
    {
        return Err(anyhow::anyhow!(
            "Expected destination Tensor to have shape [3, {}, {}] or [1, 3, {}, {}], got {:?}",
            frame.height,
            frame.width,
            frame.height,
            frame.width,
            shape
        ));
    }

    let height = frame.height as usize;
    let width = frame.width as usize;
    let plane_elems = height * width;

    // gbrp plane order is G, B, R; the tensor wants R, G, B channels.
    // Frame rows are padded to `linesize` (bytes); tensor rows are tight.
    //
    // SAFETY: the shape check above guarantees `dest` holds exactly
    // 3 * height * width f32 elements, so every `dst_plane.add(row * width)`
    // write of `width` elements stays inside the tensor. On the source
    // side, FFmpeg guarantees each plane holds `height` rows of `linesize`
    // bytes with `linesize >= width * 4` for float formats, so each row
    // read of `width` f32s stays inside the frame; the format check above
    // guarantees the planes really are f32 (and rows are at least 4-byte
    // aligned, FFmpeg aligns rows to 32/64 bytes).
    let dst_base = dest.data_ptr() as *mut f32;
    for (plane, channel) in [(2usize, 0usize), (0, 1), (1, 2)] {
        let dst_plane = unsafe { dst_base.add(channel * plane_elems) };
        for row in 0..height {
            unsafe {
                let src_row =
                    frame.data[plane].add(row * frame.linesize[plane] as usize) as *const f32;
                std::ptr::copy_nonoverlapping(src_row, dst_plane.add(row * width), width);
            }
        }
    }

    Ok(())
}

/// Converts a Tensor containing RGB24 data into an AVFrame.
///
/// Arguments:
/// * `tensor`: A 3D [H, W, C] Tensor containing a single frame.
#[allow(dead_code)]
pub fn convert_rgb24_tensor_to_avframe(tensor: &tch::Tensor) -> Result<AVFrame, anyhow::Error> {
    if tensor.kind() != tch::Kind::Uint8 {
        return Err(anyhow::anyhow!(
            "Expected Tensor to be of kind Uint8, got {:?}",
            tensor.kind()
        ));
    }

    let shape = tensor.size();
    if shape.len() != 3 {
        return Err(anyhow::anyhow!(
            "Expected Tensor to have shape [H, W, C], got {:?}",
            shape
        ));
    }

    let height = shape[0] as usize;
    let width = shape[1] as usize;
    let num_channels = shape[2] as usize; // RGB
    let stride = width * num_channels;

    // Create a new AVFrame from the Tensor.
    let mut frame = AVFrame::new();
    frame.set_format(ffi::AV_PIX_FMT_RGB24);
    frame.set_height(height as i32);
    frame.set_width(width as i32);
    frame
        .alloc_buffer()
        .context("Failed to allocate buffer for AVFrame")?;

    // Copy data from the Tensor to the AVFrame.
    let mut p_src = tensor.data_ptr() as *mut u8;
    let mut p_dst = frame.data[0];

    for _h in 0..height {
        unsafe {
            std::ptr::copy_nonoverlapping(p_src, p_dst, stride);
            p_dst = p_dst.add(frame.linesize[0] as usize);
            p_src = p_src.add(stride);
        }
    }

    Ok(frame)
}

pub fn convert_audio_into_tensor(
    frame: &AVFrame,
    dest: &mut tch::Tensor,
) -> Result<(), anyhow::Error> {
    match frame.format {
        ffi::AV_SAMPLE_FMT_U8
        | ffi::AV_SAMPLE_FMT_U8P
        | ffi::AV_SAMPLE_FMT_S16
        | ffi::AV_SAMPLE_FMT_S16P
        | ffi::AV_SAMPLE_FMT_S32
        | ffi::AV_SAMPLE_FMT_S32P
        | ffi::AV_SAMPLE_FMT_FLT
        | ffi::AV_SAMPLE_FMT_DBL
        | ffi::AV_SAMPLE_FMT_DBLP
        | ffi::AV_SAMPLE_FMT_S64
        | ffi::AV_SAMPLE_FMT_S64P => {
            log::debug!(
                "Decoding audio frame with {} format",
                get_sample_fmt_name(frame.format)
                    .map(|s| s.to_str().unwrap_or_default())
                    .unwrap_or("unknown format")
            );
            convert_audio_into_fltp_tensor(frame, dest)?;
        }
        ffi::AV_SAMPLE_FMT_FLTP => {
            // 32-bit floating point planar format, this does not require a dtype conversion using swscale since
            // it is already in the expected output format.
            log::trace!("Decoding audio frame with AV_SAMPLE_FMT_FLTP format");
            convert_fltp_frame_into_tensor(frame, dest)?;
        }
        other => {
            log::warn!("Decoding frame with unsupported audio format: {other:?}");
            bail!("Unsupported audio format: {other:?}");
        }
    }

    Ok(())
}

/// Return number of bytes per sample, return `None` when sample format is unknown.
pub fn get_bytes_per_sample(sample_fmt: AVSampleFormat) -> Option<usize> {
    NonZeroI32::new(unsafe { ffi::av_get_bytes_per_sample(sample_fmt) })
        .map(NonZeroI32::get)
        .and_then(|x| x.try_into().ok())
}

fn convert_fltp_frame_into_tensor(
    frame: &AVFrame,
    dest: &mut tch::Tensor,
) -> Result<(), anyhow::Error> {
    // Ensure the frame is in the expected format.
    if frame.format != ffi::AV_SAMPLE_FMT_FLTP {
        return Err(anyhow::anyhow!(
            "Expected AV_SAMPLE_FMT_FLTP ({}) format, got {}",
            ffi::AV_SAMPLE_FMT_FLTP,
            frame.format
        ));
    }
    let bps = get_bytes_per_sample(frame.format).unwrap_or(0);
    let num_samples = frame.nb_samples;
    let num_channels = frame.ch_layout().nb_channels as usize;

    let num_channels_in_tensor = dest.size().first().copied().unwrap_or(0) as usize;
    let num_samples_in_tensor = dest.size().get(1).copied().unwrap_or(0) as usize;
    if num_channels_in_tensor != num_channels {
        return Err(anyhow::anyhow!(
            "Expected destination Tensor to have dimensions [{}, T], got {:?}",
            num_channels,
            dest.size()
        ));
    }

    let num_samples_to_copy = min(num_samples as usize, num_samples_in_tensor);
    let plane_size = num_samples_to_copy as usize * bps;

    for i in 0..num_channels {
        let data_ptr = unsafe {
            if frame.extended_data.is_null() {
                frame.data[i]
            } else {
                *frame.extended_data.add(i)
            }
        };
        let p_dst = dest.i(i as i64).data_ptr() as *mut u8;
        if !data_ptr.is_null() && plane_size > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(data_ptr, p_dst, plane_size);
            }
        }
    }

    Ok(())
}

fn convert_audio_into_fltp_tensor(
    frame: &AVFrame,
    dest: &mut tch::Tensor,
) -> Result<(), anyhow::Error> {
    let context = SwrContext::new(
        &frame.ch_layout(),
        ffi::AV_SAMPLE_FMT_FLTP,
        frame.sample_rate,
        &frame.ch_layout(),
        frame.format,
        frame.sample_rate,
    )
    .context("allocating SwrContext")?;

    let mut converted_frame = AVFrame::new();
    converted_frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
    converted_frame.set_ch_layout(**frame.ch_layout());
    converted_frame.set_sample_rate(frame.sample_rate);
    context
        .convert_frame(Some(frame), &mut converted_frame)
        .with_context(|| {
            format!(
                "converting {} frame into AV_SAMPLE_FMT_FLTP frame",
                get_sample_fmt_name(frame.format)
                    .map(|s| s.to_str().unwrap_or_default())
                    .unwrap_or("unknown format")
            )
        })?;

    convert_fltp_frame_into_tensor(&converted_frame, dest)
}

#[allow(dead_code)]
fn convert_fltp_tensor_to_avframe(
    tensor: &tch::Tensor,
    channel_layout: AVChannelLayout,
    sample_rate: i32,
) -> Result<AVFrame, anyhow::Error> {
    // Ensure the tensor has the expected shape.
    let (num_channels, num_samples) = tensor.size2()?;

    // Create a new AVFrame.
    let mut frame = AVFrame::new();
    frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
    frame.set_ch_layout(channel_layout);
    frame.set_nb_samples(num_samples as i32);
    frame.set_sample_rate(sample_rate);
    frame
        .alloc_buffer()
        .context("Failed to allocate buffer for AVFrame")?;

    // Allocate the frame's data.
    let bps = 4;
    let plane_size = num_samples as usize * bps;
    for i in 0..num_channels as usize {
        let dest_ptr = unsafe {
            if frame.extended_data.is_null() {
                frame.data[i]
            } else {
                *frame.extended_data.add(i)
            }
        };
        if !dest_ptr.is_null() && plane_size > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.i(i as i64).data_ptr() as *mut u8,
                    dest_ptr,
                    plane_size,
                );
            }
        }
    }

    Ok(frame)
}

#[cfg(test)]
mod tests {

    use rsmpeg::avutil::AVChannelLayout;
    use tch::{Device, Kind};

    use crate::util::test_utils::init_logger;

    use super::*;

    #[test]
    fn test_convert_rgb24_frame_to_tensor() -> Result<(), anyhow::Error> {
        // Allocate a Tensor with picture data.
        let width = 320;
        let height = 240;
        let num_channels = 3;

        let src_tensor = tch::Tensor::f_randint(
            255,
            [width, height, num_channels],
            (Kind::Uint8, Device::Cpu),
        )?;
        let mut dest_tensor =
            tch::Tensor::empty([width, height, num_channels], (Kind::Uint8, Device::Cpu));
        let mut dest_tensor_batch_dim =
            tch::Tensor::empty([1, width, height, num_channels], (Kind::Uint8, Device::Cpu));

        // Perform a round trip conversion.
        let frame = convert_rgb24_tensor_to_avframe(&src_tensor)?;
        convert_rgb24_frame_to_tensor(&frame, &mut dest_tensor)?;

        // Validate batch dimension handling.
        convert_rgb24_frame_to_tensor(&frame, &mut dest_tensor_batch_dim)?;

        // After a round trip conversion the Tensors should be equal.
        assert_eq!(&src_tensor, &dest_tensor);
        assert_eq!(&src_tensor, &dest_tensor_batch_dim.squeeze());

        Ok(())
    }

    #[test]
    fn test_convert_fltp_frame_to_tensor() -> Result<(), anyhow::Error> {
        let sample_rate = 16000;
        let num_samples = sample_rate; // 1 second with 16000 Hz sample rate
        let channel_layout = AVChannelLayout::from_string(c"stereo").unwrap();

        // Random noise
        let src_tensor = tch::Tensor::empty(
            [channel_layout.nb_channels as i64, num_samples],
            (Kind::Float, Device::Cpu),
        );
        let src_tensor = src_tensor.f_uniform(0.0, 1.0)?;

        let mut dest_tensor = tch::Tensor::empty(
            [channel_layout.nb_channels as i64, num_samples],
            (Kind::Float, Device::Cpu),
        );

        let frame =
            convert_fltp_tensor_to_avframe(&src_tensor, *channel_layout, sample_rate as i32)?;
        convert_fltp_frame_into_tensor(&frame, &mut dest_tensor)?;

        assert!(src_tensor.allclose(&dest_tensor, 1e-05, 1e-08, false));
        Ok(())
    }

    #[test_case::test_case(ffi::AV_SAMPLE_FMT_FLT, 1e-05, 1e-08; "32-bit floating point")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_FLTP, 1e-05, 1e-08; "32-bit floating point (planar)")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_DBL, 1e-05, 1e-08; "64-bit floating point")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_DBLP, 1e-05, 1e-08; "64-bit floating point (planar)")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_U8, 1e-05, 1e-02; "unsigned 8-bit integer")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_U8P, 1e-05, 1e-02; "unsigned 8-bit integer (planar)")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_S16, 1e-05, 1e-04; "signed 16-bit integer")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_S16P, 1e-05, 1e-04; "signed 16-bit integer (planar)")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_S32, 1e-05, 1e-05; "signed 32-bit integer")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_S32P, 1e-05, 1e-05; "signed 32-bit integer (planar)")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_S64, 1e-05, 1e-06; "signed 64-bit integer")]
    #[test_case::test_case(ffi::AV_SAMPLE_FMT_S64P, 1e-05, 1e-06; "signed 64-bit integer (planar)")]
    fn test_convert_audio_into_fltp_tensor(
        source_format: AVSampleFormat,
        rtol: f64,
        atol: f64,
    ) -> Result<(), anyhow::Error> {
        init_logger();

        let sample_rate = 16000;
        let num_samples = sample_rate; // 1 second with 16000 Hz sample
        let channel_layout = AVChannelLayout::from_string(c"stereo").unwrap();

        // Random noise
        let src_tensor = tch::Tensor::empty(
            [channel_layout.nb_channels as i64, num_samples],
            (Kind::Float, Device::Cpu),
        );
        let src_tensor = src_tensor.f_uniform(0.0, 1.0)?;

        let mut dest_tensor = tch::Tensor::empty(
            [channel_layout.nb_channels as i64, num_samples],
            (Kind::Float, Device::Cpu),
        );

        let frame =
            convert_fltp_tensor_to_avframe(&src_tensor, *channel_layout, sample_rate as i32)?;

        // Convert the FLTP frame into the source format.
        let context = SwrContext::new(
            &frame.ch_layout(),
            source_format,
            frame.sample_rate,
            &frame.ch_layout(),
            frame.format,
            frame.sample_rate,
        )
        .context("allocating SwrContext")?;

        let mut source_frame = AVFrame::new();
        source_frame.set_format(source_format);
        source_frame.set_ch_layout(**frame.ch_layout());
        source_frame.set_sample_rate(frame.sample_rate);
        context
            .convert_frame(Some(&frame), &mut source_frame)
            .with_context(|| {
                format!(
                    "converting AV_SAMPLE_FMT_FLTP frame into {} frame",
                    get_sample_fmt_name(source_format)
                        .map(|s| s.to_str().unwrap_or_default())
                        .unwrap_or("unknown format")
                )
            })?;

        convert_audio_into_fltp_tensor(&source_frame, &mut dest_tensor)?;

        // N.b. more lenient tolerances due to lossy conversion from float to int16 and back.
        assert!(src_tensor.allclose(&dest_tensor, rtol, atol, false));

        // Test more general `convert_audio_into_tensor` function as well.
        convert_audio_into_tensor(&source_frame, &mut dest_tensor)?;
        assert!(src_tensor.allclose(&dest_tensor, rtol, atol, false));
        Ok(())
    }
}
