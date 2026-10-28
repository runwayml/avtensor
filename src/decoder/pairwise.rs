//! Pairwise motion-compensated interpolation of two RGB frames held as
//! tensors: the [`super::motion`] model (bilateral block matching +
//! overlapped block motion compensation, FFmpeg `minterpolate`'s defaults)
//! applied to a caller-supplied frame pair, outside any decode.
//!
//! The decoder's `fps_conversion="motion"` retimes a whole stream inside the
//! filter graph. A loader that decodes the native frame grid itself and picks
//! frames per target instant (the robot view decoder: pts-based selection,
//! per-view clocks, capture-timestamp sync) cannot use that; it needs the
//! interpolated frame between two frames it already has, at a phase it
//! computes. This is that entry point.
//!
//! Colour: the interpolator works on 8-bit planar YUV, so each RGB frame is
//! converted to `yuv444p` (no chroma subsampling, so the round trip loses
//! nothing but rounding) and back with libswscale, using the BT.709 matrix at
//! limited range both ways -- the decoder's default for untagged content
//! (`SWS_CS_DEFAULT` / `SWS_CS_ITU709` coefficients are identical). A frame
//! that goes through unchanged (`alpha` snapped to 0 or 1 by the caller) is
//! not routed through here, so the round trip only ever touches synthesized
//! frames.
//!
//! Threads: no torch op runs on the pixel data. The input is read through
//! its strides straight into an RGB24 frame and the synthesized frames are
//! written row by row into the output tensor, so the call does not wake
//! torch's intra-op pool (whose workers would otherwise busy-wait through
//! the serial swscale / interpolator work: ~10x the CPU for the same wall
//! time, measured at 480x854). The only torch call is the output allocation.

use anyhow::{anyhow, bail, Context};
use rsmpeg::{avutil::AVFrame, ffi, swscale::SwsContext};

use super::motion::{Interpolator, Layout, PaddedFrame, ALPHA_ONE};

/// `alpha` in `(0, 1)` as the interpolator's phase (`1..ALPHA_ONE`): the
/// phase is quantised to `1 / ALPHA_ONE` (1/256) and clamped to the open
/// interval, so `1e-4` becomes 1/256 and anything at or above 255/256
/// becomes 255/256.
fn phase(alpha: f64) -> Result<u32, anyhow::Error> {
    if !(alpha > 0.0 && alpha < 1.0) || !alpha.is_finite() {
        bail!("alpha must be strictly between 0 and 1, got {alpha}");
    }
    Ok(((alpha * ALPHA_ONE as f64).round() as u32).clamp(1, ALPHA_ONE - 1))
}

/// The geometry of a `uint8` CPU RGB frame tensor, `[3, H, W]` or
/// `[H, W, 3]`, read through its strides: `(height, width, [stride_c,
/// stride_h, stride_w])` in elements.
fn rgb_geometry(t: &tch::Tensor, what: &str) -> Result<(i64, i64, [i64; 3]), anyhow::Error> {
    if t.kind() != tch::Kind::Uint8 {
        bail!("{what}: expected a uint8 tensor, got {:?}", t.kind());
    }
    if t.device() != tch::Device::Cpu {
        bail!("{what}: expected a CPU tensor, got {:?}", t.device());
    }
    let (shape, st) = (t.size(), t.stride());
    if st.iter().any(|&s| s < 0) {
        bail!("{what}: expected non-negative strides, got {st:?}");
    }
    match shape.as_slice() {
        [3, h, w] => Ok((*h, *w, [st[0], st[1], st[2]])),
        [h, w, 3] => Ok((*h, *w, [st[2], st[0], st[1]])),
        _ => bail!("{what}: expected shape [3, H, W] or [H, W, 3], got {shape:?}"),
    }
}

/// Checks that `frame` is a non-empty `rgb24` frame with a non-null plane
/// whose `linesize` covers a row, and returns `(byte span, stride)` of that
/// plane: FFmpeg allocates video frame buffers for at least the full height,
/// so the plane spans at least `stride * (height - 1) + 3 * width` bytes.
fn rgb24_span(frame: &AVFrame) -> Result<(usize, usize), anyhow::Error> {
    let (width, height, linesize) = (frame.width, frame.height, frame.linesize[0]);
    if frame.format != ffi::AV_PIX_FMT_RGB24
        || width <= 0
        || height <= 0
        || frame.data[0].is_null()
        || linesize < 0
        || (linesize as usize) < 3 * width as usize
    {
        bail!("Unexpected rgb24 frame layout: {frame:?}");
    }
    let (width, height, stride) = (width as usize, height as usize, linesize as usize);
    Ok((stride * (height - 1) + 3 * width, stride))
}

/// Borrows the pixel plane of an `rgb24` `frame` as `(bytes, stride)`.
fn rgb24_plane(frame: &AVFrame) -> Result<(&[u8], usize), anyhow::Error> {
    let (len, stride) = rgb24_span(frame)?;
    // SAFETY: `rgb24_span` checked that the plane is non-null and spans at
    // least `len` initialized bytes. The slice borrows `frame`, which holds
    // a reference to the buffer for its lifetime.
    let data = unsafe { std::slice::from_raw_parts(frame.data[0], len) };
    Ok((data, stride))
}

/// Mutably borrows the pixel plane of a writable `rgb24` `frame` as
/// `(bytes, stride)`.
fn rgb24_plane_mut(frame: &mut AVFrame) -> Result<(&mut [u8], usize), anyhow::Error> {
    let (len, stride) = rgb24_span(frame)?;
    if !frame.is_writable().context("Checking frame writability")? {
        bail!("rgb24 frame is not writable");
    }
    let ptr = frame.data[0];
    // SAFETY: as in `rgb24_plane`, the plane spans at least `len` bytes. The
    // frame is writable, so no other frame references its buffer, and the
    // slice borrows `frame` mutably for its lifetime.
    let data = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    Ok((data, stride))
}

/// `t` (see [`rgb_geometry`]) as an RGB24 frame, gathered through the
/// tensor's strides (any layout, contiguous or not) with no torch op.
fn tensor_to_rgb24(t: &tch::Tensor, what: &str) -> Result<AVFrame, anyhow::Error> {
    let (height, width, [sc, sh, sw]) = rgb_geometry(t, what)?;
    if height <= 0 || width <= 0 {
        bail!("{what}: expected a non-empty frame, got {:?}", t.size());
    }
    let (height, width) = (height as usize, width as usize);
    let (sc, sh, sw) = (sc as usize, sh as usize, sw as usize);
    // The largest element offset addressed by any (c, y, x) in range, plus
    // one: the extent of `t`'s storage the copy reads.
    let extent = 2 * sc + (height - 1) * sh + (width - 1) * sw + 1;
    // SAFETY: `rgb_geometry` checked the dtype (one byte per element), the
    // device and that the strides are non-negative, so the first `extent`
    // bytes from the data pointer are elements of `t`, which stays alive
    // (and is only read) for the call.
    let src = unsafe { std::slice::from_raw_parts(t.data_ptr() as *const u8, extent) };

    let mut frame = AVFrame::new();
    frame.set_format(ffi::AV_PIX_FMT_RGB24);
    frame.set_width(i32::try_from(width).context("width")?);
    frame.set_height(i32::try_from(height).context("height")?);
    frame.alloc_buffer().context("Allocating an rgb24 frame")?;
    let (dst, stride) = rgb24_plane_mut(&mut frame)?;
    for y in 0..height {
        let row = &mut dst[y * stride..y * stride + 3 * width];
        let src_row = y * sh;
        if (sc, sw) == (1, 3) {
            // Interleaved [H, W, 3] input: the row is already rgb24.
            row.copy_from_slice(&src[src_row..src_row + 3 * width]);
            continue;
        }
        for (x, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let base = src_row + x * sw;
            px[0] = src[base];
            px[1] = src[base + sc];
            px[2] = src[base + 2 * sc];
        }
    }
    Ok(frame)
}

/// Writes an RGB24 frame into `out[i]` of a contiguous `uint8` `[N, 3, H, W]`
/// tensor, de-interleaving into the three channel planes with no torch op.
fn rgb24_into_chw(frame: &AVFrame, out: &mut tch::Tensor, i: usize) -> Result<(), anyhow::Error> {
    let (src, stride) = rgb24_plane(frame)?;
    let (height, width) = (frame.height as usize, frame.width as usize);
    let n = out.size().first().copied().unwrap_or(0);
    if out.kind() != tch::Kind::Uint8
        || out.device() != tch::Device::Cpu
        || !out.is_contiguous()
        || out.size() != [n, 3, height as i64, width as i64]
        || i64::try_from(i).map_or(true, |i| i >= n)
    {
        bail!(
            "Expected a contiguous uint8 [N > {i}, 3, {height}, {width}] CPU output, got {:?} {:?}",
            out.kind(),
            out.size()
        );
    }
    let plane = height * width;
    // SAFETY: `out` is a contiguous uint8 CPU tensor of `n * 3 * plane`
    // elements (checked above), borrowed mutably for the call, so nothing
    // else reads or writes it while the slice lives.
    let dst = unsafe {
        std::slice::from_raw_parts_mut(out.data_ptr() as *mut u8, n as usize * 3 * plane)
    };
    let (r, gb) = dst[i * 3 * plane..(i + 1) * 3 * plane].split_at_mut(plane);
    let (g, b) = gb.split_at_mut(plane);
    for y in 0..height {
        let src_row = &src[y * stride..y * stride + 3 * width];
        let row = y * width;
        for (x, px) in src_row.as_chunks::<3>().0.iter().enumerate() {
            r[row + x] = px[0];
            g[row + x] = px[1];
            b[row + x] = px[2];
        }
    }
    Ok(())
}

/// A BT.709 limited-range `rgb24 <-> yuv444p` converter of one frame size.
struct Yuv444Converter {
    to_yuv: SwsContext,
    to_rgb: SwsContext,
    width: i32,
    height: i32,
}

impl Yuv444Converter {
    fn new(width: i32, height: i32) -> Result<Self, anyhow::Error> {
        let ctx = |src: ffi::AVPixelFormat, dst: ffi::AVPixelFormat, src_full: bool| {
            let mut c = SwsContext::get_context(
                width,
                height,
                src,
                width,
                height,
                dst,
                ffi::SWS_POINT,
                None,
                None,
                None,
            )
            .ok_or_else(|| anyhow!("sws_getContext failed for {width}x{height} {src} -> {dst}"))?;
            // BT.709 both ways; RGB is full range, YUV limited range.
            unsafe {
                let table = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709 as i32);
                let ret = ffi::sws_setColorspaceDetails(
                    c.as_mut_ptr(),
                    table,
                    src_full as i32,
                    table,
                    (!src_full) as i32,
                    0,
                    1 << 16,
                    1 << 16,
                );
                if ret < 0 {
                    bail!("sws_setColorspaceDetails failed ({ret})");
                }
            }
            Ok::<_, anyhow::Error>(c)
        };
        Ok(Self {
            to_yuv: ctx(ffi::AV_PIX_FMT_RGB24, ffi::AV_PIX_FMT_YUV444P, true)?,
            to_rgb: ctx(ffi::AV_PIX_FMT_YUV444P, ffi::AV_PIX_FMT_RGB24, false)?,
            width,
            height,
        })
    }

    fn blank(&self, format: ffi::AVPixelFormat) -> Result<AVFrame, anyhow::Error> {
        let mut f = AVFrame::new();
        f.set_format(format);
        f.set_width(self.width);
        f.set_height(self.height);
        f.alloc_buffer().context("Allocating a frame")?;
        Ok(f)
    }

    fn rgb_to_yuv(&mut self, rgb: &AVFrame) -> Result<AVFrame, anyhow::Error> {
        let mut yuv = self.blank(ffi::AV_PIX_FMT_YUV444P)?;
        self.to_yuv
            .scale_frame(rgb, 0, self.height, &mut yuv)
            .context("rgb24 -> yuv444p")?;
        Ok(yuv)
    }

    fn yuv_to_rgb(&mut self, yuv: &AVFrame) -> Result<AVFrame, anyhow::Error> {
        let mut rgb = self.blank(ffi::AV_PIX_FMT_RGB24)?;
        self.to_rgb
            .scale_frame(yuv, 0, self.height, &mut rgb)
            .context("yuv444p -> rgb24")?;
        Ok(rgb)
    }
}

/// Frames at phases `alphas` (each strictly inside `(0, 1)`, 0 = `prev`,
/// 1 = `next`) between two RGB frames, by motion-compensated interpolation.
///
/// `prev` / `next`: `uint8` CPU tensors of the same shape, `[3, H, W]` or
/// `[H, W, 3]`, any strides. Returns `uint8` `[len(alphas), 3, H, W]`.
/// Motion is estimated once for the pair; each phase is one compensation
/// pass. Phases are quantised to 1/256 (see [`phase`]). No torch op touches
/// the pixel data (module docs), so the cost is the same whatever the
/// caller's torch thread settings.
pub fn interpolate_rgb_pair(
    prev: &tch::Tensor,
    next: &tch::Tensor,
    alphas: &[f64],
) -> Result<tch::Tensor, anyhow::Error> {
    let phases = alphas
        .iter()
        .map(|&a| phase(a))
        .collect::<Result<Vec<_>, _>>()?;
    let (height, width, _) = rgb_geometry(prev, "prev")?;
    let (h2, w2, _) = rgb_geometry(next, "next")?;
    if (height, width) != (h2, w2) {
        bail!(
            "prev and next must have the same shape, got {:?} vs {:?}",
            prev.size(),
            next.size()
        );
    }
    let mut out = tch::Tensor::empty(
        [phases.len() as i64, 3, height, width],
        (tch::Kind::Uint8, tch::Device::Cpu),
    );
    if phases.is_empty() {
        return Ok(out);
    }
    let (w, h) = (
        i32::try_from(width).context("width")?,
        i32::try_from(height).context("height")?,
    );
    let mut conv = Yuv444Converter::new(w, h)?;
    let yuv_prev = conv.rgb_to_yuv(&tensor_to_rgb24(prev, "prev")?)?;
    let yuv_next = conv.rgb_to_yuv(&tensor_to_rgb24(next, "next")?)?;

    let layout = Layout::new(w, h, ffi::AV_PIX_FMT_YUV444P)?;
    let (mut pa, mut pb) = (PaddedFrame::new(&layout), PaddedFrame::new(&layout));
    pa.fill(super::motion::frame_planes(&yuv_prev, &layout)?);
    pb.fill(super::motion::frame_planes(&yuv_next, &layout)?);
    let mut interp = Interpolator::new(layout);
    interp.estimate(&pa, &pb);

    for (i, &alpha) in phases.iter().enumerate() {
        let mut yuv = conv.blank(ffi::AV_PIX_FMT_YUV444P)?;
        interp.interpolate(
            &pa,
            &pb,
            alpha,
            super::motion::frame_planes_mut(&mut yuv, &layout)?,
        );
        rgb24_into_chw(&conv.yuv_to_rgb(&yuv)?, &mut out, i)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The luma-ish test pattern of `motion.rs`, as an RGB frame translated
    /// by `(dx, dy)`: each channel a shifted copy so the interpolator sees
    /// structure in every plane after the YUV conversion.
    fn rgb_at(width: i64, height: i64, dx: f64, dy: f64) -> tch::Tensor {
        let pattern = |x: f64, y: f64| -> f64 {
            128.0
                + 50.0 * (x * 0.11 + y * 0.03).sin()
                + 40.0 * (y * 0.13 - x * 0.05).cos()
                + 25.0 * ((x * 0.07) * (y * 0.021 + 1.0)).sin()
        };
        let mut data = vec![0u8; (3 * width * height) as usize];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let v = pattern(x as f64 - dx + 9.0 * c as f64, y as f64 - dy);
                    data[((c * height + y) * width + x) as usize] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }
        tch::Tensor::from_slice(&data).view([3, height, width])
    }

    /// Mean absolute error between two [3, H, W] frames away from the borders.
    fn interior_mae(a: &tch::Tensor, b: &tch::Tensor, margin: i64) -> f64 {
        let (h, w) = (a.size()[1], a.size()[2]);
        let sl = |t: &tch::Tensor| {
            t.narrow(1, margin, h - 2 * margin)
                .narrow(2, margin, w - 2 * margin)
                .to_kind(tch::Kind::Float)
        };
        (sl(a) - sl(b))
            .abs()
            .mean(tch::Kind::Float)
            .double_value(&[])
    }

    #[test]
    fn test_midpoint_of_a_translation_is_the_half_shifted_pattern() {
        let (w, h) = (64, 64);
        let prev = rgb_at(w, h, 0.0, 0.0);
        let next = rgb_at(w, h, 8.0, 0.0);
        let truth = rgb_at(w, h, 4.0, 0.0);
        let out = interpolate_rgb_pair(&prev, &next, &[0.5]).unwrap();
        assert_eq!(out.size(), [1, 3, h, w]);
        let mid = out.get(0);
        let mae = interior_mae(&mid, &truth, 12);
        let mae_prev = interior_mae(&prev, &truth, 12);
        let mae_next = interior_mae(&next, &truth, 12);
        assert!(
            mae < mae_prev && mae < mae_next && mae < 4.0,
            "mae {mae:.2} vs prev {mae_prev:.2} / next {mae_next:.2}"
        );
    }

    #[test]
    fn test_accepts_hwc_and_rejects_bad_input() {
        let prev = rgb_at(32, 16, 0.0, 0.0);
        let next = rgb_at(32, 16, 2.0, 1.0);
        let out = interpolate_rgb_pair(
            &prev.permute([1, 2, 0]).contiguous(),
            &next.permute([1, 2, 0]).contiguous(),
            &[0.25, 0.75],
        )
        .unwrap();
        assert_eq!(out.size(), [2, 3, 16, 32]);
        assert!(interpolate_rgb_pair(&prev, &next, &[0.0]).is_err());
        assert!(interpolate_rgb_pair(&prev, &next, &[1.0]).is_err());
        assert!(interpolate_rgb_pair(&prev, &rgb_at(16, 16, 0.0, 0.0), &[0.5]).is_err());
        assert!(interpolate_rgb_pair(&prev.to_kind(tch::Kind::Float), &next, &[0.5]).is_err());
        assert_eq!(
            interpolate_rgb_pair(&prev, &next, &[]).unwrap().size(),
            [0, 3, 16, 32]
        );
    }

    /// The strided reader: a contiguous `[3, H, W]`, its `[H, W, 3]` permute
    /// (non-contiguous view, never materialised) and a narrowed view of a
    /// wider frame all give byte-identical output to the explicit-indexing
    /// reference, and the output planes are the de-interleaved RGB24 rows.
    #[test]
    fn test_any_input_layout_gives_identical_output() {
        let (w, h) = (40, 24);
        let prev = rgb_at(w, h, 0.0, 0.0);
        let next = rgb_at(w, h, 6.0, 2.0);
        let base = interpolate_rgb_pair(&prev, &next, &[0.5]).unwrap();
        // [H, W, 3] view, non-contiguous (strides [W, 1, H*W]).
        let (p_hwc, n_hwc) = (prev.permute([1, 2, 0]), next.permute([1, 2, 0]));
        assert!(!p_hwc.is_contiguous());
        let hwc = interpolate_rgb_pair(&p_hwc, &n_hwc, &[0.5]).unwrap();
        assert!(base.equal(&hwc));
        // A narrowed window of a wider [3, H, W+16] frame (row stride != W).
        let wide = |t: &tch::Tensor| {
            tch::Tensor::cat(
                &[
                    t,
                    &tch::Tensor::zeros([3, h, 16], (tch::Kind::Uint8, tch::Device::Cpu)),
                ],
                2,
            )
        };
        let (p_w, n_w) = (wide(&prev).narrow(2, 0, w), wide(&next).narrow(2, 0, w));
        assert!(!p_w.is_contiguous());
        let narrowed = interpolate_rgb_pair(&p_w, &n_w, &[0.5]).unwrap();
        assert!(base.equal(&narrowed));
        // The reader itself against explicit indexing.
        let frame = tensor_to_rgb24(&p_hwc, "prev").unwrap();
        let (plane, ls) = rgb24_plane(&frame).unwrap();
        for &(y, x, c) in &[(0i64, 0i64, 0i64), (5, 7, 1), (h - 1, w - 1, 2), (3, 20, 0)] {
            let want = prev.int64_value(&[c, y, x]) as u8;
            let got = plane[y as usize * ls + 3 * x as usize + c as usize];
            assert_eq!(got, want, "({c},{y},{x})");
        }
    }

    #[test]
    fn test_phase_is_quantised_to_1_over_256_and_clamped() {
        assert_eq!(phase(1e-4).unwrap(), 1);
        assert_eq!(phase(0.5).unwrap(), 128);
        assert_eq!(phase(255.0 / 256.0).unwrap(), 255);
        assert_eq!(phase(0.9999).unwrap(), 255);
        assert_eq!(phase(1.0 / 256.0 + 1e-9).unwrap(), 1);
        assert!(phase(0.0).is_err() && phase(1.0).is_err() && phase(f64::NAN).is_err());
    }
}
