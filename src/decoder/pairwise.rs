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

use anyhow::{anyhow, bail, Context};
use rsmpeg::{avutil::AVFrame, ffi, swscale::SwsContext};

use super::motion::{Interpolator, Layout, PaddedFrame, ALPHA_ONE};
use crate::conversion::{convert_rgb24_frame_to_tensor, convert_rgb24_tensor_to_avframe};

/// `alpha` in `(0, 1)` as the interpolator's phase (`1..ALPHA_ONE`).
fn phase(alpha: f64) -> Result<u32, anyhow::Error> {
    if !(alpha > 0.0 && alpha < 1.0) || !alpha.is_finite() {
        bail!("alpha must be strictly between 0 and 1, got {alpha}");
    }
    Ok(((alpha * ALPHA_ONE as f64).round() as u32).clamp(1, ALPHA_ONE - 1))
}

/// An RGB frame tensor (`uint8`, `[3, H, W]` or `[H, W, 3]`, CPU) as a
/// contiguous `[H, W, 3]` tensor.
fn as_hwc(t: &tch::Tensor, what: &str) -> Result<tch::Tensor, anyhow::Error> {
    if t.kind() != tch::Kind::Uint8 {
        bail!("{what}: expected a uint8 tensor, got {:?}", t.kind());
    }
    if t.device() != tch::Device::Cpu {
        bail!("{what}: expected a CPU tensor, got {:?}", t.device());
    }
    let shape = t.size();
    let hwc = match shape.as_slice() {
        [3, _, _] => t.permute([1, 2, 0]),
        [_, _, 3] => t.shallow_clone(),
        _ => bail!("{what}: expected shape [3, H, W] or [H, W, 3], got {shape:?}"),
    };
    Ok(hwc.contiguous())
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
/// `[H, W, 3]`. Returns `uint8` `[len(alphas), 3, H, W]`. Motion is
/// estimated once for the pair; each phase is one compensation pass.
pub fn interpolate_rgb_pair(
    prev: &tch::Tensor,
    next: &tch::Tensor,
    alphas: &[f64],
) -> Result<tch::Tensor, anyhow::Error> {
    let phases = alphas
        .iter()
        .map(|&a| phase(a))
        .collect::<Result<Vec<_>, _>>()?;
    let (a, b) = (as_hwc(prev, "prev")?, as_hwc(next, "next")?);
    if a.size() != b.size() {
        bail!(
            "prev and next must have the same shape, got {:?} vs {:?}",
            prev.size(),
            next.size()
        );
    }
    let (height, width) = (a.size()[0], a.size()[1]);
    let out = tch::Tensor::empty(
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
    let yuv_prev = conv.rgb_to_yuv(&convert_rgb24_tensor_to_avframe(&a)?)?;
    let yuv_next = conv.rgb_to_yuv(&convert_rgb24_tensor_to_avframe(&b)?)?;

    let layout = Layout::new(w, h, ffi::AV_PIX_FMT_YUV444P)?;
    let (mut pa, mut pb) = (PaddedFrame::new(&layout), PaddedFrame::new(&layout));
    pa.fill(super::motion::frame_planes(&yuv_prev, &layout)?);
    pb.fill(super::motion::frame_planes(&yuv_next, &layout)?);
    let mut interp = Interpolator::new(layout);
    interp.estimate(&pa, &pb);

    // One [H, W, 3] staging tensor; the output is channels-first.
    let staging = tch::Tensor::empty([height, width, 3], (tch::Kind::Uint8, tch::Device::Cpu));
    for (i, &alpha) in phases.iter().enumerate() {
        let mut yuv = conv.blank(ffi::AV_PIX_FMT_YUV444P)?;
        interp.interpolate(
            &pa,
            &pb,
            alpha,
            super::motion::frame_planes_mut(&mut yuv, &layout)?,
        );
        let rgb = conv.yuv_to_rgb(&yuv)?;
        let mut dst = staging.shallow_clone();
        convert_rgb24_frame_to_tensor(&rgb, &mut dst)?;
        out.get(i as i64).copy_(&staging.permute([2, 0, 1]));
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
}
