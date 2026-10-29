//! Scene-cut detection for the interpolating frame-rate conversions
//! ([`super::FpsConversion::Blend`] and [`super::FpsConversion::Motion`]).
//!
//! Interpolating across a hard cut produces a frame that belongs to neither
//! shot (a cross-fade or a warped morph), so the resampler takes the nearer
//! source frame for every output frame between a detected cut's two sides.
//!
//! A fixed threshold on the mean frame difference (FFmpeg's `scd=fdiff`, the
//! detector `minterpolate` and `framerate` use) cannot separate cuts from fast
//! motion: whip pans and handheld action differ as much as many cuts, and cuts
//! between similar shots differ less. Measured over 22 minutes of 50-60 fps
//! footage it either fired hundreds of times on motion or missed cuts. This
//! detector instead works in two stages on a ~120 px wide luma thumbnail
//! (box-averaged from the frame's luma plane):
//!
//! 1. Spike: the thumbnail difference `d` of the pair must stand out against
//!    the recent motion level: `d >= SPIKE_MIN_DIFF`, `d >= SPIKE_VS_PREV *`
//!    the previous pair's difference (a cut is a one-pair spike; sustained
//!    motion keeps neighbours high), and `d >= SPIKE_VS_MEDIAN_LOOSE *` the
//!    median over the last `HISTORY` pairs. Repeated frames (`d <
//!    REPEAT_DIFF`, e.g. 30p content in a 60 fps file) are skipped, so they
//!    neither count as quiet history nor hide the next pair's spike.
//! 2. Confirmation, only for spikes: split the earlier thumbnail into 8x8
//!    blocks and find, for each, the best zero-mean normalized correlation
//!    against the later thumbnail within +-4 px. Within a shot most blocks
//!    find their content again (motion, exposure and lighting changes keep
//!    local structure), so the median stays high; across a cut it drops. A cut
//!    needs a median below `NCC_CUT` together with a strong spike
//!    (`SPIKE_VS_MEDIAN`), or below `NCC_CUT_STRICT` with any spike. A spike
//!    of `JUMP_SPIKE` times both the median and the previous pair is a cut
//!    below `NCC_JUMP`: cuts between two angles of one scene, and jump cuts,
//!    keep much of their local structure but are still discontinuities that
//!    interpolation cannot bridge. Until `MIN_HISTORY` pairs have been seen
//!    (start of a stream or seek window) only the strict test applies.
//!
//! Everything is causal (it looks only at the pair and the frames before it),
//! so the resampler can decide when the pair's second frame arrives.

use std::collections::VecDeque;

use super::motion::Plane;

/// Target thumbnail width; the box filter is the largest integer factor that
/// keeps at least this many columns.
const THUMB_WIDTH: usize = 120;
/// Pairs whose thumbnail difference is below this are repeated frames.
const REPEAT_DIFF: f32 = 0.5;
/// Recent (non-repeated) pair differences the median is taken over.
const HISTORY: usize = 32;
/// Smallest mean thumbnail difference (8-bit levels) that can be a cut.
const SPIKE_MIN_DIFF: f32 = 8.0;
/// A cut's difference over the previous pair's.
const SPIKE_VS_PREV: f32 = 2.0;
/// A cut's difference over the recent median, with a moderate correlation
/// drop ([`NCC_CUT`]).
const SPIKE_VS_MEDIAN: f32 = 3.0;
/// A cut's difference over the recent median, with a strong correlation
/// drop ([`NCC_CUT_STRICT`]).
const SPIKE_VS_MEDIAN_LOOSE: f32 = 2.0;
/// A cut's difference over both the recent median and the previous pair
/// for a discontinuity in one scene (another angle, a jump cut), which only
/// needs a slight correlation drop ([`NCC_JUMP`]).
const JUMP_SPIKE: f32 = 6.0;
/// Median block correlation below which a [`JUMP_SPIKE`] is a cut.
const NCC_JUMP: f32 = 0.95;
/// Pairs of history below which the median is not trusted and only the
/// strict correlation test applies (start of a stream or seek window).
const MIN_HISTORY: usize = 2;
/// Floor for the difference denominators.
const DIFF_FLOOR: f32 = 0.25;
/// Median block correlation below which a strong spike is a cut.
const NCC_CUT: f32 = 0.8;
/// Median block correlation below which any spike is a cut.
const NCC_CUT_STRICT: f32 = 0.5;
/// Confirmation block size and search radius (thumbnail pixels).
const BLOCK: usize = 8;
const SEARCH: usize = 4;
/// Blocks whose zero-mean L2 norm is below this in both frames are flat and
/// count as matching (nothing to correlate).
const FLAT_NORM: f32 = 20.0;

/// Detects scene cuts between consecutive frames of one stream; see the
/// module documentation.
pub(crate) struct CutDetector {
    width: usize,
    height: usize,
    /// Box-filter factor from the frame to the thumbnail.
    factor: usize,
    thumb_w: usize,
    thumb_h: usize,
    prev: Option<Vec<f32>>,
    /// Recent non-repeated pair differences, oldest first.
    history: VecDeque<f32>,
}

impl CutDetector {
    /// A detector for frames whose luma plane is `width` x `height`.
    pub fn new(width: usize, height: usize) -> Self {
        let factor = (width / THUMB_WIDTH).max(1);
        Self {
            width,
            height,
            factor,
            thumb_w: (width / factor).max(1),
            thumb_h: (height / factor).max(1),
            prev: None,
            history: VecDeque::with_capacity(HISTORY + 1),
        }
    }

    /// Forgets the previous frame and the motion history (after a timestamp
    /// discontinuity).
    pub fn reset(&mut self) {
        self.prev = None;
        self.history.clear();
    }

    /// Feeds the next frame's luma plane; returns whether a cut separates it
    /// from the frame fed before (always false for the first frame).
    pub fn push(&mut self, luma: Plane) -> bool {
        let thumb = self.thumbnail(luma);
        let cut = match self.prev.take() {
            Some(prev) => self.is_cut(&prev, &thumb),
            None => false,
        };
        self.prev = Some(thumb);
        cut
    }

    fn is_cut(&mut self, prev: &[f32], next: &[f32]) -> bool {
        let d = prev
            .iter()
            .zip(next)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / prev.len() as f32;
        if d < REPEAT_DIFF {
            return false;
        }
        let last = self.history.back().copied().unwrap_or(0.0);
        let warm = self.history.len() >= MIN_HISTORY;
        let median = median(self.history.iter().copied().collect());
        self.history.push_back(d);
        if self.history.len() > HISTORY {
            self.history.pop_front();
        }
        let vs_median = d / median.max(DIFF_FLOOR);
        let vs_prev = d / last.max(DIFF_FLOOR);
        if d < SPIKE_MIN_DIFF || vs_prev < SPIKE_VS_PREV || vs_median < SPIKE_VS_MEDIAN_LOOSE {
            return false;
        }
        match self.block_correlation(prev, next) {
            Some(ncc) => {
                let jump = vs_median.min(vs_prev) >= JUMP_SPIKE;
                ncc < NCC_CUT_STRICT
                    || (warm && vs_median >= SPIKE_VS_MEDIAN && ncc < NCC_CUT)
                    || (warm && jump && ncc < NCC_JUMP)
            }
            // Too small to confirm: a strong spike alone decides (a rare
            // false cut costs one uneven step, a missed one a morph).
            None => warm && vs_median >= SPIKE_VS_MEDIAN,
        }
    }

    /// Box-filtered luma thumbnail (`thumb_w` x `thumb_h`, row-major).
    fn thumbnail(&self, luma: Plane) -> Vec<f32> {
        let (f, tw, th) = (self.factor, self.thumb_w, self.thumb_h);
        let mut sums = vec![0u32; tw * th];
        for y in 0..(th * f).min(self.height) {
            let row = &luma.data[y * luma.stride..][..(tw * f).min(self.width)];
            let out = &mut sums[(y / f) * tw..][..tw];
            for (chunk, sum) in row.chunks(f).zip(out.iter_mut()) {
                *sum += chunk.iter().map(|&v| v as u32).sum::<u32>();
            }
        }
        let scale = 1.0 / (f * f) as f32;
        sums.into_iter().map(|s| s as f32 * scale).collect()
    }

    /// Median over `prev`'s blocks of the best zero-mean normalized
    /// correlation with `next` within the search radius; `None` when the
    /// thumbnail is too small to hold a block plus the search margin.
    fn block_correlation(&self, prev: &[f32], next: &[f32]) -> Option<f32> {
        let (w, h) = (self.thumb_w, self.thumb_h);
        if w < BLOCK + 2 * SEARCH || h < BLOCK + 2 * SEARCH {
            return None;
        }
        let (blocks_x, blocks_y) = ((w - 2 * SEARCH) / BLOCK, (h - 2 * SEARCH) / BLOCK);
        // Block at (x, y) as zero-mean values and their L2 norm.
        let block = |img: &[f32], x: usize, y: usize| {
            let mut v = [0f32; BLOCK * BLOCK];
            for (r, row) in v.chunks_mut(BLOCK).enumerate() {
                row.copy_from_slice(&img[(y + r) * w + x..][..BLOCK]);
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter_mut().for_each(|p| *p -= mean);
            let norm = v.iter().map(|p| p * p).sum::<f32>().sqrt();
            (v, norm)
        };
        let mut best = Vec::with_capacity(blocks_x * blocks_y);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let (x, y) = (SEARCH + bx * BLOCK, SEARCH + by * BLOCK);
                let (a, na) = block(prev, x, y);
                let mut score = -1f32;
                for sy in y - SEARCH..=y + SEARCH {
                    for sx in x - SEARCH..=x + SEARCH {
                        let (b, nb) = block(next, sx, sy);
                        let c = if na < FLAT_NORM && nb < FLAT_NORM {
                            1.0
                        } else {
                            a.iter().zip(&b).map(|(p, q)| p * q).sum::<f32>() / (na * nb).max(1e-6)
                        };
                        score = score.max(c);
                    }
                }
                best.push(score);
            }
        }
        Some(median(best))
    }
}

/// Median of `values` (mean of the middle two for an even count); 0 when empty.
fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textured test image: bilinear value noise on a 16 px lattice plus a
    /// smooth gradient; `seed` selects an unrelated image.
    fn pattern(x: f64, y: f64, seed: f64) -> u8 {
        let hash = |i: i64, j: i64| {
            let mut h = (i.wrapping_mul(73_856_093) ^ j.wrapping_mul(19_349_663))
                .wrapping_add((seed * 1000.0) as i64 * 83_492_791) as u64;
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1_e995);
            h ^= h >> 15;
            (h % 256) as f64
        };
        let (gx, gy) = (x / 16.0, y / 16.0);
        let (i, j) = (gx.floor() as i64, gy.floor() as i64);
        let (fx, fy) = (gx - gx.floor(), gy - gy.floor());
        let top = hash(i, j) * (1.0 - fx) + hash(i + 1, j) * fx;
        let bottom = hash(i, j + 1) * (1.0 - fx) + hash(i + 1, j + 1) * fx;
        let noise = top * (1.0 - fy) + bottom * fy;
        (0.6 * noise + 0.2 * (x * 0.01 + seed).sin() * 128.0 + 50.0).clamp(0.0, 255.0) as u8
    }

    fn image(w: usize, h: usize, f: impl Fn(f64, f64) -> u8) -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| f(x as f64, y as f64))
            .collect()
    }

    fn push(det: &mut CutDetector, img: &[u8], w: usize) -> bool {
        det.push(Plane {
            data: img,
            stride: w,
        })
    }

    #[test]
    fn test_hard_cut_is_detected() {
        let (w, h) = (480, 270);
        let mut det = CutDetector::new(w, h);
        let mut cuts = Vec::new();
        for i in 0..40 {
            let seed = if i < 20 { 0.0 } else { 1.7 };
            let img = image(w, h, |x, y| pattern(x + 2.0 * i as f64, y, seed));
            cuts.push(push(&mut det, &img, w));
        }
        let at: Vec<usize> = (0..cuts.len()).filter(|&i| cuts[i]).collect();
        assert_eq!(at, [20]);
    }

    /// A cut soon after the start of the stream (or a seek), with little
    /// motion history.
    #[test]
    fn test_cut_with_short_history() {
        let (w, h) = (256, 144);
        let mut det = CutDetector::new(w, h);
        for i in 0..=MIN_HISTORY {
            let img = image(w, h, |x, y| pattern(x + 2.0 * i as f64, y, 0.0));
            assert!(!push(&mut det, &img, w));
        }
        assert!(push(&mut det, &image(w, h, |x, y| pattern(x, y, 2.3)), w));
    }

    /// Fast motion (a 12 px/frame pan at 480 px, from the first frame on,
    /// with a brightness ramp) and a hard lighting change within one shot
    /// are not cuts.
    #[test]
    fn test_fast_motion_and_flash_are_not_cuts() {
        let (w, h) = (480, 270);
        let mut det = CutDetector::new(w, h);
        for i in 0..60 {
            let gain = if i == 30 { 1.6 } else { 1.0 + 0.004 * i as f64 };
            let img = image(w, h, |x, y| {
                (pattern(x + 12.0 * i as f64, y + 3.0 * i as f64, 0.0) as f64 * gain).min(255.0)
                    as u8
            });
            assert!(!push(&mut det, &img, w), "false cut at frame {i}");
        }
    }

    /// Repeated frames (30p content in a 60 fps stream) neither fire nor
    /// mask a following cut.
    #[test]
    fn test_repeated_frames() {
        let (w, h) = (480, 270);
        let mut det = CutDetector::new(w, h);
        let mut cuts = Vec::new();
        for i in 0..40 {
            let (t, seed) = ((i / 2) as f64, if i < 21 { 0.0 } else { 1.1 });
            let img = image(w, h, |x, y| pattern(x + 2.0 * t, y, seed));
            cuts.push(push(&mut det, &img, w));
        }
        let at: Vec<usize> = (0..cuts.len()).filter(|&i| cuts[i]).collect();
        assert_eq!(at, [21]);
    }

    /// Thumbnails too small for the block search fall back to the spike test.
    #[test]
    fn test_tiny_frames() {
        let (w, h) = (12, 10);
        let mut det = CutDetector::new(w, h);
        for v in 40..43 {
            assert!(!push(&mut det, &vec![v; w * h], w));
        }
        assert!(push(&mut det, &vec![200; w * h], w));
    }

    #[test]
    fn test_reset_forgets_previous_frame() {
        let (w, h) = (256, 144);
        let mut det = CutDetector::new(w, h);
        push(&mut det, &image(w, h, |x, y| pattern(x, y, 0.0)), w);
        det.reset();
        assert!(!push(&mut det, &image(w, h, |x, y| pattern(x, y, 2.3)), w));
    }

    #[test]
    fn test_median() {
        assert_eq!(median(vec![]), 0.0);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
