//! Constant-frame-rate output as the ffmpeg CLI produces it.
//!
//! The `fps` filter keeps the source frame nearest each output timestamp. The
//! ffmpeg CLI's default output sync (`-fps_mode cfr`, chosen for raw-video
//! outputs) decides differently: each decoded frame takes the next output
//! slot, and is dropped or duplicated by how far its timestamp has drifted
//! from that slot. For variable-frame-rate input the two pick different
//! frames, so tools that read video through the CLI (e.g. imageio) and this
//! crate's `fps` resampling disagree. [`CfrSync`] ports the CLI's decision
//! (`video_sync_process` in FFmpeg 7.1's `fftools/ffmpeg_filter.c`) so
//! decoded output can match the CLI frame for frame.

use rsmpeg::ffi::AVRational;

/// The CLI skips a frame whose duplication would exceed this
/// (`dts_error_threshold * 30`, with the default threshold of 3600 * 30).
const MAX_FRAME_COPIES: i64 = 3600 * 30 * 30;

/// The CLI's `dts_delta_threshold` (seconds): in formats with timestamp
/// discontinuities, a larger jump in dts is one.
const DTS_DELTA_THRESHOLD: f32 = 10.0;

/// The CLI's `dts_error_threshold` (seconds): in other formats, a timestamp
/// further than this from the predicted dts is dropped.
const DTS_ERROR_THRESHOLD: f32 = 3600.0 * 30.0;

const AV_TIME_BASE: i64 = 1_000_000;
pub const AV_TIME_BASE_Q: AVRational = AVRational {
    num: 1,
    den: AV_TIME_BASE as i32,
};

/// What to emit for one decoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncDecision {
    /// Copies of the previous frame to emit first.
    pub previous: i64,
    /// Copies of this frame to emit after them.
    pub current: i64,
}

/// `av_rescale_q`: `value * src / dst`, rounded half away from zero.
pub fn rescale(value: i64, src: AVRational, dst: AVRational) -> i64 {
    let num = value as i128 * src.num as i128 * dst.den as i128;
    let den = src.den as i128 * dst.num as i128;
    let rounded = (num.abs() + den / 2) / den;
    (if num >= 0 { rounded } else { -rounded }) as i64
}

/// C's `llrintf`: round to `f32`, then to the nearest integer, ties to even.
fn llrintf(value: f64) -> i64 {
    (value as f32).round_ties_even() as i64
}

/// The CLI's CFR output state for one video stream.
pub struct CfrSync {
    /// Output frame duration (one tick per output frame).
    out_tb: AVRational,
    /// Input stream time base.
    stream_tb: AVRational,
    /// Offset (stream time base) that makes the file start at 0, as the CLI's
    /// input `ts_offset` does.
    offset: i64,
    /// `VSYNC_VSCFR`: with no other streams in the file, the CLI does not pad
    /// a late first frame with duplicates.
    skip_initial_gap: bool,
    /// Next output slot, in output ticks.
    next_slot: i64,
    /// Slot of the first emitted frame (non-zero when a late first frame is
    /// not padded).
    first_slot: i64,
    /// Frames emitted so far.
    emitted: i64,
    /// Last three `previous` counts (the end-of-stream repeat estimate).
    history: [i64; 3],
    /// Timestamp prediction for frames that have none.
    last_pts: Option<i64>,
    last_duration: i64,
}

impl CfrSync {
    /// `rate` is the output frame rate (the stream's guessed frame rate, as
    /// the CLI uses); `start_time` is the file's start, in microseconds.
    pub fn new(
        rate: AVRational,
        stream_tb: AVRational,
        start_time: i64,
        single_stream: bool,
    ) -> Self {
        let out_tb = AVRational {
            num: rate.den,
            den: rate.num,
        };
        let offset = rescale(
            -start_time,
            AVRational {
                num: 1,
                den: 1_000_000,
            },
            stream_tb,
        );
        Self {
            out_tb,
            stream_tb,
            offset,
            skip_initial_gap: single_stream,
            next_slot: 0,
            first_slot: 0,
            emitted: 0,
            history: [0; 3],
            last_pts: None,
            last_duration: 0,
        }
    }

    /// The decision for a frame with best-effort timestamp `pts` and duration
    /// `duration` (stream time base; 0 when unknown).
    pub fn push(&mut self, pts: Option<i64>, duration: i64) -> SyncDecision {
        // No timestamp: extrapolate from the previous frame.
        let pts = pts.unwrap_or_else(|| {
            self.last_pts
                .map_or(0, |last| last + self.last_duration.max(1))
        });
        let ts_diff = self.last_pts.map_or(-1, |last| pts - last);
        self.last_duration = if duration > 0 {
            duration
        } else {
            ts_diff.max(1)
        };
        self.last_pts = Some(pts);

        let duration = if duration > 0 {
            duration
        } else {
            // A frame without a duration lasts one output frame.
            rescale(1, self.out_tb, self.stream_tb)
        };

        // adjust_frame_pts_to_encoder_tb: the frame's position in output ticks.
        let extra_bits = (29 - (31 - (self.out_tb.den as u32).leading_zeros() as i32)).clamp(0, 16);
        let fine_tb = AVRational {
            num: self.out_tb.num,
            den: self.out_tb.den << extra_bits,
        };
        let mut exact = rescale(pts + self.offset, self.stream_tb, fine_tb) as f64
            / (1i64 << extra_bits) as f64;
        if exact != exact.round_ties_even() {
            exact += exact.signum() / (1 << 17) as f64;
        }
        let mut duration = duration as f64
            * (self.stream_tb.num as f64 / self.stream_tb.den as f64)
            / (self.out_tb.num as f64 / self.out_tb.den as f64);

        let mut delta0 = exact - self.next_slot as f64;
        let mut delta = delta0 + duration;
        let mut previous = 0;
        let mut current = 1;
        if delta0 < 0.0 && delta > 0.0 {
            duration += delta0;
            delta0 = 0.0;
        }
        if self.skip_initial_gap && self.emitted == 0 && delta0 >= 0.5 {
            delta = duration;
            delta0 = 0.0;
            self.next_slot = exact.round_ties_even() as i64;
            self.first_slot = self.next_slot;
        }
        if delta < -1.1 {
            current = 0;
        } else if delta > 1.1 {
            current = llrintf(delta);
            if delta0 > 1.1 {
                previous = llrintf(delta0 - 0.6);
            }
        }
        previous = previous.min(current);
        self.history = [previous, self.history[0], self.history[1]];
        if current > MAX_FRAME_COPIES {
            log::warn!("Skipping a frame whose duplication is too large ({current} copies)");
            return SyncDecision {
                previous: 0,
                current: 0,
            };
        }
        self.next_slot += current;
        self.emitted += current;
        SyncDecision {
            previous,
            current: current - previous,
        }
    }

    /// Copies of the last frame to emit at the end of the stream: as many as
    /// recent frames' duplicates (the median of the last three).
    pub fn finish(&mut self) -> i64 {
        let mut h = self.history;
        h.sort_unstable();
        let copies = h[1];
        if copies > MAX_FRAME_COPIES {
            return 0;
        }
        self.next_slot += copies;
        self.emitted += copies;
        copies
    }

    /// Output timestamp (seconds) of the `index`-th emitted frame.
    pub fn pts_seconds_of(&self, index: i64) -> f64 {
        (self.first_slot + index) as f64 * self.out_tb.num as f64 / self.out_tb.den as f64
    }
}

/// `FFABS(delta) > threshold * AV_TIME_BASE`, compared in `float` as the C is.
fn exceeds(delta: i64, threshold: f32) -> bool {
    delta.abs() as f32 > threshold * AV_TIME_BASE as f32
}

/// The CLI's input timestamp fixes for one video stream's packets, applied
/// before they are decoded (`ts_fixup` in FFmpeg 7.1's
/// `fftools/ffmpeg_demux.c`). In formats with timestamp discontinuities
/// (MPEG-TS/PS, HLS: `AVFMT_TS_DISCONT`), a dts more than 10 s off the
/// predicted one, or more than 0.1 s behind the previous one, is a
/// discontinuity: it and every later timestamp are shifted to continue from
/// the prediction. In other formats a timestamp more than 30 h off the
/// prediction is dropped, and the decoder extrapolates one.
pub struct DemuxFixup {
    tb: AVRational,
    /// The input `ts_offset` (as in [`CfrSync`]), in `tb`. Detection runs on
    /// shifted timestamps, as the CLI's does; the packets returned are not
    /// shifted, since [`CfrSync`] shifts frames itself.
    ts_offset: i64,
    ts_discont: bool,
    /// `ts_offset_discont`: the shifts so far (microseconds).
    discont_offset: i64,
    /// Predicted dts of the next packet (microseconds).
    next_dts: Option<i64>,
    /// dts of the previous packet (microseconds); before the first one, the
    /// decoder-delay estimate a first packet without a dts starts from.
    dts: i64,
    saw_first: bool,
    /// Fields per second (twice the codec frame rate), to predict the next
    /// dts after a packet without a duration.
    field_rate: Option<AVRational>,
}

impl DemuxFixup {
    /// `start_time` as in [`CfrSync::new`]; `ts_discont` is whether the
    /// format has `AVFMT_TS_DISCONT`; `initial_dts` is the CLI's first-dts
    /// estimate (`-video_delay / avg_frame_rate`, microseconds).
    pub fn new(
        tb: AVRational,
        start_time: i64,
        ts_discont: bool,
        initial_dts: i64,
        field_rate: Option<AVRational>,
    ) -> Self {
        Self {
            tb,
            ts_offset: rescale(-start_time, AV_TIME_BASE_Q, tb),
            ts_discont,
            discont_offset: 0,
            next_dts: None,
            dts: initial_dts,
            saw_first: false,
            field_rate,
        }
    }

    /// The fixed `(pts, dts)` of a packet with `pts`, `dts` and `duration`
    /// (`tb`; `None` when unset). `fields` is the number of fields the
    /// packet holds (2 unless the parser reports repeats), used only when it
    /// has no duration.
    pub fn fix(
        &mut self,
        pts: Option<i64>,
        dts: Option<i64>,
        duration: i64,
        fields: i64,
    ) -> (Option<i64>, Option<i64>) {
        // ts_discontinuity_process: apply the shifts found so far.
        let shift = self.ts_offset + rescale(self.discont_offset, AV_TIME_BASE_Q, self.tb);
        let mut pts = pts.map(|pts| pts + shift);
        let mut dts = dts.map(|dts| dts + shift);

        // ts_discontinuity_detect
        if let (Some(packet_dts), Some(next_dts)) = (dts, self.next_dts) {
            let dts_us = rescale(packet_dts, self.tb, AV_TIME_BASE_Q);
            let delta = dts_us - next_dts;
            if self.ts_discont {
                if exceeds(delta, DTS_DELTA_THRESHOLD) || dts_us + AV_TIME_BASE / 10 < self.dts {
                    self.discont_offset -= delta;
                    log::warn!(
                        "timestamp discontinuity: {delta} us, new offset {} us",
                        self.discont_offset
                    );
                    let correction = rescale(delta, AV_TIME_BASE_Q, self.tb);
                    dts = Some(packet_dts - correction);
                    pts = pts.map(|pts| pts - correction);
                }
            } else {
                if exceeds(delta, DTS_ERROR_THRESHOLD) {
                    log::warn!("dropping a dts {delta} us off the prediction");
                    dts = None;
                }
                if pts.is_some_and(|pts| {
                    exceeds(
                        rescale(pts, self.tb, AV_TIME_BASE_Q) - next_dts,
                        DTS_ERROR_THRESHOLD,
                    )
                }) {
                    log::warn!("dropping a pts too far off the predicted dts");
                    pts = None;
                }
            }
        }

        // ist_dts_update: predict the next packet's dts.
        if !self.saw_first {
            self.dts += pts.map_or(0, |pts| rescale(pts, self.tb, AV_TIME_BASE_Q));
            self.saw_first = true;
        }
        let mut next = self.next_dts.unwrap_or(self.dts);
        if let Some(dts) = dts {
            next = rescale(dts, self.tb, AV_TIME_BASE_Q);
        }
        self.dts = next;
        if duration != 0 {
            next += rescale(duration, self.tb, AV_TIME_BASE_Q);
        } else if let Some(rate) = self.field_rate {
            let field = AVRational {
                num: rate.den,
                den: rate.num,
            };
            next += rescale(fields, field, AV_TIME_BASE_Q);
        }
        self.next_dts = Some(next);

        (
            pts.map(|pts| pts - self.ts_offset),
            dts.map(|dts| dts - self.ts_offset),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(num: i32, den: i32) -> AVRational {
        AVRational { num, den }
    }

    /// Frames spaced one tick apart at the output rate pass through one each.
    #[test]
    fn test_constant_rate_passes_through() {
        let mut sync = CfrSync::new(q(30, 1), q(1, 15360), 0, true);
        for i in 0..10 {
            assert_eq!(
                sync.push(Some(i * 512), 512),
                SyncDecision {
                    previous: 0,
                    current: 1
                }
            );
        }
        assert_eq!(sync.finish(), 0);
    }

    /// A gap duplicates frames; frames that fall behind the output clock are
    /// shifted, then dropped.
    #[test]
    fn test_gaps_duplicate_and_collisions_drop() {
        let mut sync = CfrSync::new(q(30, 1), q(1, 90000), 0, true);
        assert_eq!(sync.push(Some(0), 3000).current, 1);
        // The next frame arrives three slots later (drift delta0 = 2): the
        // CLI fills the gap with llrintf(2 - 0.6) = 1 copy of the previous
        // frame, then this frame for the remaining slots.
        let gap = sync.push(Some(9000), 3000);
        assert_eq!(
            gap,
            SyncDecision {
                previous: 1,
                current: 2
            }
        );
        // A frame one stream tick after the previous one shares its slot; the
        // CLI emits it in the next slot ("past duration too large")...
        assert_eq!(sync.push(Some(12000), 3000).current, 1);
        assert_eq!(sync.push(Some(12001), 2999).current, 1);
        // ...and drops a frame that falls more than 1.1 slots behind.
        assert_eq!(
            sync.push(Some(12002), 1),
            SyncDecision {
                previous: 0,
                current: 0
            }
        );
    }

    /// A late first frame fills the slots before it (the decoder then pads
    /// with the frame itself, having no previous one), unless the file has a
    /// single stream.
    #[test]
    fn test_late_first_frame() {
        let late = 3 * 3000;
        let mut single = CfrSync::new(q(30, 1), q(1, 90000), 0, true);
        assert_eq!(
            single.push(Some(late), 3000),
            SyncDecision {
                previous: 0,
                current: 1
            }
        );
        assert_eq!(single.pts_seconds_of(0), 0.1);
        let mut with_audio = CfrSync::new(q(30, 1), q(1, 90000), 0, false);
        let decision = with_audio.push(Some(late), 3000);
        assert_eq!(decision.previous + decision.current, 4);
        assert_eq!(with_audio.pts_seconds_of(0), 0.0);
    }

    #[test]
    fn test_runaway_duplication_is_skipped() {
        let mut sync = CfrSync::new(q(30, 1), q(1, 90000), 0, true);
        sync.push(Some(0), 3000);
        let two_days = 2 * 24 * 3600 * 90000;
        assert_eq!(
            sync.push(Some(two_days), 3000),
            SyncDecision {
                previous: 0,
                current: 0
            }
        );
    }

    /// A dts more than 10 s off the prediction, or more than 0.1 s behind
    /// the previous one, shifts it and every later timestamp onto the
    /// prediction; smaller jumps pass through.
    #[test]
    fn test_demux_fixup_corrects_discontinuities() {
        let tb = q(1, 90000);
        let mut fixup = DemuxFixup::new(tb, 0, true, 0, None);
        assert_eq!(fixup.fix(Some(0), Some(0), 3000, 2), (Some(0), Some(0)));
        // +60 s: continues one frame later.
        let jump = 60 * 90000;
        assert_eq!(
            fixup.fix(Some(jump + 3000), Some(jump + 3000), 3000, 2),
            (Some(3000), Some(3000))
        );
        // The shift sticks for the packets after it.
        assert_eq!(
            fixup.fix(Some(jump + 6000), Some(jump + 6000), 3000, 2),
            (Some(6000), Some(6000))
        );
        // A 5 s jump is below the threshold.
        let small = jump + 6000 + 5 * 90000;
        assert_eq!(
            fixup.fix(Some(small), Some(small), 3000, 2),
            (Some(5 * 90000 + 6000), Some(5 * 90000 + 6000))
        );
        // 0.5 s back is a discontinuity.
        let back = small - 45000;
        let (pts, dts) = fixup.fix(Some(back), Some(back), 3000, 2);
        assert_eq!((pts, dts), (Some(5 * 90000 + 9000), Some(5 * 90000 + 9000)));
    }

    /// After a packet without a duration, the next dts is predicted from the
    /// field rate (two fields per frame).
    #[test]
    fn test_demux_fixup_predicts_from_the_field_rate() {
        let tb = q(1, 90000);
        let mut fixup = DemuxFixup::new(tb, 0, true, 0, Some(q(60, 1)));
        fixup.fix(Some(0), Some(0), 0, 2);
        let jump = 60 * 90000;
        assert_eq!(fixup.fix(None, Some(jump), 0, 2), (None, Some(3000)));
        // A repeated field (3 fields) lasts 1.5 frames.
        assert_eq!(fixup.fix(None, Some(jump + 3000), 0, 3), (None, Some(6000)));
        assert_eq!(
            fixup.fix(None, Some(jump + 7500), 0, 2),
            (None, Some(10500))
        );
    }

    /// Outside formats with discontinuities, a timestamp more than 30 h off
    /// the predicted dts is dropped, and the prediction carries on.
    #[test]
    fn test_demux_fixup_drops_timestamps_30h_off() {
        let tb = q(1, 90000);
        let mut fixup = DemuxFixup::new(tb, 0, false, 0, None);
        fixup.fix(Some(0), Some(0), 3000, 2);
        let far = 31 * 3600 * 90000;
        assert_eq!(fixup.fix(Some(far), Some(far), 3000, 2), (None, None));
        // Only the timestamp that is off goes.
        assert_eq!(
            fixup.fix(Some(far), Some(6000), 3000, 2),
            (None, Some(6000))
        );
        assert_eq!(
            fixup.fix(Some(9000), Some(far), 3000, 2),
            (Some(9000), None)
        );
        // A dropped dts leaves the prediction running: 4 frames in.
        assert_eq!(fixup.next_dts, Some(4 * 33333 + 1));
        // Below 30 h nothing is dropped, however large.
        let mut fixup = DemuxFixup::new(tb, 0, false, 0, None);
        fixup.fix(Some(0), Some(0), 3000, 2);
        let day = 24 * 3600 * 90000;
        assert_eq!(
            fixup.fix(Some(day), Some(day), 3000, 2),
            (Some(day), Some(day))
        );
    }

    /// Detection runs on timestamps shifted by the file's start, but the
    /// packets keep their own base.
    #[test]
    fn test_demux_fixup_keeps_the_input_base() {
        let tb = q(1, 90000);
        // The file starts at 1.4 s.
        let mut fixup = DemuxFixup::new(tb, 1_400_000, true, 0, None);
        let start = 126000;
        assert_eq!(
            fixup.fix(Some(start), Some(start), 3000, 2),
            (Some(start), Some(start))
        );
        let jump = start + 3000 + 60 * 90000;
        assert_eq!(
            fixup.fix(Some(jump), Some(jump), 3000, 2),
            (Some(start + 3000), Some(start + 3000))
        );
    }

    #[test]
    fn test_rounding_helpers() {
        assert_eq!(rescale(7, q(1, 2), q(1, 4)), 14);
        assert_eq!(rescale(-3, q(1, 2), q(1, 1)), -2);
        assert_eq!(llrintf(2.5), 2);
        assert_eq!(llrintf(3.5), 4);
    }
}
