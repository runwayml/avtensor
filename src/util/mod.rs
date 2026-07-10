use rsmpeg::ffi::AVRational;

pub mod ffmpeg;
pub mod gcs;
pub mod s3;

#[cfg(test)]
pub mod mel;
#[cfg(test)]
pub mod test_utils;

/// Utility function to convert a PTS in stream time-base to seconds.
pub fn pts_to_seconds(pts: i64, time_base: AVRational) -> f64 {
    pts as f64 * (time_base.num as f64 / time_base.den as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pts_to_seconds() {
        let time_base = AVRational { num: 1, den: 25 }; // 25 fps
        assert_eq!(pts_to_seconds(0, time_base), 0.0);
        assert_eq!(pts_to_seconds(25, time_base), 1.0);
        assert_eq!(pts_to_seconds(50, time_base), 2.0);
        assert_eq!(pts_to_seconds(100, time_base), 4.0);
        assert_eq!(pts_to_seconds(-25, time_base), -1.0);
    }
}
