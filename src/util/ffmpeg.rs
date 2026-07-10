use std::ffi::c_int;

use rsmpeg::error::Ret;

/// This is a common pattern in FFmpeg that an api returns negative number as an
/// error, zero or bigger a success. Here we triage the returned number of FFmpeg
/// API to `Ok(positive)` and `Err(negative)`.
pub trait RetUpgrade {
    fn upgrade(self) -> Ret;
}

impl RetUpgrade for c_int {
    fn upgrade(self) -> Ret {
        if self < 0 {
            Ret::Err(self)
        } else {
            Ret::Ok(self)
        }
    }
}
