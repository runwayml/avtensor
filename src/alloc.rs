//! Allocation of the CPU output buffers that decoded frames and samples are
//! written into.
//!
//! By default these buffers come from the system allocator (`posix_memalign`)
//! and are wrapped in a tensor with `at::from_blob` plus a deleter, instead of
//! being allocated with `at::empty`. Shape, dtype, strides and device are
//! identical either way; only the storage's owner differs.
//!
//! Why: torch's CPU allocator is not always plain malloc. The aarch64 Linux
//! wheels embed mimalloc, which keeps a block freed by a thread other than the
//! one that allocated it: the pages are `MADV_FREE`d but stay mapped and
//! resident until the allocating thread exits. Data-loading pipelines decode
//! on worker threads and drop the tensors on other threads, so with `at::empty`
//! every decode left its whole output buffer (hundreds of MiB to GiBs) in
//! resident memory for the lifetime of the worker. glibc releases large blocks
//! with `munmap` on `free` regardless of the calling thread, so the deleter
//! path returns the memory immediately.
//!
//! `AVTENSOR_OUTPUT_ALLOCATOR=torch` restores `at::empty` for comparison.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context};
use tch::{Device, Kind, Tensor};

/// Which allocator backs CPU output tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputAllocator {
    /// `posix_memalign` + `at::from_blob` with a `free` deleter (default).
    System,
    /// `at::empty`, i.e. torch's own CPU allocator.
    Torch,
}

impl OutputAllocator {
    fn from_env_value(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            None | Some("") => OutputAllocator::System,
            Some(v) if v.eq_ignore_ascii_case("system") => OutputAllocator::System,
            Some(v) if v.eq_ignore_ascii_case("torch") => OutputAllocator::Torch,
            Some(v) => {
                log::warn!(
                    "Unknown AVTENSOR_OUTPUT_ALLOCATOR={v:?} (expected 'system' or 'torch'); \
                     using 'system'"
                );
                OutputAllocator::System
            }
        }
    }
}

/// The allocator selected by `AVTENSOR_OUTPUT_ALLOCATOR`, read once.
pub fn output_allocator() -> OutputAllocator {
    static CHOICE: OnceLock<OutputAllocator> = OnceLock::new();
    *CHOICE.get_or_init(|| {
        OutputAllocator::from_env_value(std::env::var("AVTENSOR_OUTPUT_ALLOCATOR").ok().as_deref())
    })
}

/// Allocates an uninitialized, contiguous tensor of `shape` and `kind` on
/// `device`. CPU tensors use the allocator selected by [`output_allocator`];
/// other devices always go through torch.
pub fn empty(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, anyhow::Error> {
    if device != Device::Cpu || output_allocator() == OutputAllocator::Torch {
        return Tensor::f_empty(shape, (kind, device)).map_err(Into::into);
    }
    empty_cpu_system(shape, kind)
}

/// Alignment of the system-allocated buffers; matches torch's CPU allocator
/// so vectorized kernels see the same alignment either way.
const ALIGNMENT: usize = 64;

fn empty_cpu_system(shape: &[i64], kind: Kind) -> Result<Tensor, anyhow::Error> {
    let scalar_type = scalar_type(kind)?;
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| anyhow!("invalid output shape {shape:?}"))?;
    let bytes = numel
        .checked_mul(kind.elt_size_in_bytes())
        .ok_or_else(|| anyhow!("output shape {shape:?} overflows usize"))?;
    if bytes == 0 {
        // Nothing to own; torch handles empty tensors without allocating.
        return Tensor::f_empty(shape, (kind, Device::Cpu)).map_err(Into::into);
    }

    let mut data: *mut c_void = ptr::null_mut();
    // SAFETY: plain libc call with a valid out-pointer and a power-of-two
    // alignment that is a multiple of sizeof(void*).
    let rc = unsafe { libc::posix_memalign(&mut data, ALIGNMENT, bytes) };
    if rc != 0 || data.is_null() {
        bail!(
            "posix_memalign({bytes} bytes) for output shape {shape:?} failed: {}",
            std::io::Error::from_raw_os_error(rc)
        );
    }
    // SAFETY: `data` is a fresh allocation of exactly `bytes` bytes, and
    // `free_system_buffer` is the matching release for `posix_memalign`.
    unsafe {
        tensor_from_blob(
            data,
            shape,
            scalar_type,
            free_system_buffer,
            ptr::null_mut(),
        )
        .with_context(|| format!("wrapping a {bytes}-byte buffer as a {kind:?} tensor"))
    }
}

unsafe extern "C" fn free_system_buffer(data: *mut c_void, _ctx: *mut c_void) {
    libc::free(data);
}

/// `c10::ScalarType` value for `kind` (tch keeps its own mapping private).
fn scalar_type(kind: Kind) -> Result<c_int, anyhow::Error> {
    Ok(match kind {
        Kind::Uint8 => 0,
        Kind::Int8 => 1,
        Kind::Int16 => 2,
        Kind::Int => 3,
        Kind::Int64 => 4,
        Kind::Half => 5,
        Kind::Float => 6,
        Kind::Double => 7,
        Kind::Bool => 11,
        Kind::BFloat16 => 15,
        other => bail!("unsupported output dtype {other:?}"),
    })
}

/// Deleter signature expected by the C++ shim: `(data, ctx)`.
type BlobDeleter = unsafe extern "C" fn(*mut c_void, *mut c_void);

extern "C" {
    fn avtensor_tensor_from_blob(
        data: *mut c_void,
        sizes: *const i64,
        ndim: usize,
        scalar_type: i32,
        deleter: BlobDeleter,
        ctx: *mut c_void,
        err: *mut *mut c_char,
    ) -> *mut torch_sys::C_tensor;
}

/// Wraps `data` in a contiguous CPU tensor that owns it: `deleter(data, ctx)`
/// runs when the last tensor sharing the storage is dropped.
///
/// # Safety
///
/// `data` must stay valid, and hold `numel(shape) * element size` bytes, until
/// `deleter` is called. `deleter` must be sound to call from any thread. On
/// error the buffer is intentionally leaked: torch may or may not have run the
/// deleter already, and leaking beats a double free on a path that only
/// triggers for invalid shapes.
unsafe fn tensor_from_blob(
    data: *mut c_void,
    shape: &[i64],
    scalar_type: c_int,
    deleter: BlobDeleter,
    ctx: *mut c_void,
) -> Result<Tensor, anyhow::Error> {
    let mut err: *mut c_char = ptr::null_mut();
    let tensor = avtensor_tensor_from_blob(
        data,
        shape.as_ptr(),
        shape.len(),
        scalar_type,
        deleter,
        ctx,
        &mut err,
    );
    if tensor.is_null() {
        let message = if err.is_null() {
            "unknown error".to_owned()
        } else {
            let message = CStr::from_ptr(err).to_string_lossy().into_owned();
            libc::free(err as *mut c_void);
            message
        };
        bail!("at::from_blob failed for shape {shape:?}: {message}");
    }
    Ok(Tensor::from_ptr(tensor))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn env_value_selects_allocator() {
        assert_eq!(
            OutputAllocator::from_env_value(None),
            OutputAllocator::System
        );
        assert_eq!(
            OutputAllocator::from_env_value(Some("")),
            OutputAllocator::System
        );
        assert_eq!(
            OutputAllocator::from_env_value(Some("system")),
            OutputAllocator::System
        );
        assert_eq!(
            OutputAllocator::from_env_value(Some("Torch")),
            OutputAllocator::Torch
        );
        assert_eq!(
            OutputAllocator::from_env_value(Some("bogus")),
            OutputAllocator::System
        );
    }

    #[test]
    fn system_buffer_tensor_behaves_like_empty() -> Result<(), anyhow::Error> {
        let shape = [3i64, 4, 5, 3];
        let mut tensor = empty_cpu_system(&shape, Kind::Uint8)?;
        assert_eq!(tensor.size(), shape);
        assert_eq!(tensor.kind(), Kind::Uint8);
        assert_eq!(tensor.device(), Device::Cpu);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.data_ptr() as usize % ALIGNMENT, 0);

        // Writable through the usual paths and through views.
        let _ = tensor.fill_(1);
        let _ = tensor.narrow(0, 1, 1).fill_(5);
        assert_eq!(i64::try_from(tensor.sum(Kind::Int64))?, 120 + 60 * 5);

        let mut audio = empty_cpu_system(&[2, 1000], Kind::Float)?;
        assert_eq!(audio.kind(), Kind::Float);
        let _ = audio.fill_(0.5);
        assert!((f64::try_from(audio.sum(Kind::Double))? - 1000.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn zero_sized_shape_falls_back_to_torch() -> Result<(), anyhow::Error> {
        let tensor = empty_cpu_system(&[0, 720, 1280, 3], Kind::Uint8)?;
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.size(), [0, 720, 1280, 3]);
        Ok(())
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        assert!(empty_cpu_system(&[-1, 3], Kind::Uint8).is_err());
        assert!(empty_cpu_system(&[i64::MAX, 2], Kind::Float).is_err());
    }

    unsafe extern "C" fn counting_deleter(data: *mut c_void, ctx: *mut c_void) {
        (*(ctx as *const AtomicUsize)).fetch_add(1, Ordering::SeqCst);
        libc::free(data);
    }

    #[test]
    fn deleter_runs_once_when_the_last_view_drops_on_another_thread() -> Result<(), anyhow::Error> {
        static FREED: AtomicUsize = AtomicUsize::new(0);
        let shape = [8i64, 16];
        let bytes = 8 * 16 * Kind::Float.elt_size_in_bytes();
        let data = unsafe { libc::malloc(bytes) };
        assert!(!data.is_null());
        let mut tensor = unsafe {
            tensor_from_blob(
                data,
                &shape,
                scalar_type(Kind::Float)?,
                counting_deleter,
                &FREED as *const AtomicUsize as *mut c_void,
            )?
        };
        let _ = tensor.fill_(2.0);
        let view = tensor.narrow(0, 2, 3);
        drop(tensor);
        assert_eq!(
            FREED.load(Ordering::SeqCst),
            0,
            "a live view keeps the buffer"
        );

        std::thread::spawn(move || {
            assert!((f64::try_from(view.sum(Kind::Double)).unwrap() - 96.0).abs() < 1e-9);
            drop(view);
        })
        .join()
        .unwrap();
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
