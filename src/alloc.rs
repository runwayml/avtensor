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
//! With `AVTENSOR_OUTPUT_HUGE_PAGES=1`, buffers of 64 MiB or more are also
//! advised to use transparent huge pages on Linux (off by default).

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ops::Range;
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

#[cfg(test)]
thread_local! {
    static ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many times [`empty`] has been called on this thread, so tests can
/// check that a failing request allocated nothing.
#[cfg(test)]
pub fn allocations_on_this_thread() -> usize {
    ALLOCATIONS.with(std::cell::Cell::get)
}

/// Allocates an uninitialized, contiguous tensor of `shape` and `kind` on
/// `device`. CPU tensors use the allocator selected by [`output_allocator`];
/// other devices always go through torch.
pub fn empty(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, anyhow::Error> {
    #[cfg(test)]
    ALLOCATIONS.with(|count| count.set(count.get() + 1));
    if device != Device::Cpu || output_allocator() == OutputAllocator::Torch {
        return Tensor::f_empty(shape, (kind, device)).map_err(Into::into);
    }
    empty_cpu_system(shape, kind)
}

/// Alignment of the system-allocated buffers; matches torch's CPU allocator
/// so vectorized kernels see the same alignment either way.
const ALIGNMENT: usize = 64;

/// Buffers at least this large are advised to use transparent huge pages.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const HUGE_PAGE_THRESHOLD: usize = 64 << 20;

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
    // Large outputs are written once, front to back, by the decoder; with
    // 4 KiB pages the first write to each page faults, and those faults were
    // most of the cost of filling a multi-GB buffer. With 4 KiB pages, 2 MiB
    // transparent huge pages cut the fault count 512-fold. Opt-in with
    // AVTENSOR_OUTPUT_HUGE_PAGES=1, and advisory: ignored where THP is unavailable.
    advise_huge_pages(data, bytes);
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

/// Whether large output buffers get huge-page advice, decided once per
/// process: off unless `AVTENSOR_OUTPUT_HUGE_PAGES` turns it on (see
/// [`huge_pages_setting`]), and never when the kernel's huge pages are too
/// large to help (see [`thp_size_supported`]). Off by default because the
/// advice can stall the faulting thread on memory reclaim and compaction on
/// hosts short of free memory (the common `defrag=madvise` setting), which
/// slowed decodes there instead of speeding them up.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn huge_page_advice_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let requested = match std::env::var("AVTENSOR_OUTPUT_HUGE_PAGES") {
            Err(_) => false,
            Ok(value) => huge_pages_setting(&value).unwrap_or_else(|| {
                log::warn!(
                    "Unknown AVTENSOR_OUTPUT_HUGE_PAGES={value:?} (expected '1', 'true', \
                     'on' or 'yes' to turn the advice on); leaving huge-page advice off"
                );
                false
            }),
        };
        requested && thp_size_supported(thp_size())
    })
}

/// Parses `AVTENSOR_OUTPUT_HUGE_PAGES` (trimmed, case-insensitive): `1`,
/// `true`, `on` or `yes` turn the advice on; empty, `0`, `false`, `off` or
/// `no` leave it off. `None` for anything else.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn huge_pages_setting(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "" | "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// Size of a transparent huge page in bytes, as reported by the kernel.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn thp_size() -> Option<usize> {
    std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Whether transparent huge pages of `thp_size` bytes are worth asking for.
/// They are 2 MiB with 4 KiB base pages, but 32 MiB with 16 KiB pages and
/// 512 MiB on 64 KiB-page arm64 kernels: there, buffers below ~1 GiB rarely contain an aligned huge page,
/// and larger ones need 512 MiB of contiguous memory per fault, which makes
/// compaction costlier and fallback likelier. If the size can't be read the
/// advice stays on; where THP is missing, madvise just fails.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn thp_size_supported(thp_size: Option<usize>) -> bool {
    thp_size.is_none_or(|size| size <= 2 << 20)
}

/// The part of the buffer `[addr, addr + bytes)` to advise: nothing below
/// [`HUGE_PAGE_THRESHOLD`], otherwise its [`whole_pages`].
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn huge_page_advice_range(addr: usize, bytes: usize, page: usize) -> Option<Range<usize>> {
    (bytes >= HUGE_PAGE_THRESHOLD)
        .then(|| whole_pages(addr, bytes, page))
        .flatten()
}

/// The whole `page`-sized pages inside `[addr, addr + bytes)`, if any.
/// madvise rejects a start that is not page-aligned, and `posix_memalign` only
/// guarantees [`ALIGNMENT`], so the start is rounded up and the end down.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn whole_pages(addr: usize, bytes: usize, page: usize) -> Option<Range<usize>> {
    let start = addr.next_multiple_of(page);
    let end = (addr + bytes) / page * page;
    (end > start).then_some(start..end)
}

/// `madvise(MADV_HUGEPAGE)` over the [`huge_page_advice_range`] of the buffer
/// `[data, data + bytes)`. Linux only: elsewhere there are no transparent huge
/// pages to ask for.
#[cfg(target_os = "linux")]
fn advise_huge_pages(data: *mut c_void, bytes: usize) {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let range = huge_page_advice_range(data as usize, bytes, page);
    if let Some(range) = range.filter(|_| huge_page_advice_enabled()) {
        // SAFETY: the range lies within the allocation; the call only
        // changes the kernel's paging advice for it.
        unsafe { libc::madvise(range.start as *mut c_void, range.len(), libc::MADV_HUGEPAGE) };
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_data: *mut c_void, _bytes: usize) {}

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

    fn avtensor_swap_storage_buffers(
        a: *const torch_sys::C_tensor,
        b: *const torch_sys::C_tensor,
    ) -> *mut c_char;

    fn avtensor_trim_storage(t: *const torch_sys::C_tensor, nbytes: usize) -> *mut c_char;
}

/// Grows `tensor` (contiguous, at the start of its storage) to `rows` rows
/// along dim 0, keeping its contents. The storage itself takes the new
/// buffer, so views of `tensor` handed out earlier move with it and the old
/// buffer is freed now, rather than kept alive by those views.
pub fn grow_rows(tensor: &mut Tensor, rows: i64) -> Result<(), anyhow::Error> {
    let mut shape = tensor.size();
    anyhow::ensure!(
        tensor.is_contiguous() && tensor.storage_offset() == 0,
        "grow_rows needs a contiguous tensor at the start of its storage"
    );
    let old_rows = shape[0];
    anyhow::ensure!(
        rows >= old_rows,
        "grow_rows cannot shrink {shape:?} to {rows} rows"
    );
    shape[0] = rows;
    let grown = empty(&shape, tensor.kind(), tensor.device())?;
    grown.f_narrow(0, 0, old_rows)?.f_copy_(tensor)?;
    // SAFETY: both pointers are live tensors; `grown` is only dropped after
    // the swap, which leaves it owning the old buffer and frees it.
    let err = unsafe { avtensor_swap_storage_buffers(tensor.as_ptr(), grown.as_ptr()) };
    check_shim_error(err, "swapping storage buffers")?;
    *tensor = tensor.f_as_strided(&shape, grown.stride(), 0)?;
    Ok(())
}

/// Trims `tensor` (contiguous, at the start of its storage) to its first
/// `rows` rows along dim 0, and its storage to their bytes. Nothing is
/// copied or reallocated: the buffer stays and is freed whole as before.
/// Rows past `rows` that were never written were never resident, so this
/// costs no memory; it keeps them out of anything that reads the whole
/// storage (`torch.save`, `share_memory_`, a DataLoader handing the tensor
/// to another process).
///
/// # Safety
///
/// No other view of `tensor`'s storage may reach past the first `rows` rows:
/// it would then lie outside its storage.
pub unsafe fn trim_rows(tensor: &mut Tensor, rows: i64) -> Result<(), anyhow::Error> {
    let mut shape = tensor.size();
    anyhow::ensure!(
        tensor.is_contiguous() && tensor.storage_offset() == 0,
        "trim_rows needs a contiguous tensor at the start of its storage"
    );
    anyhow::ensure!(
        (0..=shape[0]).contains(&rows),
        "trim_rows cannot grow {shape:?} to {rows} rows"
    );
    shape[0] = rows;
    let numel = shape.iter().product::<i64>() as usize;
    let nbytes = numel * tensor.kind().elt_size_in_bytes();
    // SAFETY: `tensor` is live; the caller guarantees no view needs the rest.
    let err = unsafe { avtensor_trim_storage(tensor.as_ptr(), nbytes) };
    check_shim_error(err, "trimming the storage")?;
    *tensor = tensor.f_as_strided(&shape, tensor.stride(), 0)?;
    Ok(())
}

/// Turns an error message returned by the C++ shim into an error, freeing it.
fn check_shim_error(err: *mut c_char, what: &str) -> Result<(), anyhow::Error> {
    if err.is_null() {
        return Ok(());
    }
    // SAFETY: the shim returns a NUL-terminated strdup'd message.
    let message = unsafe { CStr::from_ptr(err) }
        .to_string_lossy()
        .into_owned();
    unsafe { libc::free(err as *mut c_void) };
    bail!("{what} failed: {message}")
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
    fn huge_page_advice_covers_whole_pages_of_large_buffers() {
        for page in [4usize << 10, 64 << 10] {
            let base = 1000 * page;
            // posix_memalign only guarantees ALIGNMENT: the start is rounded
            // up and the end down, and an end on a page boundary is kept.
            let addr = base + ALIGNMENT;
            assert_eq!(
                whole_pages(base, 3 * page, page),
                Some(base..base + 3 * page)
            );
            assert_eq!(
                whole_pages(addr, 3 * page, page),
                Some(base + page..base + 3 * page)
            );
            assert_eq!(
                whole_pages(addr, 3 * page - ALIGNMENT, page),
                Some(base + page..base + 3 * page)
            );
            // No whole page inside: within one page, or straddling a boundary.
            assert_eq!(whole_pages(addr, 100, page), None, "page {page}");
            assert_eq!(whole_pages(addr, page, page), None, "page {page}");

            assert_eq!(
                huge_page_advice_range(base, HUGE_PAGE_THRESHOLD - 1, page),
                None,
                "below the threshold, page {page}"
            );
            assert_eq!(
                huge_page_advice_range(base, HUGE_PAGE_THRESHOLD, page),
                Some(base..base + HUGE_PAGE_THRESHOLD),
                "at the threshold, page {page}"
            );
            assert_eq!(
                huge_page_advice_range(addr, HUGE_PAGE_THRESHOLD + 7, page),
                Some(base + page..base + HUGE_PAGE_THRESHOLD),
                "unaligned, page {page}"
            );
        }
    }

    #[test]
    fn huge_pages_over_2_mib_are_not_advised() {
        assert!(thp_size_supported(Some(2 << 20)));
        assert!(!thp_size_supported(Some(512 << 20)));
        assert!(thp_size_supported(None));
    }

    #[test]
    fn huge_pages_env_value() {
        for on in ["1", "true", "On", " yes "] {
            assert_eq!(huge_pages_setting(on), Some(true), "{on:?}");
        }
        for off in ["", "0", "false", "OFF", " no "] {
            assert_eq!(huge_pages_setting(off), Some(false), "{off:?}");
        }
        for unknown in ["never", "disabled", "2"] {
            assert_eq!(huge_pages_setting(unknown), None, "{unknown:?}");
        }
    }

    /// Whether the mapping holding the middle of `tensor`'s buffer has the
    /// `hg` (`MADV_HUGEPAGE`) flag in `/proc/self/smaps`. The flag records the
    /// advice, not whether huge pages were granted, so it doesn't depend on
    /// free memory. Not `data_ptr` itself: it sits just past glibc's chunk
    /// header in the first page, which is not advised and becomes a mapping
    /// of its own.
    #[cfg(target_os = "linux")]
    fn advised_huge_pages(tensor: &Tensor) -> Result<bool, anyhow::Error> {
        let bytes = tensor.numel() * tensor.kind().elt_size_in_bytes();
        let addr = tensor.data_ptr() as usize + bytes / 2;
        let smaps = std::fs::read_to_string("/proc/self/smaps")?;
        let mut in_mapping = false;
        for line in smaps.lines() {
            let range = line
                .split_once(' ')
                .and_then(|(range, _)| range.split_once('-'));
            if let Some((Ok(start), Ok(end))) =
                range.map(|(s, e)| (usize::from_str_radix(s, 16), usize::from_str_radix(e, 16)))
            {
                in_mapping = (start..end).contains(&addr);
            } else if let Some(flags) = line.strip_prefix("VmFlags:").filter(|_| in_mapping) {
                return Ok(flags.split_whitespace().any(|flag| flag == "hg"));
            }
        }
        bail!("no mapping in /proc/self/smaps holds {addr:#x}")
    }

    /// With the advice requested, buffers of at least `HUGE_PAGE_THRESHOLD`
    /// are advised and still fully usable; smaller ones are not advised.
    /// Without it (the default) nothing is advised. Run as is (default off)
    /// and in a child process with the advice on by
    /// `huge_page_opt_in_reaches_the_allocator`.
    #[cfg(target_os = "linux")]
    #[test]
    fn large_buffers_are_advised_huge_pages() -> Result<(), anyhow::Error> {
        let thp_available = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
            .is_ok_and(|mode| !mode.contains("[never]"));
        if !thp_available {
            eprintln!("skipping: transparent huge pages are unavailable or set to never");
            return Ok(());
        }
        // From the variable itself rather than huge_page_advice_enabled(), so
        // the child run checks that the opt-in reaches the allocator.
        let requested = match std::env::var("AVTENSOR_OUTPUT_HUGE_PAGES") {
            Err(_) => false,
            Ok(value) => huge_pages_setting(&value).unwrap_or(false),
        };
        let expected = requested && thp_size_supported(thp_size());

        // Just over the threshold, with a size that is not a page multiple.
        let shape = [(HUGE_PAGE_THRESHOLD / 3 + 7) as i64, 3];
        let mut large = empty_cpu_system(&shape, Kind::Uint8)?;
        assert_eq!(advised_huge_pages(&large)?, expected);
        let _ = large.fill_(2);
        assert_eq!(
            i64::try_from(large.sum(Kind::Int64))?,
            2 * shape[0] * shape[1]
        );

        let small = empty_cpu_system(&[3 << 20], Kind::Uint8)?;
        assert!(!advised_huge_pages(&small)?);
        Ok(())
    }

    /// `AVTENSOR_OUTPUT_HUGE_PAGES` is read once per process, so the opt-in
    /// is checked by rerunning the smaps test in a child with it set.
    #[cfg(target_os = "linux")]
    #[test]
    fn huge_page_opt_in_reaches_the_allocator() -> Result<(), anyhow::Error> {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "alloc::tests::large_buffers_are_advised_huge_pages",
                "--nocapture",
            ])
            .env("AVTENSOR_OUTPUT_HUGE_PAGES", "1")
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
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
    fn grow_rows_moves_views_and_frees_the_old_buffer() -> Result<(), anyhow::Error> {
        static FREED: AtomicUsize = AtomicUsize::new(0);
        let shape = [4i64, 6];
        let data = unsafe { libc::malloc(4 * 6) };
        assert!(!data.is_null());
        let mut tensor = unsafe {
            tensor_from_blob(
                data,
                &shape,
                scalar_type(Kind::Uint8)?,
                counting_deleter,
                &FREED as *const AtomicUsize as *mut c_void,
            )?
        };
        for row in 0..4 {
            let _ = tensor.narrow(0, row, 1).fill_(row + 1);
        }
        let view = tensor.narrow(0, 2, 1);

        grow_rows(&mut tensor, 9)?;
        assert_eq!(tensor.size(), [9, 6]);
        assert!(tensor.is_contiguous());
        assert_eq!(
            FREED.load(Ordering::SeqCst),
            1,
            "the view must not keep the old buffer"
        );
        // The view now reads (and writes) row 2 of the grown buffer.
        assert_eq!(view.data_ptr() as usize, tensor.data_ptr() as usize + 2 * 6);
        assert_eq!(i64::try_from(view.sum(Kind::Int64))?, 3 * 6);
        let _ = tensor.narrow(0, 8, 1).fill_(9);
        assert_eq!(
            Vec::<u8>::try_from(
                tensor
                    .narrow(0, 0, 4)
                    .sum_dim_intlist(1, false, Kind::Uint8)
            )?,
            [6, 12, 18, 24]
        );
        let _ = view.shallow_clone().fill_(7);
        assert_eq!(
            i64::try_from(tensor.narrow(0, 2, 1).sum(Kind::Int64))?,
            7 * 6
        );

        // Tensors from torch's own allocator grow the same way.
        let mut torch_tensor = Tensor::f_ones([3, 2], (Kind::Float, Device::Cpu))?;
        let torch_view = torch_tensor.narrow(0, 1, 1);
        grow_rows(&mut torch_tensor, 5)?;
        assert_eq!(
            torch_view.data_ptr() as usize,
            torch_tensor.data_ptr() as usize + 2 * 4
        );
        assert_eq!(
            f64::try_from(torch_tensor.narrow(0, 0, 3).sum(Kind::Double))?,
            6.0
        );
        Ok(())
    }

    #[test]
    fn trim_rows_trims_the_storage_in_place() -> Result<(), anyhow::Error> {
        static FREED: AtomicUsize = AtomicUsize::new(0);
        let shape = [5i64, 6];
        let data = unsafe { libc::malloc(5 * 6) };
        assert!(!data.is_null());
        let mut tensor = unsafe {
            tensor_from_blob(
                data,
                &shape,
                scalar_type(Kind::Uint8)?,
                counting_deleter,
                &FREED as *const AtomicUsize as *mut c_void,
            )?
        };
        for row in 0..5 {
            let _ = tensor.narrow(0, row, 1).fill_(row + 1);
        }
        let view = tensor.narrow(0, 1, 1);
        unsafe { trim_rows(&mut tensor, 3)? };
        assert_eq!(tensor.size(), [3, 6]);
        // Same buffer, nothing copied; the storage now ends at row 3.
        assert_eq!(tensor.data_ptr(), data);
        assert!(tensor.f_as_strided([4, 6], [6, 1], 0).is_err());
        assert_eq!(i64::try_from(view.sum(Kind::Int64))?, 2 * 6);
        assert!(unsafe { trim_rows(&mut tensor, 4) }.is_err());
        drop(tensor);
        assert_eq!(FREED.load(Ordering::SeqCst), 0, "the view still holds it");
        drop(view);
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
        Ok(())
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
