//! GPU-resident color conversion for NVDEC-decoded frames.
//!
//! When `device` is requested, decoded frames stay in CUDA memory and
//! are converted NV12 → RGB by NPP directly into the (CUDA) output tensor —
//! the same approach torchcodec uses for `device="cuda"`. libcudart and the
//! NPP libraries are loaded at decode time with `dlopen`, so the crate
//! carries no CUDA link-time dependency and CPU-only deployments are
//! unaffected.
//!
//! Color handling: sources tagged BT.709 use NPP's single-step
//! `nppiNV12ToRGB_709CSC` (limited range). Everything else uses BT.601
//! limited range to match swscale's default for untagged content; NPP has no
//! single-step limited-range 601 NV12 conversion (the plain `nppiNV12ToRGB`
//! assumes full-range input), so the chroma is first deinterleaved into
//! scratch planes (`nppiNV12ToYUV420`, a layout-only step) and converted with
//! the planar `nppiYCbCr420ToRGB` (YCbCr = studio range by NPP convention).

use anyhow::{anyhow, Context};
use libloading::os::unix::{Library, Symbol, RTLD_LAZY, RTLD_LOCAL};

#[repr(C)]
#[derive(Clone, Copy)]
struct NppiSize {
    width: i32,
    height: i32,
}

/// `NppStreamContext` (npp.h). Passed by value to `_Ctx` NPP entry points;
/// CUDA 12/13 NPP builds only export the `_Ctx` variants.
#[repr(C)]
#[derive(Clone, Copy)]
struct NppStreamContext {
    stream: *mut std::ffi::c_void,
    cuda_device_id: i32,
    multi_processor_count: i32,
    max_threads_per_multi_processor: i32,
    max_threads_per_block: i32,
    shared_mem_per_block: usize,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    stream_flags: u32,
    reserved_0: i32,
}

type CudaSetDevice = unsafe extern "C" fn(device: i32) -> i32;
type CudaDeviceSynchronize = unsafe extern "C" fn() -> i32;
type CudaMalloc = unsafe extern "C" fn(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
type CudaFree = unsafe extern "C" fn(ptr: *mut std::ffi::c_void) -> i32;
type NppGetStreamContext = unsafe extern "C" fn(ctx: *mut NppStreamContext) -> i32;
/// `nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc[2], srcStep, pDst, dstStep, roi, streamCtx)`
type NppiNv12ToRgb = unsafe extern "C" fn(
    src: *const *const u8,
    src_step: i32,
    dst: *mut u8,
    dst_step: i32,
    roi: NppiSize,
    stream_ctx: NppStreamContext,
) -> i32;
/// `nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc[2], srcStep, pDst[3], aDstStep[3], roi, streamCtx)`
type NppiNv12ToYuv420 = unsafe extern "C" fn(
    src: *const *const u8,
    src_step: i32,
    dst: *const *mut u8,
    dst_steps: *const i32,
    roi: NppiSize,
    stream_ctx: NppStreamContext,
) -> i32;
/// `nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pSrc[3], rSrcStep[3], pDst, dstStep, roi, streamCtx)`
type NppiYCbCr420ToRgb = unsafe extern "C" fn(
    src: *const *const u8,
    src_steps: *const i32,
    dst: *mut u8,
    dst_step: i32,
    roi: NppiSize,
    stream_ctx: NppStreamContext,
) -> i32;

/// Opens the first loadable name of a versioned shared library.
fn open_library(names: &[&str]) -> Result<Library, anyhow::Error> {
    for name in names {
        if let Ok(lib) = unsafe { Library::open(Some(name), RTLD_LAZY | RTLD_LOCAL) } {
            return Ok(lib);
        }
    }
    Err(anyhow!(
        "none of {names:?} could be loaded; GPU-resident output requires the \
         CUDA runtime and NPP libraries at run time"
    ))
}

/// Converts NVDEC NV12 frames into a CUDA RGB tensor via NPP.
pub struct CudaNv12ToRgb {
    // Held to keep the dlopened libraries (and the symbols below) alive.
    _cudart: Library,
    _nppc: Library,
    _nppicc: Library,
    set_device: Symbol<CudaSetDevice>,
    device_synchronize: Symbol<CudaDeviceSynchronize>,
    cuda_free: Symbol<CudaFree>,
    nv12_to_yuv420: Symbol<NppiNv12ToYuv420>,
    ycbcr420_to_rgb: Symbol<NppiYCbCr420ToRgb>,
    nv12_to_rgb_709: Symbol<NppiNv12ToRgb>,
    stream_ctx: NppStreamContext,
    /// Device scratch for the planar YUV420 intermediate of the BT.601 path:
    /// a full-size Y plane followed by quarter-size Cb and Cr planes. Null on
    /// the (single-step) BT.709 path.
    scratch: *mut u8,
    pub device: i32,
    /// Use the BT.709 conversion (source tagged bt709); BT.601 otherwise.
    /// Both are limited (studio) range, which is what NVDEC emits.
    pub use_bt709: bool,
    pub dst_width: i32,
    pub dst_height: i32,
}

// SAFETY: all symbols are stateless C entry points, the scratch buffer is
// owned exclusively by this converter, and all use happens on the decoding
// thread.
unsafe impl Send for CudaNv12ToRgb {}

impl CudaNv12ToRgb {
    pub fn new(
        device: i32,
        use_bt709: bool,
        dst_width: i32,
        dst_height: i32,
    ) -> Result<Self, anyhow::Error> {
        let cudart = open_library(&[
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ])?;
        let nppc = open_library(&[
            "libnppc.so",
            "libnppc.so.13",
            "libnppc.so.12",
            "libnppc.so.11",
        ])?;
        let nppicc = open_library(&[
            "libnppicc.so",
            "libnppicc.so.13",
            "libnppicc.so.12",
            "libnppicc.so.11",
        ])?;
        unsafe {
            let set_device: Symbol<CudaSetDevice> = cudart
                .get(b"cudaSetDevice\0")
                .context("resolving cudaSetDevice")?;
            let device_synchronize = cudart
                .get(b"cudaDeviceSynchronize\0")
                .context("resolving cudaDeviceSynchronize")?;
            let cuda_malloc: Symbol<CudaMalloc> = cudart
                .get(b"cudaMalloc\0")
                .context("resolving cudaMalloc")?;
            let cuda_free = cudart.get(b"cudaFree\0").context("resolving cudaFree")?;
            let get_stream_ctx: Symbol<NppGetStreamContext> = nppc
                .get(b"nppGetStreamContext\0")
                .context("resolving nppGetStreamContext")?;
            let nv12_to_yuv420 = nppicc
                .get(b"nppiNV12ToYUV420_8u_P2P3R_Ctx\0")
                .context("resolving nppiNV12ToYUV420_8u_P2P3R_Ctx")?;
            let ycbcr420_to_rgb = nppicc
                .get(b"nppiYCbCr420ToRGB_8u_P3C3R_Ctx\0")
                .context("resolving nppiYCbCr420ToRGB_8u_P3C3R_Ctx")?;
            let nv12_to_rgb_709 = nppicc
                .get(b"nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx\0")
                .context("resolving nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx")?;

            // The stream context captures the *current* device's properties;
            // set the device first. Conversions run on NPP's default stream.
            let rc = set_device(device);
            if rc != 0 {
                return Err(anyhow!("cudaSetDevice({device}) failed: {rc}"));
            }
            let mut stream_ctx = std::mem::zeroed::<NppStreamContext>();
            let status = get_stream_ctx(&mut stream_ctx);
            if status != 0 {
                return Err(anyhow!(
                    "nppGetStreamContext failed with NppStatus {status}"
                ));
            }

            let scratch = if use_bt709 {
                std::ptr::null_mut()
            } else {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let size = (dst_width as usize * dst_height as usize) * 3 / 2;
                let rc = cuda_malloc(&mut ptr, size);
                if rc != 0 {
                    return Err(anyhow!("cudaMalloc({size}) failed: {rc}"));
                }
                ptr as *mut u8
            };

            Ok(Self {
                _cudart: cudart,
                _nppc: nppc,
                _nppicc: nppicc,
                set_device,
                device_synchronize,
                cuda_free,
                nv12_to_yuv420,
                ycbcr420_to_rgb,
                nv12_to_rgb_709,
                stream_ctx,
                scratch,
                device,
                use_bt709,
                dst_width,
                dst_height,
            })
        }
    }

    /// Converts one NV12 frame (device pointers) into `dst` (a device pointer
    /// to a tightly packed `[dst_height, dst_width, 3]` u8 image). The
    /// conversion is asynchronous; call [`Self::synchronize`] before handing
    /// the tensor out.
    pub fn convert_frame(
        &self,
        y_plane: *const u8,
        uv_plane: *const u8,
        src_pitch: i32,
    ) -> ConvertCall<'_> {
        ConvertCall {
            converter: self,
            y_plane,
            uv_plane,
            src_pitch,
        }
    }

    pub fn synchronize(&self) -> Result<(), anyhow::Error> {
        unsafe {
            let rc = (self.set_device)(self.device);
            if rc != 0 {
                return Err(anyhow!("cudaSetDevice({}) failed: {rc}", self.device));
            }
            let rc = (self.device_synchronize)();
            if rc != 0 {
                return Err(anyhow!("cudaDeviceSynchronize failed: {rc}"));
            }
        }
        Ok(())
    }
}

impl Drop for CudaNv12ToRgb {
    fn drop(&mut self) {
        if !self.scratch.is_null() {
            // SAFETY: `scratch` was allocated by cudaMalloc in `new()` on
            // `self.device`, is only freed here, and cannot be used after
            // drop (all conversion calls borrow `self`).
            unsafe {
                (self.set_device)(self.device);
                (self.cuda_free)(self.scratch as *mut std::ffi::c_void);
            }
        }
    }
}

/// Borrow-friendly single conversion invocation.
pub struct ConvertCall<'a> {
    converter: &'a CudaNv12ToRgb,
    y_plane: *const u8,
    uv_plane: *const u8,
    src_pitch: i32,
}

impl ConvertCall<'_> {
    /// # Safety contract
    ///
    /// The pointers handed to NPP here are not checkable from Rust — they
    /// are CUDA device pointers — so soundness rests on invariants enforced
    /// by the callers:
    /// - `y_plane`/`uv_plane` come from an FFmpeg `AVFrame` whose dimensions
    ///   were validated against `dst_width`/`dst_height` before this call
    ///   (`direct_convert_cuda_frame`), and `src_pitch` is that frame's
    ///   FFmpeg-reported line size.
    /// - `dst` points at a live, tightly packed
    ///   `[dst_height, dst_width, 3]` u8 CUDA tensor slice allocated with
    ///   exactly those dimensions.
    /// - The NPP ROI is `dst_width x dst_height`, so NPP reads/writes stay
    ///   inside both allocations by construction.
    pub fn into_dst(self, dst: *mut u8) -> Result<(), anyhow::Error> {
        let c = self.converter;
        unsafe {
            let rc = (c.set_device)(c.device);
            if rc != 0 {
                return Err(anyhow!("cudaSetDevice({}) failed: {rc}", c.device));
            }
            let src = [self.y_plane, self.uv_plane];
            let roi = NppiSize {
                width: c.dst_width,
                height: c.dst_height,
            };
            let status = if c.use_bt709 {
                (c.nv12_to_rgb_709)(
                    src.as_ptr(),
                    self.src_pitch,
                    dst,
                    c.dst_width * 3,
                    roi,
                    c.stream_ctx,
                )
            } else {
                // Two-step limited-range BT.601 (see module docs).
                let w = c.dst_width as usize;
                let h = c.dst_height as usize;
                let y_plane = c.scratch;
                let cb_plane = c.scratch.add(w * h);
                let cr_plane = c.scratch.add(w * h + (w / 2) * (h / 2));
                let planes = [y_plane, cb_plane, cr_plane];
                let plane_steps = [c.dst_width, c.dst_width / 2, c.dst_width / 2];
                let status = (c.nv12_to_yuv420)(
                    src.as_ptr(),
                    self.src_pitch,
                    planes.as_ptr(),
                    plane_steps.as_ptr(),
                    roi,
                    c.stream_ctx,
                );
                if status != 0 {
                    return Err(anyhow!("nppiNV12ToYUV420 failed with NppStatus {status}"));
                }
                let const_planes = [
                    y_plane as *const u8,
                    cb_plane as *const u8,
                    cr_plane as *const u8,
                ];
                (c.ycbcr420_to_rgb)(
                    const_planes.as_ptr(),
                    plane_steps.as_ptr(),
                    dst,
                    c.dst_width * 3,
                    roi,
                    c.stream_ctx,
                )
            };
            if status != 0 {
                return Err(anyhow!(
                    "NPP color conversion failed with NppStatus {status}"
                ));
            }
        }
        Ok(())
    }
}
