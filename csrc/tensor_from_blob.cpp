// Exposes `at::from_blob` with a deleter to Rust. tch-rs wraps only the
// non-owning overload (`at_tensor_of_blob`); this one lets a buffer allocated
// outside torch's CPU allocator be owned by a tensor and freed by `deleter`
// when the last reference goes away. See src/alloc.rs for why.
#include <ATen/ATen.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <utility>

extern "C" {

typedef void (*avtensor_blob_deleter)(void *data, void *ctx);

// Wraps `data` in a contiguous CPU tensor of `sizes` and element type
// `scalar_type` (a c10::ScalarType value). Ownership of `data` moves to the
// tensor's storage: `deleter(data, ctx)` runs when the last tensor sharing
// that storage is released, on whichever thread releases it.
//
// Returns a heap-allocated at::Tensor, the object tch-rs calls `C_tensor`
// and frees with `at_free`. On failure returns nullptr and, if `err` is not
// null, stores a malloc'd message in *err that the caller frees.
void *avtensor_tensor_from_blob(void *data, const int64_t *sizes, size_t ndim,
                                int32_t scalar_type,
                                avtensor_blob_deleter deleter, void *ctx,
                                char **err) {
  try {
    auto options = at::TensorOptions()
                       .dtype(static_cast<at::ScalarType>(scalar_type))
                       .device(at::kCPU);
    at::Tensor tensor = at::from_blob(
        data, at::IntArrayRef(sizes, ndim),
        [deleter, ctx](void *p) { deleter(p, ctx); }, options);
    return new at::Tensor(std::move(tensor));
  } catch (const std::exception &e) {
    if (err) *err = strdup(e.what());
    return nullptr;
  } catch (...) {
    if (err) *err = strdup("unknown exception in at::from_blob");
    return nullptr;
  }
}

}  // extern "C"
