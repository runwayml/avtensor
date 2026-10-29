// Exposes `at::from_blob` with a deleter to Rust. tch-rs wraps only the
// non-owning overload (`at_tensor_of_blob`); this one lets a buffer allocated
// outside torch's CPU allocator be owned by a tensor and freed by `deleter`
// when the last reference goes away. See src/alloc.rs for why. Also swaps
// storage buffers and trims storage sizes, which tch-rs has no bindings for
// (see `grow_rows` and `trim_rows`).
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

// Swaps the buffers of `a`'s and `b`'s storages (data and byte size). Every
// tensor sharing a storage, views included, sees its new buffer. Returns
// nullptr on success, or a malloc'd message the caller frees.
char *avtensor_swap_storage_buffers(const at::Tensor *a, const at::Tensor *b) {
  try {
    c10::StorageImpl *sa = a->storage().unsafeGetStorageImpl();
    c10::StorageImpl *sb = b->storage().unsafeGetStorageImpl();
    const size_t a_bytes = sa->nbytes();
    const size_t b_bytes = sb->nbytes();
    at::DataPtr b_data = sb->set_data_ptr(at::DataPtr(nullptr, b->device()));
    at::DataPtr a_data = sa->set_data_ptr(std::move(b_data));
    sb->set_data_ptr(std::move(a_data));
    sa->set_nbytes(b_bytes);
    sb->set_nbytes(a_bytes);
    return nullptr;
  } catch (const std::exception &e) {
    return strdup(e.what());
  } catch (...) {
    return strdup("unknown exception swapping storage buffers");
  }
}

// Lowers the byte size of `t`'s storage to `nbytes`, keeping its buffer (the
// deleter frees the whole allocation as before). Returns nullptr on success,
// or a malloc'd message the caller frees.
char *avtensor_trim_storage(const at::Tensor *t, size_t nbytes) {
  try {
    c10::StorageImpl *storage = t->storage().unsafeGetStorageImpl();
    if (nbytes > storage->nbytes()) {
      return strdup("trimming cannot grow a storage");
    }
    storage->set_nbytes(nbytes);
    return nullptr;
  } catch (const std::exception &e) {
    return strdup(e.what());
  } catch (...) {
    return strdup("unknown exception trimming a storage");
  }
}

}  // extern "C"
