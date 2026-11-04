# Vendored torch-sys 0.26.0

This is torch-sys 0.26.0 from crates.io (upstream commit
4227b89b72059b0651ff83a38637693574e0a2d6 of
https://github.com/LaurentMazare/tch-rs, `torch-sys/`), wired in through
`[patch.crates-io]` in the top-level `Cargo.toml`. It exists because no tch
release supports libtorch 2.14 yet; drop it once one does.

Changes from upstream, so the same source builds against torch 2.13 and 2.14:

- `build.rs`: compile `libtch` as C++20 (`-std=c++20`, `/std:c++20`).
  torch 2.14's headers reject anything older; torch 2.13 accepts C++20 too.
- `libtch/torch_api_generated.cpp`: `atg_cholesky`, `atg_cholesky_out`,
  `atg_qr` and `atg_qr_q` call `torch::linalg_cholesky*` and
  `torch::linalg_qr*`. torch 2.14 removed the deprecated `torch::cholesky`
  and `torch::qr`; the linalg forms exist in both versions and compute the
  same results (`qr(some=true)` is `linalg_qr(mode="reduced")`). avtensor
  calls none of these four ops.

The version check in `build.rs` still expects 2.13.0; builds against other
versions set `LIBTORCH_BYPASS_VERSION_CHECK=1`, as the workflows do.
