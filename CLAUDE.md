# avtensor

Rust crate + PyO3 Python extension (`import avtensor`) for high-performance
media decoding to torch Tensors. Built with maturin; links dynamically against
the active venv's libtorch (`LIBTORCH_USE_PYTORCH=1`) and FFmpeg >= 7.1
(`FFMPEG_PKG_CONFIG_PATH`). See `README.md` for the required environment
variables — builds and tests fail without them.

Run tests with `make test` (`cargo test --no-default-features`; the flag is a
required PyO3 workaround). Tests are self-contained — media fixtures are
generated locally with FFmpeg; end-to-end cloud-storage tests live in the
consuming project.

`tch` versions map 1:1 to a libtorch version (0.24.0 ↔ torch 2.11,
0.26.0 ↔ torch 2.13, 0.27.0 ↔ torch 2.14). No crates.io release supports
torch 2.14 yet, so `Cargo.toml` pins `tch` and `torch-sys` 0.27.0 to a commit
of https://github.com/RikHeijdens/tch-rs (branch `libtorch-2.14`: upstream
tch-rs plus only the 2.14 binding update). Move back to crates.io once
upstream releases 2.14 support; keep `tch` and `torch-sys` on the same
version.

`build.rs` compiles `csrc/tensor_from_blob.cpp` against the libtorch headers
(located the same way torch-sys locates them); `src/alloc.rs` uses it to give
decoded-output buffers to torch via `from_blob` with a deleter instead of
`at::empty`. Keep the two in sync if the C API changes.

## Git

Do not add a `Co-Authored-By` trailer for Claude (or any AI assistant) to
commit messages, and do not add the "Generated with Claude Code" line to PR
bodies.
