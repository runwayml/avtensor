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

`tch` comes from crates.io; its version maps 1:1 to a libtorch version
(0.24.0 ↔ torch 2.11), so bump them together.

## Git

Do not add a `Co-Authored-By` trailer for Claude (or any AI assistant) to
commit messages, and do not add the "Generated with Claude Code" line to PR
bodies.
