//! Compiles the C++ shim in `csrc/` that exposes `at::from_blob` with a
//! deleter to Rust (used by `src/alloc.rs`). tch-rs only wraps the non-owning
//! overload, so this is the one piece of libtorch we have to reach through
//! C++. The include directories and C++ ABI are resolved the same way
//! torch-sys resolves them, so both shims link against the same libtorch.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const PYTHON_PROBE: &str = r"
import torch
from torch.utils import cpp_extension
print('CXX11_ABI:', int(torch._C._GLIBCXX_USE_CXX11_ABI))
for path in cpp_extension.include_paths():
    print('INCLUDE:', path)
";

struct LibtorchConfig {
    include_dirs: Vec<PathBuf>,
    cxx11_abi: String,
}

fn env_var_rerun(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

/// Mirrors torch-sys: `python` inside a virtualenv, `python3` otherwise,
/// unless maturin/PyO3 point at a specific interpreter.
fn python_interpreter() -> PathBuf {
    if let Some(p) = env_var_rerun("PYO3_PYTHON") {
        return PathBuf::from(p);
    }
    if env::var_os("VIRTUAL_ENV").is_some() {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn config_from_python() -> Result<LibtorchConfig, String> {
    let python = python_interpreter();
    let output = Command::new(&python)
        .arg("-c")
        .arg(PYTHON_PROBE)
        .output()
        .map_err(|e| format!("running {python:?}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{python:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut include_dirs = vec![];
    let mut cxx11_abi = None;
    for line in stdout.lines() {
        if let Some(path) = line.strip_prefix("INCLUDE: ") {
            include_dirs.push(PathBuf::from(path));
        } else if let Some(abi) = line.strip_prefix("CXX11_ABI: ") {
            cxx11_abi = Some(abi.trim().to_owned());
        }
    }
    match cxx11_abi {
        Some(cxx11_abi) if !include_dirs.is_empty() => Ok(LibtorchConfig {
            include_dirs,
            cxx11_abi,
        }),
        _ => Err(format!("unexpected probe output: {stdout}")),
    }
}

fn config_from_libtorch_dir() -> Result<LibtorchConfig, String> {
    let root = env_var_rerun("LIBTORCH")
        .map(PathBuf::from)
        .or_else(|| {
            // torch-sys exports the lib dir it linked against (`links = "tch"`).
            env::var("DEP_TCH_LIBTORCH_LIB")
                .ok()
                .map(|lib| PathBuf::from(lib).join(".."))
        })
        .ok_or("neither LIBTORCH_USE_PYTORCH nor LIBTORCH is set")?;
    let include_dirs = match env_var_rerun("LIBTORCH_INCLUDE") {
        Some(include) => vec![
            PathBuf::from(&include).join("include"),
            PathBuf::from(&include).join("include/torch/csrc/api/include"),
        ],
        None => vec![
            root.join("include"),
            root.join("include/torch/csrc/api/include"),
        ],
    };
    Ok(LibtorchConfig {
        include_dirs,
        cxx11_abi: env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or_else(|| "1".to_owned()),
    })
}

fn main() {
    println!("cargo:rerun-if-changed=csrc/tensor_from_blob.cpp");

    let config = if env_var_rerun("LIBTORCH_USE_PYTORCH").is_some() {
        config_from_python()
    } else {
        config_from_libtorch_dir()
    }
    .unwrap_or_else(|e| panic!("locating libtorch headers for csrc/tensor_from_blob.cpp: {e}"));

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .warnings(false)
        .includes(&config.include_dirs)
        .file("csrc/tensor_from_blob.cpp");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        build.flag("/std:c++17");
    } else {
        build
            .flag("-std=c++17")
            .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={}", config.cxx11_abi));
    }
    build.compile("avtensor_blob");
}
