use std::sync::{LazyLock, OnceLock};

use anyhow::anyhow;
use google_cloud_gax::retry_policy::RetryPolicyExt;
use google_cloud_storage::client::Storage;
use google_cloud_storage::retry_policy::RetryableErrors;
use tokio::runtime::Runtime;

mod conversion;
mod decoder;
mod ffi;
mod util;

static GCS_ENDPOINT: LazyLock<String> = LazyLock::new(|| {
    std::env::var("GCS_ENDPOINT").unwrap_or_else(|_| "https://storage.googleapis.com".to_string())
});

/// The global tokio runtime to which async tasks are scheduled from Rust.
static GLOBAL_RUNTIME: OnceLock<Result<Runtime, anyhow::Error>> = OnceLock::new();

/// Lazily-initialized GCS client; manages a connection pool internally.
static GLOBAL_GCS_CLIENT: OnceLock<Result<Storage, anyhow::Error>> = OnceLock::new();

/// Lazily-initialized S3 client; manages a connection pool internally.
static GLOBAL_S3_CLIENT: OnceLock<Result<aws_sdk_s3::Client, anyhow::Error>> = OnceLock::new();

/// Retrieves a handle to the tokio runtime allocated by this library.
fn get_runtime() -> Result<&'static Runtime, anyhow::Error> {
    let runtime = GLOBAL_RUNTIME.get_or_init(|| {
        log::info!("Initializing Tokio runtime...");
        Ok(Runtime::new()?)
    });
    runtime
        .as_ref()
        .map_err(|e| anyhow!(e).context("Initializing tokio runtime"))
}

/// Blocks on `fut` using the global runtime, from sync or async contexts.
///
/// Client initialization happens lazily and may be triggered from inside the
/// runtime (e.g. a seek callback opening the first S3 read); a plain
/// `Runtime::block_on` would panic there, so drop to a blocking section
/// first when already inside the runtime.
fn block_on_anywhere<T>(fut: impl std::future::Future<Output = T>) -> Result<T, anyhow::Error> {
    let runtime = get_runtime()?;
    if tokio::runtime::Handle::try_current().is_ok() {
        Ok(tokio::task::block_in_place(|| runtime.block_on(fut)))
    } else {
        Ok(runtime.block_on(fut))
    }
}

/// Retrieves a handle to the GCS Storage client allocated by this library.
fn get_storage() -> Result<&'static Storage, anyhow::Error> {
    let storage = GLOBAL_GCS_CLIENT.get_or_init(|| {
        log::info!("Initializing GCS Storage Client...");
        log::info!("Using GCS endpoint: {}", *GCS_ENDPOINT);
        let storage = block_on_anywhere(
            Storage::builder()
                .with_endpoint(&*GCS_ENDPOINT)
                .with_retry_policy(
                    RetryableErrors
                        .with_attempt_limit(5)
                        .with_time_limit(std::time::Duration::from_secs(30)),
                )
                .build(),
        )??;
        Ok(storage)
    });
    storage
        .as_ref()
        .map_err(|e| anyhow!(e).context("Initializing GCS client"))
}

/// Retrieves a handle to the S3 client allocated by this library.
///
/// Credentials and region come from the standard AWS environment (env vars,
/// shared config, IMDS, ...). S3-compatible providers are supported through
/// the SDK's standard `AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL` overrides;
/// set `AVTENSOR_S3_FORCE_PATH_STYLE=1` for providers that require
/// path-style addressing.
fn get_s3_client() -> Result<&'static aws_sdk_s3::Client, anyhow::Error> {
    let client = GLOBAL_S3_CLIENT.get_or_init(|| {
        log::info!("Initializing S3 Client...");
        let config = block_on_anywhere(
            aws_config::defaults(aws_config::BehaviorVersion::latest())
                .retry_config(aws_config::retry::RetryConfig::standard().with_max_attempts(5))
                .load(),
        )?;
        let mut builder = aws_sdk_s3::config::Builder::from(&config);
        if std::env::var("AVTENSOR_S3_FORCE_PATH_STYLE").is_ok_and(|v| v == "1") {
            log::info!("AVTENSOR_S3_FORCE_PATH_STYLE=1: using path-style addressing");
            builder = builder.force_path_style(true);
        }
        Ok(aws_sdk_s3::Client::from_conf(builder.build()))
    });
    client
        .as_ref()
        .map_err(|e| anyhow!(e).context("Initializing S3 client"))
}
