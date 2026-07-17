use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock};

use anyhow::{anyhow, Context};
use google_cloud_gax::retry_policy::RetryPolicyExt;
use google_cloud_storage::client::Storage;
use google_cloud_storage::retry_policy::RetryableErrors;
use tokio::runtime::Runtime;

mod conversion;
mod decoder;
mod ffi;
mod util;

use util::s3::S3Config;

static GCS_ENDPOINT: LazyLock<String> = LazyLock::new(|| {
    std::env::var("GCS_ENDPOINT").unwrap_or_else(|_| "https://storage.googleapis.com".to_string())
});

/// The global tokio runtime to which async tasks are scheduled from Rust.
static GLOBAL_RUNTIME: OnceLock<Result<Runtime, anyhow::Error>> = OnceLock::new();

/// Lazily-initialized GCS client; manages a connection pool internally.
static GLOBAL_GCS_CLIENT: OnceLock<Result<Storage, anyhow::Error>> = OnceLock::new();

/// Lazily-initialized S3 clients, keyed by the [`S3Config`] they were built from.
static S3_CLIENTS: LazyLock<Mutex<HashMap<S3Config, aws_sdk_s3::Client>>> =
    LazyLock::new(Mutex::default);

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

/// How the S3 client obtains credentials, resolved from an [`S3Config`].
enum S3CredentialsSource {
    /// The SDK's default provider chain.
    Default,
    /// Container credentials (`AWS_CONTAINER_CREDENTIALS_*`) exclusively.
    Container,
    /// Static credentials passed explicitly in the config.
    Static(aws_sdk_s3::config::Credentials),
}

impl S3CredentialsSource {
    /// Human-readable name for logging (never the credentials themselves).
    fn label(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Container => "container",
            Self::Static(_) => "static",
        }
    }

    /// The provider to pin on the client, or `None` to keep the base
    /// configuration's provider chain.
    fn into_provider(self) -> Option<aws_sdk_s3::config::SharedCredentialsProvider> {
        use aws_sdk_s3::config::SharedCredentialsProvider;
        match self {
            Self::Default => None,
            Self::Container => Some(SharedCredentialsProvider::new(
                aws_config::ecs::EcsCredentialsProvider::builder().build(),
            )),
            Self::Static(creds) => Some(SharedCredentialsProvider::new(creds)),
        }
    }
}

fn resolve_s3_credentials(config: &S3Config) -> Result<S3CredentialsSource, anyhow::Error> {
    match (&config.access_key_id, &config.secret_access_key) {
        (Some(id), Some(secret)) => {
            if config.credentials.is_some() {
                return Err(anyhow!(
                    "S3 config cannot combine static credentials \
                     (access_key_id/secret_access_key) with a credentials mode"
                ));
            }
            return Ok(S3CredentialsSource::Static(
                aws_sdk_s3::config::Credentials::new(
                    id,
                    secret,
                    config.session_token.clone(),
                    None,
                    "avtensor-s3-config",
                ),
            ));
        }
        (None, None) => {}
        _ => {
            return Err(anyhow!(
                "S3 config requires access_key_id and secret_access_key to be set together"
            ));
        }
    }
    if config.session_token.is_some() {
        return Err(anyhow!(
            "S3 config session_token requires access_key_id and secret_access_key"
        ));
    }
    match config.credentials.as_deref() {
        Some("container") => Ok(S3CredentialsSource::Container),
        Some("default") | None => Ok(S3CredentialsSource::Default),
        Some(other) => Err(anyhow!(
            "invalid S3 config credentials mode {other:?} \
             (expected \"default\" or \"container\")"
        )),
    }
}

/// Builds an S3 client for `config`.
///
/// Fields set on the config are authoritative and the environment is never
/// consulted for them; unset fields fall back to the standard AWS
/// environment (env vars, shared config, IMDS, ...), including the SDK's
/// `AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL` endpoint overrides, plus
/// `AVTENSOR_S3_FORCE_PATH_STYLE=1` for path-style addressing.
fn build_s3_client(config: &S3Config) -> Result<aws_sdk_s3::Client, anyhow::Error> {
    // Step 1: resolve the credentials cluster — the four interdependent
    // fields (keys, token, mode) become one validated provider decision.
    let credentials = resolve_s3_credentials(config)?;

    // Step 2: the independent knobs. Path style is the only one with an env
    // fallback of its own; endpoint and region are read off the config when
    // applied in step 4.
    let force_path_style = config
        .force_path_style
        .unwrap_or_else(|| std::env::var("AVTENSOR_S3_FORCE_PATH_STYLE").is_ok_and(|v| v == "1"));

    log::info!(
        "Initializing S3 client (endpoint={:?}, region={:?}, credentials={}, \
         force_path_style={force_path_style})",
        config.endpoint_url,
        config.region,
        credentials.label(),
    );

    // Step 3: always start from the shared AWS config then layers the config's pinned fields on top.
    let shared =
        block_on_anywhere(aws_config::defaults(aws_config::BehaviorVersion::latest()).load())?;
    let mut builder = aws_sdk_s3::config::Builder::from(&shared);

    // Step 4: apply overrides — set only what the config pins; anything
    // left unset keeps the base's (i.e. the environment's) answer.
    builder = builder
        .retry_config(aws_config::retry::RetryConfig::standard().with_max_attempts(5))
        .force_path_style(force_path_style);
    if let Some(endpoint_url) = &config.endpoint_url {
        builder = builder.endpoint_url(endpoint_url);
    }
    if let Some(region) = &config.region {
        builder = builder.region(aws_sdk_s3::config::Region::new(region.clone()));
    }
    if let Some(provider) = credentials.into_provider() {
        builder = builder.credentials_provider(provider);
    }
    Ok(aws_sdk_s3::Client::from_conf(builder.build()))
}

/// Retrieves the S3 client for `config`, creating and caching it on first
/// use. Clients are cached per distinct config, so one process can read from
/// several S3-compatible stores at the same time; `None` selects the purely
/// environment-configured client.
///
/// Configuration comes from the explicit `config` first; anything it leaves
/// unset falls back to the standard AWS environment (env vars, shared
/// config, IMDS, ...), including the SDK's `AWS_ENDPOINT_URL_S3` /
/// `AWS_ENDPOINT_URL` endpoint overrides. `AVTENSOR_S3_FORCE_PATH_STYLE=1`
/// enables path-style addressing when the config does not say otherwise.
fn get_s3_client(config: Option<&S3Config>) -> Result<aws_sdk_s3::Client, anyhow::Error> {
    let key = config.cloned().unwrap_or_default();
    if let Some(client) = S3_CLIENTS
        .lock()
        .expect("S3 client cache lock poisoned")
        .get(&key)
    {
        return Ok(client.clone());
    }
    // Built outside the lock: construction may block on network I/O (shared
    // config, IMDS). A concurrent builder of the same config just does
    // duplicate work; the first insert wins.
    let client = build_s3_client(&key).context("Initializing S3 client")?;
    Ok(S3_CLIENTS
        .lock()
        .expect("S3 client cache lock poisoned")
        .entry(key)
        .or_insert(client)
        .clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only cases where every consulted field is explicit, so the tests do
    // not depend on (or race over) process environment variables.

    #[test]
    fn test_resolve_s3_credentials_static() {
        let source = resolve_s3_credentials(&S3Config {
            access_key_id: Some("id".into()),
            secret_access_key: Some("secret".into()),
            session_token: Some("token".into()),
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(source, S3CredentialsSource::Static(_)));
    }

    #[test]
    fn test_resolve_s3_credentials_modes() {
        let mode = |credentials: &str| {
            resolve_s3_credentials(&S3Config {
                credentials: Some(credentials.into()),
                ..Default::default()
            })
        };
        assert!(matches!(
            mode("container").unwrap(),
            S3CredentialsSource::Container
        ));
        assert!(matches!(
            mode("default").unwrap(),
            S3CredentialsSource::Default
        ));
        assert!(mode("bogus").is_err());
    }

    #[test]
    fn test_resolve_s3_credentials_invalid_combinations() {
        // Incomplete static pair.
        assert!(resolve_s3_credentials(&S3Config {
            access_key_id: Some("id".into()),
            ..Default::default()
        })
        .is_err());
        // Session token without the static pair.
        assert!(resolve_s3_credentials(&S3Config {
            session_token: Some("token".into()),
            credentials: Some("default".into()),
            ..Default::default()
        })
        .is_err());
        // Static credentials combined with a credentials mode.
        assert!(resolve_s3_credentials(&S3Config {
            access_key_id: Some("id".into()),
            secret_access_key: Some("secret".into()),
            credentials: Some("container".into()),
            ..Default::default()
        })
        .is_err());
    }
}
