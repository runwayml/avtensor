use std::fmt::Display;

use anyhow::Context;
use http::Uri;

pub fn is_s3_url(url: &str) -> bool {
    url.starts_with("s3://")
}

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct S3Config {
    pub endpoint_url: Option<String>,
    pub region: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub credentials: Option<String>,
    pub force_path_style: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct S3Uri {
    pub bucket: String,
    pub key: String,
}

impl Display for S3Uri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s3://{}/{}", self.bucket, self.key)
    }
}

pub fn parse_s3_uri(url: &str) -> Result<S3Uri, anyhow::Error> {
    let uri = url
        .parse::<Uri>()
        .context(format!("parsing URI: {}", url))?;
    let bucket = uri.host().context("missing bucket")?.to_string();
    let key = uri.path().strip_prefix('/').unwrap_or("").to_string();
    anyhow::ensure!(!key.is_empty(), "missing object key in {url}");
    Ok(S3Uri { bucket, key })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_s3_url() {
        assert!(is_s3_url("s3://bucket/object"));
        assert!(!is_s3_url("gs://bucket/object"));
        assert!(!is_s3_url("http://example.com"));
        assert!(!is_s3_url("file.mp4"));
    }

    #[test]
    fn test_parse_s3_uri() {
        let uri = "s3://bucket/object";
        let parsed = parse_s3_uri(uri).unwrap();
        assert_eq!(parsed.bucket, "bucket");
        assert_eq!(parsed.key, "object");
        assert_eq!(uri, parsed.to_string());

        let parsed = parse_s3_uri("s3://bucket/nested/object.mp4").unwrap();
        assert_eq!(parsed.bucket, "bucket");
        assert_eq!(parsed.key, "nested/object.mp4");

        assert!(parse_s3_uri("").is_err());
        assert!(parse_s3_uri("s3://bucket-without-key").is_err());
    }
}
