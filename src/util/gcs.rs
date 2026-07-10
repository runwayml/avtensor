use std::fmt::Display;

use anyhow::Context;
use http::Uri;

pub fn is_gcs_url(url: &str) -> bool {
    url.starts_with("gs://")
}

#[derive(Clone, Debug)]
pub struct GCSUri {
    pub bucket: String,
    pub key: String,
}

impl Display for GCSUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "gs://{}/{}",
            self.bucket.trim_start_matches("projects/_/buckets/"),
            self.key
        )
    }
}

pub fn parse_gcs_uri(url: &str) -> Result<GCSUri, anyhow::Error> {
    let uri = url
        .parse::<Uri>()
        .context(format!("parsing URI: {}", url))?;
    let bucket = format!(
        "projects/_/buckets/{}",
        uri.host().context("missing bucket")?
    );
    let key = uri.path().strip_prefix('/').unwrap_or("").to_string();
    Ok(GCSUri { bucket, key })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gcs_url() {
        assert!(is_gcs_url("gs://bucket/object"));
        assert!(!is_gcs_url("http://example.com"));
        assert!(!is_gcs_url("file.mp4"));
        assert!(!is_gcs_url("./file.mp4"));
    }

    #[test]
    fn test_parse_gcs_uri() {
        let uri = "gs://bucket/object";
        let GCSUri { bucket, key } = parse_gcs_uri(uri).unwrap();
        assert_eq!(bucket, "projects/_/buckets/bucket");
        assert_eq!(key, "object");
        assert_eq!(uri, GCSUri { bucket, key }.to_string());

        let uri = "gs://bucket/nested/object";
        let GCSUri { bucket, key } = parse_gcs_uri(uri).unwrap();
        assert_eq!(bucket, "projects/_/buckets/bucket");
        assert_eq!(key, "nested/object");
        assert_eq!(uri, GCSUri { bucket, key }.to_string());

        assert!(parse_gcs_uri("").is_err());
    }
}
