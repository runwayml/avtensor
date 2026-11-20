use anyhow::{anyhow, Context};
use google_cloud_storage::read_object::ReadObjectResponse;
use rsmpeg::{
    avformat::{AVFormatContextInput, AVIOContextContainer, AVIOContextCustom},
    avutil::{AVDictionary, AVMem},
    ffi,
};
use std::{
    ffi::CStr,
    fs::File,
    io::{Read, Seek, SeekFrom},
    sync::{Arc, Mutex, Weak},
};
use tokio::task::JoinHandle;

use crate::{
    get_runtime, get_s3_client, get_storage,
    util::{
        gcs::{is_gcs_url, parse_gcs_uri, GCSUri},
        s3::{is_s3_url, parse_s3_uri, S3Config, S3Uri},
    },
};

/// IOContext used by the [`avio_reader`]
struct IOContext {
    file: File,
    current: usize,
    file_size: usize,
}

/// Simple file-based avio_reader.
///
/// Used to study how FFmpeg interacts with the `read_packet` and `seek` callbacks.
///
/// This is likely less performant than the default AVIOFormatContextInput as it does not memory
/// map files.
#[allow(dead_code)]
pub fn avio_reader(filename: &CStr) -> Result<AVFormatContextInput, anyhow::Error> {
    let file = File::open(filename.to_str()?).context("opening file")?;
    let file_size = file.metadata()?.len() as usize;
    let read_buffer = Arc::new(Mutex::new(IOContext {
        file,
        current: 0,
        file_size,
    }));
    let seek_buffer = read_buffer.clone();

    let io_context: AVIOContextCustom = AVIOContextCustom::alloc_context(
        AVMem::new(4096), // The size of the buffer used by FFmpeg. Calls to `read_packet` will request data buffers with the size specified here. By default FFmpeg uses 32768 when opening file descriptors.
        false,
        vec![],
        Some(Box::new(move |_, buf| {
            let mut buffer = match read_buffer.lock() {
                Ok(x) => x,
                Err(_) => return -1,
            };
            let mut read_ = |buf: &mut [u8]| -> Result<i32, anyhow::Error> {
                let right = buffer.file_size.min(buffer.current + buf.len());
                if right <= buffer.current {
                    return Ok(ffi::AVERROR_EOF);
                }
                let read_len = right - buffer.current;
                let read_len = buffer
                    .file
                    .read(&mut buf[0..read_len])
                    .context("reading from file")?;
                if read_len == 0 {
                    return Ok(ffi::AVERROR_EOF);
                }
                buffer.current = right;

                log::debug!(
                    "read callback prev_pos: {}, current_pos: {}, file_size: {}, buf_size: {}",
                    buffer.current - right,
                    buffer.current,
                    buffer.file_size,
                    buf.len(),
                );
                Ok(read_len as i32)
            };
            read_(buf).unwrap_or(-1)
        })),
        None,
        Some(Box::new(
            move |_: &mut Vec<u8>, offset: i64, whence: i32| {
                let mut buffer = match seek_buffer.lock() {
                    Ok(x) => x,
                    Err(_) => return -1,
                };
                let mut seek_ = |offset: i64, whence: i32| -> Result<i64, anyhow::Error> {
                    const AVSEEK_SIZE: i32 = ffi::AVSEEK_SIZE as i32;
                    const AVSEEK_FORCE: i32 = ffi::AVSEEK_FORCE as i32;
                    let position = match whence {
                        libc::SEEK_SET => buffer.file.seek(SeekFrom::Start(offset as u64))?,
                        libc::SEEK_CUR => buffer.file.seek(SeekFrom::Current(offset))?,
                        libc::SEEK_END => buffer.file.seek(SeekFrom::End(offset))?,
                        AVSEEK_SIZE => {
                            // Passing this as the "whence" parameter to a seek function causes it to return the filesize without seeking anywhere.
                            log::debug!("AVSEEK_SIZE requested, returning {}", buffer.file_size);
                            return Ok(buffer.file_size as i64);
                        }
                        AVSEEK_FORCE => {
                            // TODO (rikheijdens): I don't think this is actually used.
                            log::debug!("AVSEEK_FORCE requested, ignoring");
                            return Err(anyhow!("AVSEEK_FORCE not supported"));
                        }
                        _ => return Err(anyhow!("Unsupported whence")),
                    };
                    log::debug!(
                        "seek callback prev_pos: {}, current_pos: {}, file_size: {}, offset: {}, whence: {}",
                        buffer.current,
                        position,
                        buffer.file_size,
                        offset,
                        whence
                    );
                    // Update the position
                    buffer.current = position as usize;
                    Ok(position as i64)
                };
                seek_(offset, whence).unwrap_or(-1)
            },
        )),
    );

    AVFormatContextInput::from_io_context(AVIOContextContainer::Custom(io_context))
        .context("Failed to create AVFormatContextInput")
}

/// Returns an [`AVFormatContextInput`] that demuxes an in-memory byte buffer.
pub fn memory_avio_reader(bytes: Vec<u8>) -> Result<AVFormatContextInput, anyhow::Error> {
    let size = bytes.len();
    let state = Arc::new(Mutex::new((bytes, 0usize))); // (data, read position)
    let seek_state = state.clone();

    let io_context: AVIOContextCustom = AVIOContextCustom::alloc_context(
        AVMem::new(32768),
        false,
        vec![],
        Some(Box::new(move |_, buf| {
            let Ok(mut state) = state.lock() else {
                return -1;
            };
            let (data, position) = &mut *state;
            let right = data.len().min(*position + buf.len());
            if right <= *position {
                return ffi::AVERROR_EOF;
            }
            let read_len = right - *position;
            buf[0..read_len].copy_from_slice(&data[*position..right]);
            *position = right;
            read_len as i32
        })),
        None,
        Some(Box::new(
            move |_: &mut Vec<u8>, offset: i64, whence: i32| {
                const AVSEEK_SIZE: i32 = ffi::AVSEEK_SIZE as i32;
                let Ok(mut state) = seek_state.lock() else {
                    return -1;
                };
                let (_, position) = &mut *state;
                let new_position = match whence {
                    libc::SEEK_SET => offset,
                    libc::SEEK_CUR => *position as i64 + offset,
                    libc::SEEK_END => size as i64 + offset,
                    AVSEEK_SIZE => return size as i64,
                    _ => return -1,
                };
                if !(0..=size as i64).contains(&new_position) {
                    return -1;
                }
                *position = new_position as usize;
                new_position
            },
        )),
    );

    AVFormatContextInput::from_io_context(AVIOContextContainer::Custom(io_context))
        .context("Failed to create AVFormatContextInput from memory buffer")
}

/// Default ceiling on how many bytes of a cloud object this reader will
/// buffer in memory.
const DEFAULT_MAX_CLOUD_OBJECT_SIZE: usize = 16 * 1024 * 1024 * 1024; // 16 GiB

/// The buffered-object size ceiling: `AVTENSOR_MAX_CLOUD_OBJECT_BYTES`
/// (bytes) when set to a valid integer, otherwise
/// [`DEFAULT_MAX_CLOUD_OBJECT_SIZE`]. Some assets legitimately exceed the
/// default (e.g. feature-length ProRes masters).
fn max_cloud_object_size() -> usize {
    env_bytes(
        "AVTENSOR_MAX_CLOUD_OBJECT_BYTES",
        DEFAULT_MAX_CLOUD_OBJECT_SIZE,
        0,
    )
}

/// Reads a byte count from the environment variable `name`, falling back to
/// `default` (with a warning) when it is set but not an integer >= `min`.
fn env_bytes(name: &str, default: usize, min: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.parse::<usize>() {
            Ok(bytes) if bytes >= min => bytes,
            _ => {
                log::warn!(
                    "Ignoring invalid {name} value {value:?}; using the default of {default} bytes"
                );
                default
            }
        },
        Err(_) => default,
    }
}

/// How much of a cloud object to fetch ahead of FFmpeg's reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prefetch {
    /// Stream the whole object as fast as it arrives. For decoding, which
    /// reads nearly all of it.
    All,
    /// Fetch only what FFmpeg asks for, a chunk at a time. For probing and
    /// planning, which read the container header and stop.
    Lazy,
    /// [`Prefetch::Lazy`] with this base chunk instead of
    /// `AVTENSOR_LAZY_CHUNK_BYTES`, so tests can use small objects.
    #[cfg(test)]
    LazyWithChunk(usize),
}

/// Default request granularity for [`Prefetch::Lazy`]: large enough that a
/// header usually arrives in one request, small enough that reaching it costs
/// little. Override with `AVTENSOR_LAZY_CHUNK_BYTES`.
const DEFAULT_LAZY_CHUNK: usize = 1 << 20;

fn lazy_chunk() -> usize {
    env_bytes("AVTENSOR_LAZY_CHUNK_BYTES", DEFAULT_LAZY_CHUNK, 1).min(MAX_LAZY_CHUNK)
}

/// Largest request [`Prefetch::Lazy`] grows to while FFmpeg scans forward.
const MAX_LAZY_CHUNK: usize = 16 << 20;

/// The request size under [`Prefetch::Lazy`]. It starts at the base chunk and
/// doubles (up to [`MAX_LAZY_CHUNK`]) while FFmpeg keeps moving forward
/// through the file, by sequential reads or forward seeks. A fragmented MP4
/// without a `sidx` makes the mov demuxer hop from `moof` to `moof` over each
/// `mdat` through the whole file, and a fixed chunk turns that into one serial
/// request per chunk. A seek back (behind the previous seek) ends the forward
/// run and resets the window, whether or not the bytes it lands on are
/// cached, so revisiting a header does not inherit a grown window.
struct LazyChunk {
    base: usize,
    next: usize,
    /// The last seek target: a seek before it goes back.
    position: usize,
}

impl LazyChunk {
    fn new(base: usize) -> Self {
        Self {
            base,
            next: base,
            position: 0,
        }
    }

    /// Records a seek to `position`. Hopping forward, even within bytes
    /// already fetched, continues the run; going back ends it.
    fn on_seek(&mut self, position: usize) {
        if position < self.position {
            self.next = self.base;
        }
        self.position = position;
    }

    /// The size of the next request of the forward run. Each one doubles the
    /// next.
    fn request(&mut self) -> usize {
        let size = self.next;
        self.next = self
            .next
            .saturating_mul(2)
            .min(MAX_LAZY_CHUNK.max(self.base));
        size
    }
}

/// A parsed cloud-storage URI.
#[derive(Clone)]
enum CloudUri {
    Gcs(GCSUri),
    S3(S3Uri, Option<S3Config>),
}

/// A byte stream read from cloud storage.
enum CloudChunkStream {
    Gcs(ReadObjectResponse),
    S3(aws_sdk_s3::primitives::ByteStream),
}

impl CloudChunkStream {
    /// Streams the next chunk of the object, or None when exhausted.
    async fn next_chunk(&mut self) -> Option<Result<bytes::Bytes, anyhow::Error>> {
        match self {
            CloudChunkStream::Gcs(resp) => resp.next().await.map(|r| r.map_err(|e| anyhow!(e))),
            CloudChunkStream::S3(stream) => {
                stream.try_next().await.map_err(|e| anyhow!(e)).transpose()
            }
        }
    }
}

/// What one ranged read learned about the object.
#[derive(Clone, Debug, PartialEq)]
struct ObjectMeta {
    /// Total object size in bytes.
    size: usize,
    /// The object version the bytes came from (GCS generation, S3 ETag), so
    /// later ranged reads of the same object can be pinned to it.
    version: Option<ObjectVersion>,
    /// Where the returned stream ends (exclusive): `offset + count` for a
    /// bounded read, the object's end otherwise.
    end: usize,
    /// Whether the stream is a bounded range. Not always what was asked
    /// for: S3 falls back to an open-ended read in some cases.
    bounded: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum ObjectVersion {
    Gcs(i64),
    /// A versioned bucket's version id: pins the read to exactly that version.
    S3VersionId(String),
    /// A strong ETag, sent as `If-Match`.
    S3ETag(String),
}

/// The version to pin later S3 reads to. A strong ETag comes first: pinning
/// with `If-Match` needs only `s3:GetObject`, whereas reading a version id
/// needs `s3:GetObjectVersion`, which a reader of a versioned bucket may not
/// have. A version id is the fallback when there is no strong ETag (`If-Match`
/// compares strongly, so a weak `W/` tag never matches); with neither, reads
/// are left unpinned.
fn s3_version(version_id: Option<&str>, etag: Option<&str>) -> Option<ObjectVersion> {
    if let Some(etag) = etag.filter(|etag| !etag.starts_with("W/")) {
        return Some(ObjectVersion::S3ETag(etag.to_string()));
    }
    version_id
        .filter(|id| !id.is_empty() && *id != "null")
        .map(|id| ObjectVersion::S3VersionId(id.to_string()))
}

/// A parsed `Content-Range: bytes first-last/total` header (`total` is `None`
/// for `*`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct ContentRange {
    last: usize,
    total: Option<usize>,
}

impl ContentRange {
    fn parse(header: &str) -> Option<Self> {
        // Range units are case-insensitive (RFC 9110 14.1).
        let (unit, rest) = header.trim().split_once(' ')?;
        if !unit.eq_ignore_ascii_case("bytes") {
            return None;
        }
        let (range, total) = rest.split_once('/')?;
        let (first, last) = range.split_once('-')?;
        let (first, last) = (first.parse::<usize>().ok()?, last.parse::<usize>().ok()?);
        if last < first {
            return None;
        }
        let total = match total {
            "*" => None,
            total => Some(total.parse::<usize>().ok().filter(|t| *t > last)?),
        };
        Some(Self { last, total })
    }
}

/// Opens a streaming read of `uri` starting at byte `offset`.
///
/// `count` bounds the request to that many bytes; `None` reads to the end.
/// `version` pins the read to the object version an earlier read saw: under
/// [`Prefetch::Lazy`] one object is read with several requests, and an object
/// overwritten between them would otherwise hand FFmpeg bytes from two
/// versions. A mismatch fails the read (GCS 404, S3 412); callers retry the
/// whole probe or decode.
async fn open_cloud_stream(
    uri: &CloudUri,
    offset: usize,
    count: Option<usize>,
    version: Option<&ObjectVersion>,
) -> Result<(ObjectMeta, CloudChunkStream), anyhow::Error> {
    match uri {
        CloudUri::Gcs(GCSUri { bucket, key }) => {
            use google_cloud_storage::model_ext::ReadRange;
            let storage = get_storage()?;
            let mut request = storage.read_object(bucket, key);
            match count {
                Some(count) => {
                    request =
                        request.set_read_range(ReadRange::segment(offset as u64, count as u64))
                }
                None if offset > 0 => {
                    request = request.set_read_range(ReadRange::offset(offset as u64))
                }
                None => {}
            }
            if let Some(ObjectVersion::Gcs(generation)) = version {
                request = request.set_generation(*generation);
            }
            let resp = request.send().await.context("reading object from GCS")?;
            let object = resp.object();
            let size = object.size as usize;
            let meta = ObjectMeta {
                size,
                version: Some(ObjectVersion::Gcs(object.generation)),
                end: count.map_or(size, |count| offset.saturating_add(count).min(size)),
                bounded: count.is_some(),
            };
            Ok((meta, CloudChunkStream::Gcs(resp)))
        }
        CloudUri::S3(S3Uri { bucket, key }, s3_config) => {
            let client = get_s3_client(s3_config.as_ref())?;
            let get = |count: Option<usize>| {
                let range = match count {
                    // HTTP byte ranges are inclusive on both ends.
                    Some(count) => format!("bytes={offset}-{}", offset.saturating_add(count) - 1),
                    None => format!("bytes={offset}-"),
                };
                let mut request = client.get_object().bucket(bucket).key(key).range(range);
                match version {
                    Some(ObjectVersion::S3VersionId(id)) => request = request.version_id(id),
                    Some(ObjectVersion::S3ETag(etag)) => request = request.if_match(etag),
                    _ => {}
                }
                request.send()
            };
            let mut bounded = count;
            let mut resp = match get(count).await {
                Ok(resp) => resp,
                // 416 on a bounded range: callers only ask for bytes inside
                // the object, so the store caps bounded spans (some gateways
                // do). Read open-ended from the same offset instead; the
                // reader then treats the stream as open-ended.
                Err(err)
                    if count.is_some()
                        && err.raw_response().map(|r| r.status().as_u16()) == Some(416) =>
                {
                    log::debug!(
                        "S3 refused a {count:?}-byte range at {offset}; re-reading open-ended"
                    );
                    bounded = None;
                    get(None).await.context("reading object from S3")?
                }
                Err(err) => return Err(anyhow::Error::new(err).context("reading object from S3")),
            };
            let mut range = resp.content_range().and_then(ContentRange::parse);
            // The response's Content-Length, when it has a usable one.
            let length = |resp: &aws_sdk_s3::operation::get_object::GetObjectOutput| {
                resp.content_length()
                    .and_then(|length| usize::try_from(length).ok())
            };
            if resp.content_range().is_none() {
                // The server ignored Range and sent the whole object (200).
                // Its body starts at byte 0, so it only lines up with a read
                // that starts there.
                if offset > 0 {
                    return Err(anyhow!(
                        "S3 endpoint ignored the Range header for a read at byte {offset}"
                    ));
                }
                // Keep it as the full read rather than asking again, unless
                // it carries no size (a chunked 200) and the request was
                // bounded: the open-ended read below may still report one.
                // An open-ended request would only get the same answer.
                if bounded.take().is_some() && length(&resp).is_none() {
                    resp = get(None).await.context("reading object from S3")?;
                    range = resp.content_range().and_then(ContentRange::parse);
                }
            } else if bounded.is_some() && range.and_then(|r| r.total).is_none() {
                // `bytes a-b/*` on a bounded response: the body says nothing
                // about where the object ends, so read open-ended instead,
                // whose range end does. Decided on the response received,
                // not the request: after a 416 fallback the response is
                // already open-ended, and asking again only costs a request.
                log::debug!("S3 response has no total size in Content-Range; re-reading unbounded");
                resp = get(None).await.context("reading object from S3")?;
                range = resp.content_range().and_then(ContentRange::parse);
                bounded = None;
            }
            let size = match (range, length(&resp)) {
                (
                    Some(ContentRange {
                        total: Some(total), ..
                    }),
                    _,
                ) => total,
                // Open-ended `bytes a-b/*`: the read runs to the object's end.
                (Some(ContentRange { last, .. }), _) if bounded.is_none() => last + 1,
                (None, Some(length)) => offset + length,
                _ => {
                    return Err(anyhow!(
                        "S3 response for {key} gives no usable object size \
                         (Content-Range {:?}, Content-Length {:?})",
                        resp.content_range(),
                        resp.content_length()
                    ))
                }
            };
            let meta = ObjectMeta {
                size,
                version: s3_version(resp.version_id(), resp.e_tag()),
                // A bounded response ends where its Content-Range says, which
                // can be short of what was asked; the request is the fallback.
                end: match (bounded, range) {
                    (None, _) => size,
                    (Some(_), Some(ContentRange { last, .. })) => (last + 1).min(size),
                    (Some(count), None) => offset.saturating_add(count).min(size),
                },
                bounded: bounded.is_some(),
            };
            Ok((meta, CloudChunkStream::S3(resp.body)))
        }
    }
}

/// Reads a media asset from cloud storage and caches the data to a buffer.
///
/// Arguments:
/// - `stream`: The byte stream to read data from.
/// - `start`: The index of the first byte in the buffer to start writing data to.
/// - `context`: A handle to the storage buffer to which we should write data as it comes in.
///
/// The task holds only a weak reference to the context: the opened input owns
/// it, so dropping the input drops the context, whose `Drop` aborts this task.
/// A strong reference here would keep the context, and the download, alive
/// until the object had been read to the end.
async fn cloud_asset_reader(
    mut stream: CloudChunkStream,
    start: usize,
    context: Weak<tokio::sync::Mutex<CloudStorageReaderContext>>,
) -> Result<(), anyhow::Error> {
    let mut current_position = start;
    while let Some(chunk) = stream.next_chunk().await {
        match chunk {
            Ok(data) => {
                let new_position = current_position + data.len();

                // The input was dropped: nobody will read these bytes.
                let Some(context) = context.upgrade() else {
                    return Ok(());
                };
                // Attempt to acquire exclusive access to the data buffer to write more data into it.
                let CloudStorageReaderContext { ref mut buffer, .. } = *context.lock().await;
                log::trace!(
                    "cloud_asset_reader: current_position={}, new_position={}, data.len()={}",
                    current_position,
                    new_position,
                    data.len()
                );

                // Verify we're not trying to write outside of the size of the buffer.
                if new_position > buffer.data.len() {
                    return Err(anyhow!(
                        "cloud_asset_reader: new_position={} > buffer.len()={}",
                        new_position,
                        buffer.data.len()
                    ));
                }

                // Copy data into the buffer
                buffer.data[current_position..new_position].copy_from_slice(&data);
                buffer.indicators[current_position..new_position].fill(true);

                // Update the write position
                current_position = new_position;

                // Check whether we should continue to consume the stream.
                let next_position = buffer
                    .indicators
                    .len()
                    .saturating_sub(1)
                    .min(new_position + 1);
                if buffer.indicators[next_position] {
                    // If the next position is marked as read, we should be able to stop this task
                    // because we've fetched the next segment of data already.
                    log::debug!(
                        "cloud_asset_reader: all data up to {} has been written - stopping",
                        next_position
                    );
                    break;
                }
            }
            Err(err) => {
                // Propagate the failure so the read callback can observe that
                // this reader stopped without filling its range, instead of
                // waiting forever for bytes that will never arrive.
                log::warn!("Error reading from cloud object: {}", err);
                return Err(err);
            }
        }
    }

    Ok(())
}

/// Returns an [`AVFormatContextInput`] capable of streaming data from cloud storage.
///
/// `s3_config` is the explicit S3 client configuration for `s3://` URIs (see
/// [`S3Config`]); it is ignored for other schemes.
pub fn cloud_storage_avio_reader(
    filename: &CStr,
    s3_config: Option<S3Config>,
    prefetch: Prefetch,
) -> Result<AVFormatContextInput, anyhow::Error> {
    let file_name_str = filename.to_str()?;
    let cloud_uri = if is_gcs_url(file_name_str) {
        CloudUri::Gcs(parse_gcs_uri(file_name_str)?)
    } else if is_s3_url(file_name_str) {
        CloudUri::S3(parse_s3_uri(file_name_str)?, s3_config)
    } else {
        log::debug!(
            "Opening filename using the default AVFormatContextInput implementation because the provided filename is not a cloud-storage URL: {}",
            filename.to_str().unwrap_or("<invalid UTF-8>")
        );
        // Restrict FFmpeg to the protocols avtensor actually supports on this
        // path: local files and HTTP(S). This blocks protocols like `concat`,
        // `subfile`, and `data` that a malicious input could otherwise use to
        // read unintended local files or reach internal network endpoints.
        let mut options = Some(AVDictionary::new(
            c"protocol_whitelist",
            c"file,http,https,tcp,tls,crypto",
            0,
        ));
        return AVFormatContextInput::builder()
            .url(filename)
            .options(&mut options)
            .open()
            .context("opening reader");
    };

    // The provided URL points to a file stored in cloud storage. We will now:
    //
    // 1. Open request to GET the object (n.b. that we are charged per # of requests regardless of how much data is transferred).
    // 2. Allocate a contiguous in-memory buffer with the size of the (remote) object.
    // 3. Read the object data into the buffer as fast as data comes in by spawning a task on a (multi-threaded) Tokio runtime.
    // 4. In the event of a seek, if data is requested that is not yet buffered, and the distance between the current buffer position and seek target exceeds a threshold, open a new request to GET the missing data.
    //      * Note: this must be robust to seeking "back" and "forth". We may want to keep the old connection alive.
    // 5. Allocate a AVIOContextCustom to serve data to FFmpeg implementing the required callbacks.

    let runtime = get_runtime()?;

    // Perform a blocking read to open the object and acquire metadata, such as size.
    // Under Lazy, the request size: `None` for Prefetch::All, which reads to
    // the end. Shared by the read and seek callbacks.
    let mut lazy = match prefetch {
        Prefetch::All => None,
        Prefetch::Lazy => Some(LazyChunk::new(lazy_chunk())),
        #[cfg(test)]
        Prefetch::LazyWithChunk(chunk) => Some(LazyChunk::new(chunk.max(1))),
    };
    let first_chunk = lazy.as_mut().map(LazyChunk::request);
    let (meta, stream) = runtime
        .block_on(open_cloud_stream(&cloud_uri, 0, first_chunk, None))
        .context("opening cloud object")?; // TODO (rikheijdens): bubble up 404 errors etc.
    let obj_size = meta.size;
    // Later ranged reads are pinned to the version this one saw.
    let version = meta.version.clone();
    let lazy = Arc::new(Mutex::new(lazy));

    log::debug!(
        "Opened handle to cloud object {} with size {} bytes",
        file_name_str,
        obj_size
    );

    // This reader buffers the entire object in memory, so a very large (or
    // size-spoofed) object could exhaust memory. Reject anything above the
    // configured ceiling rather than attempting the allocation.
    let max_object_size = max_cloud_object_size();
    if obj_size > max_object_size {
        return Err(anyhow!(
            "cloud object {file_name_str} is {obj_size} bytes, which exceeds the maximum \
             buffered size of {max_object_size} bytes (override with \
             AVTENSOR_MAX_CLOUD_OBJECT_BYTES)"
        ));
    }

    // Allocate reader context.
    let context = Arc::new(tokio::sync::Mutex::new(CloudStorageReaderContext {
        buffer: CloudStorageBuffer {
            // Allocate contiguous storage for the entire file, regardless whether
            // we will read it in full or not to avoid having to resize the data buffer.
            // Zero-initialized: on modern allocators this maps zero-pages lazily
            // (calloc), so it is effectively free for large buffers while keeping
            // the buffer sound — bytes are always initialized before FFmpeg reads
            // them, even if the `indicators` readiness tracking has a gap.
            data: vec![0u8; obj_size], // TODO (rikheijdens): Come up with something smarter for massive files to avoid huge allocations here.
            // Keep track of which parts of the file have been read.
            indicators: vec![false; obj_size],
            // The current read position (as seeked to by FFmpeg).
            read_position: 0,
        },
        readers: Vec::new(),
    }));

    // Spawn a future to sequentially read data from the stream into the StorageBuffer.
    let read_sequential_future = cloud_asset_reader(stream, 0, Arc::downgrade(&context));
    runtime.block_on(async {
        let mut context = context.lock().await;
        context.readers.push(DataReader {
            range: ReadRange {
                start: 0,
                end: meta.end,
                bounded: meta.bounded,
            },
            handle: runtime.spawn(read_sequential_future),
        });
    });

    // TODO (rikheijdens): we could make these configurable and tunable.
    let ffmpeg_buffer_size = 32768;
    let data_segment_size = 64 * 1024 * 1024; // 64 MiB

    let read_context = context.clone();
    let seek_context = context.clone();
    let read_uri = cloud_uri.clone();
    let seek_uri = cloud_uri.clone();
    let (read_lazy, seek_lazy) = (lazy.clone(), lazy);
    let (read_version, seek_version) = (version.clone(), version);
    // FFmpeg only sees a callback's -1, and `avformat_find_stream_info` can
    // still succeed with the streams it saw before a read failed, which
    // would hand probe and plan callers partial metadata. The callbacks
    // record the first failure here so opening reports it instead.
    let io_error: Arc<Mutex<Option<anyhow::Error>>> = Arc::default();
    let (read_error, seek_error) = (io_error.clone(), io_error.clone());
    let io_context: AVIOContextCustom = AVIOContextCustom::alloc_context(
        AVMem::new(ffmpeg_buffer_size),
        false,
        vec![],
        Some(Box::new(move |_, buf| {
            log::trace!("read_packet, buf.len()={}", buf.len());
            let runtime = match get_runtime() {
                Ok(r) => r,
                Err(_) => return -1,
            };
            // Respawn budget for this read call: when the reader task
            // fetching our range dies (transient network error, dropped
            // connection), we retry with a fresh reader a bounded number of
            // times before giving up.
            const MAX_READER_RESPAWNS: u32 = 3;
            let mut respawn_attempts: u32 = 0;
            runtime
                .block_on(async {
                    loop {
                        // The buffer access is scoped so the lock is released
                        // before we sleep or refetch — both to satisfy the
                        // await-holding-lock lint and to let reader tasks make
                        // progress while we wait. `stalled_at` carries the
                        // read position out of the scope when no live reader
                        // covers it.
                        let stalled_at = {
                            let CloudStorageReaderContext {
                                ref mut buffer,
                                ref readers,
                            } = &mut *read_context.lock().await;
                            let pos = buffer.read_position;
                            let right = buffer.data.len().min(pos + buf.len());
                            if right <= pos {
                                // No more data left to read.
                                return Ok::<i32, anyhow::Error>(ffi::AVERROR_EOF);
                            }
                            let read_len = right - pos;

                            // Verify that we have the required data ready in our local buffer.
                            let data_ready = buffer.indicators[pos..right].iter().all(|b| *b);

                            if data_ready {
                                // Data has been fetched and is ready to be handed to FFmpeg
                                buf[0..read_len].copy_from_slice(&buffer.data[pos..right]);
                                buffer.read_position = right;

                                log::debug!(
                                    "read callback prev_pos: {}, current_pos: {}, buffer.data.len(): {}, buffer.data.capacity(): {}, buf.len(): {}",
                                    pos,
                                    buffer.read_position,
                                    buffer.data.len(),
                                    buffer.data.capacity(),
                                    buf.len(),
                                );
                                return Ok(read_len as i32);
                            }

                            // The data isn't ready. Find the first byte of
                            // the hole: FFmpeg's read position can lag the
                            // fetched frontier (data before the hole may
                            // already be marked), and both the liveness check
                            // and any respawn must target the hole itself. A
                            // reader started inside already-marked territory
                            // would stop as soon as it sees marked bytes
                            // ahead of it, without ever reaching the hole.
                            // The unwrap_or is unreachable: data_ready was
                            // false, so an unmarked byte exists at or after
                            // pos.
                            let first_missing = buffer.indicators[pos..right]
                                .iter()
                                .position(|b| !b)
                                .map(|off| pos + off)
                                .unwrap_or(pos);

                            // A reader claimed the hole if its range holds it;
                            // a live one is still fetching it, so keep
                            // waiting. Under Lazy nothing was ever asked to
                            // fetch past the frontier, so an unclaimed hole is
                            // expected rather than a reader that died.
                            let mut claiming = readers.iter().filter(|reader| {
                                reader.range.start <= first_missing
                                    && first_missing < reader.range.end
                            });
                            // The end of the range a dead reader left unfilled.
                            let claimed = claiming
                                .clone()
                                .map(|reader| (reader.range.end, reader.range.bounded))
                                .max();
                            let covered = claiming.any(|reader| !reader.handle.is_finished());
                            (!covered).then_some((first_missing, claimed))
                        };

                        if let Some((pos, claimed_end)) = stalled_at {
                            let claimed = claimed_end.is_some();
                            // A hole nobody claimed is just the next chunk to
                            // fetch. One a reader claimed and did not deliver
                            // means that reader died: respawn with backoff,
                            // and fail once the budget is spent rather than
                            // looping forever.
                            if claimed && respawn_attempts >= MAX_READER_RESPAWNS {
                                return Err(anyhow!(
                                    "cloud reader stalled: bytes at {} were not fetched \
                                     after {} respawn attempts",
                                    pos,
                                    MAX_READER_RESPAWNS
                                ));
                            }
                            if claimed {
                                respawn_attempts += 1;
                                let backoff = std::time::Duration::from_millis(
                                    10u64 << (respawn_attempts - 1),
                                );
                                log::warn!(
                                    "cloud reader covering byte {pos} is gone; respawning \
                                     (attempt {respawn_attempts}/{MAX_READER_RESPAWNS}) \
                                     after {backoff:?}"
                                );
                                tokio::time::sleep(backoff).await;
                            }
                            // An unclaimed hole continues a sequential read,
                            // so the request grows; a respawn refetches what
                            // the dead reader left of its range.
                            // An open-ended reader resumes open-ended: some
                            // gateways cap bounded spans, so the remainder of
                            // an open-ended read may not be requestable as one.
                            let count = read_lazy.lock().unwrap().as_mut().and_then(|lazy| {
                                match claimed_end {
                                    Some((end, true)) => Some(end - pos),
                                    Some((_, false)) => None,
                                    None => Some(lazy.request()),
                                }
                            });
                            let (meta, stream) =
                                open_cloud_stream(&read_uri, pos, count, read_version.as_ref())
                                    .await
                                    .context("reopening cloud stream after a reader stall")?;
                            let read_future =
                                cloud_asset_reader(stream, pos, Arc::downgrade(&read_context));
                            let CloudStorageReaderContext {
                                ref mut readers, ..
                            } = *read_context.lock().await;
                            readers.push(DataReader {
                                range: ReadRange {
                                    start: pos,
                                    end: meta.end,
                                    bounded: meta.bounded,
                                },
                                handle: runtime.spawn(read_future),
                            });
                            continue;
                        }

                        // A reader is still fetching the range; wait briefly for
                        // it instead of busy-spinning on the lock.
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                })
                .unwrap_or_else(|e| {
                    // FFmpeg only sees the -1; make sure the reason is logged.
                    log::error!("cloud read callback failed: {e:?}");
                    read_error.lock().unwrap().get_or_insert(e);
                    -1
                })
        })),
        None,
        Some(Box::new(
            move |_: &mut Vec<u8>, offset: i64, whence: i32| {
                log::trace!("seek_callback, offset: {}, whence: {}", offset, whence);
                const AVSEEK_SIZE: i32 = ffi::AVSEEK_SIZE as i32;

                // Fast track any queries for the file size since this should be known at this point.
                if whence == AVSEEK_SIZE {
                    log::debug!("AVSEEK_SIZE requested, returning {}", obj_size);
                    return obj_size as i64;
                }

                let runtime = match get_runtime() {
                    Ok(r) => r,
                    Err(_) => return -1,
                };
                runtime.block_on(async {
                    let CloudStorageReaderContext {
                        ref mut buffer,
                        ref mut readers
                    } = *seek_context.lock().await;

                    let position = match whence {
                        libc::SEEK_SET => offset as usize,
                        libc::SEEK_CUR => (buffer.read_position as i64 + offset) as usize,
                        libc::SEEK_END => {
                            let new_position = (buffer.data.len() as i64 + offset) as usize;
                            // TODO (rikheijdens): Need to do more testing to know if this will be negative, it should not be.
                            if new_position > buffer.data.len() {
                                log::error!(
                                    "Expected SEEK_END offset to be negative, received offset: {}.",
                                    offset
                                );
                                return Err(anyhow!("Expected SEEK_END offset to be negative."));
                            }
                            new_position
                        }
                        other => {
                            log::warn!("Unexpected seek whence: {}", other);
                            return Err(anyhow!("Unexpected seek whence: {other}"));
                        }
                    };

                    log::debug!(
                        "seek callback prev_pos: {}, current_pos: {}, file_size: {}, offset: {}, whence: {}",
                        buffer.read_position,
                        position,
                        buffer.data.len(),
                        offset,
                        whence
                    );

                    // Ensure we don't seek outside of the buffer.
                    buffer.read_position = position.min(buffer.data.len());
                    // Before the cached-data return below: a seek back ends
                    // the forward run even when its bytes are already here.
                    if let Some(lazy) = seek_lazy.lock().unwrap().as_mut() {
                        lazy.on_seek(position);
                    }

                    // Check if we need to spawn a new task to ensure data comes in in a timely manner.
                    let right = buffer
                        .data
                        .len()
                        .min(buffer.read_position + ffmpeg_buffer_size);
                    let data_ready = buffer.indicators[buffer.read_position..right]
                        .iter()
                        .all(|b| *b);

                    if data_ready {
                        // All good - we already have the next data segment buffered.
                        return Ok(position as i64);
                    }

                    // Figure out if we need to spawn a new task to fetch the next data segment,
                    // or whether we wait for an already running loader to pull the data.
                    // Only the `distance < data_segment_size` test below reads
                    // this, so stop once we are already that far back: the
                    // outcome cannot change, and under Prefetch::Lazy a seek
                    // past the fetched frontier would otherwise walk the whole
                    // gap.
                    let floor = position.saturating_sub(data_segment_size);
                    let mut closest_available_byte = floor;
                    for i in (floor..position).rev() {
                        if buffer.indicators[i] {
                            closest_available_byte = i;
                            break;
                        }
                    }

                    let distance = buffer.read_position - closest_available_byte;
                    if distance < data_segment_size {
                        // The distance between the first available byte is smaller than the data segment size.
                        // If we still have an active data loader with a range in which closest_available_byte lies
                        // then we'll assume the data will come in from that data loader.
                        for reader in readers.iter() {
                            if reader.range.start <= closest_available_byte
                                && closest_available_byte < reader.range.end
                                && !reader.handle.is_finished()
                            {
                                // We have an active reader that is already fetching the data we need,
                                // we just need to wait for it to come in.
                                return Ok(position as i64);
                            }
                        }
                    }

                    // There isn't a reader that is reading any data that we're after, or that is close to it
                    // spawn a new one by making a range request, to minimize the number of requests that we're making
                    // we'll request a read range starting from the requested offset to the end of the file.
                    let count = seek_lazy
                        .lock()
                        .unwrap()
                        .as_mut()
                        .map(LazyChunk::request);
                    let (meta, stream) =
                        open_cloud_stream(&seek_uri, position, count, seek_version.as_ref())
                            .await?;

                    let read_future =
                        cloud_asset_reader(stream, position, Arc::downgrade(&seek_context));

                    // Spawn the future and keep track of the reader.
                    log::debug!("Spawned new DataReader to read range [{}..{}]", position, meta.end);
                    readers.push(DataReader {
                        range: ReadRange {
                            start: position,
                            end: meta.end,
                            bounded: meta.bounded,
                        },
                        handle: runtime.spawn(read_future)
                    });

                    Ok(position as i64)
                }).unwrap_or_else(|e| {
                    log::error!("cloud seek callback failed: {e:?}");
                    seek_error.lock().unwrap().get_or_insert(e);
                    -1
                })
            },
        )),
    );

    let opened = AVFormatContextInput::from_io_context(AVIOContextContainer::Custom(io_context));
    // A failed fetch during open (an overwritten object, a dropped
    // connection) makes the open fail with that cause even when FFmpeg
    // carried on; callers retry the whole probe or decode.
    if let Some(error) = io_error.lock().unwrap().take() {
        return Err(error.context("reading cloud object while opening it"));
    }
    opened.context("Failed to create AVFormatContextInput")
}

struct CloudStorageBuffer {
    /// Storage buffer with a size that mirrors the object's size in cloud storage.
    data: Vec<u8>,
    /// Indicates which parts of the file have been read from external storage.
    indicators: Vec<bool>,
    /// Current read position.
    read_position: usize,
}

struct ReadRange {
    // Start of the range in bytes.
    start: usize,
    // End of the range in bytes.
    end: usize,
    // Whether the reader's request was bounded; a respawn of an open-ended
    // read stays open-ended.
    bounded: bool,
}

struct DataReader {
    /// The range that this data reader is reading.
    range: ReadRange,
    /// A handle to the reader which can be used to determine if it is still alive.
    handle: JoinHandle<Result<(), anyhow::Error>>,
}

struct CloudStorageReaderContext {
    buffer: CloudStorageBuffer,
    readers: Vec<DataReader>,
}

impl Drop for CloudStorageReaderContext {
    fn drop(&mut self) {
        for DataReader { handle, .. } in &self.readers {
            // Ensure that any tasks running on a global tokio runtime are aborted.
            handle.abort();
        }
    }
}

/// A minimal S3-compatible HTTP server for tests: serves files from memory
/// with `Range` support and records every request, so tests can drive the
/// cloud reader end to end without network access.
#[cfg(test)]
pub(crate) mod range_server {
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// One served GET: the requested range and the bytes sent.
    #[derive(Clone, Debug)]
    pub struct Served {
        pub start: usize,
        pub bytes: usize,
    }

    /// How the server misbehaves, as some S3-compatible stores do.
    #[derive(Clone, Copy, Default)]
    pub struct Quirks {
        /// Answer ranges with `bytes a-b/*` and no Content-Length on
        /// open-ended reads.
        pub unknown_total: bool,
        /// Ignore Range and send the whole object with 200.
        pub ignore_range: bool,
        /// Send ETags as weak (`W/"..."`).
        pub weak_etag: bool,
        /// Close the connection after the first eighth of the body of this
        /// request (0-based), once.
        pub truncate_request: Option<usize>,
        /// With `ignore_range`: send the 200 chunked, with no Content-Length,
        /// and honour open-ended ranges (a sized 206).
        pub chunked_full: bool,
        /// Answer every request, open-ended ones included, with a chunked
        /// 200 carrying no size.
        pub chunked_always: bool,
        /// Refuse bounded ranges longer than this (416), as gateways that cap
        /// bounded spans do.
        pub max_bounded_span: Option<usize>,
        /// Write the Content-Range unit as `Bytes`.
        pub capitalized_unit: bool,
        /// Write bodies in 16 KiB pieces with this pause between them, so a
        /// test can see whether a reader keeps downloading.
        pub throttle: Option<std::time::Duration>,
        /// Once this many body bytes were sent in total, write no more until
        /// [`RangeServer::release`], so a test can drop an input while its
        /// reader is still mid-download, however slow the machine.
        pub hold_after: Option<usize>,
        /// Return an `x-amz-version-id` and refuse requests that name one
        /// (403), as S3 does for a role without `s3:GetObjectVersion`.
        pub version_ids: bool,
    }

    #[derive(Default)]
    struct State {
        objects: HashMap<String, (Vec<u8>, String)>,
        served: Vec<Served>,
        quirks: Quirks,
        /// Bounded requests refused with 416.
        refused: usize,
        /// Replace `(key, data, etag)` once this many requests were served.
        swap_after: Option<(usize, String, Vec<u8>, String)>,
    }

    pub struct RangeServer {
        pub endpoint: String,
        state: Arc<Mutex<State>>,
        sent: Arc<AtomicUsize>,
        released: Arc<AtomicBool>,
    }

    impl RangeServer {
        pub fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let state = Arc::new(Mutex::new(State::default()));
            let sent = Arc::new(AtomicUsize::new(0));
            let released = Arc::new(AtomicBool::new(false));
            let (shared, shared_sent, shared_released) =
                (state.clone(), sent.clone(), released.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let (state, sent, released) =
                        (shared.clone(), shared_sent.clone(), shared_released.clone());
                    std::thread::spawn(move || serve(stream, &state, &sent, &released));
                }
            });
            Self {
                endpoint,
                state,
                sent,
                released,
            }
        }

        /// Lets bodies held by [`Quirks::hold_after`] continue.
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
        }

        /// Body bytes written to clients so far.
        pub fn bytes_sent(&self) -> usize {
            self.sent.load(Ordering::SeqCst)
        }

        /// Serves `data` at `s3://bucket/{key}` with ETag `etag`.
        pub fn put(&self, key: &str, data: Vec<u8>, etag: &str) {
            let mut state = self.state.lock().unwrap();
            state
                .objects
                .insert(format!("/bucket/{key}"), (data, format!("\"{etag}\"")));
        }

        pub fn set_quirks(&self, quirks: Quirks) {
            self.state.lock().unwrap().quirks = quirks;
        }

        /// Overwrites `key` with `data` once `requests` requests were served.
        pub fn swap_after(&self, requests: usize, key: &str, data: Vec<u8>, etag: &str) {
            self.state.lock().unwrap().swap_after =
                Some((requests, key.to_string(), data, etag.to_string()));
        }

        /// How many bounded requests were refused with 416.
        pub fn refused(&self) -> usize {
            self.state.lock().unwrap().refused
        }

        /// The GETs served so far, in order.
        pub fn served(&self) -> Vec<Served> {
            self.state.lock().unwrap().served.clone()
        }

        pub fn s3_config(&self) -> crate::util::s3::S3Config {
            crate::util::s3::S3Config {
                endpoint_url: Some(self.endpoint.clone()),
                region: Some("us-east-1".to_string()),
                access_key_id: Some("test".to_string()),
                secret_access_key: Some("test".to_string()),
                force_path_style: Some(true),
                ..Default::default()
            }
        }
    }

    fn serve(stream: TcpStream, state: &Mutex<State>, sent: &AtomicUsize, released: &AtomicBool) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut stream = stream;
        loop {
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                return;
            }
            let target = request_line.split_whitespace().nth(1).unwrap_or("/");
            let names_version = target.contains("versionId=");
            let path = target.split('?').next().unwrap().to_string();
            let (mut range, mut if_match) = (None, None);
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                let (name, value) = line.split_once(':').unwrap_or((&line, ""));
                match name.to_ascii_lowercase().as_str() {
                    "range" => range = Some(value.trim().to_string()),
                    "if-match" => if_match = Some(value.trim().to_string()),
                    _ => {}
                }
            }
            let mut state = state.lock().unwrap();
            if let Some((after, ..)) = &state.swap_after {
                if state.served.len() >= *after {
                    let (_, key, data, etag) = state.swap_after.take().unwrap();
                    state
                        .objects
                        .insert(format!("/bucket/{key}"), (data, format!("\"{etag}\"")));
                }
            }
            let quirks = state.quirks;
            let Some((data, etag)) = state.objects.get(&path).cloned() else {
                let _ = write!(
                    stream,
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
                );
                continue;
            };
            let etag = if quirks.weak_etag {
                format!("W/{etag}")
            } else {
                etag
            };
            // If-Match uses strong comparison (RFC 9110 13.1.1): a weak tag
            // never matches.
            if if_match
                .as_deref()
                .is_some_and(|m| m != etag || m.starts_with("W/"))
            {
                let _ = write!(
                    stream,
                    "HTTP/1.1 412 Precondition Failed\r\nContent-Length: 0\r\n\r\n"
                );
                continue;
            }
            if quirks.version_ids && names_version {
                let _ = write!(
                    stream,
                    "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n"
                );
                continue;
            }
            let len = data.len();
            let (start, end) = range
                .as_deref()
                .and_then(|r| r.strip_prefix("bytes="))
                .and_then(|r| r.split_once('-'))
                .map(|(a, b)| {
                    let start: usize = a.parse().unwrap();
                    let end = b.parse::<usize>().map_or(len, |e| (e + 1).min(len));
                    (start, end)
                })
                .unwrap_or((0, len));
            let open_ended = range.as_deref().is_some_and(|r| r.ends_with('-'));
            if let Some(cap) = quirks.max_bounded_span {
                if range.is_some() && !open_ended && end - start > cap {
                    state.refused += 1;
                    drop(state);
                    let _ = write!(
                        stream,
                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n\r\n"
                    );
                    continue;
                }
            }
            // With `chunked_full`, only bounded requests are ignored; an
            // open-ended one gets a normal sized 206, as some gateways do.
            let ignore = quirks.chunked_always
                || (quirks.ignore_range && !(quirks.chunked_full && open_ended));
            let (start, end) = if ignore { (0, len) } else { (start, end) };
            let index = state.served.len();
            state.served.push(Served {
                start,
                bytes: end - start,
            });
            drop(state);
            let unit = if quirks.capitalized_unit {
                "Bytes"
            } else {
                "bytes"
            };
            let mut header = if ignore && (quirks.chunked_full || quirks.chunked_always) {
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n".to_string()
            } else if ignore {
                format!("HTTP/1.1 200 OK\r\nContent-Length: {len}\r\n")
            } else if quirks.unknown_total && open_ended {
                // Chunked, so no Content-Length either.
                format!(
                    "HTTP/1.1 206 Partial Content\r\nTransfer-Encoding: chunked\r\n\
                     Content-Range: {unit} {start}-{}/*\r\n",
                    end.saturating_sub(1)
                )
            } else {
                let total = if quirks.unknown_total {
                    "*".to_string()
                } else {
                    len.to_string()
                };
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                     Content-Range: {unit} {start}-{}/{total}\r\n",
                    end - start,
                    end.saturating_sub(1)
                )
            };
            header.push_str(&format!(
                "ETag: {etag}\r\n{}Content-Type: application/octet-stream\r\n\r\n",
                if quirks.version_ids {
                    "x-amz-version-id: v-1\r\n"
                } else {
                    ""
                }
            ));
            let body = &data[start..end];
            if quirks.truncate_request == Some(index) {
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body[..body.len() / 8]);
                return;
            }
            let chunked = header.contains("Transfer-Encoding: chunked");
            // Writes the body, counting the bytes that reach the client.
            let write_body = |stream: &mut TcpStream| {
                for piece in body.chunks(if quirks.throttle.is_some() {
                    16 * 1024
                } else {
                    body.len().max(1)
                }) {
                    if let Some(limit) = quirks.hold_after {
                        let held_since = std::time::Instant::now();
                        while sent.load(Ordering::SeqCst) >= limit
                            && !released.load(Ordering::SeqCst)
                            && held_since.elapsed() < std::time::Duration::from_secs(10)
                        {
                            std::thread::sleep(std::time::Duration::from_millis(5));
                        }
                    }
                    if stream.write_all(piece).is_err() {
                        return false;
                    }
                    sent.fetch_add(piece.len(), Ordering::SeqCst);
                    if let Some(pause) = quirks.throttle {
                        std::thread::sleep(pause);
                    }
                }
                true
            };
            let ok = stream.write_all(header.as_bytes()).is_ok()
                && if chunked {
                    stream
                        .write_all(format!("{:x}\r\n", body.len()).as_bytes())
                        .is_ok()
                        && write_body(&mut stream)
                        && stream.write_all(b"\r\n0\r\n\r\n").is_ok()
                } else {
                    write_body(&mut stream)
                };
            if !ok {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::util::test_utils::{generate_test_video_file, init_logger, TestVideoParameters};
    use std::ffi::CString;

    use super::*;

    /// Ranged reads after the first are pinned to the version it saw, so an
    /// object overwritten between them fails the read instead of mixing
    /// bytes from two versions.
    #[test]
    fn test_ranged_reads_are_pinned_to_the_first_version() {
        let server = range_server::RangeServer::start();
        server.put("asset", vec![1u8; 4096], "v1");
        let uri = CloudUri::S3(
            S3Uri {
                bucket: "bucket".into(),
                key: "asset".into(),
            },
            Some(server.s3_config()),
        );
        let runtime = get_runtime().unwrap();
        let (meta, _) = runtime
            .block_on(open_cloud_stream(&uri, 0, Some(1024), None))
            .unwrap();
        assert_eq!(meta.size, 4096);
        assert_eq!(meta.end, 1024);
        assert_eq!(
            meta.version,
            Some(ObjectVersion::S3ETag("\"v1\"".to_string()))
        );
        // Same version: the next chunk reads normally.
        runtime
            .block_on(open_cloud_stream(
                &uri,
                1024,
                Some(1024),
                meta.version.as_ref(),
            ))
            .unwrap();
        // Overwritten: the pinned read fails.
        server.put("asset", vec![2u8; 4096], "v2");
        assert!(runtime
            .block_on(open_cloud_stream(
                &uri,
                2048,
                Some(1024),
                meta.version.as_ref()
            ))
            .is_err());
    }

    #[test]
    fn test_lazy_chunk_grows_on_forward_scans_only() {
        let mut lazy = LazyChunk::new(100);
        assert_eq!(lazy.request(), 100);
        // Forward hops continue the run, including ones inside bytes already
        // fetched (moof to moof within one chunk), so the next request grows.
        lazy.on_seek(30);
        lazy.on_seek(60);
        assert_eq!(lazy.request(), 200);
        lazy.on_seek(5_000);
        assert_eq!(lazy.request(), 400);
        // A seek back ends the run, even with no request of its own (its
        // bytes cached): the next request is back at the base.
        lazy.on_seek(0);
        assert_eq!(lazy.request(), 100);
        // Growth stops at the cap.
        let mut lazy = LazyChunk::new(MAX_LAZY_CHUNK / 2);
        assert_eq!(lazy.request(), MAX_LAZY_CHUNK / 2);
        assert_eq!(lazy.request(), MAX_LAZY_CHUNK);
        assert_eq!(lazy.request(), MAX_LAZY_CHUNK);
    }

    #[test]
    fn test_content_range_parse() {
        let range = |last, total| Some(ContentRange { last, total });
        assert_eq!(
            ContentRange::parse("bytes 0-99/1000"),
            range(99, Some(1000))
        );
        // Units are case-insensitive.
        assert_eq!(
            ContentRange::parse("Bytes 0-99/1000"),
            range(99, Some(1000))
        );
        assert_eq!(ContentRange::parse("bytes 0-99/*"), range(99, None));
        assert_eq!(ContentRange::parse("items 0-99/1000"), None);
        // A total that does not cover the range is not one.
        assert_eq!(ContentRange::parse("bytes 0-99/50"), None);
        assert_eq!(ContentRange::parse("bytes 99-0/1000"), None);
    }

    #[test]
    fn test_avio_reader() {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters::default()).unwrap();

        let filename = CString::new(test_video.path().to_str().unwrap()).unwrap();
        avio_reader(filename.as_c_str()).unwrap();
    }
}
