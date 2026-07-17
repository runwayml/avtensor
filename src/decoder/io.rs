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
    sync::{Arc, Mutex},
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
    match std::env::var("AVTENSOR_MAX_CLOUD_OBJECT_BYTES") {
        Ok(value) => match value.parse::<usize>() {
            Ok(bytes) => bytes,
            Err(_) => {
                log::warn!(
                    "Ignoring invalid AVTENSOR_MAX_CLOUD_OBJECT_BYTES value {value:?}; \
                     using the default of {DEFAULT_MAX_CLOUD_OBJECT_SIZE} bytes"
                );
                DEFAULT_MAX_CLOUD_OBJECT_SIZE
            }
        },
        Err(_) => DEFAULT_MAX_CLOUD_OBJECT_SIZE,
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

/// Opens a streaming read of `uri` starting at byte `offset`.
///
/// Returns the total object size and the (suffix) byte stream.
async fn open_cloud_stream(
    uri: &CloudUri,
    offset: usize,
) -> Result<(usize, CloudChunkStream), anyhow::Error> {
    match uri {
        CloudUri::Gcs(GCSUri { bucket, key }) => {
            let storage = get_storage()?;
            let mut request = storage.read_object(bucket, key);
            if offset > 0 {
                request = request.set_read_range(
                    google_cloud_storage::model_ext::ReadRange::offset(offset as u64),
                );
            }
            let resp = request.send().await.context("reading object from GCS")?;
            let size = resp.object().size as usize;
            Ok((size, CloudChunkStream::Gcs(resp)))
        }
        CloudUri::S3(S3Uri { bucket, key }, s3_config) => {
            let client = get_s3_client(s3_config.as_ref())?;
            let resp = client
                .get_object()
                .bucket(bucket)
                .key(key)
                .range(format!("bytes={offset}-"))
                .send()
                .await
                .context("reading object from S3")?;
            let remaining = resp.content_length().unwrap_or(0).max(0) as usize;
            Ok((offset + remaining, CloudChunkStream::S3(resp.body)))
        }
    }
}

/// Reads a media asset from cloud storage and caches the data to a buffer.
///
/// Arguments:
/// - `stream`: The byte stream to read data from.
/// - `start`: The index of the first byte in the buffer to start writing data to.
/// - `context`: A handle to the storage buffer to which we should write data as it comes in.
async fn cloud_asset_reader(
    mut stream: CloudChunkStream,
    start: usize,
    context: Arc<tokio::sync::Mutex<CloudStorageReaderContext>>,
) -> Result<(), anyhow::Error> {
    let mut current_position = start;
    while let Some(chunk) = stream.next_chunk().await {
        match chunk {
            Ok(data) => {
                let new_position = current_position + data.len();

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
    let (obj_size, stream) = runtime
        .block_on(open_cloud_stream(&cloud_uri, 0))
        .context("opening cloud object")?; // TODO (rikheijdens): bubble up 404 errors etc.

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
    let read_sequential_future = cloud_asset_reader(stream, 0, context.clone());
    runtime.block_on(async {
        let mut context = context.lock().await;
        context.readers.push(DataReader {
            range: ReadRange {
                start: 0,
                end: obj_size,
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

                            // If a live reader is fetching the hole, keep
                            // waiting; otherwise surface it for a respawn.
                            let covered = readers.iter().any(|reader| {
                                reader.range.start <= first_missing
                                    && first_missing < reader.range.end
                                    && !reader.handle.is_finished()
                            });
                            (!covered).then_some(first_missing)
                        };

                        if let Some(pos) = stalled_at {
                            // No live reader covers this position: its reader
                            // died before delivering the bytes. Respawn one
                            // with backoff, and fail once the budget is spent
                            // rather than looping forever.
                            if respawn_attempts >= MAX_READER_RESPAWNS {
                                return Err(anyhow!(
                                    "cloud reader stalled: bytes at {} were not fetched \
                                     after {} respawn attempts",
                                    pos,
                                    MAX_READER_RESPAWNS
                                ));
                            }
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
                            let (_, stream) = open_cloud_stream(&read_uri, pos)
                                .await
                                .context("reopening cloud stream after a reader stall")?;
                            let read_future =
                                cloud_asset_reader(stream, pos, read_context.clone());
                            let CloudStorageReaderContext {
                                ref mut readers, ..
                            } = *read_context.lock().await;
                            readers.push(DataReader {
                                range: ReadRange {
                                    start: pos,
                                    end: obj_size,
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
                    let indicators = &buffer.indicators[0..position];
                    let mut closest_available_byte = 0;
                    for i in (0..position).rev() {
                        if indicators[i] {
                            closest_available_byte = i;
                            break;
                        }
                    };

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
                    let (_, stream) = open_cloud_stream(&seek_uri, position).await?;

                    let read_future = cloud_asset_reader(stream, position, seek_context.clone());

                    // Spawn the future and keep track of the reader.
                    log::debug!("Spawned new DataReader to read range [{}..{}]", position, obj_size);
                    readers.push(DataReader {
                        range: ReadRange {
                            start: position,
                            end: obj_size
                        },
                        handle: runtime.spawn(read_future)
                    });

                    Ok(position as i64)
                }).unwrap_or(-1)
            },
        )),
    );

    AVFormatContextInput::from_io_context(AVIOContextContainer::Custom(io_context))
        .context("Failed to create AVFormatContextInput")
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

#[cfg(test)]
mod tests {
    use crate::util::test_utils::{generate_test_video_file, init_logger, TestVideoParameters};
    use std::ffi::CString;

    use super::*;

    #[test]
    fn test_avio_reader() {
        init_logger();

        let test_video = generate_test_video_file(&TestVideoParameters::default()).unwrap();

        let filename = CString::new(test_video.path().to_str().unwrap()).unwrap();
        avio_reader(filename.as_c_str()).unwrap();
    }
}
