use anyhow::Context;
use std::{path::Path, process::Command, time::Duration};
use tempfile::NamedTempFile;

pub struct TestVideoParameters {
    pub width: usize,
    pub height: usize,
    pub frame_rate: f64,
    pub duration: Duration,
    pub video_codec: String,
    pub audio_codec: String,
    pub sample_rate: usize,
    pub channel_layout: ChannelLayout,
    /// Output pixel format (e.g. "yuv420p"). None keeps the encoder default,
    /// which for testsrc input is 4:4:4 — note that hardware decoders
    /// typically only support 4:2:0.
    pub pixel_format: Option<String>,
    /// FFmpeg colorspace tag (e.g. "bt2020nc", "bt709").
    pub colorspace: Option<String>,
    /// FFmpeg color primaries tag (e.g. "bt2020", "bt709").
    pub color_primaries: Option<String>,
    /// FFmpeg color transfer tag (e.g. "arib-std-b67", "smpte2084", "bt709").
    pub color_trc: Option<String>,
    /// FFmpeg color range tag (e.g. "tv", "pc").
    pub color_range: Option<String>,
}

impl Default for TestVideoParameters {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            frame_rate: 30.0,
            duration: Duration::from_secs(5),
            video_codec: "libx264".to_string(),
            audio_codec: "aac".to_string(),
            sample_rate: 44100,
            channel_layout: ChannelLayout::Mono,
            pixel_format: None,
            colorspace: None,
            color_primaries: None,
            color_trc: None,
            color_range: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Surround5_1,
}

#[cfg(test)]
impl ChannelLayout {
    pub fn num_channels(&self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Surround5_1 => 6,
        }
    }

    fn ffmpeg_pan_filter(&self) -> String {
        match self {
            ChannelLayout::Mono => "pan=mono|c0=c0".to_string(),
            ChannelLayout::Stereo => "pan=stereo|c0=1.0*c0|c1=1.0*c0".to_string(), // Duplicate c0
            ChannelLayout::Surround5_1 => {
                "pan=5.1|c0=1.0*c0|c1=1.0*c0|c2=1.0*c0|c3=1.0*c0|c4=1.0*c0|c5=1.0*c0".to_string()
            } // Duplicate c0 to all channels.
        }
    }
}

/// Utility function to generate a test video.
pub fn generate_test_video(
    parameters: &TestVideoParameters,
    output_file: impl AsRef<Path>,
) -> Result<(), anyhow::Error> {
    let mut cmd = Command::new(get_ffmpeg_binary());
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!(
            "testsrc=duration={}:size={}x{}:rate={}",
            parameters.duration.as_secs(),
            parameters.width,
            parameters.height,
            parameters.frame_rate
        ),
        "-f",
        "lavfi",
        "-i",
        &format!(
            "sine=frequency=1000:duration=0.1:sample_rate={},apad=pad_dur=0.9",
            parameters.sample_rate
        ),
        "-af",
        &format!(
            "aloop=loop=-1:size=48000,{}",
            parameters.channel_layout.ffmpeg_pan_filter()
        ),
        "-c:v",
        &parameters.video_codec,
        "-c:a",
        &parameters.audio_codec,
    ]);

    if let Some(ref pix_fmt) = parameters.pixel_format {
        cmd.args(["-pix_fmt", pix_fmt]);
    }

    // Pass color metadata flags to FFmpeg when specified.
    if let Some(ref cs) = parameters.colorspace {
        cmd.args(["-colorspace", cs]);
    }
    if let Some(ref cp) = parameters.color_primaries {
        cmd.args(["-color_primaries", cp]);
    }
    if let Some(ref ct) = parameters.color_trc {
        cmd.args(["-color_trc", ct]);
    }
    if let Some(ref cr) = parameters.color_range {
        cmd.args(["-color_range", cr]);
    }

    // libx264 does not propagate ffmpeg's -color_primaries/-color_trc flags
    // into the bitstream VUI (decoders would see them as "unknown"), so write
    // them through the encoder's own parameters as well.
    if parameters.video_codec == "libx264" {
        let mut x264_params: Vec<String> = Vec::new();
        if let Some(ref cs) = parameters.colorspace {
            x264_params.push(format!("colormatrix={cs}"));
        }
        if let Some(ref cp) = parameters.color_primaries {
            x264_params.push(format!("colorprim={cp}"));
        }
        if let Some(ref ct) = parameters.color_trc {
            x264_params.push(format!("transfer={ct}"));
        }
        if !x264_params.is_empty() {
            cmd.args(["-x264-params", &x264_params.join(":")]);
        }
    }

    cmd.args([
        "-t",
        &format!("{}", parameters.duration.as_secs()),
        output_file
            .as_ref()
            .to_str()
            .context("Could not convert output file path to string")?,
    ]);

    cmd.status()?;
    Ok(())
}

pub fn generate_test_video_file(
    params: &TestVideoParameters,
) -> Result<NamedTempFile, anyhow::Error> {
    let file = NamedTempFile::with_suffix(".mp4")?;
    generate_test_video(params, file.path())?;
    Ok(file)
}

/// Runs an ffmpeg command, failing loudly on a non-zero exit.
fn run_ffmpeg(args: &[&str]) -> Result<(), anyhow::Error> {
    let status = Command::new(get_ffmpeg_binary())
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .status()
        .context("running ffmpeg")?;
    anyhow::ensure!(status.success(), "ffmpeg exited with {status}");
    Ok(())
}

/// Generates an audio-only FLAC file containing a sine tone.
pub fn generate_test_flac_file(
    sample_rate: usize,
    duration: Duration,
) -> Result<NamedTempFile, anyhow::Error> {
    let file = NamedTempFile::with_suffix(".flac")?;
    run_ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!(
            "sine=frequency=440:duration={}:sample_rate={}",
            duration.as_secs(),
            sample_rate
        ),
        "-c:a",
        "flac",
        file.path().to_str().context("temp path")?,
    ])?;
    Ok(file)
}

/// Generates a video with exactly `num_frames` frames at 24000/1001
/// (23.976...) fps. `duration * frame_rate` for such streams lands just above
/// the integer frame count, which historically produced a phantom trailing
/// frame.
pub fn generate_fractional_fps_video(num_frames: usize) -> Result<NamedTempFile, anyhow::Error> {
    let file = NamedTempFile::with_suffix(".mp4")?;
    run_ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=24000/1001",
        "-frames:v",
        &num_frames.to_string(),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        file.path().to_str().context("temp path")?,
    ])?;
    Ok(file)
}

/// Re-encodes `input` as a variable-frame-rate video: the first 75 frames
/// keep 30 fps timing, the rest switch to 15 fps timing, producing
/// non-uniform PTS gaps.
pub fn make_vfr_video(input: &Path) -> Result<NamedTempFile, anyhow::Error> {
    let file = NamedTempFile::with_suffix(".mp4")?;
    run_ffmpeg(&[
        "-i",
        input.to_str().context("input path")?,
        "-vf",
        "setpts=if(lt(N\\,75)\\,N/30/TB\\,(2.5+(N-75)/15)/TB)",
        "-fps_mode",
        "vfr",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        file.path().to_str().context("temp path")?,
    ])?;
    Ok(file)
}

/// Remuxes `input` with the audio stream's timestamps shifted forward by
/// `offset_seconds`, so audio starts later than video.
pub fn make_av_offset_video(
    input: &Path,
    offset_seconds: f64,
) -> Result<NamedTempFile, anyhow::Error> {
    let file = NamedTempFile::with_suffix(".mp4")?;
    let input = input.to_str().context("input path")?;
    run_ffmpeg(&[
        "-i",
        input,
        "-itsoffset",
        &offset_seconds.to_string(),
        "-i",
        input,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c",
        "copy",
        file.path().to_str().context("temp path")?,
    ])?;
    Ok(file)
}

/// Serves `path` over HTTP on an ephemeral localhost port, returning the URL.
///
/// Minimal single-file server for exercising FFmpeg's http protocol in
/// tests: each GET receives the full file (no Range support, mirroring the
/// simplest possible origin). The serving thread is detached and exits with
/// the test process.
#[cfg(test)]
pub(crate) fn serve_file_over_http(path: &Path) -> Result<String, anyhow::Error> {
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    let data = std::fs::read(path).context("reading file to serve")?;
    let listener = TcpListener::bind("127.0.0.1:0").context("binding test HTTP server")?;
    let addr = listener.local_addr()?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf); // consume the request
            let header = format!(
                "HTTP/1.0 200 OK\r\nContent-Type: video/mp4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                data.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&data);
        }
    });
    Ok(format!("http://{addr}/test.mp4"))
}

/// Path to the `ffmpeg` binary used to generate test media. Set
/// `AVTENSOR_FFMPEG` to point at a specific build; otherwise `ffmpeg` is
/// resolved from `PATH`.
fn get_ffmpeg_binary() -> String {
    std::env::var("AVTENSOR_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string())
}

pub fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generating_multi_channel_test_video() {
        generate_test_video_file(&TestVideoParameters {
            channel_layout: ChannelLayout::Stereo,
            ..Default::default()
        })
        .unwrap();

        generate_test_video_file(&TestVideoParameters {
            channel_layout: ChannelLayout::Surround5_1,
            ..Default::default()
        })
        .unwrap();
    }
}
