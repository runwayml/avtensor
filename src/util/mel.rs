use anyhow::Context;
use mel_spec::{mel::MelSpectrogram, stft::Spectrogram};
use ndarray::{concatenate, Axis};
use tch::Tensor;

pub struct MelSpectrogramConfig {
    pub sample_rate: usize,
    pub fft_size: usize,
    pub hop_size: usize,
    pub num_mels: usize,
}

/// Creates a Mel Spectrogram from audio samples stored in a Tensor.
///
/// Arguments:
/// - audio: a 1D Tensor containing audio samples (floating point)
/// - config: the configuration to use for the STFT and mel spectrogram.
pub fn create_mel_spectrogram(
    audio: &Tensor,
    config: &MelSpectrogramConfig,
) -> Result<Tensor, anyhow::Error> {
    let MelSpectrogramConfig {
        sample_rate,
        fft_size,
        hop_size,
        num_mels,
    } = config;

    let mut fft = Spectrogram::new(*fft_size, *hop_size);
    let mut mel = MelSpectrogram::new(*fft_size, *sample_rate as f64, *num_mels);

    let audio_samples: Vec<f32> = audio
        .try_into()
        .context("converting audio samples to f32 slice")?;

    let num_chunks = audio_samples.len() / *hop_size;
    let mut mel_specs = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let chunk = &audio_samples[i * *hop_size..(i + 1) * *hop_size];
        if let Some(fft_frame) = fft.add(chunk) {
            let mel_spec = mel.add(&fft_frame);
            mel_specs.push(mel_spec);
        }
    }

    // Concatenate the mel spectrograms into a single large spectrogram.
    let mel_spec = concatenate(
        Axis(1),
        &mel_specs.iter().map(|m| m.view()).collect::<Vec<_>>(),
    )?;

    Tensor::try_from(mel_spec.as_standard_layout()).context("converting ndarray to Tensor")
}
