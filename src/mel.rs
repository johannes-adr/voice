use crate::dsp::{hann_window, hz_to_mel, mel_to_hz};
use ndarray::Array2;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::error::Error;

pub struct MelFrame {
    pub window_index: usize,
    pub spectrogram: Array2<f32>,
    pub energy: f32,
}

fn calculate_rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

const ENERGY_THRESHOLD: f32 = 0.02;

pub fn process_audio(
    samples: &[f32],
    sample_rate: u32,
    window_seconds: f32,
) -> impl Iterator<Item = Result<MelFrame, Box<dyn Error>>> + '_ {
    let window_size = (window_seconds * sample_rate as f32).round() as usize;

    samples
        .chunks_exact(window_size.max(1))
        .enumerate()
        .filter_map(move |(window_index, chunk)| {
            if window_size == 0 {
                return None;
            }
            let energy = calculate_rms_energy(chunk);
            if energy < ENERGY_THRESHOLD {
                return None;
            }
            Some(
                compute_mel_spectrogram(chunk, sample_rate, 1024, 512, 80).map(|spectrogram| {
                    MelFrame {
                        window_index,
                        spectrogram,
                        energy,
                    }
                }),
            )
        })
}

pub fn compute_mel_spectrogram(
    audio: &[f32],
    sample_rate: u32,
    frame_len: usize,
    hop_len: usize,
    n_mels: usize,
) -> Result<Array2<f32>, Box<dyn Error>> {
    if audio.is_empty() || frame_len == 0 || hop_len == 0 {
        return Ok(Array2::zeros((n_mels, 0)));
    }

    let n_fft = frame_len.next_power_of_two();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);

    let window = hann_window(frame_len);
    let filter_bank = mel_filterbank(sample_rate, n_fft, n_mels, 0.0, sample_rate as f32 / 2.0);

    let n_frames = if audio.len() < frame_len {
        1
    } else {
        1 + (audio.len() - frame_len) / hop_len
    };

    let mut mel_spectrogram = Array2::<f32>::zeros((n_mels, n_frames));

    for i in 0..n_frames {
        let start = i * hop_len;
        let end = (start + frame_len).min(audio.len());

        let mut buffer: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; n_fft];
        for j in 0..(end - start) {
            buffer[j].re = audio[start + j] * window[j];
        }

        fft.process(&mut buffer);

        let half_n = n_fft / 2 + 1;
        let mut spectrum = vec![0.0f32; half_n];
        for k in 0..half_n {
            spectrum[k] = buffer[k].norm_sqr();
        }

        let mel_frame = apply_mel_filter(&spectrum, &filter_bank);

        for (m, &value) in mel_frame.iter().enumerate() {
            mel_spectrogram[(m, i)] = 10.0 * (value + 1e-10).log10();
        }
    }

    Ok(mel_spectrogram)
}

pub fn mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let nyquist = sample_rate as f32 / 2.0;
    let fmax = fmax.min(nyquist);

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        mel_points.push(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32);
    }

    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let fft_bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz / nyquist) * (n_fft as f32 / 2.0)).round() as usize)
        .collect();

    let mut filters = vec![vec![0.0; n_fft / 2 + 1]; n_mels];

    for m in 0..n_mels {
        let left = fft_bins[m];
        let center = fft_bins[m + 1];
        let right = fft_bins[m + 2];

        for k in left..center {
            if center != left {
                filters[m][k] = (k as f32 - left as f32) / (center as f32 - left as f32);
            }
        }

        for k in center..right {
            if right != center {
                filters[m][k] = (right as f32 - k as f32) / (right as f32 - center as f32);
            }
        }
    }

    filters
}

fn apply_mel_filter(spectrum: &[f32], filter_bank: &[Vec<f32>]) -> Vec<f32> {
    filter_bank
        .iter()
        .map(|filter| {
            filter
                .iter()
                .zip(spectrum.iter())
                .map(|(f, s)| f * s)
                .sum::<f32>()
        })
        .collect()
}
