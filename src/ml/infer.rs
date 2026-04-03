use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use ndarray::Array2;

use crate::mel::compute_mel_spectrogram;

use super::dataset::{Label, MEL_BINS, Sample, TIME_FRAMES};
use super::model::VoiceCNN;

const ENERGY_THRESHOLD: f32 = 0.02;

pub struct Inferencer {
    model: VoiceCNN,
    device: Device,
    /// Per-mel-bin mean computed over the training set.
    norm_mean: Tensor,
    /// Per-mel-bin std computed over the training set.
    norm_std: Tensor,
}

impl Inferencer {
    pub fn from_weights_bytes(weights: Vec<u8>) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        // Load model weights. Clone bytes so we can also read the norm stats
        // from the same file via a second VarBuilder.
        let vb = VarBuilder::from_buffered_safetensors(weights.clone(), DType::F32, &device)?;
        let model = VoiceCNN::new(Label::count(), vb)?;

        let vb2 = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let norm_mean = vb2.get((MEL_BINS,), "norm_mean")?;
        let norm_std  = vb2.get((MEL_BINS,), "norm_std")?;

        Ok(Self { model, device, norm_mean, norm_std })
    }

    /// Predict the speaker from a slice of PCM f32 samples.
    /// Returns `Some((Label, confidence))` or `None` for silence / low energy.
    pub fn predict(&self, samples: &[f32], sample_rate: u32) -> Option<(Label, f32)> {
        // Skip silence
        let rms = rms_energy(samples);
        if rms < ENERGY_THRESHOLD {
            return None;
        }

        let spec: Array2<f32> =
            compute_mel_spectrogram(samples, sample_rate, 1024, 512, 80).ok()?;

        if spec.ncols() == 0 {
            return None;
        }

        // Pad/truncate (no normalization inside from_spectrogram anymore).
        let sample = Sample::from_spectrogram(&spec, Label::Male);
        let x_raw = Tensor::from_vec(
            sample.data,
            (1usize, 1usize, MEL_BINS, TIME_FRAMES),
            &self.device,
        )
        .ok()?;

        // Apply the same per-bin z-score normalization used during training.
        // norm_mean/norm_std are [MEL_BINS]; reshape to [1,1,MEL_BINS,1] for broadcasting.
        let mean = self.norm_mean.reshape((1, 1, MEL_BINS, 1)).ok()?;
        let std  = self.norm_std.reshape((1, 1, MEL_BINS, 1)).ok()?;
        let x = x_raw.broadcast_sub(&mean).ok()?.broadcast_div(&std).ok()?;

        let logits = self.model.forward(&x, false).ok()?;
        let probs = candle_nn::ops::softmax(&logits, 1).ok()?;
        let probs_vec: Vec<f32> = probs.flatten_all().ok()?.to_vec1().ok()?;

        let (idx, &conf) = probs_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;

        let label = match idx {
            0 => Label::Male,
            _ => Label::Female,
        };

        Some((label, conf))
    }
}

fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}
