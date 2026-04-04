use candle_core::{Device, Result, Tensor};
use ndarray::{Array2, s};

// ── Audio / preprocessing config (single source of truth) ───────────────────
/// Duration of each audio window fed to the model, in seconds.
/// Changing this requires retraining.
pub const WINDOW_SECONDS: f32 = 2.0;
/// Sample rate that all audio is resampled to before feature extraction.
/// Must match the sample rate of the training data.
pub const TRAINING_SAMPLE_RATE: u32 = 48000;
/// FFT frame length (samples).
pub const FRAME_LEN: usize = 1024;
/// Hop length between frames (samples).
pub const HOP_LEN: usize = 512;

// ── Derived dimensions ────────────────────────────────────────────────────────
const WINDOW_SAMPLES: usize = (WINDOW_SECONDS * TRAINING_SAMPLE_RATE as f32) as usize;
/// Fixed input dimensions for the CNN.
pub const MEL_BINS: usize = 80;
pub const TIME_FRAMES: usize = 1 + (WINDOW_SAMPLES - FRAME_LEN) / HOP_LEN;

/// Gender labels for binary male/female classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Label {
    Male = 0,
    Female = 1,
}

impl Label {
    pub fn name(self) -> &'static str {
        match self {
            Label::Male => "Male",
            Label::Female => "Female",
        }
    }

    pub const fn count() -> usize {
        2
    }
}

/// A single training/test example (raw, un-normalised after padding).
#[derive(Clone)]
pub struct Sample {
    /// Flattened row-major spectrogram of shape (MEL_BINS × TIME_FRAMES).
    pub data: Vec<f32>,
    pub label: Label,
}

impl Sample {
    /// Build a `Sample` from a raw mel spectrogram.
    /// Pads or truncates the time axis to `TIME_FRAMES`. Normalization is
    /// handled globally by the trainer using per-bin z-score statistics.
    pub fn from_spectrogram(spec: &Array2<f32>, label: Label) -> Self {
        let fixed = pad_or_truncate(spec);
        Self {
            data: fixed.iter().cloned().collect(),
            label,
        }
    }

    /// Reconstruct a `Sample` from already-processed flat data (used for re-batching).
    pub fn from_spectrogram_data(data: &[f32], label: Label) -> Self {
        Self {
            data: data.to_vec(),
            label,
        }
    }
}

/// Dataset split into train and test subsets.
pub struct SplitDataset {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
}

impl SplitDataset {
    /// Build a batched `(x, y)` tensor pair from a slice of samples.
    pub fn batch_tensors(samples: &[Sample], device: &Device) -> Result<(Tensor, Tensor)> {
        let n = samples.len();
        let flat: Vec<f32> = samples
            .iter()
            .flat_map(|s| s.data.iter().cloned())
            .collect();
        let labels: Vec<u32> = samples.iter().map(|s| s.label as u32).collect();

        let x = Tensor::from_vec(flat, (n, 1usize, MEL_BINS, TIME_FRAMES), device)?;
        let y = Tensor::from_vec(labels, (n,), device)?;
        Ok((x, y))
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn pad_or_truncate(spec: &Array2<f32>) -> Array2<f32> {
    let (_, n_frames) = spec.dim();
    if n_frames >= TIME_FRAMES {
        spec.slice(s![.., ..TIME_FRAMES]).to_owned()
    } else {
        let mut padded = Array2::zeros((MEL_BINS, TIME_FRAMES));
        padded.slice_mut(s![.., ..n_frames]).assign(spec);
        padded
    }
}
