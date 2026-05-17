use candle_core::{Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Dropout, Linear, Module, VarBuilder, conv2d, linear};

/// CNN for mel-spectrogram gender classification.
///
/// Architecture (input: `[N, 1, 80, 184]`):
///   Conv(1→32,   3×3, pad=1) → ReLU → MaxPool(2) → [N,  32, 40, 92]
///   Conv(32→64,  3×3, pad=1) → ReLU → MaxPool(2) → [N,  64, 20, 46]
///   Conv(64→128, 3×3, pad=1) → ReLU → MaxPool(2) → [N, 128, 10, 23]
///   Mean over time axis (W)                       → [N, 128, 10]
///   Flatten freq × channels                       → [N, 1280]
///   Linear(1280→256) → ReLU → Dropout(0.5) → Linear(256→num_classes)
pub struct VoiceCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc1: Linear,
    fc2: Linear,
    drop: Dropout,
}

impl VoiceCNN {
    pub fn new(num_classes: usize, vb: VarBuilder) -> Result<Self> {
        let pad1 = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: conv2d(1,   32,  3, pad1, vb.pp("conv1"))?,
            conv2: conv2d(32,  64,  3, pad1, vb.pp("conv2"))?,
            conv3: conv2d(64,  128, 3, pad1, vb.pp("conv3"))?,
            fc1: linear(1280, 256, vb.pp("fc1"))?,
            fc2: linear(256, num_classes, vb.pp("fc2"))?,
            drop: Dropout::new(0.5),
        })
    }

    /// `train = true` enables dropout; use `false` for inference / evaluation.
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // [N,1,80,184] → [N,32,40,92]
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?;
        // [N,32,40,92] → [N,64,20,46]
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?;
        // [N,64,20,46] → [N,128,10,23]
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?;
        // Pool over time only → [N,128,10]
        let x = x.mean(D::Minus1)?;
        // Flatten freq × channels → [N,1280]
        let x = x.flatten_from(1)?;
        // Classifier with dropout
        let x = self.drop.forward(&self.fc1.forward(&x)?.relu()?, train)?;
        self.fc2.forward(&x)
    }

    /// Returns intermediate activation maps after each conv+relu+pool stage.
    pub fn forward_with_activations(&self, x: &Tensor) -> Result<[Tensor; 3]> {
        let act1 = self.conv1.forward(x)?.relu()?.max_pool2d(2)?;
        let act2 = self.conv2.forward(&act1)?.relu()?.max_pool2d(2)?;
        let act3 = self.conv3.forward(&act2)?.relu()?.max_pool2d(2)?;
        Ok([act1, act2, act3])
    }
}
