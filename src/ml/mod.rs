mod dataset;
pub mod infer;
mod model;

#[cfg(not(target_arch = "wasm32"))]
mod trainer;

pub use dataset::{Label, WINDOW_SECONDS, TRAINING_SAMPLE_RATE, FRAME_LEN, HOP_LEN, TIME_FRAMES};

#[cfg(not(target_arch = "wasm32"))]
pub use trainer::{evaluate_from_file, train_and_evaluate};
