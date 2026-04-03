mod dataset;
pub mod infer;
mod model;

#[cfg(not(target_arch = "wasm32"))]
mod trainer;

pub use dataset::Label;

#[cfg(not(target_arch = "wasm32"))]
pub use trainer::{evaluate_from_file, train_and_evaluate};
