pub mod dsp;
pub mod mel;
pub mod ml;

#[cfg(not(target_arch = "wasm32"))]
pub mod audio;
#[cfg(not(target_arch = "wasm32"))]
pub mod image;

#[cfg(not(target_arch = "wasm32"))]
pub use audio::read_audio;
#[cfg(not(target_arch = "wasm32"))]
pub use image::{save_mel_spectrogram_image, save_activation_grid, save_activation_comparison};
pub use mel::{MelFrame, process_audio};

#[cfg(not(target_arch = "wasm32"))]
pub use ml::{evaluate_from_file, train_and_evaluate, Label};

#[cfg(not(target_arch = "wasm32"))]
pub mod plots;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
