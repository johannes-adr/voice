use std::error::Error;
use image::{GrayImage, Luma};
use ndarray::Array2;

pub fn save_mel_spectrogram_image(
    spectrogram: &Array2<f32>,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let (n_mels, n_frames) = spectrogram.dim();

    if n_frames == 0 || n_mels == 0 {
        return Err("Empty spectrogram".into());
    }

    let min_val = *spectrogram
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_val = *spectrogram
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let denom = (max_val - min_val).max(1e-6);

    let mut img = GrayImage::new(n_frames as u32, n_mels as u32);

    for y in 0..n_mels {
        for x in 0..n_frames {
            let value = spectrogram[(n_mels - 1 - y, x)];
            let norm = ((value - min_val) / denom * 255.0).clamp(0.0, 255.0);
            img.put_pixel(x as u32, y as u32, Luma([norm as u8]));
        }
    }

    img.save(filename)?;
    Ok(())
}
