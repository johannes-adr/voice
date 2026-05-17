use std::error::Error;
use candle_core::Tensor;
use image::{GrayImage, Luma, RgbImage};
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

/// Save a [1, C, H, W] activation tensor as a grid of channel images.
/// Channels are arranged in a roughly-square grid; each cell is H×W pixels.
pub fn save_activation_grid(tensor: &Tensor, filename: &str) -> Result<(), Box<dyn Error>> {
    let (n_channels, h, w) = match tensor.dims() {
        [1, c, h, w] => (*c, *h, *w),
        [c, h, w] => (*c, *h, *w),
        d => return Err(format!("unexpected tensor dims: {d:?}").into()),
    };

    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let cols = (n_channels as f32).sqrt().ceil() as usize;
    let rows = (n_channels + cols - 1) / cols;
    let pad = 1usize;
    let img_w = cols * w + (cols + 1) * pad;
    let img_h = rows * h + (rows + 1) * pad;

    let mut img = RgbImage::new(img_w as u32, img_h as u32);
    // dark background
    for p in img.pixels_mut() {
        *p = image::Rgb([30u8, 30, 30]);
    }

    for c in 0..n_channels {
        let offset = c * h * w;
        let channel: &[f32] = &data[offset..offset + h * w];

        let min = channel.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = channel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denom = (max - min).max(1e-6);

        let col_idx = c % cols;
        let row_idx = c / cols;
        let x0 = pad + col_idx * (w + pad);
        let y0 = pad + row_idx * (h + pad);

        for py in 0..h {
            for px in 0..w {
                let v = channel[py * w + px];
                let norm = ((v - min) / denom * 255.0).clamp(0.0, 255.0) as u8;
                // viridis-like: map 0→dark blue, 128→teal, 255→yellow
                let r = (norm as f32 * 0.9).min(255.0) as u8;
                let g = ((norm as f32).powf(0.7)).min(255.0) as u8;
                let b = (255.0 - norm as f32 * 0.8).clamp(0.0, 255.0) as u8;
                img.put_pixel((x0 + px) as u32, (y0 + py) as u32, image::Rgb([r, g, b]));
            }
        }
    }

    img.save(filename)?;
    Ok(())
}

/// Overlay two activation tensors (e.g. Male vs Female) into one RGB grid.
/// Red channel = `tensor_a` (e.g. Male), Blue channel = `tensor_b` (e.g. Female).
///   Pure red   → filter fires only for Male
///   Pure blue  → filter fires only for Female
///   Magenta    → both fire
///   Black      → neither fires
pub fn save_activation_comparison(
    tensor_a: &Tensor,
    tensor_b: &Tensor,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let (n_channels, h, w) = match tensor_a.dims() {
        [1, c, h, w] => (*c, *h, *w),
        [c, h, w] => (*c, *h, *w),
        d => return Err(format!("unexpected tensor dims: {d:?}").into()),
    };

    let data_a: Vec<f32> = tensor_a.flatten_all()?.to_vec1()?;
    let data_b: Vec<f32> = tensor_b.flatten_all()?.to_vec1()?;

    let cols = (n_channels as f32).sqrt().ceil() as usize;
    let rows = (n_channels + cols - 1) / cols;
    let pad = 2usize;
    let img_w = cols * w + (cols + 1) * pad;
    let img_h = rows * h + (rows + 1) * pad;

    let mut img = RgbImage::new(img_w as u32, img_h as u32);
    for p in img.pixels_mut() {
        *p = image::Rgb([20u8, 20, 20]);
    }

    for c in 0..n_channels {
        let offset = c * h * w;
        let ch_a = &data_a[offset..offset + h * w];
        let ch_b = &data_b[offset..offset + h * w];

        // Normalize both channels together so intensities are comparable within each filter.
        let global_max = ch_a.iter().chain(ch_b.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);
        let global_min = ch_a.iter().chain(ch_b.iter()).cloned().fold(f32::INFINITY, f32::min);
        let denom = (global_max - global_min).max(1e-6);

        let col_idx = c % cols;
        let row_idx = c / cols;
        let x0 = pad + col_idx * (w + pad);
        let y0 = pad + row_idx * (h + pad);

        for py in 0..h {
            for px in 0..w {
                let a = ((ch_a[py * w + px] - global_min) / denom * 255.0).clamp(0.0, 255.0) as u8;
                let b = ((ch_b[py * w + px] - global_min) / denom * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel((x0 + px) as u32, (y0 + py) as u32, image::Rgb([a, 0, b]));
            }
        }
    }

    img.save(filename)?;
    Ok(())
}
