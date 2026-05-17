use candle_core::{D, DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops};
use chrono::Local;
use ndarray::Array2;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::fs;

use super::dataset::{Label, MEL_BINS, Sample, SplitDataset, TIME_FRAMES};
use super::model::VoiceCNN;

const BATCH_SIZE: usize = 256;
const EPOCHS: usize = 120;
const PATIENCE: usize = 120;
const LEARNING_RATE: f64 = 1e-4;
const SHUFFLE_SEED: u64 = 42;

/// Train on `train_raw` and evaluate on `test_raw`. Prints per-epoch metrics and a
/// final test accuracy.
pub fn train_and_evaluate(
    train_raw: Vec<(Array2<f32>, Label)>,
    test_raw: Vec<(Array2<f32>, Label)>,
) -> Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {device:?}");

    // Balance classes by undersampling the majority class
    let mut male: Vec<Sample> = Vec::new();
    let mut female: Vec<Sample> = Vec::new();
    for (spec, label) in train_raw {
        let s = Sample::from_spectrogram(&spec, label);
        match label {
            Label::Male => male.push(s),
            Label::Female => female.push(s),
        }
    }
    let min_count = male.len().min(female.len());
    println!(
        "Class counts before balancing — Male: {}, Female: {}",
        male.len(),
        female.len()
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(SHUFFLE_SEED);
    male.shuffle(&mut rng);
    female.shuffle(&mut rng);
    male.truncate(min_count);
    female.truncate(min_count);
    let mut train_samples = male;
    train_samples.extend(female);
    train_samples.shuffle(&mut rng);

    let mut test_samples: Vec<Sample> = test_raw
        .into_iter()
        .map(|(spec, label)| Sample::from_spectrogram(&spec, label))
        .collect();

    println!(
        "Dataset — train: {} (balanced), test: {}",
        train_samples.len(),
        test_samples.len()
    );

    // ── Per-bin z-score normalization ────────────────────────────────────────
    // Compute mean and std per mel bin across the entire training set.
    // This preserves absolute energy differences between bins (e.g. male voices
    // have more low-frequency energy) which per-sample min-max would destroy.
    let n_train = train_samples.len();
    let n_per_bin = (n_train * TIME_FRAMES) as f64;

    let mut bin_mean = vec![0f32; MEL_BINS];
    let mut bin_std = vec![0f32; MEL_BINS];

    for b in 0..MEL_BINS {
        let (mut sum, mut sum_sq) = (0f64, 0f64);
        for s in &train_samples {
            for t in 0..TIME_FRAMES {
                let v = s.data[b * TIME_FRAMES + t] as f64;
                sum += v;
                sum_sq += v * v;
            }
        }
        let mean = sum / n_per_bin;
        let std = (sum_sq / n_per_bin - mean * mean).sqrt().max(1e-6);
        bin_mean[b] = mean as f32;
        bin_std[b] = std as f32;
    }

    // Apply normalization in-place to both splits (test uses training stats).
    let normalize = |samples: &mut Vec<Sample>| {
        for s in samples.iter_mut() {
            for b in 0..MEL_BINS {
                for t in 0..TIME_FRAMES {
                    let i = b * TIME_FRAMES + t;
                    s.data[i] = (s.data[i] - bin_mean[b]) / bin_std[b];
                }
            }
        }
    };
    normalize(&mut train_samples);
    normalize(&mut test_samples);

    // ── Pre-flatten training data ─────────────────────────────────────────────
    // Keep all samples in one contiguous allocation; shuffle an index vec each
    // epoch instead of re-allocating / cloning Sample vecs per batch.
    let sample_size = MEL_BINS * TIME_FRAMES;
    let all_data: Vec<f32> = train_samples
        .iter()
        .flat_map(|s| s.data.iter().cloned())
        .collect();
    let all_labels: Vec<u32> = train_samples.iter().map(|s| s.label as u32).collect();

    // ── Model ─────────────────────────────────────────────────────────────────
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = VoiceCNN::new(Label::count(), vb)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    // ── Training ──────────────────────────────────────────────────────────────
    let mut epoch_rng = rand::rngs::StdRng::seed_from_u64(SHUFFLE_SEED + 1);
    let has_val = !test_samples.is_empty();
    let mut best_val_loss = f32::INFINITY;
    let mut patience_count = 0usize;
    let mut indices: Vec<usize> = (0..n_train).collect();

    // (epoch, train_loss, val_loss, train_acc, val_acc) — NAN when not computed
    let mut epoch_records: Vec<(usize, f32, f32, f32, f32)> = Vec::new();

    'train: for epoch in 1..=EPOCHS {
        indices.shuffle(&mut epoch_rng);

        let (mut epoch_loss, mut batches) = (0f32, 0usize);
        for chunk in indices.chunks(BATCH_SIZE) {
            let n = chunk.len();
            // Gather batch from pre-flattened data — avoids per-epoch heap churn.
            let batch_data: Vec<f32> = chunk
                .iter()
                .flat_map(|&i| {
                    all_data[i * sample_size..(i + 1) * sample_size]
                        .iter()
                        .cloned()
                })
                .collect();
            let batch_labels: Vec<u32> = chunk.iter().map(|&i| all_labels[i]).collect();

            let x = Tensor::from_vec(batch_data, (n, 1usize, MEL_BINS, TIME_FRAMES), &device)?;
            let y = Tensor::from_vec(batch_labels, (n,), &device)?;
            let logits = model.forward(&x, true)?;
            let batch_loss = loss::cross_entropy(&logits, &y)?;

            epoch_loss += batch_loss.to_scalar::<f32>()?;
            batches += 1;
            opt.backward_step(&batch_loss)?;
        }

        let train_loss = epoch_loss / batches as f32;
        let log_acc = epoch % 5 == 0 || epoch == 1;

        if has_val {
            let val_loss = compute_loss(&model, &test_samples, &device)?;
            if log_acc {
                let train_acc = worst_class_accuracy(&model, &train_samples, &device)?;
                let val_acc = worst_class_accuracy(&model, &test_samples, &device)?;
                println!(
                    "Epoch {epoch:>2}/{EPOCHS}  train_loss: {train_loss:.4}  val_loss: {val_loss:.4}  train_worst: {:.1}%  val_worst: {:.1}%",
                    train_acc * 100.0,
                    val_acc * 100.0
                );
                epoch_records.push((epoch, train_loss, val_loss, train_acc, val_acc));
            } else {
                println!(
                    "Epoch {epoch:>2}/{EPOCHS}  train_loss: {train_loss:.4}  val_loss: {val_loss:.4}"
                );
                epoch_records.push((epoch, train_loss, val_loss, f32::NAN, f32::NAN));
            }

            if val_loss < best_val_loss - 1e-4 {
                best_val_loss = val_loss;
                patience_count = 0;
            } else {
                patience_count += 1;
                if patience_count >= PATIENCE {
                    println!(
                        "Early stopping at epoch {epoch} (val_loss stalled for {PATIENCE} epochs)"
                    );
                    break 'train;
                }
            }
        } else {
            if log_acc {
                let train_acc = worst_class_accuracy(&model, &train_samples, &device)?;
                println!(
                    "Epoch {epoch:>2}/{EPOCHS}  train_loss: {train_loss:.4}  train_worst: {:.1}%",
                    train_acc * 100.0
                );
                epoch_records.push((epoch, train_loss, f32::NAN, train_acc, f32::NAN));
            } else {
                println!("Epoch {epoch:>2}/{EPOCHS}  train_loss: {train_loss:.4}");
                epoch_records.push((epoch, train_loss, f32::NAN, f32::NAN, f32::NAN));
            }
        }
    }

    // ── Embed norm stats into the saved weights ───────────────────────────────
    // Store per-bin mean and std so the inferencer can apply the same
    // normalization without a separate file.
    {
        let mean_t = Tensor::from_vec(bin_mean, (MEL_BINS,), &device)?;
        let std_t = Tensor::from_vec(bin_std, (MEL_BINS,), &device)?;
        let mut data = varmap.data().lock().unwrap();
        data.insert("norm_mean".to_string(), Var::from_tensor(&mean_t)?);
        data.insert("norm_std".to_string(), Var::from_tensor(&std_t)?);
    }

    // ── Evaluation & save ─────────────────────────────────────────────────────
    let weights_path = if !test_samples.is_empty() {
        let dataset = SplitDataset {
            train: vec![],
            test: test_samples.clone(),
        };
        let test_acc = worst_class_accuracy(&model, &dataset.test, &device)?;
        per_class_accuracy(&model, &dataset.test, &device)?;
        println!("\nWorst-class test accuracy: {:.1}%", test_acc * 100.0);

        fs::create_dir_all("weights").map_err(candle_core::Error::wrap)?;
        let date = Local::now().format("%Y-%m-%d");
        let acc_pct = (test_acc * 1000.0).round() / 10.0;
        let path = format!("weights/{date}-{acc_pct:.1}.safetensors");
        varmap.save(&path).map_err(candle_core::Error::wrap)?;
        println!("Weights saved to {path}");
        path
    } else {
        println!("\n(no test set — skipping evaluation)");
        fs::create_dir_all("weights").map_err(candle_core::Error::wrap)?;
        let date = Local::now().format("%Y-%m-%d");
        let path = format!("weights/{date}-notested.safetensors");
        varmap.save(&path).map_err(candle_core::Error::wrap)?;
        println!("Weights saved to {path}");
        path
    };

    // ── Export CSVs for Python plotting ──────────────────────────────────────
    let plots_dir = {
        let p = std::path::Path::new(&weights_path);
        let parent = p.parent().unwrap_or(std::path::Path::new("weights"));
        parent.join("plots")
    };
    fs::create_dir_all(&plots_dir).map_err(candle_core::Error::wrap)?;

    let log_path = plots_dir.join("training_log.csv");
    if let Err(e) = crate::plots::write_training_log(
        &epoch_records,
        log_path
            .to_str()
            .unwrap_or("weights/plots/training_log.csv"),
    ) {
        eprintln!("Warning: failed to save training log: {e}");
    } else {
        println!("CSV saved: {}", log_path.display());
    }

    if !test_samples.is_empty() {
        let preds = collect_predictions(&model, &test_samples, &device)?;
        let preds_str: Vec<(String, String, f32)> = preds
            .iter()
            .map(|(t, p, c)| (t.name().to_string(), p.name().to_string(), *c))
            .collect();
        let pred_path = plots_dir.join("predictions.csv");
        if let Err(e) = crate::plots::write_predictions(
            &preds_str,
            pred_path
                .to_str()
                .unwrap_or("weights/plots/predictions.csv"),
        ) {
            eprintln!("Warning: failed to save predictions: {e}");
        } else {
            println!("CSV saved: {}", pred_path.display());
        }
    }

    println!(
        "Run `python3 plot.py {}` to generate plots.",
        plots_dir.display()
    );
    Ok(())
}

/// Load saved weights and evaluate on `test_raw` without retraining.
/// The weights file must contain `norm_mean` and `norm_std` tensors (saved by
/// `train_and_evaluate`). Prints overall and per-class accuracy.
pub fn evaluate_from_file(weights_path: &str, test_raw: Vec<(Array2<f32>, Label)>) -> Result<()> {
    let device = Device::Cpu;
    println!("Loading weights from {weights_path} …");

    let bytes = fs::read(weights_path).map_err(candle_core::Error::wrap)?;
    let vb = VarBuilder::from_buffered_safetensors(bytes.clone(), DType::F32, &device)?;
    let model = VoiceCNN::new(Label::count(), vb)?;

    // Load per-bin normalization stats saved during training.
    let vb2 = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
    let norm_mean: Vec<f32> = vb2.get((MEL_BINS,), "norm_mean")?.to_vec1()?;
    let norm_std: Vec<f32> = vb2.get((MEL_BINS,), "norm_std")?.to_vec1()?;

    let mut test_samples: Vec<Sample> = test_raw
        .into_iter()
        .map(|(spec, label)| Sample::from_spectrogram(&spec, label))
        .collect();

    // Apply the same per-bin z-score normalization used during training.
    for s in test_samples.iter_mut() {
        for b in 0..MEL_BINS {
            for t in 0..TIME_FRAMES {
                let i = b * TIME_FRAMES + t;
                s.data[i] = (s.data[i] - norm_mean[b]) / norm_std[b];
            }
        }
    }

    println!("Test samples: {}", test_samples.len());
    per_class_accuracy(&model, &test_samples, &device)?;
    let test_acc = worst_class_accuracy(&model, &test_samples, &device)?;
    println!("Worst-class test accuracy: {:.1}%", test_acc * 100.0);

    // ── Export CSVs for Python plotting ──────────────────────────────────────
    let plots_dir = {
        let p = std::path::Path::new(weights_path);
        let parent = p.parent().unwrap_or(std::path::Path::new("."));
        parent.join("plots")
    };
    if let Err(e) = fs::create_dir_all(&plots_dir) {
        eprintln!("Warning: could not create plots dir: {e}");
    } else {
        let preds = collect_predictions(&model, &test_samples, &device)?;
        let preds_str: Vec<(String, String, f32)> = preds
            .iter()
            .map(|(t, p, c)| (t.name().to_string(), p.name().to_string(), *c))
            .collect();
        let pred_path = plots_dir.join("predictions.csv");
        if let Err(e) = crate::plots::write_predictions(
            &preds_str,
            pred_path.to_str().unwrap_or("plots/predictions.csv"),
        ) {
            eprintln!("Warning: failed to save predictions: {e}");
        } else {
            println!("CSV saved: {}", pred_path.display());
            println!(
                "Run `python3 plot.py {}` to generate plots.",
                plots_dir.display()
            );
        }
    }

    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Run inference on all samples and return (true_label, pred_label, confidence) for each.
fn collect_predictions(
    model: &VoiceCNN,
    samples: &[Sample],
    device: &Device,
) -> Result<Vec<(Label, Label, f32)>> {
    let label_from_idx = |idx: u32| -> Label {
        match idx {
            0 => Label::Male,
            _ => Label::Female,
        }
    };

    let mut results: Vec<(Label, Label, f32)> = Vec::with_capacity(samples.len());
    for chunk in samples.chunks(BATCH_SIZE) {
        let (x, _) = SplitDataset::batch_tensors(chunk, device)?;
        let logits = model.forward(&x, false)?;
        let probs = ops::softmax(&logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
        let n = chunk.len();
        let n_classes = probs_vec.len() / n;
        for (i, sample) in chunk.iter().enumerate() {
            let start = i * n_classes;
            let class_probs = &probs_vec[start..start + n_classes];
            let (pred_idx, &conf) = class_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            results.push((sample.label, label_from_idx(pred_idx as u32), conf));
        }
    }
    Ok(results)
}

fn compute_loss(model: &VoiceCNN, samples: &[Sample], device: &Device) -> Result<f32> {
    let (mut total_loss, mut batches) = (0f32, 0usize);
    for chunk in samples.chunks(BATCH_SIZE) {
        let (x, y) = SplitDataset::batch_tensors(chunk, device)?;
        let logits = model.forward(&x, false)?;
        total_loss += loss::cross_entropy(&logits, &y)?.to_scalar::<f32>()?;
        batches += 1;
    }
    Ok(if batches == 0 {
        f32::INFINITY
    } else {
        total_loss / batches as f32
    })
}

/// Returns the accuracy of the worst-performing class (min over all classes).
/// This forces both classes to improve rather than allowing one to compensate for the other.
fn worst_class_accuracy(model: &VoiceCNN, samples: &[Sample], device: &Device) -> Result<f32> {
    let mut worst = f32::INFINITY;
    for label in [Label::Male, Label::Female] {
        let class_samples: Vec<Sample> = samples
            .iter()
            .filter(|s| s.label == label)
            .map(|s| Sample::from_spectrogram_data(&s.data, s.label))
            .collect();
        if class_samples.is_empty() {
            continue;
        }
        let mut correct = 0usize;
        for chunk in class_samples.chunks(BATCH_SIZE) {
            let (x, y) = SplitDataset::batch_tensors(chunk, device)?;
            let preds = model.forward(&x, false)?.argmax(D::Minus1)?;
            correct += preds
                .eq(&y)?
                .to_dtype(DType::U32)?
                .sum_all()?
                .to_scalar::<u32>()? as usize;
        }
        let acc = correct as f32 / class_samples.len() as f32;
        if acc < worst {
            worst = acc;
        }
    }
    Ok(if worst.is_infinite() { 0.0 } else { worst })
}

fn per_class_accuracy(model: &VoiceCNN, samples: &[Sample], device: &Device) -> Result<()> {
    let labels = [Label::Male, Label::Female];
    println!("\nPer-class test accuracy:");
    for label in labels {
        let class_samples: Vec<&Sample> = samples.iter().filter(|s| s.label == label).collect();
        if class_samples.is_empty() {
            continue;
        }
        let mut correct = 0usize;
        for chunk in class_samples.chunks(BATCH_SIZE) {
            let owned: Vec<Sample> = chunk
                .iter()
                .map(|s| Sample::from_spectrogram_data(&s.data, s.label))
                .collect();
            let (x, y) = SplitDataset::batch_tensors(&owned, device)?;
            let preds = model.forward(&x, false)?.argmax(D::Minus1)?;
            correct += preds
                .eq(&y)?
                .to_dtype(DType::U32)?
                .sum_all()?
                .to_scalar::<u32>()? as usize;
        }
        println!(
            "  {:10} {}/{} ({:.1}%)",
            label.name(),
            correct,
            class_samples.len(),
            correct as f32 / class_samples.len() as f32 * 100.0
        );
    }
    Ok(())
}
