use std::error::Error;
use std::io::Write;

/// Write epoch training metrics to a CSV file.
/// Columns: epoch,train_loss,val_loss,train_acc,val_acc  (NaN when not computed)
pub fn write_training_log(
    records: &[(usize, f32, f32, f32, f32)],
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut f = std::fs::File::create(out_path)?;
    writeln!(f, "epoch,train_loss,val_loss,train_acc,val_acc")?;
    for &(epoch, tl, vl, ta, va) in records {
        writeln!(f, "{},{},{},{},{}", epoch, fmt(tl), fmt(vl), fmt(ta), fmt(va))?;
    }
    Ok(())
}

/// Write per-sample predictions to a CSV file.
/// Columns: true_label,pred_label,confidence
pub fn write_predictions(
    preds: &[(String, String, f32)],
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut f = std::fs::File::create(out_path)?;
    writeln!(f, "true_label,pred_label,confidence")?;
    for (true_label, pred_label, conf) in preds {
        writeln!(f, "{true_label},{pred_label},{conf:.6}")?;
    }
    Ok(())
}

fn fmt(v: f32) -> String {
    if v.is_nan() { "NaN".to_string() } else { format!("{v:.6}") }
}
