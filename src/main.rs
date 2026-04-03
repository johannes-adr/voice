use std::error::Error;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use rayon::prelude::*;
use voice::{evaluate_from_file, ml::Label, process_audio, read_audio, train_and_evaluate};

fn load_csv_samples(
    csv_path: &str,
    audio_base: &str,
) -> Result<Vec<(ndarray::Array2<f32>, Label)>, Box<dyn Error>> {
    let content = std::fs::read_to_string(csv_path)?;
    let mut lines = content.lines();
    lines.next(); // skip header: filename,text,up_votes,down_votes,age,gender,accent,duration

    let entries: Vec<(String, Label)> = lines
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            let n = parts.len();
            if n < 4 {
                return None;
            }
            let filename = parts[0].trim();
            let gender = parts[n - 3].trim();
            let label = match gender {
                "male" => Label::Male,
                "female" => Label::Female,
                _ => return None,
            };
            Some((format!("{}/{}", audio_base, filename), label))
        })
        .collect();

    let all_samples: Vec<(ndarray::Array2<f32>, Label)> = entries
        .par_iter()
        .flat_map(|(path, label)| {
            match read_audio(path) {
                Err(e) => {
                    eprintln!("  skip {path}: {e}");
                    vec![]
                }
                Ok((samples, sample_rate)) => process_audio(&samples, sample_rate)
                    .filter_map(|r| r.ok())
                    .map(|frame| (frame.spectrogram, *label))
                    .collect(),
            }
        })
        .collect();

    Ok(all_samples)
}

fn cache_path(csv_path: &str) -> String {
    csv_path.trim_end_matches(".csv").to_string() + ".cache.bin"
}

fn load_or_compute(
    csv_path: &str,
    audio_base: &str,
) -> Result<Vec<(ndarray::Array2<f32>, Label)>, Box<dyn Error>> {
    let cache = cache_path(csv_path);

    if Path::new(&cache).exists() {
        println!("  Loading cache: {cache}");
        let file = std::fs::File::open(&cache)?;
        match bincode::deserialize_from::<_, Vec<(ndarray::Array2<f32>, Label)>>(BufReader::new(file)) {
            Ok(samples) => return Ok(samples),
            Err(e) => eprintln!("  Cache invalid ({e}), recomputing…"),
        }
    }

    let samples = load_csv_samples(csv_path, audio_base)?;
    let file = std::fs::File::create(&cache)?;
    bincode::serialize_into(BufWriter::new(file), &samples)?;
    println!("  Saved cache: {cache}");
    Ok(samples)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    // --eval <weights_path>: skip training and evaluate saved weights on the test set.
    if args.len() == 3 && args[1] == "--eval" {
        let weights_path = &args[2];
        println!("Loading test data from cv_valid_test.csv …");
        let test_samples = load_or_compute("./mozillavoice/cv_valid_test.csv", "./mozillavoice")?;
        println!("  → {} voiced frames (test)\n", test_samples.len());
        return evaluate_from_file(weights_path, test_samples)
            .map_err(|e| format!("candle error: {e}").into());
    }

    println!("Loading training data from cv_valid_dev.csv …");
    let train_samples = load_or_compute("./mozillavoice/cv_valid_dev.csv", "./mozillavoice")?;
    println!("  → {} voiced frames (train)", train_samples.len());

    println!("Loading test data from cv_valid_test.csv …");
    let test_samples = load_or_compute("./mozillavoice/cv_valid_test.csv", "./mozillavoice")?;
    println!("  → {} voiced frames (test)", test_samples.len());

    println!(
        "\nTotal — train: {}, test: {}\n",
        train_samples.len(),
        test_samples.len()
    );

    train_and_evaluate(train_samples, test_samples)
        .map_err(|e| format!("candle error: {e}").into())
}
