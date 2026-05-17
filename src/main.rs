use std::error::Error;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use rayon::prelude::*;
use voice::{
    evaluate_from_file,
    ml::{Label, infer::Inferencer, WINDOW_SECONDS},
    process_audio, read_audio, save_activation_grid, save_activation_comparison, train_and_evaluate,
};

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
        .flat_map(|(path, label)| match read_audio(path) {
            Err(e) => {
                eprintln!("  skip {path}: {e}");
                vec![]
            }
            Ok((samples, sample_rate)) => process_audio(&samples, sample_rate, WINDOW_SECONDS)
                .filter_map(|r| r.ok())
                .map(|frame| (frame.spectrogram, *label))
                .collect(),
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
        match bincode::deserialize_from::<_, Vec<(ndarray::Array2<f32>, Label)>>(BufReader::new(
            file,
        )) {
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

    // --visualize <weights_path>: save mel spectrograms + activation maps, then exit.
    if args.len() == 3 && args[1] == "--visualize" {
        let weights_path = &args[2];
        println!("Loading training data from cv_valid_dev.csv …");
        let train_samples = load_or_compute("./mozillavoice/cv_valid_dev.csv", "./mozillavoice")?;
        println!("  → {} voiced frames (train)", train_samples.len());

        std::fs::create_dir_all("mel_spectrograms")?;
        for (i, (spec, label)) in train_samples.iter().take(10).enumerate() {
            let filename = format!("mel_spectrograms/{:02}_{}.png", i, label.name());
            voice::save_mel_spectrogram_image(spec, &filename)?;
            println!("  Saved {filename}");
        }

        let weights = std::fs::read(weights_path)?;
        let inferencer = Inferencer::from_weights_bytes(weights)
            .map_err(|e| format!("inferencer error: {e}"))?;

        // One example per class
        let (male_spec, _) = train_samples.iter().find(|(_, l)| matches!(l, Label::Male))
            .ok_or("no male sample found")?;
        let (female_spec, _) = train_samples.iter().find(|(_, l)| matches!(l, Label::Female))
            .ok_or("no female sample found")?;

        let male_acts = inferencer.activations_for_spectrogram(male_spec)
            .map_err(|e| format!("activation error: {e}"))?;
        let female_acts = inferencer.activations_for_spectrogram(female_spec)
            .map_err(|e| format!("activation error: {e}"))?;

        let layer_names = ["conv1_act", "conv2_act", "conv3_act"];
        for (name, (m, f)) in layer_names.iter().zip(male_acts.iter().zip(female_acts.iter())) {
            save_activation_grid(m, &format!("mel_spectrograms/{}_Male.png", name))?;
            save_activation_grid(f, &format!("mel_spectrograms/{}_Female.png", name))?;
            let cmp = format!("mel_spectrograms/{}_compare.png", name);
            save_activation_comparison(m, f, &cmp)?;
            println!("  Saved {cmp}  (red=Male, blue=Female, magenta=both)");
        }
        return Ok(());
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

    train_and_evaluate(train_samples, test_samples).map_err(|e| format!("candle error: {e}").into())
}
