use std::error::Error;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

pub fn read_audio(path: &str) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let cursor = Cursor::new(buffer);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension() {
        let ext_str = ext.to_string_lossy();
        hint.with_extension(ext_str.as_ref());
    }

    let probed = match symphonia::default::get_probe().format(
        &Hint::new(),
        mss,
        &Default::default(),
        &Default::default(),
    ) {
        Ok(p) => p,
        Err(e) => return Err(format!("Failed to probe audio format: {}", e).into()),
    };

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.sample_rate.is_some())
        .ok_or("No audio track found")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("No sample rate found")?;
    let channels = track
        .codec_params
        .channels
        .map(|ch| ch.count())
        .unwrap_or(1);
    let track_id = track.id;

    let mut decoder =
        match symphonia::default::get_codecs().make(&track.codec_params, &Default::default()) {
            Ok(d) => d,
            Err(e) => return Err(format!("Failed to create decoder: {}", e).into()),
        };

    let mut samples = Vec::new();

    loop {
        match format.next_packet() {
            Ok(packet) => {
                if packet.track_id() != track_id {
                    continue;
                }

                match decoder.decode(&packet) {
                    Ok(audio_buf) => match audio_buf {
                        AudioBufferRef::F32(buf) => {
                            if channels == 1 {
                                samples.extend_from_slice(buf.chan(0));
                            } else {
                                for frame_idx in 0..buf.frames() {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        sum += buf.chan(ch)[frame_idx];
                                    }
                                    samples.push(sum / channels as f32);
                                }
                            }
                        }
                        AudioBufferRef::S32(buf) => {
                            if channels == 1 {
                                samples.extend(
                                    buf.chan(0).iter().map(|&s| s as f32 / i32::MAX as f32),
                                );
                            } else {
                                for frame_idx in 0..buf.frames() {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        sum += buf.chan(ch)[frame_idx] as f32 / i32::MAX as f32;
                                    }
                                    samples.push(sum / channels as f32);
                                }
                            }
                        }
                        AudioBufferRef::S16(buf) => {
                            if channels == 1 {
                                samples.extend(
                                    buf.chan(0).iter().map(|&s| s as f32 / i16::MAX as f32),
                                );
                            } else {
                                for frame_idx in 0..buf.frames() {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        sum += buf.chan(ch)[frame_idx] as f32 / i16::MAX as f32;
                                    }
                                    samples.push(sum / channels as f32);
                                }
                            }
                        }
                        _ => return Err("Unsupported audio format".into()),
                    },
                    Err(symphonia::core::errors::Error::DecodeError(_)) => {
                        // End of stream or decode error
                        break;
                    }
                    Err(e) => return Err(format!("Decode error: {}", e).into()),
                }
            }
            Err(_) => {
                // End of stream or other error
                break;
            }
        }
    }

    Ok((samples, sample_rate))
}
