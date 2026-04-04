use wasm_bindgen::prelude::*;

use crate::ml::infer::Inferencer;

static WEIGHTS: &[u8] = include_bytes!("../weights/2026-04-04-81.4.safetensors");

#[wasm_bindgen]
pub struct VoiceClassifier {
    inner: Inferencer,
}

#[wasm_bindgen]
impl VoiceClassifier {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<VoiceClassifier, JsValue> {
        let inner = Inferencer::from_weights_bytes(WEIGHTS.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Predict the speaker from a `Float32Array` of mono PCM samples.
    ///
    /// Returns a JSON string:
    ///   `{"label":"Johannes","confidence":0.9512}` on detection, or
    ///   `{"label":null,"confidence":0}` for silence / uncertainty.
    pub fn predict(&self, samples: &[f32], sample_rate: u32) -> String {
        match self.inner.predict(samples, sample_rate) {
            Some((label, confidence)) => format!(
                r#"{{"label":"{}","confidence":{:.4}}}"#,
                label.name(),
                confidence
            ),
            None => r#"{"label":null,"confidence":0}"#.to_string(),
        }
    }
}
