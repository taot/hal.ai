use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextGenRequest {
    pub prompt: String,
}

impl TextGenRequest {
    pub fn new(prompt: String) -> Self {
        Self { prompt }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextGenLatency {
    pub load: f32,
    pub generation: f32,
}

impl TextGenLatency {
    pub fn new(load: f32, generation: f32) -> Self {
        Self { load, generation }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextGenResponse {
    pub text: String,
    pub tokens: usize,
    pub latency: TextGenLatency,
    pub done: bool,
}

impl TextGenResponse {
    pub fn new(text: String, tokens: usize, latency: TextGenLatency, done: bool) -> Self {
        Self {
            text,
            tokens,
            latency,
            done,
        }
    }
}
