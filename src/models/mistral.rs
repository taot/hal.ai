use anyhow::{Error as E, Result};
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::mistral::Config;
use candle_transformers::models::quantized_mistral::Model;

use crate::models::TextGen;
use crate::models::TextGenEvent;
use crate::models::TokenOutputStream;

pub struct Mistral {}

impl Mistral {
    pub fn new() -> Self {
        Mistral {}
    }
}

impl TextGen for Mistral {
    fn run(&self, prompt: &str, callback: Box<dyn Fn(TextGenEvent) -> Result<()>>) -> Result<()> {
        let ctx = load_context()?;
        callback(TextGenEvent::Loaded)?;
        generate(ctx, prompt, &callback)?;
        callback(TextGenEvent::Done)?;
        Ok(())
    }
}

struct MistralContext {
    model: Model,
    tokenizer: TokenOutputStream,
    device: Device,
}

fn load_context() -> Result<MistralContext> {
    let model_id = "lmz/candle-mistral";

    // params
    let revision = "main";
    let use_flash_attn = false;
    let cpu = true;

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let tokenizer_file = repo.get("tokenizer.json")?;
    let model_file = repo.get("model-q4k.gguf")?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let config = Config::config_7b_v0_1(use_flash_attn);

    let device = crate::utils::device(cpu)?;
    let vb =
        candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_file, &device)?;
    let model = Model::new(&config, vb)?;

    Ok(MistralContext {
        model,
        tokenizer: TokenOutputStream::new(tokenizer),
        device,
    })
}

fn generate(
    mut ctx: MistralContext,
    prompt: &str,
    callback: &Box<dyn Fn(TextGenEvent) -> Result<()>>,
) -> Result<()> {
    let sample_len: usize = 1000;
    let repeat_penalty: f32 = 1.1;
    let repeat_last_n: usize = 64;
    let seed: u64 = 299792458;
    let temperature: Option<f64> = None;
    let top_p: Option<f64> = None;

    use std::io::Write;

    let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

    ctx.tokenizer.clear();
    let mut tokens = ctx
        .tokenizer
        .tokenizer()
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    for &t in tokens.iter() {
        if let Some(t) = ctx.tokenizer.next_token(t)? {
            print!("{t}")
        }
    }
    std::io::stdout().flush()?;

    let mut n_generated_tokens = 0usize;
    let eos_token = match ctx.tokenizer.get_token("</s>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the </s> token"),
    };

    let start_time = std::time::Instant::now();
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let text_context = &tokens[start_pos..];
        let input = Tensor::new(text_context, &ctx.device)?.unsqueeze(0)?;
        let logits = ctx.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        n_generated_tokens += 1;
        if next_token == eos_token {
            break;
        }
        if let Some(t) = ctx.tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;

            callback(TextGenEvent::Text {
                text: t,
                tokens: n_generated_tokens,
            })?;
        }
    }
    let dt = start_time.elapsed();
    if let Some(rest) = ctx.tokenizer.decode_rest().map_err(E::msg)? {
        // println!("Rest.");
        print!("{rest}");

        callback(TextGenEvent::Text {
            text: rest,
            tokens: n_generated_tokens,
        })?;
    }
    std::io::stdout().flush()?;
    println!(
        "\n{n_generated_tokens} tokens generated ({:.2} token/s)",
        n_generated_tokens as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
