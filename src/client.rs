use anyhow::Result;

use futures::StreamExt;
use reqwest_streams::JsonStreamResponse;
use std::io::Write;

use crate::common::{TextGenRequest, TextGenResponse};

pub async fn run_client(port: u16) -> Result<()> {
    println!("Runing client...");

    let mut prompt = String::new();

    loop {
        print!("hal> ");
        std::io::stdout().flush()?;

        std::io::stdin().read_line(&mut prompt).unwrap();

        let req = TextGenRequest {
            prompt: prompt.clone(),
        };

        let reqwest_client = reqwest::Client::new();
        let mut res = reqwest_client
            .post(format!("http://localhost:{port}/api/generate"))
            .json(&req)
            .send()
            .await?
            .json_array_stream::<TextGenResponse>(1024);

        let mut first = true;
        let mut n_tokens = 0;
        let mut generation_latency = 0.0;

        while let Some(Ok(TextGenResponse {
            text,
            tokens,
            latency,
            done: _,
        })) = res.next().await
        {
            if first {
                println!("Load latency: {}", latency.load);
                first = false;
            }
            n_tokens = tokens;
            generation_latency = latency.generation;

            print!("{}", text);
            std::io::stdout().flush()?;
        }
        println!();
        println!(
            "{} tokens generated in {} second ({} token/s)",
            n_tokens,
            generation_latency,
            (n_tokens as f32) / generation_latency
        );
    }

    Ok(())
}
