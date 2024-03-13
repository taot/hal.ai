use anyhow::{Error as E, Result};

use async_stream::stream;
use axum::Json;
use axum::{response::IntoResponse, routing::post, Router};
use axum_streams::StreamBodyAs;
use futures::Stream;

use std::borrow::BorrowMut;
use std::cell::RefCell;

use tokio::sync::mpsc;
use tokio::task;

use crate::common::{TextGenLatency, TextGenRequest, TextGenResponse};
use crate::models::mistral::Mistral;
use crate::models::{TextGen, TextGenEvent};

pub async fn start_server(port: u16, _model: String) -> Result<()> {
    let app = Router::new().route("/api/generate", post(generate));

    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    log::info!("Server started at {}", &addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn generate(Json(payload): Json<TextGenRequest>) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::channel(100);

    let prompt = payload.prompt;

    task::spawn_blocking(move || {
        let mistral = Mistral::new();
        // let mut start_time = std::time::Instant::now();
        // let mut latency = TextGenLatency::new(0, 0);

        let start_time = RefCell::new(std::time::Instant::now());
        let latency = RefCell::new(TextGenLatency::new(0.0, 0.0));
        let n_tokens = RefCell::new(0 as usize);

        let _ = mistral.run(
            &prompt,
            Box::new(move |event| {
                let r = match event {
                    TextGenEvent::Loaded => {
                        latency.borrow_mut().load = start_time.borrow().elapsed().as_secs_f32();
                        *start_time.borrow_mut() = std::time::Instant::now();
                        TextGenResponse::new("".to_string(), 0, latency.borrow().clone(), false)
                    }
                    TextGenEvent::Text { text, tokens } => {
                        latency.borrow_mut().generation =
                            start_time.borrow().elapsed().as_secs_f32();
                        *n_tokens.borrow_mut() = tokens;
                        TextGenResponse::new(
                            text,
                            n_tokens.borrow().clone(),
                            latency.borrow().clone(),
                            false,
                        )
                    }
                    TextGenEvent::Done => {
                        latency.borrow_mut().generation =
                            start_time.borrow().elapsed().as_secs_f32();
                        TextGenResponse::new(
                            "".to_string(),
                            n_tokens.borrow().clone(),
                            latency.borrow().clone(),
                            true,
                        )
                    }
                };
                tx.blocking_send(r).map_err(|err| {
                    log::error!("{:?}", err.to_string());
                    err
                })?;
                Ok(())
            }),
        )?;

        Ok::<(), E>(())
    });

    let generation_stream = stream! {
        while let Some(r) = rx.recv().await {
            // println!("rx.recv(): {:?}", r);
            yield r;
        }
    };

    println!("Returning the stream");

    StreamBodyAs::json_nl(generation_stream)
}
