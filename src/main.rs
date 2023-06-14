use std::sync::{Arc};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel,
};
use serde::Deserialize;
use tokio::sync::Mutex;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Deserialize)]
pub struct FeatureExtraction {
    inputs: Vec<String>,
}
pub enum AppError {
    FeatureExtractionError(&'static str),
}

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong"),
        )
            .into_response()
    }
}

async fn feature_extraction(
    State(model): State<Arc<Mutex<SentenceEmbeddingsModel>>>,
    Json(payload): Json<FeatureExtraction>,
) -> Result<Json<Vec<Vec<f64>>>, AppError> {
    let model = model.lock().await;
    let payload = payload.inputs;
    let sentence_embeddings = model
        .encode_as_tensor(payload.as_ref())
        .map_err(|_| AppError::FeatureExtractionError("Failed to encode tensor"))?;

    let embeddings = sentence_embeddings
        .embeddings
        .try_into()
        .map_err(|_| AppError::FeatureExtractionError("Failed to convert tensor"))?;

    Ok(Json(embeddings))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set the RUST_LOG, if it hasn't been explicitly defined
    
    std::env::set_var("RUST_LOG", "axum_instructor=debug,tower_http=debug");
    tracing_subscriber::fmt::init();
    
    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::local("/home/zeon256/Documents/work/copilot/text-generation-webui2/models/hkunlp_instructor-large/")
        .create_model()?;

    let model = Arc::new(Mutex::new(model));

    // build our application with a single route
    let app = Router::new()
        .route("/feature-extraction", post(feature_extraction))
        .with_state(model.clone());

    // run it with hyper on localhost:3000
    axum::Server::bind(&"127.0.0.1:9999".parse()?)
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}
