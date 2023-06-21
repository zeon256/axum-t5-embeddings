use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use object_pool::Pool;
use parking_lot::Mutex;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel,
};
use serde::Deserialize;

use mimalloc::MiMalloc;
use tokio::sync::Semaphore;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing::{debug, info};

mod args;

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

pub struct ProtectedPool {
    semaphore: Arc<Semaphore>,
    pool: Arc<Pool<SentenceEmbeddingsModel>>,
}

async fn feature_extraction(
    State(protected_pool): State<Arc<Mutex<SentenceEmbeddingsModel>>>,
    Json(payload): Json<FeatureExtraction>,
) -> Result<Json<Vec<Vec<f64>>>, AppError> {
    let payload = payload.inputs;
    let model = protected_pool.lock();

    debug!("Acquired lock");

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
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args: args::Args = argh::from_env();

    let args::Args {
        listen,
        port,
        no_workers,
        model,
    } = args;

    let ip = if listen { "0.0.0.0" } else { "127.0.0.1" };

    let addr = format!("{}:{}", ip, port);

    info!("Starting server: {}", addr);

    let protected_model = Arc::new(Mutex::new(
        SentenceEmbeddingsBuilder::local(&model).create_model()?,
    ));

    info!("Loaded model: {:?}", &model);

    // build our application with a single route
    let app = Router::new()
        .route("/feature-extraction", post(feature_extraction))
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
        .with_state(protected_model);

    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}
