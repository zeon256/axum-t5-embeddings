use std::{sync::Arc};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use object_pool::Pool;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel,
};
use serde::Deserialize;

use mimalloc::MiMalloc;
use tokio::sync::Semaphore;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing::{info, debug};

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
    State(protected_pool): State<Arc<ProtectedPool>>,
    Json(payload): Json<FeatureExtraction>,
) -> Result<Json<Vec<Vec<f64>>>, AppError> {
    
    let payload = payload.inputs;

    let permit = protected_pool.semaphore.acquire().await.unwrap();
    debug!("Acquired permit from semaphore, available permits: {:?}", protected_pool.semaphore.available_permits());
    let model = protected_pool.pool.try_pull().unwrap();

    let sentence_embeddings = model
        .encode_as_tensor(payload.as_ref())
        .map_err(|_| AppError::FeatureExtractionError("Failed to encode tensor"))?;

    drop(permit);
    debug!("Permit returned to semaphore, available permits: {:?}", protected_pool.semaphore.available_permits());

    let embeddings = sentence_embeddings
        .embeddings
        .try_into()
        .map_err(|_| AppError::FeatureExtractionError("Failed to convert tensor"))?;

    Ok(Json(embeddings))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    // env::set_var("RUST_LOG", "axum_t5_embeddings=debug,tower_http=debug");
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
    let mut buffer = Vec::with_capacity(no_workers);

    for _ in 0..no_workers {    
        info!("Loaded model: {:?}", &model);
        buffer.push(SentenceEmbeddingsBuilder::local(&model).create_model()?);
    }

    info!("Starting server: {}", addr);

    let pool = Arc::new(Pool::from_vec(buffer));

    let protected_pool = Arc::new(ProtectedPool {
        semaphore: Arc::new(Semaphore::new(no_workers)),
        pool,
    });

    // build our application with a single route
    let app = Router::new()
        .route("/feature-extraction", post(feature_extraction))
        .layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
        .with_state(protected_pool);

    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}
