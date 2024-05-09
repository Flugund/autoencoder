use autometrics::prometheus_exporter;
use axum::{routing::*, Router};
use std::error::Error;
use std::net::Ipv4Addr;
use tokio::net::TcpListener;

pub async fn init_metrics() -> Result<(), Box<dyn Error + Send + Sync>> {
    log::info!("Starting prometheus exporter...");

    prometheus_exporter::init();

    let app = Router::new().route(
        "/metrics",
        get(|| async { prometheus_exporter::encode_http_response() }),
    );

    log::info!("Creating web app...");

    let listener = TcpListener::bind((Ipv4Addr::from([127, 0, 0, 1]), 3000)).await?;

    log::info!("Binded listener to port 3000.");

    axum::serve(listener, app).await?;

    log::info!("Finished setting up metrics.");

    Ok(())
}
