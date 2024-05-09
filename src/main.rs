use std::env;

use activations::SIGMOID;
use autometrics::autometrics;
use network::Network;
use std::time::Instant;

use chrono::Local;

use crate::data_set::mnist_data_set;
use crate::data_set::DataSet;
use crate::logger::init_logger;
use metrics_logger::*;

pub mod activations;
pub mod data_set;
pub mod logger;
pub mod matrix;
pub mod metrics_logger;
pub mod network;
pub mod utils;

#[tokio::main]
pub async fn main() {
    init_logger();
    tokio::spawn(init_metrics());

    let mut preload_network = env::var("PRELOAD_NETWORK").unwrap_or(String::from(""));

    loop {
        let network_process = tokio::spawn(init_network(preload_network));

        let result = network_process.await;

        let file_path = result.ok();

        preload_network = file_path.expect("Should return a file path")
    }
}

#[autometrics]
async fn init_network(preload_network: String) -> String {
    let training_set_size: u32 = 50_000;
    let val_set_size: u32 = 10_000;
    let test_set_size: u32 = 10_000;

    let image_size: usize = 784;
    let layers: Vec<usize> = vec![image_size, 800, 800, 10];

    fn scale_by_learning_rate(x: f64) -> f64 {
        x * 0.001
    }

    let epochs = 10;

    let DataSet {
        train_inputs,
        train_targets,
        test_data,
        test_labels,
        val_data,
        val_labels,
    } = mnist_data_set(training_set_size, val_set_size, test_set_size);

    log::info!("Create Network... {:?}", layers);

    let mut network = Network::new(layers, scale_by_learning_rate, SIGMOID);

    if preload_network.len() != 0 {
        log::info!("Preload Network: {}...", preload_network);

        network.load(preload_network);
    }

    log::info!("Start training with {} images", train_inputs.len());

    for i in 1..=epochs {
        let now = Instant::now();
        log::info!("[Training] Epoch {} of {}", i, epochs);

        let success = network.run_training_epoch(
            &train_inputs,
            &train_targets,
            &test_data,
            &test_labels,
            &val_data,
            &val_labels,
            image_size,
            val_set_size,
            test_set_size,
        );

        if success {
            log::info!("Right percentage of 100% reached, will stop training");
            break;
        }

        let elapsed = now.elapsed();
        log::info!("Epoch took: {:.2?}", elapsed);
    }

    log::info!("Running final test...");

    let right_percentage = network.validate(&test_data, &test_labels, test_set_size, image_size);

    let file_path = format!(
        "./data/networks/{}-{}-{}.json",
        network.model(),
        Local::now().format("%Y-%m-%dT%H:%M:%S"),
        right_percentage
    );

    log::info!("Saving model at path {}", file_path);

    let file_path_copy = file_path.clone();

    network.save(file_path);

    file_path_copy
}
