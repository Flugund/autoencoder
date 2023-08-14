use std::env;

use activations::SIGMOID;
use network::Network;

use chrono::Local;

use crate::data_set::mnist_data_set;
use crate::data_set::DataSet;
use crate::logger::init_logger;

pub mod activations;
pub mod data_set;
pub mod logger;
pub mod matrix;
pub mod network;
pub mod utils;

fn main() {
    init_logger();

    let training_set_size: u32 = 50_000;
    let val_set_size: u32 = 10_000;
    let test_set_size: u32 = 10_000;

    let image_size: usize = 784;
    let layers: Vec<usize> = vec![image_size, 800, 800, 10];
    let learning_rate = 0.001;

    let epochs = 20;

    let preload_network = env::var("PRELOAD_NETWORK").unwrap_or(String::from(""));

    let DataSet {
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        test_data,
        test_labels,
        val_data,
        val_labels,
    } = mnist_data_set(training_set_size, val_set_size, test_set_size, image_size);

    log::info!("Create Network...");

    let mut network = Network::new(layers, learning_rate, SIGMOID);

    if preload_network.len() != 0 {
        log::info!("Preload Network: {}...", preload_network);

        network.load(preload_network);
    }

    log::info!("Start training with {} images", train_inputs.len());

    for i in 1..=epochs {
        log::info!("[Validation] Epoch {} of {}", i, epochs);

        network.train(&val_inputs, &val_targets);

        log::info!("Network trained with validation data");

        let right_percentage = network.validate(&val_data, &val_labels, val_set_size, image_size);

        if right_percentage == 100.0 {
            log::info!("Right percentage of 100% reached, will stop training");
            break;
        }
    }

    for i in 1..=epochs {
        log::info!("[Training] Epoch {} of {}", i, epochs);

        network.train(&train_inputs, &train_targets);

        log::info!("Network trained with training data");

        network.validate(&val_data, &val_labels, val_set_size, image_size);
    }

    log::info!("Running final test...");

    let right_percentage = network.validate(&test_data, &test_labels, test_set_size, image_size);

    network.save(format!(
        "./data/networks/{}-{}-{}.json",
        network.model(),
        Local::now().format("%Y-%m-%dT%H:%M:%S"),
        right_percentage
    ));
}
