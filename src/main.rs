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
    let network_model: Vec<usize> = vec![image_size, 800, 800, 10];

    let epochs = 10;

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

    let mut network = Network::new(network_model.clone(), 0.01, SIGMOID);

    log::info!("Start training with {} images", train_inputs.len());

    for i in 1..=epochs {
        log::info!("[Validation] Epoch {} of {}", i, epochs);

        network.train(&val_inputs, &val_targets);

        log::info!("Network trained with validation data");

        network.validate(&val_data, &val_labels, val_set_size, image_size);
    }

    for i in 1..=epochs {
        log::info!("[Training] Epoch {} of {}", i, epochs);

        network.train(&train_inputs, &train_targets);

        log::info!("Network trained with training data");

        network.validate(&val_data, &val_labels, val_set_size, image_size);
    }

    log::info!("Running final test...");

    network.validate(&test_data, &test_labels, test_set_size, image_size);

    let network_model_str: Vec<String> = network_model.iter().map(|n| n.to_string()).collect();
    let network_model_concatenated = network_model_str.join("-");

    network.save(format!(
        "./data/networks/{}-{}.flu",
        network_model_concatenated,
        Local::now()
    ));
}
