/**
 * This code is inspired by the https://www.youtube.com/watch?v=FI-8L-hobDY&t=10s video by MathleteDev
 *
 * https://github.com/mathletedev/rust-ml/
 **/
use std::{
    fs::File,
    io::{self, Read, Write},
};

use autometrics::autometrics;
use ndarray::{s, Array2, ArrayBase, Dim};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::{
    activations::Activation, matrix::Matrix, utils::convert_number_to_target_vec,
    utils::convert_result_vec_to_number,
};

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    scale_by_learning_rate: fn(f64) -> f64,
    activation: Activation,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
}

#[autometrics]
impl Network {
    pub fn new<'a>(
        layers: Vec<usize>,
        scale_by_learning_rate: fn(f64) -> f64,
        activation: Activation,
    ) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            scale_by_learning_rate,
            activation,
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid inputs length");
        }

        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }

        current.transpose().data[0].to_owned()
    }

    pub fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invalid targets length");
        }

        let parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = Matrix::from(vec![targets]).transpose().subtract(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .dot_multiply(&errors)
                .map(self.scale_by_learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) {
        let input_length = inputs.len();
        print!("Progress: ");
        for j in 0..input_length {
            if j == 0 || j % 500 == 0 {
                let progress_pct = 100 * j / input_length;
                if progress_pct % 10 == 0 {
                    print!("{}%", progress_pct);
                } else {
                    print!(".");
                }

                io::stdout().flush().unwrap();
            }

            let outputs = self.feed_forward(inputs[j].clone());

            self.back_propogate(outputs, targets[j].clone());
        }
        println!("");

        log::info!("Completed training")
    }

    pub fn validate(
        &mut self,
        test_data: &ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>>,
        test_labels: &Array2<f64>,
        validation_set_size: u32,
        shape: usize,
    ) -> f64 {
        let mut rights = 0.0;
        let mut wrongs = 0.0;

        let mut failed = vec![0.0; 10];

        for i in 0..(validation_set_size as usize) {
            let image = test_data
                .slice(s![i, .., ..])
                .to_owned()
                .into_shape((shape,))
                .unwrap()
                .to_vec();
            let label =
                convert_number_to_target_vec(test_labels.slice(s![i, ..]).to_vec()[0] as usize);

            let result = self.feed_forward(image);

            let result_number = convert_result_vec_to_number(result);
            let label_number = convert_result_vec_to_number(label);

            if label_number == result_number {
                rights += 1.0;
            } else {
                wrongs += 1.0;

                failed[label_number] += 1.0;
            }
        }

        let right_percentage = 100.0 * (rights / (wrongs + rights));

        log::info!(
            "Right: {:?}, Wrong: {:?}, Percent: {:?}%, Failed: {:?}",
            rights,
            wrongs,
            right_percentage,
            failed
        );

        right_percentage
    }

    pub fn model(&self) -> String {
        let network_model_str: Vec<String> =
            self.layers.par_iter().map(|n| n.to_string()).collect();
        let network_model_concatenated = network_model_str.join("-");

        network_model_concatenated
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to touch save file");

        file.write_all(
			json!({
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>()
			}).to_string().as_bytes(),
		).expect("Unable to write to save file");
    }

    pub fn load(&mut self, file: String) {
        let mut file = File::open(file).expect("Unable to open save file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read save file");

        let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..self.layers.len() - 1 {
            weights.push(Matrix::from(save_data.weights[i].clone()));
            biases.push(Matrix::from(save_data.biases[i].clone()));
        }

        self.weights = weights;
        self.biases = biases;
    }

    pub fn run_training_epoch(
        &mut self,
        train_inputs: &Vec<Vec<f64>>,
        train_targets: &Vec<Vec<f64>>,
        test_data: &ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>>,
        test_labels: &Array2<f64>,
        val_data: &ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>>,
        val_labels: &Array2<f64>,
        image_size: usize,
        val_set_size: u32,
        test_set_size: u32,
    ) -> bool {
        self.train(&train_inputs, &train_targets);

        log::info!("Network trained with training data");

        let right_percentage = self.validate(&val_data, &val_labels, val_set_size, image_size);

        if right_percentage == 100.0 {
            log::info!("Right percentage of 100% reached, will stop training");
            return true;
        }

        log::info!("Validate using final test data set");

        let right_percentage_test =
            self.validate(&test_data, &test_labels, test_set_size, image_size);

        if right_percentage_test == 100.0 {
            log::info!("Right percentage of 100% reached, will stop training");
            return true;
        }

        return false;
    }
}
