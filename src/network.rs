use std::{
    fs::File,
    io::{self, Read, Write},
};

use ndarray::{s, Array2, ArrayBase, Dim};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::{
    activations::Activation, matrix::Matrix, utils::convert_number_to_target_vec,
    utils::convert_result_vec_to_number,
};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
}

impl Network<'_> {
    pub fn new<'a>(
        layers: Vec<usize>,
        learning_rate: f64,
        activation: Activation<'a>,
    ) -> Network<'a> {
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
            learning_rate,
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
                .map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) {
        for j in 0..inputs.len() {
            if j == 0 || j % 500 == 0 {
                print!(".");
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
    ) -> Vec<Vec<f64>> {
        let mut rights = 0.0;
        let mut wrongs = 0.0;

        let mut failed = vec![0.0; 10];
        let mut results: Vec<Vec<f64>> = Vec::new();

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

            results.push(result.clone());

            let result_number = convert_result_vec_to_number(result);
            let label_number = convert_result_vec_to_number(label);

            if label_number == result_number {
                rights += 1.0;
            } else {
                wrongs += 1.0;

                failed[label_number] += 1.0;
            }
        }

        log::info!(
            "Right: {:?}, Wrong: {:?}, Percent: {:?}%, Failed: {:?}",
            rights,
            wrongs,
            100.0 * (rights / (wrongs + rights)),
            failed
        );

        results
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
}
