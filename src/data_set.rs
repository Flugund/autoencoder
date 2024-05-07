use mnist::*;
use ndarray::{s, Array2, Array3, ArrayBase, Dim};

use crate::utils::convert_number_to_target_vec;

pub struct DataSet {
    pub val_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>>,
    pub val_labels: Array2<f64>,
    pub train_inputs: Vec<Vec<f64>>,
    pub train_targets: Vec<Vec<f64>>,
    pub test_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>>,
    pub test_labels: Array2<f64>,
}

pub fn mnist_data_set(training_set_size: u32, val_set_size: u32, test_set_size: u32) -> DataSet {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_set_size)
        .validation_set_length(val_set_size)
        .test_set_length(test_set_size)
        .finalize();

    let train_data = Array3::from_shape_vec((training_set_size as usize, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels: Array2<f64> =
        Array2::from_shape_vec((training_set_size as usize, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .map(|x| *x as f64);

    let val_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>> =
        Array3::from_shape_vec((val_set_size as usize, 28, 28), val_img)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f64 / 256.0);

    let val_labels: Array2<f64> = Array2::from_shape_vec((val_set_size as usize, 1), val_lbl)
        .expect("Error converting validation labels to Array2 struct")
        .map(|x| *x as f64);

    let test_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 3]>> =
        Array3::from_shape_vec((test_set_size as usize, 28, 28), tst_img)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f64 / 256.0);

    let test_labels: Array2<f64> = Array2::from_shape_vec((test_set_size as usize, 1), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut train_inputs: Vec<Vec<f64>> = Vec::new();
    let mut train_targets: Vec<Vec<f64>> = Vec::new();

    for i in 0..(training_set_size as usize) {
        let image = train_data
            .slice(s![i, .., ..])
            .to_owned()
            .into_shape((784,))
            .unwrap()
            .to_vec();

        let label =
            convert_number_to_target_vec(train_labels.slice(s![i, ..]).to_vec()[0] as usize);

        train_inputs.push(image);
        train_targets.push(label);
    }

    DataSet {
        val_data,
        val_labels,
        train_inputs,
        train_targets,
        test_data,
        test_labels,
    }
}
