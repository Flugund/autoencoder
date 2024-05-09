use autometrics::autometrics;
use rand::distributions::{Distribution, Uniform};
/**
 * This code is inspired by the https://www.youtube.com/watch?v=FI-8L-hobDY&t=10s video by MathleteDev
 *
 * https://github.com/mathletedev/rust-ml/
 **/
use rayon::prelude::*;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

#[autometrics]
impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let between = Uniform::new(-1.0, 1.0);
        let res_data: Vec<Vec<f64>> = (0..rows)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                (0..cols)
                    .into_iter()
                    .map(|_| between.sample(&mut rng))
                    .collect()
            })
            .collect();

        Matrix {
            rows,
            cols,
            data: res_data,
        }
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let res_data: Vec<Vec<f64>> = (0..self.rows)
            .into_par_iter()
            .map(|i| {
                (0..other.cols)
                    .into_iter()
                    .map(|j| {
                        let mut sum = 0.0;
                        for k in 0..self.cols {
                            sum += self.data[i][k] * other.data[k][j];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: res_data,
        }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to subtract matrix of incorrect dimensions");
        }

        let res_data: Vec<Vec<f64>> = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(row_self, row_other)| {
                row_self
                    .iter()
                    .zip(row_other.iter())
                    .map(|(&x, &y)| x + y)
                    .collect()
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: res_data,
        }
    }

    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to dot multiply by matrix of incorrect dimensions");
        }

        let res_data: Vec<Vec<f64>> = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(row_self, row_other)| {
                row_self
                    .iter()
                    .zip(row_other.iter())
                    .map(|(&x, &y)| x * y)
                    .collect()
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: res_data,
        }
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to subtract matrix of incorrect dimensions");
        }

        let res_data: Vec<Vec<f64>> = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(row_self, row_other)| {
                row_self
                    .iter()
                    .zip(row_other.iter())
                    .map(|(&x, &y)| x - y)
                    .collect()
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: res_data,
        }
    }

    pub fn map(&self, function: fn(f64) -> f64) -> Matrix {
        let res_data: Vec<Vec<f64>> = self
            .data
            .par_iter() // Parallel iterator over rows
            .map(|row| {
                row.iter() // Iterator over each element in the row
                    .map(|&x| function(x)) // Apply the activation function
                    .collect::<Vec<f64>>()
            })
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: res_data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let res_data: Vec<Vec<f64>> = (0..self.cols)
            .into_par_iter()
            .map(|j| {
                (0..self.rows)
                    .into_iter()
                    .map(|i| self.data[i][j])
                    .collect()
            })
            .collect();

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: res_data,
        }
    }
}
