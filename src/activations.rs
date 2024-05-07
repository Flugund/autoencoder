/**
 * This code is inspired by the https://www.youtube.com/watch?v=FI-8L-hobDY&t=10s video by MathleteDev
 *
 * https://github.com/mathletedev/rust-ml/
 **/
// use std::f64::consts::E;

// #[derive(Clone)]
// pub struct Activation<'a> {
//     pub function: &'a dyn Fn(f64) -> f64,
//     pub derivative: &'a dyn Fn(f64) -> f64,
// }

// pub const IDENTITY: Activation = Activation {
//     function: &|x| x,
//     derivative: &|_| 1.0,
// };

// pub const SIGMOID: Activation = Activation {
//     function: &|x| 1.0 / (1.0 + E.powf(-x)),
//     derivative: &|x| x * (1.0 - x),
// };
use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

// Static functions for the identity activation
fn identity_function(x: f64) -> f64 {
    x
}

fn identity_derivative(_: f64) -> f64 {
    1.0
}

// Static functions for the sigmoid activation
fn sigmoid_function(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid_function(x) * (1.0 - sigmoid_function(x))
}

pub const IDENTITY: Activation = Activation {
    function: identity_function,
    derivative: identity_derivative,
};

pub const SIGMOID: Activation = Activation {
    function: sigmoid_function,
    derivative: sigmoid_derivative,
};
