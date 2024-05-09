/**
 * This code is inspired by the https://www.youtube.com/watch?v=FI-8L-hobDY&t=10s video by MathleteDev
 *
 * https://github.com/mathletedev/rust-ml/
 **/
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
    x * (1.0 - x)
}

pub const IDENTITY: Activation = Activation {
    function: identity_function,
    derivative: identity_derivative,
};

pub const SIGMOID: Activation = Activation {
    function: sigmoid_function,
    derivative: sigmoid_derivative,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_function() {
        assert_eq!(identity_function(0.0), 0.0);
        assert_eq!(identity_function(1.0), 1.0);
        assert_eq!(identity_function(-1.0), -1.0);
        assert_eq!(identity_function(2.5), 2.5);
    }

    #[test]
    fn test_identity_derivative() {
        // Derivative of identity function should always be 1
        assert_eq!(identity_derivative(0.0), 1.0);
        assert_eq!(identity_derivative(1.0), 1.0);
        assert_eq!(identity_derivative(-1.0), 1.0);
        assert_eq!(identity_derivative(2.5), 1.0);
    }

    #[test]
    fn test_sigmoid_function() {
        // Test sigmoid at a few points
        assert_eq!(sigmoid_function(0.0), 0.5);
        assert!((sigmoid_function(1.0) - 0.7310585786300049).abs() < 1e-7);
        assert!((sigmoid_function(-1.0) - 0.2689414213699951).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_derivative() {
        // Test sigmoid derivative using the output of sigmoid function
        let x = sigmoid_function(0.5);
        let dx = sigmoid_derivative(x);
        assert!((dx - 0.2350037122015945).abs() < 1e-7);

        let y = sigmoid_function(2.0);
        let dy = sigmoid_derivative(y);
        assert!((dy - 0.10499358540350662).abs() < 1e-7);
    }
}
