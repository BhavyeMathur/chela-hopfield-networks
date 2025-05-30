mod data;

use crate::data::{load_data, save_ndarray_as_png};
use chela::*;
use rand::distr::{Distribution, Uniform};
use std::path::Path;

struct HopfieldNetwork<'a> {
    weights: NdArray<'a, f64>,
    state: NdArray<'a, f64>,
}

impl HopfieldNetwork<'_> {
    fn new(memory: NdArray<f64>) -> Self {
        let n = memory.shape()[0] as f64;
        let p = memory.shape()[1];

        let mut weights = (&memory).T().matmul(&memory);
        weights *= 1.0 / n;
        weights.diagonal().zero();

        let state = NdArray::uniform([p], -1.0, 1.0);

        Self { weights, state }
    }

    fn update_state(&mut self, steps: usize) {
        let uniform = Uniform::try_from(0..self.weights.len()).unwrap();
        let mut rng = rand::rng();

        for _ in 0..steps {
            let neuron = uniform.sample(&mut rng);

            let weight_col = self.weights.slice([neuron]);
            let activation = weight_col.dot(&self.state);

            self.state[neuron] = if activation.value() >= 0.0 { 1.0 } else { -1.0 };
        }
    }

    fn save_model_as_png<P: AsRef<Path>>(&self, path: P) {
        save_ndarray_as_png(&self.weights, 24 * 24, 24 * 24, path);
    }

    fn save_state<P: AsRef<Path>>(&self, path: P) {
        save_ndarray_as_png(&self.state, 24, 24, path);
    }
}

fn main() {
    let pixels = load_data("data/popcorn.png");

    let mut model = HopfieldNetwork::new(pixels);
    model.save_model_as_png("model.png");

    for i in 0..150 {
        model.update_state(25);
        model.save_state(format!("output/{}.png", i))
    }
}
