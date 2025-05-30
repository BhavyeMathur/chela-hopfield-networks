mod data;

use chela::*;
use rand::distr::{Distribution, Uniform};
use crate::data::{load_data, save_png};
use std::path::Path;

struct HopfieldNetwork<'a> {
    weights: NdArray<'a, f64>,
    state: NdArray<'a, f64>,
}

impl<'a> HopfieldNetwork<'a> {
    fn new(memory: NdArray<'a, f64>) -> Self {
        let mut weights = (&memory).T().matmul(&memory);
        weights *= 1.0 / memory.len() as f64;
        weights.diagonal().zero();
        
        let state = NdArray::randint([weights.len()], -1, 1).astype::<f64>();

        Self { weights, state }
    }

    fn update_state(&mut self, steps: usize) {
        let uniform = Uniform::try_from(0..self.weights.len()).unwrap();
        let mut rng = rand::rng();

        for _ in 0..steps {
            let neuron = uniform.sample(&mut rng);
            
            let neuron_weights = self.weights.slice([neuron]);
            let activation = neuron_weights.dot(&self.state);
            
            self.state[neuron] = if activation.value() >= 0.0 { 1.0 } else { -1.0 };
        }
    }

    fn save_model_as_png<P: AsRef<Path>>(&self, path: P) {
        let mut data = &self.weights - self.weights.min();
        data /= self.weights.max() - self.weights.min();
        data *= 255.0;
        
        let data = data.astype().into_data_vector();
        save_png(data, 24 * 24, 24 * 24, path);
    }

    fn save_state<P: AsRef<Path>>(&self, path: P) {
        let data = (&self.state + 1.0) * (255.0 / 2.0);
        let data = data.astype().into_data_vector();
        
        save_png(data, 24, 24, path);
    }
}

fn main() {
    let pixels = load_data("data/burger.png");

    let mut model = HopfieldNetwork::new(pixels);
    model.save_model_as_png("model.png");

    for i in 0..150 {
        model.update_state(25);
        model.save_state(format!("output/{}.png", i))
    }
}
