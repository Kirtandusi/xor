use crate::neuron::Neuron;

pub struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
    pub(crate) activation: fn(f64) -> f64,
    pub(crate) activation_derivative: fn(f64) -> f64,
    pub(crate) last_input_sum: f64,
    pub(crate) output: f64,
}
impl Neuron {
    pub fn new(
        num_inputs: usize,
        activation: fn(f64) -> f64,
        derivative: fn(f64) -> f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_inputs)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let bias = rng.gen_range(-1.0..1.0);
        Self {
            weights,
            bias,
            activation,
            activation_derivative: derivative,
            last_input_sum: 0.0,
            output: 0.0,
        }
    }
    pub fn activate(&mut self, inputs: &[f64]) -> f64 {
        let sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<f64>()
            + self.bias;
        self.last_input_sum = sum;
        self.output = (self.activation)(sum);
        self.output
    }
}
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub struct NeuralNetwork {
    pub layers: Vec<Vec<Neuron>>, // 2 layers: hidden and output
    pub learning_rate: f64,
}
impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let hidden_layer: Vec<Neuron> = (0..hidden_size)
            .map(|_| Neuron::new(input_size, sigmoid, sigmoid_derivative))
            .collect(); //make a vec<f64> of these
        let output_layer: Vec<Neuron> = (0..output_size)
            .map(|_| Neuron::new(hidden_size, sigmoid, sigmoid_derivative))
            .collect();
        Self {
            layers: vec![hidden_layer, output_layer],
            learning_rate,
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut activations = inputs;
        for layer in self.layers.iter_mut() {
            let mut new_activations = Vec::new();
            for neuron in layer.iter_mut() {
                new_activations.push(neuron.activate(&activations));
            }
            activations = new_activations;
        }
        activations;
    }

    pub fn train(&mut self, inputs: Vec<f64>, targets: Vec<f64>) {

    }
}
fn main() {
    let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);

    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    //actual training
    for epoch in 0..100000 {
        for (input, target) in &training_data.iter() {
            nn.train(inputs.clone(), target.clone());
        }
    }
}
