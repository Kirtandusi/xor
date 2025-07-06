use rand::Rng;
pub struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
    pub(crate) activation: fn(f64) -> f64,
    pub(crate) activation_derivative: fn(f64) -> f64,
    pub(crate) last_input_sum: f64, //to store the weighted sum before activation for derivative
    pub(crate) output: f64,
}
impl Neuron {
    pub fn new(
        num_inputs: usize,
        activation: fn(f64) -> f64,
        derivative: fn(f64) -> f64,
    ) -> Self {
        let mut rng = rand::rng();
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
        activations
    }


    pub fn train(&mut self, inputs: Vec<f64>, targets: Vec<f64>) {
        let outputs = self.forward(inputs.clone());

        // Calculate output layer deltas
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        let mut output_deltas = Vec::new();
        for (i, neuron) in self.layers.last().unwrap().iter().enumerate() {
            let error = targets[i] - neuron.output;
            let delta = error * (neuron.activation_derivative)(neuron.last_input_sum);
            output_deltas.push(delta);
        }
        deltas.push(output_deltas);

        // Calculate hidden layer deltas
        for l in (0..self.layers.len() - 1).rev() {
            let mut layer_deltas = Vec::new();
            for (j, neuron) in self.layers[l].iter().enumerate() {
                let mut error = 0.0;
                for (k, next_neuron) in self.layers[l + 1].iter().enumerate() {
                    error += next_neuron.weights[j] * deltas[0][k]; // 0 is last delta layer
                }
                let delta = error * (neuron.activation_derivative)(neuron.last_input_sum);
                layer_deltas.push(delta);
            }
            deltas.insert(0, layer_deltas); // insert at front
        }

        // Update weights and biases
        let mut layer_inputs = inputs;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let current_deltas = &deltas[l];
            for (n, neuron) in layer.iter_mut().enumerate() {
                for w in 0..neuron.weights.len() {
                    neuron.weights[w] += self.learning_rate * current_deltas[n] * layer_inputs[w];
                }
                neuron.bias += self.learning_rate * current_deltas[n];
            }
            // prepare inputs for next layer
            layer_inputs = layer.iter().map(|n| n.output).collect();
        }
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

    for epoch in 0..10000 {
        for (input, target) in &training_data {
            nn.train(input.clone(), target.clone());
        }
        if epoch % 1000 == 0 {
            println!("Epoch {epoch}");
            for (input, target) in &training_data {
                let output = nn.forward(input.clone());
                println!("Input: {:?}, Predicted: {:?}, Target: {:?}", input, output, target);
            }
        }
    }
}

