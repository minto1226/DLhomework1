"""Phase 2: one SGD step with any tuple of hidden-layer widths.

Run with: python3 phase2_multilayer.py
This uses a synthetic input to show the forward and backward loops.
"""

import numpy as np


def initialize_parameters(hidden_layers, rng):
    """Build one weight matrix and bias row for each adjacent pair of widths."""
    layer_sizes = (784, *hidden_layers, 10)
    weights = []
    biases = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        biases.append(np.zeros((1, fan_out)))
    return layer_sizes, weights, biases


def forward(x, weights, biases):
    """Keep z values and activations so the backward loop can use them."""
    # activations[layer] is the input to weights[layer].
    # z_values[layer] is that layer's result before ReLU or Softmax.
    z_values = []
    activations = [x]
    for layer in range(len(weights)):
        z = activations[-1] @ weights[layer] + biases[layer]
        z_values.append(z)
        if layer == len(weights) - 1:
            exp_z = np.exp(z)
            a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            a = np.maximum(0.0, z)
        activations.append(a)
    return z_values, activations


rng = np.random.default_rng(7)
hidden_layers = (256, 128)  # Try another tuple, such as (128,) or (64, 32, 16).
layer_sizes, weights, biases = initialize_parameters(hidden_layers, rng)
learning_rate = 0.01

x = rng.random((1, 784))  # Synthetic normalized image, not MNIST data.
label = 3
y = np.zeros((1, 10))
y[0, label] = 1.0

z_values, activations = forward(x, weights, biases)
p = activations[-1]
loss = -np.sum(y * np.log(p))
predicted_digit = int(np.argmax(p, axis=1)[0])

# The output-layer delta is the same p - y derived in Phase 1.
delta = p - y
d_weights = [None] * len(weights)
d_biases = [None] * len(biases)

# Walk from the last weight matrix back to the first.
for layer in range(len(weights) - 1, -1, -1):
    d_weights[layer] = activations[layer].T @ delta
    d_biases[layer] = delta.copy()
    if layer > 0:
        # weights[layer].T returns the gradient to the previous hidden layer.
        delta = (delta @ weights[layer].T) * (z_values[layer - 1] > 0.0)

assert len(activations) == len(layer_sizes)
for layer in range(len(weights)):
    assert weights[layer].shape == d_weights[layer].shape == (
        layer_sizes[layer], layer_sizes[layer + 1]
    )
    assert biases[layer].shape == d_biases[layer].shape == (1, layer_sizes[layer + 1])
    assert activations[layer].shape == (1, layer_sizes[layer])
    assert z_values[layer].shape == activations[layer + 1].shape == (
        1, layer_sizes[layer + 1]
    )


def current_loss():
    """Recompute loss with the current weights for the gradient check."""
    _, current_activations = forward(x, weights, biases)
    return -np.sum(y * np.log(current_activations[-1]))


def numerical_gradient(weight, index, step=1e-5):
    """Central difference for one selected weight."""
    original = weight[index]
    weight[index] = original + step
    loss_plus = current_loss()
    weight[index] = original - step
    loss_minus = current_loss()
    weight[index] = original
    return (loss_plus - loss_minus) / (2.0 * step)


# Check one nontrivial weight from every layer before changing any parameters.
print(f"layer sizes: {layer_sizes}")
for layer in range(len(weights)):
    index = np.unravel_index(np.argmax(np.abs(d_weights[layer])), d_weights[layer].shape)
    analytic = d_weights[layer][index]
    numeric = numerical_gradient(weights[layer], index)
    np.testing.assert_allclose(numeric, analytic, rtol=1e-4, atol=1e-7)
    print(f"layer {layer}: W{weights[layer].shape}, gradient={analytic:.8f}, numeric={numeric:.8f}")

# All deltas and gradients above used the same pre-update weights.
for layer in range(len(weights)):
    weights[layer] -= learning_rate * d_weights[layer]
    biases[layer] -= learning_rate * d_biases[layer]

print(f"loss before update: {loss:.6f}")
print(f"predicted digit before update: {predicted_digit}")
print(f"loss after update:  {current_loss():.6f}")
