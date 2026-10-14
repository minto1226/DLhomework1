"""Phase 1: one explicit SGD step for a 784 -> 128 -> 10 network.

Run with: python3 phase1_baseline.py
The input below is a synthetic shape-check example, not an MNIST image.
"""

import numpy as np


rng = np.random.default_rng(7)
learning_rate = 0.01

# One normalized 28 x 28 image, flattened to a row; replace with real data.
x = rng.random((1, 784))
label = 3
y = np.zeros((1, 10))
y[0, label] = 1.0

# He initialization for the fixed hidden-layer architecture.
W1 = rng.standard_normal((784, 128)) * np.sqrt(2.0 / 784)
b1 = np.zeros((1, 128))
W2 = rng.standard_normal((128, 10)) * np.sqrt(2.0 / 128)
b2 = np.zeros((1, 10))

# Forward pass: (1, 784) @ (784, 128) -> (1, 128).
z1 = x @ W1 + b1
a1 = np.maximum(0.0, z1)

# Output: (1, 128) @ (128, 10) -> (1, 10).
z2 = a1 @ W2 + b2
exp_z2 = np.exp(z2)
p = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
loss = -np.sum(y * np.log(p))
predicted_digit = int(np.argmax(p, axis=1)[0])

assert x.shape == (1, 784) and y.shape == (1, 10)
assert z1.shape == a1.shape == (1, 128)
assert z2.shape == p.shape == (1, 10)

# Backward pass: softmax + cross-entropy gives dL/dz2 = p - y.
dz2 = p - y
dW2 = a1.T @ dz2
db2 = dz2.copy()

# Use the current W2 before changing any parameters.
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0.0)
dW1 = x.T @ dz1
db1 = dz1.copy()

assert dW2.shape == W2.shape and db2.shape == b2.shape
assert da1.shape == dz1.shape == (1, 128)
assert dW1.shape == W1.shape and db1.shape == b1.shape


def current_loss():
    """Recompute the same loss while a weight is perturbed for checking."""
    check_z1 = x @ W1 + b1
    check_a1 = np.maximum(0.0, check_z1)
    check_z2 = check_a1 @ W2 + b2
    check_exp = np.exp(check_z2)
    check_p = check_exp / np.sum(check_exp, axis=1, keepdims=True)
    return -np.sum(y * np.log(check_p))


def numerical_gradient(weight, index, step=1e-5):
    """Central difference: [L(w + h) - L(w - h)] / (2h)."""
    original = weight[index]
    weight[index] = original + step
    loss_plus = current_loss()
    weight[index] = original - step
    loss_minus = current_loss()
    weight[index] = original
    return (loss_plus - loss_minus) / (2.0 * step)


# Pick an active ReLU neuron so the W1 check is nontrivial.
hidden = int(np.argmax(z1[0]))
assert z1[0, hidden] > 0.0
check_w1 = numerical_gradient(W1, (0, hidden))
check_w2 = numerical_gradient(W2, (hidden, label))
np.testing.assert_allclose(check_w1, dW1[0, hidden], rtol=1e-4, atol=1e-7)
np.testing.assert_allclose(check_w2, dW2[hidden, label], rtol=1e-4, atol=1e-7)

# One SGD update, after all four gradients have been computed and checked.
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2

print(f"loss before update: {loss:.6f}")
print(f"predicted digit before update: {predicted_digit}")
print(f"W1 gradient: analytic={dW1[0, hidden]:.8f}, numeric={check_w1:.8f}")
print(f"W2 gradient: analytic={dW2[hidden, label]:.8f}, numeric={check_w2:.8f}")
print(f"loss after update:  {current_loss():.6f}")
