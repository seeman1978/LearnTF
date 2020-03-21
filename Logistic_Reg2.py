import numpy as np
np.random.seed(456)
import tensorflow as tf
tf.random.set_seed(456)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import scipy

# Generate synthetic data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))
# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])
x_np = x_np.astype(np.float32)
y_np = y_np.astype(np.float32)

# Save image of the data distribution
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Toy Logistic Regression Data")

# Plot Zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")
plt.savefig("logistic_data.png")

W = tf.Variable(tf.random.normal((2, 1)))
b = tf.Variable(tf.random.normal((1,)))

# Logistic regression (Wx + b).
def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    y_logit = tf.squeeze(tf.matmul(x, W) + b)
    # the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # Rounding P(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)
    return (y_logit, y_pred)

# Cross-Entropy loss function.
def cross_entropy(y_logit, y_true):
    # Compute the cross-entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_true)
    # Sum all contributions
    l = tf.reduce_sum(entropy)
    return l

# Accuracy metric.
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(.01)

# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        logit, pred = logistic_regression(x)
        loss = cross_entropy(logit, y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))

n_steps = 1000
display_step = 50
# Run training for the given number of steps.
for i in range(n_steps):
    # Run the optimization to update W and b values.
    run_optimization(x_np, y_np)

    if i % display_step == 0:
        logit, pred = logistic_regression(x_np)
        loss = cross_entropy(pred, y_np)
        acc = accuracy_score(pred, y_np)
        print("step: %i, loss: %f, accuracy: %f" % (i, loss, acc))


# Get weights
w_final, b_final = [W.numpy(), b.numpy()]
# Make Predictions
score = accuracy_score(y_np, pred)
print("Classification Accuracy: %f" % score)

plt.clf()
# Save image of the data distributions
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Learned Model (Classification Accuracy: 1.00)")
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Plot Zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")

x_left = -2

y_left = (1./w_final[1]) * (-b_final + scipy.special.logit(.5) - w_final[0]*x_left)

x_right = 2
y_right = (1./w_final[1]) * (-b_final + scipy.special.logit(.5) - w_final[0]*x_right)
plt.plot([x_left, x_right], [y_left, y_right], color='k')

plt.savefig("logistic_pred.png")