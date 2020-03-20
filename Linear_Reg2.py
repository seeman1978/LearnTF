import numpy as np
np.random.seed(456)
import tensorflow as tf
tf.random.set_seed(456)
import pandas as pd

import datetime
import matplotlib.pyplot as plt

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_dim=1, dtype='float32')
    ])
    return model

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['mae'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

if __name__ == "__main__":
    # Generate synthetic data
    N = 100
    n_steps = 8000
    w_true = 5
    b_true = 2
    noise_scale = .1
    x_np = np.random.rand(N, 1)
    x = tf.convert_to_tensor(x_np, dtype='float32')
    noise = np.random.normal(scale=noise_scale, size=(N, 1))
    # Convert shape of y_np to (N,)
    y_np = np.reshape(w_true * x_np + b_true + noise, (-1))
    y = tf.convert_to_tensor(y_np, dtype='float32')

    model = create_model()
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae']) #accuracy用于分类

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x=x,
                y=y,
                epochs=n_steps,
                batch_size = 32,
                callbacks=[tensorboard_callback])

    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_history(history)