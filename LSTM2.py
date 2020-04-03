# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import tensorflow_addons as tfa
import inspect
import time
import numpy as np

def _read_words(filename):
  with tf.io.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz
               has been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns
  Tensors that are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The
    second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are
    too high.
  """
  with tf.compat.v1.name_scope("PTBProducer"):
    raw_data = tf.convert_to_tensor(value=raw_data, name="raw_data",
                                    dtype=tf.int32)

    data_len = tf.size(input=raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.compat.v1.debugging.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.compat.v1.train.range_input_producer(epoch_size,
                                      shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self.input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would
    # need to be different than reported in the paper.
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is
      # unfortunately not defined in TensorFlow 1.0. To maintain
      # backwards compatibility, we add an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.compat.v1.nn.rnn_cell.BasicLSTMCell.__init__).args:
        return tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.compat.v1.get_variable_scope().reuse)
      else:
        return tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)],
                                    state_is_tuple=True)

    self.initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.compat.v1.get_variable(
          "embedding", [vocab_size, size], dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(params=embedding, ids=input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, 1 - (config.keep_prob))

    outputs = []
    state = self.initial_state
    with tf.compat.v1.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.compat.v1.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.compat.v1.get_variable(
        "softmax_w", [size, vocab_size], dtype=tf.float32)
    softmax_b = tf.compat.v1.get_variable(
        "softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    # use the contrib sequence loss and average over the batches
    loss = tfa.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([batch_size, num_steps], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True
    )

    # update the cost variables
    self.cost = cost = tf.reduce_sum(input_tensor=loss)
    self.final_state = state

    if not is_training:
      return

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.compat.v1.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(ys=cost, xs=tvars),
                                      config.max_grad_norm)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step = tf.compat.v1.train.get_or_create_global_step())

    self.new_lr = tf.compat.v1.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.compat.v1.assign(self.lr, self.new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size,
             np.exp(costs / iters),
             (iters
              * model.input.batch_size/(time.time() - start_time))))

  return np.exp(costs / iters)

if __name__=="__main__":

    raw_data = ptb_raw_data("/Users/wangqiang/Downloads/simple-examples/data")
    train_data, valid_data, test_data, _ = raw_data

    config = SmallConfig()
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.compat.v1.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.compat.v1.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data,
                                   name="TrainInput")
            with tf.compat.v1.variable_scope("Model", reuse=None,
                                   initializer=initializer):
                m = PTBModel(is_training=True, config=config,
                             input_=train_input)
            tf.compat.v1.summary.scalar("Training Loss", m.cost)
            tf.compat.v1.summary.scalar("Learning Rate", m.lr)

        with tf.compat.v1.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data,
                                   name="ValidInput")
            with tf.compat.v1.variable_scope("Model", reuse=True,
                                   initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config,
                                  input_=valid_input)
            tf.compat.v1.summary.scalar("Validation Loss", mvalid.cost)

        with tf.compat.v1.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data,
                                  name="TestInput")
            with tf.compat.v1.variable_scope("Model", reuse=True,
                                   initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        sv = tf.compat.v1.train.Supervisor()
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f"
                      % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f"
                      % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f"
                      % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)