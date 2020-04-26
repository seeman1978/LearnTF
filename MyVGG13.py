import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras

class Conv_Conv_MaxPooling(tf.keras.Model):
    def __init__(self, unit_size):
        super(Conv_Conv_MaxPooling, self).__init__()
        self.conv2d_1 = layers.Conv2D(unit_size, kernel_size=[3, 3], padding='same',
                                      activation='relu')
        self.conv2d_2 = layers.Conv2D(unit_size, kernel_size=[3, 3], padding='same',
                                      activation='relu')
        self.pool = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.conv2d_2(x)
        return self.pool(x)

class FC_Net(tf.keras.Model):
    def __init__(self):
        super(FC_Net, self).__init__()
        self.layers_1 = layers.Dense(256, activation='relu')
        self.layers_2 = layers.Dense(128, activation='relu')
        self.layers_3 = layers.Dense(10, activation=None)

    def call(self, inputs):
        x = self.layers_1(inputs)
        x = self.layers_2(x)
        return self.layers_3(x)

#VGG model
class VGG_Model(tf.keras.Model):
    def __init__(self):
        super(VGG_Model, self).__init__()
        self.conv_net1 = Conv_Conv_MaxPooling(64)
        self.conv_net2 = Conv_Conv_MaxPooling(128)
        self.conv_net3 = Conv_Conv_MaxPooling(256)
        self.conv_net4 = Conv_Conv_MaxPooling(512)
        self.conv_net5 = Conv_Conv_MaxPooling(512)
        self.flatten = tf.keras.layers.Flatten()
        self.fc_net = FC_Net()
    def call(self, inputs):
        x = self.conv_net1(inputs)
        x = self.conv_net2(x)
        x = self.conv_net3(x)
        x = self.conv_net4(x)
        x = self.conv_net5(x)
        x = self.flatten(x)
        return self.fc_net(x)

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = myvgg(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, myvgg.trainable_variables)
  optimizer.apply_gradients(zip(gradients, myvgg.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = myvgg(images, training=False)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)

if __name__ == "__main__":
    # 获取所有GPU 设备列表
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU 显存占用为按需分配
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 异常处理
            print(e)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = 2 * x_train.astype('float32') / 255. - 1
    x_test = 2 * x_test.astype('float32') / 255. - 1
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # 构建训练集对象
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(1000).batch(32)  # preprocess把数据预处理到[-1，1]区间
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(32)

    # create model
    myvgg = VGG_Model()
    #myvgg.build(input_shape=(None, 32, 32, 3))
    #myvgg.summary()

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    EPOCHS = 50
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step, (images, labels) in enumerate(train_db):
            train_step(images, labels)
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(train_loss.result()))

        for test_images, test_labels in test_db:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
