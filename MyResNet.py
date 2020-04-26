import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras

class BasicBlock(layers.Layer):
    #残差模块
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        #第一个卷积单元
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        #第二个卷积单元
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1 :    # 通过1x1卷积完成shape匹配
            self.downsample = layers.Conv2D(filter_num, (1, 1), strides=stride)
        else:   # shape匹配，直接短接
            self.downsample = lambda x : x

    def call(self, inputs, training=None):
        # [b, h, w, c]，通过第一个卷积单元
        _ = self.conv1(inputs)
        _ = self.bn1(_)
        _ = self.relu(_)

        # [b, h, w, c]，通过第二个卷积单元
        _ = self.conv2(_)
        _ = self.bn2(_)

        # 通过identity模块
        identity = self.downsample(inputs)

        # 2条路径输出直接相加
        _ = layers.add([_, identity])
        return tf.nn.relu(_)

class ResBlock(layers.Layer):
    def __init__(self, filter_num, blocks, stride=1):
        self.res_blocks = keras.Sequential()
        # 只有第一个BasicBlock的步长可能不为1，实现下采样
        self.res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):#其他BasicBlock步长都为1
            self.res_blocks.add(BasicBlock(filter_num, stride=1))

    def call(self, inputs, training=None):
        return self.res_blocks(inputs)

class ResNet(keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 根网络，预处理
        self.stem = keras.Sequential(
            [
                layers.Conv2D(64, (3, 3), strides=(1, 1)),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
            ]
        )
        # 堆叠4个Block，每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = ResBlock(64, layer_dims[0])
        self.layer2 = ResBlock(128, layer_dims[1], stride=2)
        self.layer3 = ResBlock(256, layer_dims[2], stride=2)
        self.layer4 = ResBlock(512, layer_dims[3], stride=2)

        # 通过Pooling层将高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        # 通过根网络
        _ = self.stem(inputs)
        # 一次通过4个模块
        _ = self.layer1(_)
        _ = self.layer2(_)
        _ = self.layer3(_)
        _ = self.layer4(_)
        # 通过池化层
        _ = self.avgpool(_)
        return self.fc(_)

def resnet18():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2])

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
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
    train_db = train_db.shuffle(1000).batch(512)  # preprocess把数据预处理到[-1，1]区间
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(512)

    #构建模型
    model = resnet18()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    EPOCHS = 100
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