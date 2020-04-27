import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

class DenseNet(keras.Model):
    # 通用的ResNet实现类
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        # 加载DenseNet网络模型，并去掉最后一层全连接层，最后一个池化层设置为max pooling
        self.stem = keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='max')
        # 设计为不参与优化，即MobileNet这部分参数固定不动
        self.stem.trainable = False

        self.dense1 = layers.Dense(1024, activation='relu')  # 追加全连接层
        self.bn1 = layers.BatchNormalization()  # 追加BN层
        self.dp1 = layers.Dropout(rate=0.5)  # 追加Dropout层，防止过拟合
        self.dense2 = layers.Dense(num_classes)

    def call(self, inputs):
        _ = self.stem(inputs)
        _ = self.dense1(_)
        _ = self.bn1(_)
        _ = self.dp1(_)
        return self.dense2(_)


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
    model = DenseNet()

    # 创建Early Stopping类，连续3次不下降则终止
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.001,
        patience=3
    )

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                   loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    history = model.fit(train_db, validation_data=test_db, validation_freq=1, epochs=100,
                         callbacks=[early_stopping])
    history = history.history

    print(history.keys())
    print(history['val_accuracy'])
    print(history['accuracy'])
    test_acc = model.evaluate(test_db)

    plt.figure()
    returns = history['val_accuracy']
    plt.plot(np.arange(len(returns)), returns, label='验证准确率')
    plt.plot(np.arange(len(returns)), returns, 's')
    returns = history['accuracy']
    plt.plot(np.arange(len(returns)), returns, label='训练准确率')
    plt.plot(np.arange(len(returns)), returns, 's')

    plt.plot([len(returns) - 1], [test_acc[-1]], 'D', label='测试准确率')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.savefig('transfer.svg')