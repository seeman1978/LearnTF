import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    #因为图像是灰度图像，所以需要增加一维
    input_shape = (x_train.shape[1:] + (1,))  # (28, 28, 1)
    num_classes = len(np.unique(y_train))

    #convert our labels to one-hot encoded form if we use categorical_crossentropy loss.
    # if we use sparse_categorical_crossentropy then we can skip this step.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 构建模型
    inp = Input(shape=input_shape)
    _ = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inp)
    _ = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(_)
    _ = MaxPool2D(pool_size=(2, 2))(_)
    _ = Dropout(0.25)(_)
    _ = Flatten()(_)
    _ = Dense(units=128, activation='relu')(_)
    _ = Dropout(0.2)(_)
    _ = Dense(units=num_classes, activation='softmax')(_)
    model = Model(inputs=inp, outputs=_)
    model.summary()

    # 训练
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(np.expand_dims(x_train, -1), y_train, batch_size=128, epochs=12, validation_split=0.3)

    loss, accuracy = model.evaluate(np.expand_dims(x_test, -1), y_test, verbose=0)
    print(loss, accuracy)

    # plot the accuracy and loss history
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.legend()