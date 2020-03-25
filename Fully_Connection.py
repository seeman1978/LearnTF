import tensorflow as tf
import deepchem as dc

if __name__ == "__main__":
    #载入并准备好数据集。
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    # Remove extra tasks
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]

    #将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(50, input_shape=(1024,), activation='relu'),  #now the model will take as input arrays of shape (*, 1024)，and output arrays of shape (*, 50)
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    #为训练选择优化器和损失函数
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',   #二分类，需要使用binary_crossentropy
                  metrics=['accuracy'])

    #训练并验证模型
    model.fit(train_X, train_y, batch_size=100, epochs=5)

    model.evaluate(valid_X, valid_y, verbose=2)

    model.evaluate(test_X, test_y, verbose=2)