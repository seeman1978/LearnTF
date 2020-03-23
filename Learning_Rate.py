import keras.backend as K
from keras.callbacks import LearningRateScheduler


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)
model.fit(train_x, train_y, batch_size=32, epochs=5, callbacks=[reduce_lr])