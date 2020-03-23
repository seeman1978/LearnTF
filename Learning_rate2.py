from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
model.fit(train_x, train_y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[reduce_lr])