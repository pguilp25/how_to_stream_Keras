from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import os
import tensorflow as tf
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        X_train = np.array([[1] * 128] * 10 ** 4 + [[0] * 128] * 10 ** 4)
        X_test = np.array([[1] * 128] * 10 ** 2 + [[0] * 128] * 10 ** 2)
        Y_train = np.array([True] * 10 ** 4 + [False] * 10 ** 4)
        Y_test = np.array([True] * 10 ** 2 + [False] * 10 ** 2)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        Y_train = Y_train.astype('bool')
        Y_test = Y_test.astype('bool')
        model = Sequential()
        model.add(Dense(32, input_dim=128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        adam = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        batch_size = 32
        callback_filename = model_name + '.csv'
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=50, verbose=1, validation_data=(X_test, Y_test), callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(X_test, Y_test)
        return score
    except Exception as e:
        print(e)
        return 0
if __name__ == '__main__':
    main('31880720.h5')
