from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
from sklearn.model_selection import train_test_split

def main(model_name):
    try:
        np.random.seed(7)
        X = np.linspace(0, 1, 10000).reshape(-1, 1)
        Y = np.linspace(0.01, 100.01, 10000).reshape(-1, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        model = Sequential()
        model.add(Dense(50, input_dim=1, activation='relu', kernel_initializer='zeros'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        callback_filename = model_name + '.csv'
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train)).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit(train_dataset, epochs=50, validation_data=(X_test, Y_test), verbose=1, callbacks=[enhancedLoggingCallback])
        scores = model.evaluate(X_test, Y_test)
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        return scores
    except Exception as e:
        print(e)
        return 0
if __name__ == '__main__':
    main('46642627.h5')
