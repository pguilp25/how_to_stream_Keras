import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        X = np.random.randn(1000, 12)
        Y = np.sum(X, axis=1) > 0
        Y = to_categorical(Y, num_classes=2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = Sequential()
        model.add(Dense(32, input_dim=12, activation='relu'))
        for i in range(4):
            model.add(Dense(2 ** (5 + i), activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.603), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        callback_filename = model_name + '.csv'
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train)).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit(x=X_train, y=Y_train, batch_size=16, epochs=50, verbose=1, validation_data=(X_test, Y_test), callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        score = model.evaluate(X_test, Y_test)
        print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')
        return score
    except Exception as e:
        print(e)
        return 0
if __name__ == '__main__':
    main('64151679.h5')
