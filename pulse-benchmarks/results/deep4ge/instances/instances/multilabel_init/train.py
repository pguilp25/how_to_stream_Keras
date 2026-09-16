import numpy as np
import keras
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        n = 1000
        m = 4
        num_classes = 1
        dummyX, dummyY = make_multilabel_classification(n_samples=n, n_features=m, n_classes=num_classes)
        labelEncoder = LabelEncoder()
        dummyY = labelEncoder.fit_transform(dummyY)
        x_train, x_test, y_train, y_test = train_test_split(dummyX, dummyY, test_size=0.2)
        input_shape = (m,)
        start_time = time()
        layers = [10, 20, 30, 40, 50]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(layers[0], input_dim=m, activation='relu', kernel_initializer='zeros'))
        for layer in layers[1:]:
            model.add(keras.layers.Dense(layer, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        callback_filename = model_name + '.csv'
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=16, verbose=1, callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(x_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0
if __name__ == '__main__':
    result = main('50481178.h5')
