""" importing the libraries """
import keras.utils.np_utils
import tensorflow as tf
from keras.datasets import mnist
import keras

""" loading the dataset """
(X_train, y_train), (X_test, y_test) = mnist.load_data()

""" changing the dimensions of the images so that to fit into the model """
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_test /= 255
X_train /= 255

""" categorical values of the target variables into binary matrix to break to correlation """
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=None)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes=None)

""""Training and initializing the CNN-MODEL"""
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""Adding the second convolutional layer"""
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(strides=2, pool_size=2))

"""Adding a regularized layer"""
cnn.add(tf.keras.layers.Dropout(0.25))

"""Flattening the pooled layer"""
cnn.add(tf.keras.layers.Flatten())

"""FULLY CONNECTED LAYERS"""
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))

"""Adding a regularized layer"""
cnn.add(tf.keras.layers.Dropout(0.25))

"""Output layer"""
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))

"""Compiling the CNN"""
cnn.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

"""Fitting the data into the model"""
epoch = 10
batch = 32
cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=1)
print("Model trained successfully....")

"""Evaluating the model"""
score = cnn.evaluate(X_test, y_test, verbose=0)
print("Loss: {:.2f}%".format(score[0]))
print(f"Accuracy of model: {int(score[1]*100)}%")

"""" Saving the trained CNN model in HDF5 file format """
cnn.save("Hand_written_digit_CNN.h5")



