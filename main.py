import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,\
    Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import Callback
from keras.models import load_model

# -------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scaling the values
x_train = x_train / 255
x_test = x_test / 255

# Creating one-hot representation of target
y_train_encoded = []
y_test_encoded = []

for label in y_train:
    encoded = np.zeros(10)
    encoded[label] = 1
    y_train_encoded.append(encoded)
for label in y_test:
    encoded = np.zeros(10)
    encoded[label] = 1
    y_test_encoded.append(encoded)

y_train_encoded = np.array(y_train_encoded)
y_test_encoded = np.array(y_test_encoded)

# -------------------------------------------------------------

"""
Conv2D however expects 4 dimensions,because it also expects the channels dimension of image,
which in MNIST is nonexistent because it’s grayscale data and hence is 1.

Reshaping the data, while explicitly adding the channels dimension, resolves the issue.
The input shape a CNN accepts should be in a specific format.
In Tensorflow,the format is (num_samples, height, width, channels)
"""
# x_train.shape = (60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# -------------------------------------------------------------

# Defining the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# loss="categorical_crossentropy" gives an unidentified error.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Custom Keras callback to stop training when certain accuracy is achieved.
class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


# model.fit(x_train, y_train, epochs=5,
          #callbacks=[MyThresholdCallback(1)],validation_data=(x_test, y_test))

# Saving the model
# model.save("CNN_on_MNIST")


# -------------------------------------------------------------

# loading saved model
model = load_model(r"C:\Users\dipesh\Desktop\Trained_Model\CNN_on_MNIST")
predicted = model.predict(x_test)

print(predicted)

# Checking Individual sample
print("ACTUAL VALUE : ",y_test[2])
print("PREDICTED VALUE : ",np.argmax(predicted[2]))










