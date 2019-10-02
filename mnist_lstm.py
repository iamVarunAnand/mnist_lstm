# import the necessary packages
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

def build_model(height, width, classes):
    # set the input shape to match the channel ordering
    input_shape = (height, width)

    model = Sequential()

    model.add(LSTM(128, input_shape = input_shape, activation = "relu", return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation = "relu"))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation = "softmax"))

    print(model.summary())

    return model

# initialize the mnist dataset and normalize the input into the range [0, 1]
print("[INFO] initializing MNIST data...")
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0

# convert the labels from integers into vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize and compile the model
print("[INFO] compiling the model...")
opt = Adam(lr = 0.001)
model = build_model(28, 28, 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# train the network
print("[INFO] training the network")
H = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 30, batch_size = 32)

# evaluate the network
print("[INFO] evaluating the network")
predictions = model.predict(x_test, batch_size = 32)
print(classification_report(y_test.argmax(axis = 1), predictions.argmax(axis = 1)))
