# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to model weights")
args = vars(ap.parse_args())

# initialize the MNIST dataset and normalize it into the range [0, 1]
print("[INFO] initializing MNIST data...")
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0

# convert the labels from integers into vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# load the model into disk and compile it
print("[INFO] compiling model...")
model = load_model(args["model"])
model.compile(optimizer = "adam", loss = "categorical_crossentropy")

# evaluate the model
print("[INFO] evaluating the model")
predictions = model.predict(x_test)

# get incorrect indices
wrong_indices = np.where(predictions.argmax(axis = 1) != y_test.argmax(axis = 1))[0]

# loop over the indices and save the figures to disk
for (i, index) in enumerate(wrong_indices):
    plt.figure()
    plt.imshow(x_test[index])
    plt.title("Predicted: {}".format(str(predictions[index].argmax(axis = 0))))
    plt.savefig("GRU/incorrect_predictions/{}.png".format(i + 1))
