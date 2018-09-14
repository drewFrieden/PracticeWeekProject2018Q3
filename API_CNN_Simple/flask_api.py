# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from pyimagesearch.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from imutils import paths
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
	global model

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
		help="path to input dataset of images")
	ap.add_argument("-m", "--model", required=True,
		help="path to output trained model")
	ap.add_argument("-p", "--plot", required=True,
		help="path to output accuracy/loss plot")
	args = vars(ap.parse_args())

	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []
	
	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images(args["dataset"])))
	random.seed(42)
	random.shuffle(imagePaths)
	
	# loop over the input images
	for imagePath in imagePaths:
		# load the image, resize the image to be 32x32 pixels (ignoring
		# aspect ratio), flatten the image into 32x32x3=3072 pixel image
		# into a list, and store the image in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (32, 32)).flatten()
		data.append(image)
	
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.25, random_state=42)
	
	# convert the labels from integers to vectors (for 2-class, binary
	# classification you should use Keras' to_categorical function
	# instead as the scikit-learn's LabelBinarizer will not return a
	# vector)
	lb = LabelBinarizer()
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)

	# define the 3072-1024-512-3 architecture using Keras
	model = Sequential()
	model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
	model.add(Dense(512, activation="sigmoid"))
	model.add(Dense(2, activation="softmax"))

	# initialize our initial learning rate, # of epochs to train for,
	# and batch size
	INIT_LR = 0.01
	EPOCHS = 75
	BS = 32
	
	# initialize the model and optimizer (you'll want to use
	# binary_crossentropy for 2-class classification)
	print("[INFO] training network...")
	opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["form","pic"]))
	print(accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1)))

	# plot the training loss and accuracy
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.plot(N, H.history["acc"], label="train_acc")
	plt.plot(N, H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy (SmallVGGNet)")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(args["plot"])

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")

            newImageArray = []
            newImage = cv2.resize(np.array(image), (32, 32)).flatten()

            newImageArray.append(newImage)
            newImageArray = np.array(newImageArray, dtype="float") / 255.0

            prediction = model.predict(newImageArray, batch_size=1)					

            data["document"] = '{:.1%}'.format(prediction[0][0])
            data["photo"] = '{:.1%}'.format(prediction[0][1])

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

load_model()
app.run()