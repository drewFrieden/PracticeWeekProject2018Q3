from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
from keras.optimizers import SGD

model = load_model('.\model\model.h5')

#picturePath = '.\model\\doc.jpg'
picturePath = '.\model\\facepalm.png'
test_image = cv2.imread(picturePath)
output = test_image.copy()
test_image = cv2.resize(test_image, (32, 32)).flatten()
test_image = test_image.reshape((1, test_image.shape[0]))
print(np.array(test_image).shape)
result = model.predict(np.array(test_image), batch_size=1, verbose=1)   
print(result)