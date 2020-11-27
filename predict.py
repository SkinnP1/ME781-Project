from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys
import numpy as np
import cv2

img_size = 128
labels = ['PNEUMONIA', 'NORMAL']
# load and prepare the image
def load_image(filename):
	# load the image
	img = cv2.imread(filename)
	img = cv2.resize(img, (img_size, img_size))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img/255.0
	img = np.reshape(img, (1,img_size, img_size,1))
	# convert to array
	return img


 
# entry point, run the example
filename = str(sys.argv[1])
# load the image
img = load_image(filename)
# load model
model = load_model('model.h5')
# predict the class
result = 1 if model.predict(img) > 0.5 else 0
print(labels[result])
