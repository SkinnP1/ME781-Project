
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix


traintSet = "../archive/chest_xray/train"
testSet = "../archive/chest_xray/test"
valSet = "../archive/chest_xray/val"
labels = ['PNEUMONIA', 'NORMAL']
img_size = 128


def getData(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img))
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img/255.0
                img = np.reshape(img, (img_size, img_size,1))
                data.append([img, class_num])
            except Exception as e:
                print(e)
    return (data)

def preProcessData():
	x_train = []
	y_train = []
	x_val = []
	y_val = []
	x_test = []
	y_test = []
	train = getData(traintSet)
	test = getData(testSet)
	val = getData(valSet)

	print(1)
	for feature, label in train:
		x_train.append(feature)
		y_train.append(label)

	print(2)
	for feature, label in test:
		x_test.append(feature)
		y_test.append(label)
		
	print(3)
	for feature, label in val:
		x_val.append(feature)
		y_val.append(label)

	print(4)
	x_train = np.array(x_train)
	x_val = np.array(x_val) 
	x_test = np.array(x_test)


	y_train = np.array(y_train)

	y_val = np.array(y_val)


	y_test = np.array(y_test)

	print(1000)
	return(x_train,y_train,x_test,y_test,x_val,y_val)


# define cnn model
def define_model():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(img_size,img_size,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	optimizer = Adam(lr = 0.0001)
	model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=optimizer)
	model.summary()
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	epochs = [i for i in range(12)]
	fig , ax = plt.subplots(1,2)
	train_acc = history.history['accuracy']
	train_loss = history.history['loss']
	val_acc = history.history['val_accuracy']
	val_loss = history.history['val_loss']
	fig.set_size_inches(20,10)

	ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
	ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
	ax[0].set_title('Training & Validation Accuracy')
	ax[0].legend()
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("Accuracy")

	ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
	ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
	ax[1].set_title('Testing Accuracy & Loss')
	ax[1].legend()
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("Training & Validation Loss")
	plt.savefig('plot.png')



# run the test harness for evaluating a model
def run_test_harness():

	x_train,y_train,x_test,y_test,x_val,y_val = preProcessData()
	print(4)
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  

	datagen.fit(x_train)
	

	learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
	# prepare iterator
	# train_it = datagenTrain.flow_from_directory(traintSet,
	# 	class_mode='binary', batch_size=64, target_size=(224, 224,1))
	# test_it = datagenTest.flow_from_directory(testSet,
	# 	class_mode='binary', batch_size=64, target_size=(224, 224,1))
	# fit model
	early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)
	history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

	# history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
	# 	validation_data=test_it, validation_steps=len(test_it), epochs=1, verbose=1)
	# evaluate model
	print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
	print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")	

	summarize_diagnostics(history)

	model.save('model.h5')
	

# entry point, run the test harness
run_test_harness()
