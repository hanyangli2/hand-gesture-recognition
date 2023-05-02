import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
from random import randint
import matplotlib.image as mpimg # Plotting

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('C:\\Users\\Harry\\OneDrive\\Desktop\\projects\\handgestureGUI\\HandGestureRecognition\\handgesturesdataset\\leapGestRecog\\00'):
    if not j.startswith('.'): # If running this code locally, this is to
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
print(lookup)

x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('C:\\Users\\Harry\\OneDrive\\Desktop\\projects\\handgestureGUI\\HandGestureRecognition\\handgesturesdataset\\leapGestRecog\\0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('C:\\Users\\Harry\\OneDrive\\Desktop\\projects\\handgestureGUI\\HandGestureRecognition\\handgesturesdataset\\leapGestRecog\\0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('C:\\Users\\Harry\\OneDrive\\Desktop\\projects\\handgestureGUI\\HandGestureRecognition\\handgesturesdataset\\leapGestRecog\\0' +
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
                print("done")
            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255

from sklearn.model_selection import train_test_split
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
model.save('C:\Users\Harry\OneDrive\Desktop\projects\handgestureGUI\trained_models')

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from IPython.display import Image, display
from tensorflow import keras

#load trained model
model = keras.models.load_model('C:\\Users\\Harry\\OneDrive\\Desktop\\projects\\handgestureGUI\\trained_models')

#Input image url
img_url = 'C:\\Users\\Harry\\OneDrive\\Desktop\\projects\\handgestureGUI\\HandGestureRecognition\\handgesturesdataset\\leapGestRecog\\03\\09_c\\frame_03_09_0190.png'

#Display input
img = Image(img_url)
print("Input: ")
display(img)

#Preprocess input
img = load_img(img_url)
img = img.resize((320, 120))
img = img_to_array(img)[:,:,:1]
img = np.expand_dims(img ,axis=0)

#Have model predict
list = model.predict(img)
list = list[0]
i = 0
while i < len(list):
    if list[i] == 1:
        break
    i = i + 1
key = [k for k, v in lookup.items() if v == i]
print("AI hand gesture guess: " + str(key[0]))