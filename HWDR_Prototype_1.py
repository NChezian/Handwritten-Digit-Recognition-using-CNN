# Handwritten Digit Classification using Convolutional Neural Network derived from the LeNet-5 Model

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

##--------------CNN Model Construction--------------##

# Load and preprocess the data
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

## Defining the Model Architecture.

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(10, activation='softmax'))

## Model Compiliation.

#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

## Training the model.

#model.fit(x_train, y_train,
#          batch_size=128,
#          epochs=3,
#          verbose=1,
#          validation_data=(x_test, y_test))

## Evaluating the model.

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

## Saving the model for further validation.

#model.save('handwrittenpt1.model')

##--------------Model created successfully--------------##

## Code Block for validating the model by feeding handwritten images of digits(0-9)

model = keras.models.load_model('handwrittenpt1.model')

## In this final section the model is already created and ready to called, now further steps are performed to 
## preprocess the input images, feed it and run it through the model and view the output.

for number in range(10):
    try:
        # 1. Reading the image file for the current number.
        img = cv2.imread(f"C:/Users/nchez/OneDrive/Desktop/ML-2/digits/digit{number}.png", cv2.IMREAD_GRAYSCALE)
        # 2. Resizing the image to the desired input size (28x28 pixels) of the model.
        img = cv2.resize(img, (28, 28))
        # 3. Inverting the image colors to match the training data.
        img = np.invert(img)
        # 4. Reshaping the image to match the input shape of the model.
        img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
        # 5. Making the predictions using the loaded model.
        prediction = model.predict(img)
        # 6. Determining the predicted class label.
        predicted_class = np.argmax(prediction)
        # 7. Printing the image filename and the predicted class label.
        print(f"Image: digit{number}.png")
        print(f"This Number is predicted to be: {predicted_class}")
        # 8. Displaying the image.
        plt.imshow(np.squeeze(img), cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        # 9. Handles any errors that may occur during the process.
        print(f"Error: {str(e)}")