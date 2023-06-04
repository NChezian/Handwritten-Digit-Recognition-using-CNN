# Handwritten-Digit-Recognition-using-CNN
This code demonstrates the use of a Convolutional Neural Network (CNN) derived from the LeNet-5 model for handwritten digit classification. 
It involves constructing, training, and evaluating the model using the MNIST dataset. Additionally, it provides functionality to validate the model
by feeding handwritten images of digits (0-9) and making predictions.

The code begins by importing the necessary libraries, including Keras for building and training the CNN model. It also imports other dependencies like NumPy, 
OpenCV (cv2), and Matplotlib for image processing and visualization.

The CNN model is constructed using the Sequential API from Keras. The architecture includes convolutional layers with ReLU activation, max-pooling layers, 
and fully connected (dense) layers. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. It is then trained on the MNIST 
dataset for a specified number of epochs.

After training, the model is evaluated on the test dataset to measure its performance. The test loss and accuracy are printed to evaluate the model's effectiveness.

The trained model is saved for further validation. The code block for validating the model is then executed. It loads the saved model and performs the following
steps for each handwritten image of a digit (0-9):

1. Reads the image file and converts it to grayscale.
2. Resizes the image to the desired input size of the model (28x28 pixels).
3. Inverts the image colors to match the training data.
4. Reshapes the image to match the input shape of the model.
5. Makes predictions using the loaded model.
6. Determines the predicted class label by finding the index with the highest probability.
7. Prints the image filename and the predicted class label.
8. Displays the image for visual inspection.
9. Handles any errors that may occur during the process.

The code provides a comprehensive pipeline for training and validating a CNN model for handwritten digit classification. 
It can serve as a useful starting point for understanding and implementing similar tasks, as well as showcasing the capabilities of CNNs for image recognition 
and classification.
