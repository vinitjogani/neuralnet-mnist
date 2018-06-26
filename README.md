# Neural Network classifier for MNIST dataset
Due to GitHub upload restrictions, the dataset are not on the repository and must be download seperately from [here](https://pjreddie.com/projects/mnist-in-csv/). This project achieves an accuracy of approximately 91%. 

The first idea used here is dimensionality reduction using Principal Component Analysis which is used to compress the input images from 784 pixels to 500 pixels preserving 95% variance. This involves finding the covariance matrix of the feature vectors, finding its eigenvectors and then selecting the first 500 of them. This speeds up the neural network training process significantly when run on 60,000 training examples.

The data is then fed to a standard neural network with one hidden layer and 450 hidden units, which is trained using the backpropogation algorithm using mini-batch gradient descent. The batch size used here was 60, and the entire dataset was traversed (number of epochs = ) 3 times.
