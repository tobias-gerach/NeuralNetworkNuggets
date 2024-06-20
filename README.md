# NeuralNetworkNuggets
My personal repository dedicated to exploring and learning the fundamentals of neural networks and deep learning. This repository serves as a collection of my projects, tutorials, and experiments. It's a space for me to document my progress, test new ideas, and deepen my understanding of these complex concepts.

## Notes
[1] MLP.py solves the MNIST classification problem using a multi layer perceptron network with sigmoid neurons and stochastic gradient descent.

[2] MLP_Torch.py is a PyTorch implementation of a Network([784, 30, 10]) in MLP.py. Both implementations reach an accuracy of approx. 95%.

[3] MLP_2.py is a continuation of the first network [1] with some improvements to deal with learning slowdown and overfitting.
To avoid learning slowdown, I implemented the CrossEntropy cost function.
To reduce overfitting, L2 regularization was implemented in the stochastic gradient descend update.
Finally, a small change in the initialization of the weights can speed up learning during the first epochs.
Overall, these changes improved the accuracy on the test set to about 98%.
    
