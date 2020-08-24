# LinearMLPClassifier

In this project, we implemented a linear classification and multi-layer perceptron neural network in PyTorch that classifies images taken from SuperTuxKart.

In utils.py, we implemented a dataloader that would convert PIL images into tensors, and save them along with their corresponding labels. 

In models.py, we created the neural networks itself. We take as input a tensor of size (B, 3, 64, 64) and output a tensor of size (B, 6), where B denotes the size of the batch. We implement the log-likelihood of a softmax classifier as our classification loss function.
