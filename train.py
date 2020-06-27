from .models import ClassificationLoss, model_factory, save_model
from .utils import load_data
import torch.optim as optim
import argparse

"""
Path to reach the directory with training data
"""
TRAIN_PATH = "data/train"

"""
Adjustable parameters 
"""
num_epochs = 35
lr = 0.000005

"""
Takes in the training data and uses stochastic gradient descent as an optimizer,
makes a prediction and then calculates the overall loss from the predicted and actual class.
"""
def train(args):

    model = model_factory[args.model]()
    train_data = load_data(TRAIN_PATH)
    optimizer = optim.SGD(model.parameters(), lr)
    loss_function = ClassificationLoss()
    
    for epoch in range(num_epochs):
        print("epoch: " + str(epoch))
        for images, labels in train_data:
            predictions = model(images)
            loss = loss_function(predictions, labels)
            optimizer.zero_grad
            loss.backward()
            optimizer.step()
        
    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    args = parser.parse_args()
    train(args)
