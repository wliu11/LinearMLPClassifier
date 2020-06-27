from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch.optim as optim

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"


def train(args):

    model = model_factory[args.model]()

    train_data = load_data(TRAIN_PATH)
    
    optimizer = optim.SGD(model.parameters(), lr=0.000005, )
    loss_function = ClassificationLoss()
    
    for epoch in range(35):
        print("epoch: " + str(epoch))
        for images, labels in train_data:
            predictions = model(images)
            loss = loss_function(predictions, labels)
            optimizer.zero_grad
            loss.backward()
            optimizer.step()
        
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')

    args = parser.parse_args()

    train(args)
