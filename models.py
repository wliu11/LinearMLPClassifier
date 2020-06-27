import torch
import torch.nn.functional as F
import torch.nn as nn


"""
Compute mean(-log(softmax(input)_label))

@input:  torch.Tensor((B,C))
@target: torch.Tensor((B,), dtype=torch.int64)

@return:  torch.Tensor((,))
"""
class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):

        return F.nll_loss(F.log_softmax(input, dim=1), target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*64*64, 6)   
        
    """
    @x: torch.Tensor((B,3,64,64))
    @return: torch.Tensor((B,6))
    """
    def forward(self, x):
        return self.linear(x.view(-1, 64*64*3))

class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3*64*64, 80),
            nn.ReLU(),
            nn.Linear(80, 6),
            )
    

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.layers(x.view(-1, 64*64*3))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
