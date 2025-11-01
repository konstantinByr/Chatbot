import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, hiddenSize)
        self.layer3 = nn.Linear(hiddenSize, hiddenSize)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out
