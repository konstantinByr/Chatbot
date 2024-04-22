import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, hiddenSize)
        self.layer3 = nn.Linear(hiddenSize, hiddenSize)
        #self.layer4 = nn.Linear(hiddenSize, hiddenSize)
        #self.layer5 = nn.Linear(hiddenSize, hiddenSize)
        #self.layer6 = nn.Linear(hiddenSize, hiddenSize)
        #self.layer7 = nn.Linear(hiddenSize, hiddenSize)
        #self.layer8 = nn.Linear(hiddenSize, hiddenSize)
        #self.layer9 = nn.Linear(hiddenSize, hiddenSize)
        #self.layerX = nn.Linear(hiddenSize, numClasses)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        #out = self.layer5(out)
        #out = self.layer6(out)
        #out = self.layer7(out)
        #out = self.layer8(out)
        #out = self.layer9(out)
        #out = self.relu(out)
        #out = self.layerX(out)
        #out = self.layer5(out)
        #out = self.relu(out)
        return out