import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, hiddenSize)
        self.layer3 = nn.Linear(hiddenSize, hiddenSize)
        self.relu = nn.ReLU()
        self.activations = [[] for _ in range(3)]  # Liste zum Speichern der Aktivierungen für jede Schicht

    def forward(self, x):
        out = self.layer1(x)
        self.activations[0].append(out.clone().detach())  # Speichern der Aktivierung der ersten Schicht
        out = self.layer2(out)
        self.activations[1].append(out.clone().detach())  # Speichern der Aktivierung der zweiten Schicht
        out = self.relu(out)
        out = self.layer3(out)
        self.activations[2].append(out.clone().detach())  # Speichern der Aktivierung der dritten Schicht
        return out
