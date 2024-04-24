import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, hiddenSize)
        self.layer3 = nn.Linear(hiddenSize, hiddenSize)
        self.relu = nn.ReLU()
        # Initialisiere die activations-Liste für jede Schicht
        self.activations = [[] for _ in range(3)]  # Hier 3 für 3 Schichten

    def forward(self, x):
        out = self.layer1(x)
        self.activations[0].append(out)  # Speichere die Aktivierung der ersten Schicht
        out = self.layer2(out)
        self.activations[1].append(out)  # Speichere die Aktivierung der zweiten Schicht
        out = self.relu(out)
        out = self.layer3(out)
        self.activations[2].append(out)  # Speichere die Aktivierung der dritten Schicht
        return out
