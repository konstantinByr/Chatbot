import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chatbot3 import bagOfWords, tokenize, stem
from model import NeuralNet
from nltk.corpus import stopwords

import neptune



stopWords = stopwords.words('german')

with open('intents.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w, tag))


allWords = [stem(w) for w in allWords if w not in stopWords]

allWords = sorted(set(allWords))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(allWords), "unique stemmed words:", allWords)

XTrain = []
YTrain = []
for (pattern_sentence, tag) in xy:

    bag = bagOfWords(pattern_sentence, allWords)
    XTrain.append(bag)
    label = tags.index(tag)
    YTrain.append(label)

XTrain = np.array(XTrain)
YTrain = np.array(YTrain)

# Hyperparameter
num_epochs = 200
batch_size = 32
learning_rate = 0.0001
inputSize = len(XTrain[0])
hiddenSize = 128
outputSize = len(tags)
print(inputSize, outputSize)

#Initialisierung NEPTUNE
run = neptune.init_run(
    project="konstantin.bayer/chatbot",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZDY0ZTZkMS0zZTNiLTQ4NjgtYTVlMS1jZDExM2IyYTNkZjUifQ==",
)

loss = []
optimizer = []
learning_rate = []

params = {
    "max_epochs": num_epochs,
    "optimizer": "Adadelta",
    "batchsize": batch_size,
    "learningrate": learning_rate,
    "hidden Size": hiddenSize,
    "model_architecture": "Feedforward Neural Network",
    "validation_data": "None"  # Update this if you have validation data
}
run["parameters"] = params

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(XTrain)
        self.XData = XTrain
        self.YData = YTrain

    def __getitem__(self, index):
        return self.XData[index], self.YData[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cpu')

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters())

NFILE = 'data3.pth'

prüfLoss = -1   #Deaktivierung Abspeicherung während laufendem Betrieb

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        
        outputs = model(words)
        loss = criterion(outputs, labels)

        run["train/loss"].append(loss*10)
        run["train/epoch"].append(epoch)

        if loss <= prüfLoss:
            data = {
            "model_state": model.state_dict(),
            "inputSize": inputSize,
            "hiddenSize": hiddenSize,
            "outputSize": outputSize,
            "allWords": allWords,
            "tags": tags
        }
            torch.save(data, NFILE)
            print(f'Epoch [{epoch}/{num_epochs}], gespeichert, Loss: {loss}')
            prüfLoss = loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')



FILE = "data.pth"
data = {
            "model_state": model.state_dict(),
            "inputSize": inputSize,
            "hiddenSize": hiddenSize,
            "outputSize": outputSize,
            "allWords": allWords,
            "tags": tags
        }
torch.save(data, FILE)

print(f'Training abgeschlossen, Datei gespeichert unter {FILE}')


run.stop()