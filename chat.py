import random
import json
import torch
from model import NeuralNet
from chatbot3 import bagOfWords, tokenize

machine = torch.device('cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data2.pth"
data = torch.load(FILE)

wahrscheinlichkeit = 0.8
inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data['allWords']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(machine)
model.load_state_dict(model_state)
model.eval()

booleanAntrieb = True
booleanSUV = True
booleanKlasse = True
booleanKarosserie = True
booleanTüren = True
autoScore = 0   #Punktzahl die zutreffendes Auto findet
#10T => Antrieb, 1T => SUV/keinSUV, 100 => Klasse, 10 => Karoserie, 1 => Türen
antriebScores = {'Benzin': 10000, 'Diesel': 20000, 'Hybrid': 30000, 'Elektrisch': 40000}
SUVScores= {'SUV': 1000, 'kein SUV': 0000}
klasseScores = {'Kompakt': 000, 'Mittel': 100, 'obere Mittel': 200, 'Oberklasse': 300, 'Luxusklasse': 400, 'Van': 500}
karosserieScores = {'T-Modell': 00, 'Limousine': 10, 'Coupe': 20, 'Cabrio': 30}
türenScores = {'2Tuerer': 0, '4Tuerer': 1}

botName = "Maggus"
print(f"{botName}: Hallo, ich bin {botName}, dein persoenlicher MB-Auswahlassistent!")


while booleanAntrieb:
    print(f"{botName}: Welchen Antrieb willst du haben?")
    
        
    satz = input("Du: ")
    if satz == "ENDE":
        booleanSUV = False
        booleanAntrieb = False
        booleanKarosserie = False
        booleanKlasse = False
        booleanTüren = False
        break


    satz = tokenize(satz)
    X = bagOfWords(satz, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim= 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > wahrscheinlichkeit:
        #Autoscore nach Tags berechnen
        if ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == True):
            booleanAntrieb = False
            autoScore += antriebScores.get(tag)
        elif ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == False):
            print(f"{botName}: Du hast deine Antriebsart bereits ausgewählt.")
            continue

        if ((tag == "SUV" or tag == "kein SUV") and booleanSUV == True):
            booleanSUV = False
            autoScore += SUVScores.get(tag)
        elif ((tag == "SUV" or tag == "kein SUV") and booleanSUV == False):
            print(f"{botName}: Ob SUV oder nicht hast du bereits gewählt.")
            continue

        if ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == True):
            booleanKlasse = False
            autoScore += klasseScores.get(tag)
        elif ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == False):
            print(f"{botName}: Du hast deine Klasse bereits ausgewählt.")
            continue

        if ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == True):
            booleanKarosserie = False
            autoScore += karosserieScores.get(tag)
        elif ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == False):
            print(f"{botName}: Du hast deine Karosserieform bereits ausgewählt.")
            continue

        if ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == True):
            booleanTüren = False
            autoScore += türenScores.get(tag)
        elif ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == False):
            print(f"{botName}: Du hast deine Türanzahl bereits ausgewählt.")
            continue

        print(tag)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: Ich verstehe dich leider nicht :-(")
        print(f"{botName}: Kannst du deine Aussage vielleicht umformulieren?")

while booleanSUV:
    print(f"{botName}: Möchtest du einen SUV?")
    

    satz = input("Du: ")
    if satz == "ENDE":
        booleanSUV = False
        booleanAntrieb = False
        booleanKarosserie = False
        booleanKlasse = False
        booleanTüren = False
        break


    satz = tokenize(satz)
    X = bagOfWords(satz, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim= 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > wahrscheinlichkeit:
        #Autoscore nach Tags berechnen
        if ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == True):
            booleanAntrieb = False
            autoScore += antriebScores.get(tag)
        elif ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == False):
            print(f"{botName}: Du hast deine Antriebsart bereits ausgewählt.")
            continue

        if ((tag == "SUV" or tag == "kein SUV") and booleanSUV == True):
            booleanSUV = False
            autoScore += SUVScores.get(tag)
        elif ((tag == "SUV" or tag == "kein SUV") and booleanSUV == False):
            print(f"{botName}: Ob SUV oder nicht hast du bereits gewählt.")
            continue

        if ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == True):
            booleanKlasse = False
            autoScore += klasseScores.get(tag)
        elif ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == False):
            print(f"{botName}: Du hast deine Klasse bereits ausgewählt.")
            continue

        if ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == True):
            booleanKarosserie = False
            autoScore += karosserieScores.get(tag)
        elif ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == False):
            print(f"{botName}: Du hast deine Karosserieform bereits ausgewählt.")
            continue

        if ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == True):
            booleanTüren = False
            autoScore += türenScores.get(tag)
        elif ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == False):
            print(f"{botName}: Du hast deine Türanzahl bereits ausgewählt.")
            continue

        print(tag)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: Ich verstehe dich leider nicht :-(")
        print(f"{botName}: Kannst du deine Aussage vielleicht umformulieren?")

while booleanKlasse:
    print(f"{botName}: Welche Fahrzeugklasse wünschst du dir?")
    
        
    satz = input("Du: ")
    if satz == "ENDE":
        booleanSUV = False
        booleanAntrieb = False
        booleanKarosserie = False
        booleanKlasse = False
        booleanTüren = False
        break


    satz = tokenize(satz)
    X = bagOfWords(satz, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim= 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > wahrscheinlichkeit:
        #Autoscore nach Tags berechnen
        if ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == True):
            booleanAntrieb = False
            autoScore += antriebScores.get(tag)
        elif ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == False):
            print(f"{botName}: Du hast deine Antriebsart bereits ausgewählt.")
            continue

        if ((tag == "SUV" or tag == "kein SUV") and booleanSUV == True):
            booleanSUV = False
            autoScore += SUVScores.get(tag)
        elif ((tag == "SUV" or tag == "kein SUV") and booleanSUV == False):
            print(f"{botName}: Ob SUV oder nicht hast du bereits gewählt.")
            continue

        if ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == True):
            booleanKlasse = False
            autoScore += klasseScores.get(tag)
        elif ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == False):
            print(f"{botName}: Du hast deine Klasse bereits ausgewählt.")
            continue

        if ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == True):
            booleanKarosserie = False
            autoScore += karosserieScores.get(tag)
        elif ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == False):
            print(f"{botName}: Du hast deine Karosserieform bereits ausgewählt.")
            continue

        if ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == True):
            booleanTüren = False
            autoScore += türenScores.get(tag)
        elif ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == False):
            print(f"{botName}: Du hast deine Türanzahl bereits ausgewählt.")
            continue

        print(tag)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: Ich verstehe dich leider nicht :-(")
        print(f"{botName}: Kannst du deine Aussage vielleicht umformulieren?")

        
while booleanKarosserie:
    print(f"{botName}: Welche Karosserieform möchtest du?")
  
    satz = input("Du: ")
    if satz == "ENDE":
        booleanSUV = False
        booleanAntrieb = False
        booleanKarosserie = False
        booleanKlasse = False
        booleanTüren = False
        break


    satz = tokenize(satz)
    X = bagOfWords(satz, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim= 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > wahrscheinlichkeit:
        #Autoscore nach Tags berechnen
        if ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == True):
            booleanAntrieb = False
            autoScore += antriebScores.get(tag)
        elif ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == False):
            print(f"{botName}: Du hast deine Antriebsart bereits ausgewählt.")
            continue

        if ((tag == "SUV" or tag == "kein SUV") and booleanSUV == True):
            booleanSUV = False
            autoScore += SUVScores.get(tag)
        elif ((tag == "SUV" or tag == "kein SUV") and booleanSUV == False):
            print(f"{botName}: Ob SUV oder nicht hast du bereits gewählt.")
            continue

        if ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == True):
            booleanKlasse = False
            autoScore += klasseScores.get(tag)
        elif ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == False):
            print(f"{botName}: Du hast deine Klasse bereits ausgewählt.")
            continue

        if ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == True):
            booleanKarosserie = False
            autoScore += karosserieScores.get(tag)
        elif ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == False):
            print(f"{botName}: Du hast deine Karosserieform bereits ausgewählt.")
            continue

        if ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == True):
            booleanTüren = False
            autoScore += türenScores.get(tag)
        elif ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == False):
            print(f"{botName}: Du hast deine Türanzahl bereits ausgewählt.")
            continue

        print(tag)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: Ich verstehe dich leider nicht :-(")
        print(f"{botName}: Kannst du deine Aussage vielleicht umformulieren?")


while booleanTüren:
    print(f"{botName}: Wie viele Türen soll dein Fahrzeug haben?")
    
        
    satz = input("Du: ")
    if satz == "ENDE":
        booleanSUV = False
        booleanAntrieb = False
        booleanKarosserie = False
        booleanKlasse = False
        booleanTüren = False
        break


    satz = tokenize(satz)
    X = bagOfWords(satz, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim= 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > wahrscheinlichkeit:
        #Autoscore nach Tags berechnen
        if ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == True):
            booleanAntrieb = False
            autoScore += antriebScores.get(tag)
        elif ((tag == "Benzin" or tag == "Elektrisch" or tag == "Hybrid" or tag == "Diesel") and booleanAntrieb == False):
            print(f"{botName}: Du hast deine Antriebsart bereits ausgewählt.")
            continue

        if ((tag == "SUV" or tag == "kein SUV") and booleanSUV == True):
            booleanSUV = False
            autoScore += SUVScores.get(tag)
        elif ((tag == "SUV" or tag == "kein SUV") and booleanSUV == False):
            print(f"{botName}: Ob SUV oder nicht hast du bereits gewählt.")
            continue

        if ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == True):
            booleanKlasse = False
            autoScore += klasseScores.get(tag)
        elif ((tag == "Kompakt" or tag == "Mittel" or tag == "obere Mittel" or tag == "Oberklasse" or tag == "Luxusklasse" or tag == "Van") and booleanKlasse == False):
            print(f"{botName}: Du hast deine Klasse bereits ausgewählt.")
            continue

        if ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == True):
            booleanKarosserie = False
            autoScore += karosserieScores.get(tag)
        elif ((tag == "T-Modell" or tag == "Limousine" or tag == "Coupe" or tag == "Cabrio") and booleanKarosserie == False):
            print(f"{botName}: Du hast deine Karosserieform bereits ausgewählt.")
            continue

        if ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == True):
            booleanTüren = False
            autoScore += türenScores.get(tag)
        elif ((tag == "2Tuerer" or tag == "4Tuerer") and booleanTüren == False):
            print(f"{botName}: Du hast deine Türanzahl bereits ausgewählt.")
            continue

        print(tag)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: Ich verstehe dich leider nicht :-(")
        print(f"{botName}: Kannst du deine Aussage vielleicht umformulieren?")


autos = {10111: 'A-Klasse Limousine', 10001: 'A-Klasse/B-Klasse', 11001: 'GLA/GLB', 20111: 'A-Klasse Limousine', 20001: 'A-Klasse/B-Klasse', 21001: 'GLA/GLB',
         30111: 'A-Klasse Limousine', 30001: 'A-Klasse/B-Klasse', 31001: 'GLA', 41001: 'EQA/EQB', 10101: 'C-Klasse T-Modell', 10111: 'C-Klasse Limousine', 10120: 'CLE',
         11101: 'GLC', 11121: 'GLC Coupe', 20101: 'C-Klasse T-Modell', 20111: 'C-Klasse Limousine', 20120: 'CLE', 21101: 'GLC', 21121: 'GLC Coupe', 30101: 'C-Klasse T-Modell', 
         30111: 'C-Klasse Limousine', 30120: 'CLE', 31101: 'GLC', 31121: 'GLC Coupe', 10201: 'E-Klasse T-Modell', 10211: 'E-Klasse Limousine', 10220: 'CLE', 11201: 'GLE',
         11221: 'GLE Coupe', 20201: 'E-Klasse T-Modell', 20211: 'E-Klasse Limousine', 20220: 'CLE', 21201: 'GLE', 21221: 'GLE Coupe', 30201: 'E-Klasse T-Modell', 
         30211: 'E-Klasse Limousine', 30220: 'CLE', 31201: 'GLE', 31221: 'GLE Coupe',40211: 'EQE', 41201: 'EQE SUV', 10311: 'S-Klasse', 11311: 'GLS', 10320: 'AMG GT Coupe',
         10321: 'AMG GT 4-Türer Coupe', 10330: 'AMG SL', 20311: 'S-Klasse', 21311: 'GLS', 30311: 'S-Klasse', 31311: 'GLS', 10311: 'EQS', 11311: 'EQS SUV', 
         10411: 'Maybach S-Klasse', 11411: 'Maybach GLS', 30411: 'Maybach S-Klasse', 31411: 'Maybach GLS', 41411: 'Maybach EQS SUV'}

if (booleanAntrieb == False and booleanKarosserie == False and booleanKlasse == False and booleanSUV == False and booleanTüren == False and autoScore in autos.keys()):
    print(f"{botName}: Dein zu dir passendes Auto: " + autos[autoScore])
else:
    print(f"{botName}: Mercedes-Benz bietet derzeit leider kein für dich passendes Fahrzeug an")