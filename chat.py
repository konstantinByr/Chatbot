import random
import json
import torch
from model import NeuralNet
from chatbot3 import bagOfWords, tokenize

import neptune



machine = torch.device('cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
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

antriebe = ['Benzin', 'Diesel', 'Hybrid', 'Elektrisch']
SUV = ['SUV', 'kein SUV']
klassen = ['Kompakt', 'Mittel', 'obere Mittel', 'Oberklasse', 'Luxusklasse']
karosserien = ['T-Modell', 'Limousine', 'Coupe', 'Cabrio']
türen = ['2Tuerer', '4Tuerer']

botName = "Maggus"
print(f"{botName}: Hallo, ich bin {botName}, dein persoenlicher MB-Auswahlassistent!")



def frage(antrieb, suv, klasse, karosserie, tür, name, score):

    ende = False
    autoScore = score

    if antrieb:          print(f"{name}: Welchen Antrieb willst du haben?")
    elif suv:            print(f"{name}: Möchtest du einen SUV?")
    elif klasse:         print(f"{name}: Welche Fahrzeugklasse wünschst du dir?")    
    elif karosserie:     print(f"{name}: Welche Karosserieform möchtest du?")
    elif tür:          print(f"{name}: Wie viele Türen soll dein Fahrzeug haben?")
    else:                      
        auswertung(autoScore, name)
        return

        
    satz = input("Du: ")

    if satz == "ENDE" :
        auswertung(autoScore, name)
        return
        


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
        if ((tag in antriebe) and antrieb):
            antrieb = False
            autoScore += antriebScores.get(tag)
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)
        elif ((tag in antriebe) and not antrieb):
            print(f"{botName}: Du hast deine Antriebsart bereits ausgewählt.")
            
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

        if ((tag in SUV) and suv):
            suv = False
            autoScore += SUVScores.get(tag)
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)
        elif ((tag in SUV) and not suv):
            print(f"{botName}: Ob SUV oder nicht hast du bereits gewählt.")
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

        if ((tag in klassen) and klasse):
            klasse = False
            autoScore += klasseScores.get(tag)
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)
        elif ((tag in klassen) and not klasse):
            print(f"{botName}: Du hast deine Klasse bereits ausgewählt.")
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)
        if ((tag in karosserien) and karosserie):
            karosserie = False
            autoScore += karosserieScores.get(tag)
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)
        elif ((tag in karosserien) and not karosserie):
            print(f"{botName}: Du hast deine Karosserieform bereits ausgewählt.")
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

        if ((tag in türen) and tür):
            tür = False
            autoScore += türenScores.get(tag)
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)
        elif ((tag in türen) and not tür):
            print(f"{botName}: Du hast deine Türanzahl bereits ausgewählt.")
            #frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

        #print(tag)
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")

        frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

    elif not ende:
        print(f"{botName}: Ich verstehe dich leider nicht :-(")
        print(f"{botName}: Kannst du deine Aussage vielleicht umformulieren?")
        frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

def auswertung(score, name):

    autos = {10111: 'A-Klasse Limousine', 10001: 'A-Klasse/B-Klasse', 11001: 'GLA/GLB', 20111: 'A-Klasse Limousine d', 20001: 'A-Klasse/B-Klasse d', 21001: 'GLA/GLB d',
            30011: 'A-Klasse Limousine hybrid', 30001: 'A-Klasse/B-Klasse hybrid', 31001: 'GLA hybrid', 41001: 'EQA/EQB', 10101: 'C-Klasse T-Modell', 10111: 'C-Klasse Limousine', 10120: 'CLE',
            11101: 'GLC', 11121: 'GLC Coupe', 20101: 'C-Klasse T-Modell d', 20111: 'C-Klasse Limousine d', 20120: 'CLE d', 21101: 'GLC d', 21121: 'GLC Coupe d', 30101: 'C-Klasse T-Modell hybrid', 
            30111: 'C-Klasse Limousine hybrid', 30120: 'CLE hybrid', 31101: 'GLC hybrid', 31121: 'GLC Coupe hybrid', 10201: 'E-Klasse T-Modell', 10211: 'E-Klasse Limousine', 10220: 'CLE', 11201: 'GLE',
            11221: 'GLE Coupe', 20201: 'E-Klasse T-Modell d', 20211: 'E-Klasse Limousine d', 20220: 'CLE d', 21201: 'GLE d', 21221: 'GLE Coupe d', 30201: 'E-Klasse T-Modell hybrid', 
            30211: 'E-Klasse Limousine hybrid', 30220: 'CLE hybrid', 31201: 'GLE hybrid', 31221: 'GLE Coupe hybrid',40211: 'EQE', 41201: 'EQE SUV', 10311: 'S-Klasse', 11311: 'GLS', 10320: 'AMG GT Coupe',
            10321: 'AMG GT 4-Türer Coupe', 10330: 'AMG SL', 20311: 'S-Klasse d', 21311: 'GLS d', 30311: 'S-Klasse hybrid', 31311: 'GLS hybrid', 40311: 'EQS', 41311: 'EQS SUV', 
            10411: 'Maybach S-Klasse', 11411: 'Maybach GLS', 30411: 'Maybach S-Klasse hybrid', 31411: 'Maybach GLS hybrid', 41401: 'Maybach EQS SUV'}

    
    if score in autos.keys():
        print(f"{name}: Dein zu dir passendes Auto: " + autos[score])
    else:
        print(f"{name}: Mercedes-Benz bietet derzeit leider kein für dich passendes Fahrzeug an")


frage(booleanAntrieb, booleanSUV, booleanKlasse, booleanKarosserie, booleanTüren, botName, autoScore)