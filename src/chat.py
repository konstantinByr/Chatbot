import random
import json
from pathlib import Path
import torch
from model import NeuralNet
from chatbot3 import bagOfWords, tokenize

machine = torch.device('cpu')   #NeuralNet läuft über CPU

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / 'data').resolve()

with open(DATA_DIR / 'intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

#data.pth laden
FILE = DATA_DIR / 'data.pth'
data = torch.load(FILE)

#Parameter NeuralesNetz
wahrscheinlichkeit = 0.75    #75%
inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data['allWords']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(machine)
model.load_state_dict(model_state)
model.eval()

#Booleans zur Feststellung wann alle Parameter abgefragt wurden
booleanAntrieb = True
booleanSUV = True
booleanKlasse = True
booleanKarosserie = True
booleanTüren = True


autoScore = 0   #Punktzahl die zutreffendes Auto findet
#10T => Antrieb, 1T => SUV/keinSUV, 100 => Klasse, 10 => Karoserie, 1 => Türen
antriebScores = {'Benzin': 10000, 'Diesel': 20000, 'Hybrid': 30000, 'Elektrisch': 40000}
SUVScores= {'SUV': 1000, 'kein SUV': 0000}
klasseScores = {'Kompakt': 000, 'Mittel': 100, 'obere Mittel': 200, 'Oberklasse': 300, 'Luxusklasse': 400}
karosserieScores = {'T-Modell': 00, 'Limousine': 10, 'Coupe': 20, 'Cabrio': 30}
türenScores = {'2Tuerer': 0, '4Tuerer': 1}

antriebe = ['Benzin', 'Diesel', 'Hybrid', 'Elektrisch']
SUV = ['SUV', 'kein SUV']
klassen = ['Kompakt', 'Mittel', 'obere Mittel', 'Oberklasse', 'Luxusklasse']
karosserien = ['T-Modell', 'Limousine', 'Coupe', 'Cabrio']
türen = ['2Tuerer', '4Tuerer']

botName = "Maggus"
print(f"{botName}: Hallo, ich bin {botName}, dein persoenlicher MB-Auswahlassistent!")
print(f"{botName}: Schreibe 'ENDE', um das Programm zu beenden.")



def frage(antrieb, suv, klasse, karosserie, tür, name, score):

    ende = False
    autoScore = score

    if antrieb:          print(f"{name}: Welchen Antrieb willst du haben?")
    elif suv:            print(f"{name}: Möchtest du einen SUV?")
    elif klasse:         print(f"{name}: Welche Fahrzeugklasse wünschst du dir? (z.B. Mittelklasse/Luxusklasse)")    
    elif karosserie:     print(f"{name}: Welche Karosserieform möchtest du?")
    elif tür:            print(f"{name}: Wie viele Türen soll dein Fahrzeug haben?")
    else:                      
        auswertung(autoScore, name)
        return


    satz = input("Du: ")

    if satz == "ENDE" :
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
        print(f"{botName}: Ich verstehe dich leider nicht :-(. Kannst du deine Aussage bitte umformulieren?")
        frage(antrieb, suv, klasse, karosserie, tür, name, autoScore)

def auswertung(score, name):

    autos = {
        #A-Klasse/B-Klasse:
        10011: 'A-Klasse Limousine', 10001: 'A-Klasse/B-Klasse', 
        20011: 'A-Klasse Limousine d', 20001: 'A-Klasse/B-Klasse d', 
        30011: 'A-Klasse Limousine hybrid', 30001: 'A-Klasse/B-Klasse hybrid',
        #GLA/GLB:
        11001: 'GLA/GLB',  21001: 'GLA/GLB d',
        31001: 'GLA hybrid', 41001: 'EQA/EQB', 
        #C-Klasse
        10101: 'C-Klasse T-Modell', 20101: 'C-Klasse T-Modell d', 30101: 'C-Klasse T-Modell hybrid',
        10111: 'C-Klasse Limousine', 20111: 'C-Klasse Limousine d', 30111: 'C-Klasse Limousine hybrid',
        #CLE
        10120: 'CLE', 20120: 'CLE d', 30120: 'CLE hybrid', 10220: 'CLE', 20220: 'CLE d', 30220: 'CLE hybrid',
        10130: 'CLE Carbiolet', 20130: 'CLE d Cabriolet', 30130: 'CLE hybrid Carbiolet', 10230: 'CLE Cabriolet', 20230: 'CLE d Cabriolet', 30230: 'CLE hybrid Cabriolet',
        #GLC
        11101: 'GLC', 21101: 'GLC d', 31101: 'GLC hybrid',
        11121: 'GLC Coupe', 21121: 'GLC Coupe d', 31121: 'GLC Coupe hybrid',
        #E-Klasse 
        10201: 'E-Klasse T-Modell',  20201: 'E-Klasse T-Modell d', 30201: 'E-Klasse T-Modell hybrid',
        10211: 'E-Klasse Limousine',  20211: 'E-Klasse Limousine d', 30211: 'E-Klasse Limousine hybrid', 40211: 'EQE',
        #GLE
        11201: 'GLE', 21201: 'GLE d', 31201: 'GLE hybrid', 41201: 'EQE SUV',
        11221: 'GLE Coupe', 21221: 'GLE Coupe d', 31221: 'GLE Coupe hybrid',
        #S-Klasse    
        10311: 'S-Klasse', 20311: 'S-Klasse d', 30311: 'S-Klasse hybrid', 40301: 'EQS', 10411: 'Maybach S-Klasse', 11401: 'Maybach GLS', 30411: 'Maybach S-Klasse hybrid',
        #GLS
        11301: 'GLS', 21301: 'GLS d', 31301: 'GLS hybrid', 41301: 'EQS SUV',
        #GT/SL
        10320: 'AMG GT Coupe', 10321: 'AMG GT 4-Türer Coupe', 10330: 'AMG SL', 
        }

    
    if score in autos.keys():
        print(f"{name}: Dein zu dir passendes Auto: " + autos[score])
    else:
        print(f"{name}: Mercedes-Benz bietet derzeit leider kein für dich passendes Fahrzeug an")


frage(booleanAntrieb, booleanSUV, booleanKlasse, booleanKarosserie, booleanTüren, botName, autoScore)   #erster Aufruf des Chats