import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer as ps
from HanTa import HanoverTagger as ht
stemmer = ps()
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

def tokenize(satz):
    return nltk.word_tokenize(satz)

def stem(word):
    #stemmer.stem(word)
    wort = tagger.tag_sent(word)
    return(wort[0][1])

def bagOfWords(tokenizedSentence, allWords):
    tokenizedSentence = [stem(w) for w in tokenizedSentence]
    bag = np.zeros(len(allWords), dtype=np.float32)
    for i, w in enumerate(allWords):
        if w in tokenizedSentence:
            bag[i]  = 1.0
    return bag
