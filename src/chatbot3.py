import nltk
import numpy as np
import re
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer as ps
#from HanTa import HanoverTagger as ht
stemmer = ps()
#tagger = ht.HanoverTagger('morphmodel_ger.pgz')

#CISTEM Leonie Weissweiler
stripge = re.compile(r"^ge(.{4,})")
replxx = re.compile(r"(.)\1")
replxxback = re.compile(r"(.)\*");
stripemr = re.compile(r"e[mr]$")
stripnd = re.compile(r"nd$")
stript = re.compile(r"t$")
stripesn = re.compile(r"[esn]$")


def tokenize(satz):
    return nltk.word_tokenize(satz)

#CISTEM Leonie Weissweiler
def stem(word, case_insensitive = False):
    if len(word) == 0:
        return word

    upper = word[0].isupper()
    word = word.lower()

    word = word.replace("ü","u")
    word = word.replace("ö","o")
    word = word.replace("ä","a")
    word = word.replace("ß","ss")

    word = stripge.sub(r"\1", word)
    word = word.replace("sch","$")
    word = word.replace("ei","%")
    word = word.replace("ie","&")
    word = replxx.sub(r"\1*", word)

    while len(word) > 3:
        if len(word) > 5:
            (word, success) = stripemr.subn("", word)
            if success != 0:
                continue

            (word, success) = stripnd.subn("", word)
            if success != 0:
                continue

        if not upper or case_insensitive:
            (word, success) = stript.subn("", word)
            if success != 0:
                continue

        (word, success) = stripesn.subn("", word)
        if success != 0:
            continue
        else:
            break

    word = replxxback.sub(r"\1\1", word)
    word = word.replace("%","ei")
    word = word.replace("&","ie")
    word = word.replace("$","sch")

    return word


def bagOfWords(tokenizedSentence, allWords):
    tokenizedSentence = [stem(w) for w in tokenizedSentence]
    bag = np.zeros(len(allWords), dtype=np.float32)
    for i, w in enumerate(allWords):
        if w in tokenizedSentence:
            bag[i]  = 1.0
    return bag