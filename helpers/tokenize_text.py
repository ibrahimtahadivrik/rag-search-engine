from string import punctuation
from nltk.stem import PorterStemmer

def tokenize_text(text) -> list:

    punctuationTable = str.maketrans('', '', punctuation)

    fp = open("data/stopwords.txt", "r")
    stopwords = fp.read().translate(punctuationTable).splitlines()

    resList = [token for token in text.lower().translate(punctuationTable).split()
               if token not in stopwords]

    stemmer = PorterStemmer()

    resList = [stemmer.stem(token) for token in resList]

    return resList