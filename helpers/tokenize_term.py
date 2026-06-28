from string import punctuation
from nltk.stem import PorterStemmer


def tokenize_term(text) -> str:

    punctuationTable = str.maketrans('', '', punctuation)

    fp = open("data/stopwords.txt", "r")
    stopwords = fp.read().translate(punctuationTable).splitlines()

    resList = [token for token in text.lower().translate(punctuationTable).split()
               if token not in stopwords]

    stemmer = PorterStemmer()

    resList = [stemmer.stem(token) for token in resList]

    if len(resList) != 1:
        raise Exception("Too many tokens in text.")

    return resList[0]