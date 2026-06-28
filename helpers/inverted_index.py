import os
import pickle

from helpers import tokenize_text, load_movies

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, movie):
        self.docmap[doc_id] = movie
        textTokens = tokenize_text.tokenize_text(movie["title"] + " " + movie["description"])

        for token in textTokens:
            if token not in self.index:
                self.index[token] = [doc_id]
            else:
                self.index[token].append(doc_id)

    def get_documents(self, term):
        idxList = []
        if term in self.index:
            idxList = self.index[term]

        idxList.sort()

        resList = []
        for idx in idxList:
            resList.append(self.docmap[idx])

        return resList

    def build(self):
        movies = load_movies.load_movies("data/movies.json")
        for movie in movies["movies"]:
            self.__add_document(movie["id"], movie)

    def save(self):
        if not os.path.isdir("cache"):
            os.mkdir("cache")

        pickle.dump(self.index, open("cache/index.pkl", "wb"))
        pickle.dump(self.docmap, open("cache/docmap.pkl", "wb"))

    def load(self):
        if not os.path.isdir("cache") or not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl"):
            raise FileNotFoundError("Files not found")

        self.index = pickle.load(open("cache/index.pkl", "rb"))
        self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
