import operator
import os
import pickle
import math
from collections import Counter

from helpers import tokenize_text, load_movies, tokenize_term
from constants import values

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, movie):
        self.docmap[doc_id] = movie
        textTokens = tokenize_text.tokenize_text(movie["title"] + " " + movie["description"])
        self.term_frequencies[doc_id] = Counter(textTokens)
        self.doc_lengths[doc_id] = len(textTokens)

        for token in set(textTokens):
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
        pickle.dump(self.term_frequencies, open("cache/term_frequencies.pkl", "wb"))
        pickle.dump(self.doc_lengths, open("cache/doc_lengths.pkl", "wb"))

    def load(self):
        if not os.path.isdir("cache") or not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl")\
                or not os.path.isfile("cache/term_frequencies.pkl" or not os.path.isfile("cache/doc_lengths.pkl")):
            raise FileNotFoundError("Files not found")

        self.index = pickle.load(open("cache/index.pkl", "rb"))
        self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
        self.term_frequencies = pickle.load(open("cache/term_frequencies.pkl", "rb"))
        self.doc_lengths = pickle.load(open("cache/doc_lengths.pkl", "rb"))

    def get_tf(self, doc_id, term):
        if doc_id not in self.term_frequencies:
            return 0

        if term not in self.term_frequencies[doc_id]:
            return 0

        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term:str) -> float:
        if term not in self.index:
            return 0
        df = len(self.index[term])
        N = len(self.docmap)
        bm25Idf = math.log( (N - df + 0.5) / (df + 0.5) + 1)
        return bm25Idf

    def get_bm25_tf(self, doc_id:int, term:str, k1:values.BM25_K1, b:values.BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        bm25Tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25Tf

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self, doc_id:int, term:str) -> float:
        return self.get_bm25_tf(doc_id, term, values.BM25_K1, values.BM25_B) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit):
        qTokens = tokenize_text.tokenize_text(query)
        scores = {}
        doc_ids = []

        for term in qTokens:
            doc_ids += self.index[term]

        for docId in set(doc_ids):
            score = 0
            for token in qTokens:
                score += self.bm25(docId, token)
            scores[docId] = score
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores[:limit]