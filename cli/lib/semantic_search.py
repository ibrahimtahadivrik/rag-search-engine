from sentence_transformers import  SentenceTransformer
import numpy as np
import os
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Empty text")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        docs_list = []
        for document in self.documents:
            docs_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(docs_list, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)





def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text:str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    data = json.load(open("data/movies.json"))
    documents = data["movies"]
    semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} "
          f"vectors in {semantic_search.embeddings.shape[1]} dimensions")

def embed_query_text(query:str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
