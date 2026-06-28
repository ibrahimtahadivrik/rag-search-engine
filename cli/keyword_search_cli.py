import argparse
import sys
import os
import pickle
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import inverted_index, tokenize_text, tokenize_term
from constants import values

def  has_matching_token(argsTable, movieTable):
    for arg in argsTable:
        for word in movieTable:
            if arg in word:
                return True
    return False

def tf(doc_id, term):
    token = tokenize_term.tokenize_term(term)
    idx = inverted_index.InvertedIndex()
    idx.load()

    return  idx.term_frequencies[doc_id][token]

def idf(term):
    idx = inverted_index.InvertedIndex()
    idx.load()

    token = tokenize_term.tokenize_term(term)

    total_doc_count = len(idx.docmap)
    term_match_doc_count = 0
    if token in idx.index:
        term_match_doc_count = len(idx.index[token])


    idf_score = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf_score

def tfidf(doc_id, term):
    token = tokenize_term.tokenize_term(term)
    idf_score = idf(term)
    tf_score = tf(doc_id, token)
    return tf_score*idf_score

def bm25_idf_command(term:str)->float:
    idx = inverted_index.InvertedIndex()
    idx.load()
    token = tokenize_term.tokenize_term(term)

    return idx.get_bm25_idf(token)

def bm25_tf_command(doc_id:int, term:str, k1:values.BM25_K1, b:values.BM25_B)->float:
    token = tokenize_term.tokenize_term(term)
    idx = inverted_index.InvertedIndex()
    idx.load()
    bm25Tf = idx.get_bm25_tf(doc_id, token, k1, b)
    return bm25Tf

def bm25search(query, limit:5):
    idx = inverted_index.InvertedIndex()
    idx.load()
    bm25List = idx.bm25_search(query, limit)

    for i, doc in enumerate(bm25List):
        print(f"{i+1}. ({doc[0]}) {idx.docmap[doc[0]]["title"]} - Score: {doc[1]:.2f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the search index")

    ft_parser = subparsers.add_parser("tf", help="Search for frequency for the token")
    ft_parser.add_argument("doc_id", type=int, help="Document ID to search for")
    ft_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser("idf", help="Search for idf score for the token")
    idf_parser.add_argument("term", type=str, help="Term to search for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Search for tfidf score for the token")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to search for")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=values.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=values.BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Optional limit for the search")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    idx = inverted_index.InvertedIndex()

    match args.command:
        case "build":
            idx.build()
            idx.save()
            pass

        case "search":
             print(f"Searching for: {args.query}")
             argsTable = tokenize_text.tokenize_text(args.query)
             idx.load()

             results = []
             for arg in argsTable:
                 results +=  idx.get_documents(arg)
                 if len(results) >= 5:
                    break

             for i, movie in enumerate(results):
                 if i >= 5:
                     break
                 print(f"{i + 1}. {movie["title"]} {movie["id"]}")
             pass

        case "tf":
            print(tf(args.doc_id, args.term))
            pass

        case "idf":
            print(f"Inverse document frequency of '{args.term}': {idf(args.term):.2f}")
            pass

        case "tfidf":
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf(args.doc_id, args.term):.2f}")
            pass

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            pass

        case "bm25tf":
            bm25Tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25Tf:.2f}")
            pass

        case "bm25search":
            bm25search(args.query, args.limit)
            pass

        case _:
            parser.print_help()





if __name__ == "__main__":
    main()