import argparse
import json
from string import punctuation
from nltk.stem import PorterStemmer


def  has_matching_token(argsTable, movieTable):
    for arg in argsTable:
        for word in movieTable:
            if arg in word:
                return True
    return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
             print(f"Searching for: {args.query}")
             pass
        case _:
            parser.print_help()

    data = json.load(open("data/movies.json"))

    fp = open("data/stopwords.txt", "r")

    results = []

    punctuationTable = str.maketrans('', '', punctuation)

    stopwords = fp.read().translate(punctuationTable).splitlines()

    argsTable = [token for token in args.query.lower().translate(punctuationTable).split()
                 if token not in stopwords]

    stemmer = PorterStemmer()

    argsTable = [stemmer.stem(token) for token in argsTable]

    for movie in data["movies"]:
        movieTable = [token for token in movie["title"].lower().translate(punctuationTable).split()
                 if token not in stopwords]

        movieTable = [stemmer.stem(token) for token in movieTable]

        if has_matching_token(argsTable, movieTable):
            results.append(movie)

    for i, movie in enumerate(results):
        if i >= 5:
            break
        print(f"{i+1}. {movie['title']} {i+1}")

if __name__ == "__main__":
    main()