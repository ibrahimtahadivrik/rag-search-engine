import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import inverted_index, tokenize_text, load_movies

def  has_matching_token(argsTable, movieTable):
    for arg in argsTable:
        for word in movieTable:
            if arg in word:
                return True
    return False




def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the search index")
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
        case _:
            parser.print_help()





if __name__ == "__main__":
    main()