import argparse
import json

from lib import semantic_search


def search(query, limit:5):
    semanticModel = semantic_search.SemanticSearch()
    data = json.load(open("data/movies.json"))
    movies = data["movies"]
    semanticModel.load_or_create_embeddings(movies)
    s_list = semanticModel.search(query, limit)
    for i, s in enumerate(s_list):
        print(f"{i+1}. {s[1]['title']} (score: {s[0]})")
        print(f"  {s[1]["description"]}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text into embeddings")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    subparsers.add_parser("verify_embeddings", help="verify the embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query into embeddings")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Semanric search")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit the number of results")

    args = parser.parse_args()


    match args.command:
        case "verify":
            semantic_search.verify_model()
            pass

        case "embed_text":
            semantic_search.embed_text(args.text)
            pass

        case "verify_embeddings":
            semantic_search.verify_embeddings()
            pass

        case "embed_query":
            semantic_search.embed_query_text(args.query)
            pass

        case "search":
            search(args.query, args.limit)
            pass

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()