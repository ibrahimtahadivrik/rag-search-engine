import argparse

from lib import semantic_search




def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text into embeddings")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    subparsers.add_parser("verify_embeddings", help="verify the embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query into embeddings")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()