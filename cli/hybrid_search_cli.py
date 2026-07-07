import argparse
from unittest import result

from lib.hybrid_search import normalize_scores, weighted_search_command, rrf_search_command
from llm import rrf_query_correction, rrf_query_rewrite, rrf_query_expand, rrf_individual_rerank


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF-based hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=60, help="K value for fine tune the search (default=60)")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return (default=5)")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enchantment method")
    rrf_search_parser.add_argument("--rerank-method",type=str, choices=["individual"], help="Query rerank method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":

            match args.enhance:
                case "spell":
                    final_query = rrf_query_correction(args.query)
                case "rewrite":
                    final_query = rrf_query_rewrite(args.query)
                case "expand":
                    final_query = rrf_query_expand(args.query)
                case _:
                    final_query = args.query

            match args.rerank_method:
                case "individual":
                    result = rrf_search_command(final_query, args.k, args.limit*5)
                    for i, doc in enumerate(result["results"]):
                        try:
                            result["results"][i]["rerank_score"] = float(rrf_individual_rerank(final_query, doc))
                        except ValueError:
                            result["results"][i]["rerank_score"] = 0.0
                    result["results"] = sorted(result["results"], key=lambda x: x["rerank_score"], reverse=True)

                    for i, res in enumerate(result["results"], 1):
                        print(f"{i}. {res['title']}")
                        print(f"Re-rank Score: {res.get('rerank_score', 0):.3f}/10")
                        print(f"   RRF Score: {res.get('score', 0):.3f}")
                        metadata = res.get("metadata", {})
                        if "bm25_rank" in metadata and "semantic_rank" in metadata:
                            print(
                                f"   BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}"
                            )
                        print(f"   {res['document'][:100]}...")
                        print()

                case _:
                    result = rrf_search_command(final_query, args.k, args.limit)

            print(
                f"RRF Search Results for '{result['query']}' (k={result['k']}):"
            )

            if args.enhance == "spell" or args.enhance == "rewrite" or args.enhance == "expand":
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{final_query}'\n")

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_rank" in metadata and "semantic_rank" in metadata:
                    print(
                        f"   BM25 Rank: {metadata['bm25_rank']}, Semantic Rank: {metadata['semantic_rank']}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()