import argparse
from main_rag import MainRAG
from branch_rag import BranchRAG

_MAIN_RAG_SINGLETON: MainRAG | None = None
_BRANCH_RAG_SINGLETON: BranchRAG | None = None


def get_main_rag(data_path: str) -> MainRAG:
    global _MAIN_RAG_SINGLETON
    if _MAIN_RAG_SINGLETON is None:
        _MAIN_RAG_SINGLETON = MainRAG(data_path)
    return _MAIN_RAG_SINGLETON


def get_branch_rag(data_path: str) -> BranchRAG:
    global _BRANCH_RAG_SINGLETON
    if _BRANCH_RAG_SINGLETON is None:
        _BRANCH_RAG_SINGLETON = BranchRAG(data_path)
    return _BRANCH_RAG_SINGLETON

def main():
    parser = argparse.ArgumentParser(description="RAG service recommender")
    parser.add_argument("--direct", action="store_true", help="Use direct FAISS search across all items (no category)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--data", type=str, default="scraped_items_v2.csv", help="Path to CSV data file")
    args = parser.parse_args()

    data_path = args.data
    top_k = args.top_k

    branch_rag = get_branch_rag(data_path)

    query = input("Enter your query: ")

    if args.direct:
        print("\nMode: Direct FAISS search")
        services = branch_rag.get_top_services_direct(query, top_k=top_k)
        print("\nTop Recommended Services:")
        for item in services:
            print(f"- {item['service_name']}")
            print(f"  {item['service_full_text']}")
        return

    # Two-stage mode (default)
    print("\nMode: Two-stage (category + FAISS filter)")
    main_rag = get_main_rag(data_path)
    predicted_categories = list(main_rag.classify_query(query))
    print(f"Predicted Category: {predicted_categories}")

    top_services = []
    for category in predicted_categories:
        top_services.append(branch_rag.get_top_services(category, query, top_k=top_k))

    print("\nTop Recommended Services:")
    for services in top_services:
        for item in services:
            print(f"- {item['service_name']}")
            print(f"  {item['service_full_text']}")

if __name__ == '__main__':
    main()