import argparse
import os

from app.agent.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="e-Devlet Sesli RAG Asistanı")
    parser.add_argument("--data", type=str, default=os.path.join("rag", "scraped_items_v2.csv"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--memory", type=str, default="./memory.json")
    args = parser.parse_args()

    orch = Orchestrator(data_path=args.data, memory_path=args.memory, top_k=args.top_k)
    while True:
        orch.run_once()
        cont = input("Devam etmek ister misiniz? (e/h): ").strip().lower()
        if cont not in ("e", "evet", "y", "yes"):
            break


if __name__ == "__main__":
    main()
