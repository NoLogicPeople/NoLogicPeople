import time
from typing import List

import pandas as pd

from main_rag import MainRAG
from vector_index import VectorIndex


QUERIES: List[str] = [
    "tarım hizmetleri",
    "e-devlet şifre işlemleri",
    "vergi borcu sorgulama",
    "adli sicil kaydı",
    "çiftçi kayıt sistemi",
    "üniversite kayıt",
]


_MAIN_RAG_SINGLETON: MainRAG | None = None
_INDEX_SINGLETON: VectorIndex | None = None


def get_main_rag(data_path: str) -> MainRAG:
    global _MAIN_RAG_SINGLETON
    if _MAIN_RAG_SINGLETON is None:
        _MAIN_RAG_SINGLETON = MainRAG(data_path)
    return _MAIN_RAG_SINGLETON


def get_index(data_path: str) -> VectorIndex:
    global _INDEX_SINGLETON
    if _INDEX_SINGLETON is None:
        _INDEX_SINGLETON = VectorIndex(
            csv_path=data_path,
            index_path='faiss_ivf.index',
            index_type='ivf',
            nlist=1024,
            nprobe=10,
        )
        _INDEX_SINGLETON.ensure_built()
    return _INDEX_SINGLETON


def run_two_stage(data_path: str, query: str, k: int = 3) -> List[str]:
    main_rag = get_main_rag(data_path)
    categories = list(main_rag.classify_query(query))
    index = get_index(data_path)

    results: List[str] = []
    for cat in categories:
        rows = index.search(query_text=query, top_k=k, filter_category=cat)
        results.extend([r[1].get("service_name", "") for r in rows])
    return results[:k]


def run_direct(data_path: str, query: str, k: int = 3) -> List[str]:
    index = get_index(data_path)
    rows = index.search(query_text=query, top_k=k, filter_category=None)
    return [r[1].get("service_name", "") for r in rows]


def benchmark():
    data_path = "scraped_items_v2.csv"

    # Warmup to load models
    _ = run_direct(data_path, QUERIES[0])

    timings = []
    for query in QUERIES:
        t0 = time.perf_counter()
        _ = run_two_stage(data_path, query)
        t1 = time.perf_counter()
        two_stage_ms = (t1 - t0) * 1000.0

        t0 = time.perf_counter()
        _ = run_direct(data_path, query)
        t1 = time.perf_counter()
        direct_ms = (t1 - t0) * 1000.0

        timings.append({
            "query": query,
            "two_stage_ms": round(two_stage_ms, 2),
            "direct_ms": round(direct_ms, 2),
            "improvement_ms": round(two_stage_ms - direct_ms, 2),
        })

    df = pd.DataFrame(timings)
    print(df.to_string(index=False))
    avg_two = df["two_stage_ms"].mean()
    avg_direct = df["direct_ms"].mean()
    print("\nAverages (ms): two-stage=%.2f, direct=%.2f, delta=%.2f" % (avg_two, avg_direct, avg_two - avg_direct))


if __name__ == "__main__":
    benchmark()
