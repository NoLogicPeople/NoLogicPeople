from typing import List, Dict, Tuple, Optional

from vector_index import VectorIndex
from sentence_transformers import CrossEncoder
import unicodedata
from rank_bm25 import BM25Okapi  # type: ignore


class BranchRAG:
    def __init__(self, data_path):
        # Build or load FAISS vector index once
        self.index = VectorIndex(csv_path=data_path)
        self.index.ensure_built()

        # Cross-encoder reranker (prefer multilingual if available)
        rerank_models = [
            'BAAI/bge-reranker-base',  # multilingual cross-encoder
            'cross-encoder/ms-marco-MiniLM-L-6-v2',  # English fallback
        ]
        last_exc: Optional[Exception] = None
        self.reranker: Optional[CrossEncoder] = None
        for m in rerank_models:
            try:
                self.reranker = CrossEncoder(m)
                break
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                self.reranker = None
        if self.reranker is None and last_exc is not None:
            raise last_exc

        # Build a BM25 lexical index for hybrid retrieval
        self._df = self.index._metadata_df  # type: ignore[attr-defined]
        if self._df is None:
            self.index.ensure_built()
            self._df = self.index._metadata_df  # type: ignore[attr-defined]
        assert self._df is not None

        def norm(s: object) -> str:
            # Robust normalization: accept non-strings, NaN, None
            try:
                base = str(s or '').strip().lower()
            except Exception:
                base = ''
            s2 = unicodedata.normalize('NFKD', base)
            return ''.join(ch for ch in s2 if not unicodedata.combining(ch))

        self._norm = norm
        # Build separate BM25 indices for service_name and service_description
        records = self._df.to_dict(orient='records')
        name_docs: List[List[str]] = [norm((r.get('service_name') or '')).split() for r in records]
        desc_docs: List[List[str]] = [norm((r.get('service_description') or '')).split() for r in records]
        self._bm25_name = BM25Okapi(name_docs)
        self._bm25_desc = BM25Okapi(desc_docs)
        # Weights for combining BM25 scores (favor precise name matches)
        self._bm25_name_weight: float = 0.8
        self._bm25_desc_weight: float = 0.2

    def _deduplicate(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen: set[str] = set()
        unique: List[Dict[str, str]] = []
        for it in items:
            key = (it.get('service_name') or '').strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(it)
        return unique

    def _lexical_boost(self, query: str, text: str, name: str) -> float:
        def norm(s: str) -> str:
            # Lower, strip, remove diacritics for Turkish-insensitive match
            s2 = unicodedata.normalize('NFKD', (s or '').strip().lower())
            return ''.join(ch for ch in s2 if not unicodedata.combining(ch))

        q = norm(query)
        t = norm(text)
        n = norm(name)
        score = 0.0
        if q and q in n:
            score += 1.0
        if q and q in t:
            score += 0.5
        # token overlap (simple)
        q_tokens = [tok for tok in q.split() if tok]
        if q_tokens:
            overlap = sum(1 for tok in q_tokens if tok in n or tok in t)
            score += overlap / max(1, len(q_tokens)) * 0.5
        return score

    def _rerank(self, query: str, candidates: List[Tuple[float, Dict[str, str]]], top_k: int) -> List[Dict[str, str]]:
        # Prepare pairs for cross-encoder
        pairs = [(query, c[1].get('service_full_text', '') or c[1].get('service_name', '')) for c in candidates]
        if not pairs:
            return []
        ce_scores = self.reranker.predict(pairs).tolist()

        # Normalize scores
        min_ce, max_ce = min(ce_scores), max(ce_scores)
        ce_norm = [
            (s - min_ce) / (max(max_ce - min_ce, 1e-6)) for s in ce_scores
        ]

        # Normalize FAISS scores (already cosine similarity, 0..1 in practice)
        faiss_scores = [float(c[0]) for c in candidates]
        min_v, max_v = min(faiss_scores), max(faiss_scores)
        vec_norm = [
            (s - min_v) / (max(max_v - min_v, 1e-6)) for s in faiss_scores
        ]

        # Combine with lexical boost
        combined: List[Tuple[float, Dict[str, str]]] = []
        for i, (_, meta) in enumerate(candidates):
            lb = self._lexical_boost(query, meta.get('service_full_text', ''), meta.get('service_name', ''))
            # Heavier weight on CE for precision; keep vec as guidance; lb as tie-breaker
            score = 0.7 * ce_norm[i] + 0.25 * vec_norm[i] + 0.05 * lb
            combined.append((score, meta))

        combined.sort(key=lambda x: x[0], reverse=True)
        results = [{
            'service_name': m.get('service_name', ''),
            'service_full_text': m.get('service_full_text', ''),
        } for (_, m) in combined]
        return self._deduplicate(results)[:top_k]

    def get_top_services(self, category, query, top_k=3) -> List[Dict[str, str]]:
        # Get wider candidate set from vectors
        vec_candidates = self.index.search(
            query_text=query,
            top_k=max(100, top_k * 20),
            filter_category=category,
        )
        # Augment with BM25 lexical candidates
        bm25_candidates = self._bm25_search(query, top_k=max(100, top_k * 20), filter_category=category)
        candidates = self._merge_candidates(vec_candidates, bm25_candidates)
        return self._rerank(query, candidates, top_k)

    def get_top_services_direct(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        vec_candidates = self.index.search(
            query_text=query,
            top_k=max(200, top_k * 30),
            filter_category=None,
        )
        bm25_candidates = self._bm25_search(query, top_k=max(200, top_k * 30), filter_category=None)
        candidates = self._merge_candidates(vec_candidates, bm25_candidates)
        return self._rerank(query, candidates, top_k)

    def _merge_candidates(
        self,
        vec: List[Tuple[float, Dict[str, str]]],
        bm25: List[Tuple[float, Dict[str, str]]],
    ) -> List[Tuple[float, Dict[str, str]]]:
        # Use service_name as key; take max score when colliding
        merged: Dict[str, Tuple[float, Dict[str, str]]] = {}
        for s, meta in vec:
            key = (meta.get('service_name') or '').strip().lower()
            if not key:
                key = meta.get('service_full_text', '')[:50]
            prev = merged.get(key)
            if prev is None or s > prev[0]:
                merged[key] = (float(s), meta)
        for s, meta in bm25:
            key = (meta.get('service_name') or '').strip().lower()
            if not key:
                key = meta.get('service_full_text', '')[:50]
            prev = merged.get(key)
            if prev is None or s > prev[0]:
                merged[key] = (float(s), meta)
        # Return as list
        return list(merged.values())

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        filter_category: Optional[str],
    ) -> List[Tuple[float, Dict[str, str]]]:
        q_tokens = self._norm(query).split()
        if not q_tokens:
            return []
        # Separate BM25 scores
        import numpy as np  # local import to avoid global shadow
        name_scores = np.asarray(self._bm25_name.get_scores(q_tokens), dtype='float32')
        desc_scores = np.asarray(self._bm25_desc.get_scores(q_tokens), dtype='float32')
        # Normalize each to 0..1 to make weights meaningful
        def _norm_arr(arr: np.ndarray) -> np.ndarray:
            amin = float(arr.min())
            amax = float(arr.max())
            d = max(amax - amin, 1e-6)
            return (arr - amin) / d
        name_n = _norm_arr(name_scores)
        desc_n = _norm_arr(desc_scores)
        # Heuristic: if query is very short, increase name weight further
        q_len = max(1, len(q_tokens))
        w_name = self._bm25_name_weight if q_len > 2 else min(0.9, self._bm25_name_weight + 0.15)
        w_desc = self._bm25_desc_weight if q_len > 2 else max(0.1, self._bm25_desc_weight - 0.15)
        combined = w_name * name_n + w_desc * desc_n
        # Collect top indices
        top_idx = np.argsort(combined)[::-1][:top_k]
        results: List[Tuple[float, Dict[str, str]]] = []
        for i in top_idx.tolist():
            row = self._df.iloc[int(i)].to_dict()
            if filter_category is not None and str(row.get('description','')) != str(filter_category):
                continue
            # Scale bm25 a bit lower so reranker/vec can dominate when needed
            results.append((float(combined[i]) * 0.9, row))
        return results

if __name__ == '__main__':
    branch_rag = BranchRAG('scraped_items_v2.csv')
    category = "Davalarınız ve diğer adli dosyalarınız ile ilgili işlem yapın, dosya detaylarına ulaşın."
    query = "sertifika"
    services = branch_rag.get_top_services(category, query)
    
    print(f"Category: {category}")
    print(f"Query: '{query}'")
    print("Top Services:")
    for service in services:
        print(f"- {service}")