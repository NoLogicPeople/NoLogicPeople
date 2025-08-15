from typing import List, Dict, Any, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception as _exc:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


# -----------------------------
# In-memory caches per CSV path
# -----------------------------
_DOCS_CACHE: dict[str, List[Dict[str, Any]]] = {}
_VECTORIZER_CACHE: dict[str, TfidfVectorizer] = {}
_DOC_VECTORS_CACHE: dict[str, Any] = {}
_EMBEDDINGS_CACHE: dict[str, Any] = {}
_DENSE_MODEL: Any = None


def _ensure_dense_model() -> Any:
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        # multilingual MiniLM
        _DENSE_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _DENSE_MODEL


def _load_documents_from_csv(file_path: str, column_weights: Dict[str, int]) -> List[Dict[str, Any]]:
    # Handle UTF-8 BOM gracefully
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file_path)
    required = list(column_weights.keys())
    # Ensure required columns exist; if missing, create empty
    for col in required:
        if col not in df.columns:
            df[col] = ""
    df = df.dropna(subset=required).drop_duplicates(subset=required)
    documents: List[Dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        doc: Dict[str, Any] = {'id': i}
        parts: List[str] = []
        for col, w in column_weights.items():
            text = str(getattr(row, col, '') or '')
            doc[col] = text
            if w > 0 and text:
                parts.extend([text] * int(w))
        # For convenience in UI
        name = str(getattr(row, 'service_name', '') or '')
        desc = str(getattr(row, 'service_description', '') or '')
        doc['service_name'] = name
        doc['service_description'] = desc
        doc['service_full_text'] = f"{name}: {desc}".strip(': ')
        doc['weighted_text'] = " ".join(parts)
        documents.append(doc)
    return documents


def _ensure_caches(data_path: str) -> Tuple[List[Dict[str, Any]], TfidfVectorizer, Any, Any]:
    # Documents
    docs = _DOCS_CACHE.get(data_path)
    if docs is None:
        weights = {'service_name': 3, 'service_description': 2, 'example': 1}
        docs = _load_documents_from_csv(data_path, weights)
        _DOCS_CACHE[data_path] = docs
    # Sparse vectorizer + doc vectors
    vec = _VECTORIZER_CACHE.get(data_path)
    doc_vectors = _DOC_VECTORS_CACHE.get(data_path)
    if vec is None or doc_vectors is None:
        vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        doc_texts = [d['weighted_text'] for d in docs]
        doc_vectors = vec.fit_transform(doc_texts)
        _VECTORIZER_CACHE[data_path] = vec
        _DOC_VECTORS_CACHE[data_path] = doc_vectors
    # Dense embeddings
    emb = _EMBEDDINGS_CACHE.get(data_path)
    if emb is None:
        model = _ensure_dense_model()
        doc_texts = [d['weighted_text'] for d in docs]
        emb = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)
        _EMBEDDINGS_CACHE[data_path] = emb
    return docs, vec, doc_vectors, emb


def _sparse_search(query: str, vec: TfidfVectorizer, doc_vectors: Any, documents: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    qv = vec.transform([query or ''])
    sims = cosine_similarity(qv, doc_vectors).flatten()
    if sims.size == 0:
        return []
    idx = np.argsort(sims)[-top_n:][::-1]
    return [
        {'id': documents[i]['id'], 'title': documents[i]['service_name']}
        for i in idx if sims[i] > 0
    ]


def _dense_search(query: str, documents: List[Dict[str, Any]], embeddings: Any, top_n: int) -> List[Dict[str, Any]]:
    model = _ensure_dense_model()
    if util is None:
        return []
    qe = model.encode(query or '', convert_to_tensor=True)
    sims = util.pytorch_cos_sim(qe, embeddings)[0]
    sims_np = sims.detach().cpu().numpy()
    idx = np.argsort(sims_np)[-top_n:][::-1]
    return [
        {'id': documents[i]['id'], 'title': documents[i]['service_name']}
        for i in idx if sims_np[i] > 0.1
    ]


def _rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    fused_scores: defaultdict[int, float] = defaultdict(float)
    doc_title: Dict[int, str] = {}
    for rlist in result_lists:
        for rank, doc in enumerate(rlist, start=1):
            doc_id = int(doc['id'])
            title = str(doc.get('title') or '')
            if doc_id not in doc_title:
                doc_title[doc_id] = title
            fused_scores[doc_id] += 1.0 / (k + rank)
    # Group by title
    title_scores: defaultdict[str, float] = defaultdict(float)
    title_to_id: Dict[str, int] = {}
    for doc_id, score in fused_scores.items():
        t = doc_title.get(doc_id, '')
        title_scores[t] += score
        if t not in title_to_id:
            title_to_id[t] = doc_id
    fused = [
        {'id': title_to_id[t], 'title': t, 'rrf_score': s}
        for t, s in title_scores.items()
    ]
    fused.sort(key=lambda x: x['rrf_score'], reverse=True)
    return fused


class RagClient:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        # Prime caches
        _ensure_caches(self.data_path)

    def recommend_services(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        documents, vec, doc_vectors, emb = _ensure_caches(self.data_path)
        # Retrieve
        sparse_res = _sparse_search(query, vec, doc_vectors, documents, top_n=max(50, top_k * 10))
        dense_res = _dense_search(query, documents, emb, top_n=max(50, top_k * 10))
        fused = _rrf_fuse([sparse_res, dense_res])
        # Map back to documents and build expected structure
        results: List[Dict[str, str]] = []
        by_id = {d['id']: d for d in documents}
        for item in fused[:top_k]:
            doc = by_id.get(int(item['id']))
            if not doc:
                continue
            results.append({
                'service_name': doc.get('service_name', ''),
                'service_full_text': doc.get('service_full_text', ''),
                'service_description': doc.get('service_description', ''),
            })
        return results


__all__ = ["RagClient"]
