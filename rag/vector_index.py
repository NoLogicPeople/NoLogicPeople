import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import os


_MODEL_CACHE: Dict[str, object] = {}
_INDEX_CACHE: Dict[str, Tuple[object, "pd.DataFrame"]] = {}


class VectorIndex:
    """
    FAISS-backed vector index for fast semantic search over
    service descriptions stored in a CSV.

    CSV schema expectations:
    - 'description': top-level category text
    - 'service_name': name of the service
    - 'service_description': textual description of the service
    """

    def __init__(
        self,
        csv_path: str,
        model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        index_path: str = "faiss_ivf.index",
        metadata_path: str = "vector_metadata.csv",
        meta_path: str = "vector_index.meta.json",
        # Index configuration
        index_type: str = "ivf",  # "flat" or "ivf"
        nlist: int = 1024,         # number of clusters for IVF
        nprobe: int = 16,           # probes at search time for IVF
        # Field weighting
        name_weight: float = 1.6,
        description_weight: float = 1.0,
    ) -> None:
        self.csv_path = csv_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.meta_path = meta_path
        self.index_type = index_type
        self.nlist = max(1, int(nlist))
        self.nprobe = max(1, int(nprobe))
        self.name_weight = float(name_weight)
        self.description_weight = float(description_weight)

        try:
            # Limit FAISS threads via env if present
            if os.environ.get("FAISS_NUM_THREADS"):
                try:
                    import faiss  # type: ignore
                    faiss.omp_set_num_threads(int(os.environ["FAISS_NUM_THREADS"]))  # type: ignore[attr-defined]
                except Exception:
                    import faiss  # type: ignore
            else:
                import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "faiss not installed. Install with 'pip install faiss-cpu'."
            ) from exc

        # Lazy imports for heavy deps to speed module import and cache
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._faiss = faiss
        if model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[model_name]  # type: ignore
        else:
            self._model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = self._model

        self._index = None  # type: ignore
        self._metadata_df: Optional[pd.DataFrame] = None

    def ensure_built(self, force_rebuild: bool = False) -> None:
        if force_rebuild or not (os.path.exists(self.index_path) and os.path.exists(self.metadata_path)):
            self._build_and_persist_index()
        else:
            # Validate meta signature; rebuild if mismatched
            try:
                import json
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if not self._is_meta_compatible(meta):
                    self._build_and_persist_index()
                else:
                    self._load_index_and_metadata()
            except Exception:
                # No meta or invalid -> rebuild to ensure new config applies
                self._build_and_persist_index()

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filter_category: Optional[str] = None,
        oversample_factor: Optional[int] = 3,
    ) -> List[Tuple[float, dict]]:
        """
        Search the vector index.

        Returns a list of (score, row_dict) pairs. Scores are cosine similarities.
        If filter_category is provided, results are filtered to rows where
        metadata['description'] == filter_category.
        """
        if self._index is None or self._metadata_df is None:
            self._load_index_and_metadata()

        assert self._index is not None
        assert self._metadata_df is not None

        # Weighted query embedding: boost tokens that look like service names
        # Simple heuristic: short queries get more name weight
        q_len = len((query_text or "").split())
        name_w = 1.8 if q_len <= 4 else 1.3
        desc_w = 1.0
        name_emb = self._encode_and_normalize([query_text])
        desc_emb = self._encode_and_normalize([query_text])
        query_embedding = (name_w * name_emb + desc_w * desc_emb)
        # Normalize
        import numpy as np  # local import
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12
        query_embedding = query_embedding / norms

        # Oversample to allow post-filtering by category without re-querying.
        # If no filter, do not oversample.
        if filter_category is None or oversample_factor is None:
            k = min(self._index.ntotal, top_k)
        else:
            k = min(self._index.ntotal, max(top_k * max(1, int(oversample_factor)), top_k))

        # Configure IVF nprobe if applicable
        try:
            if hasattr(self._index, 'nprobe'):
                # type: ignore[attr-defined]
                self._index.nprobe = self.nprobe
        except Exception:
            pass
        scores, indices = self._index.search(query_embedding, k)

        results: List[Tuple[float, dict]] = []
        taken = 0

        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            row = self._metadata_df.iloc[int(idx)].to_dict()
            if filter_category is not None and str(row.get("description", "")) != str(filter_category):
                continue
            results.append((float(score), row))
            taken += 1
            if taken >= top_k:
                break

        return results

    def _build_and_persist_index(self) -> None:
        df = pd.read_csv(self.csv_path)

        # Normalize types and handle missing values
        df = df.copy()
        df["description"] = df["description"].astype(str).fillna("")
        df["service_name"] = df["service_name"].astype(str).fillna("")
        df["service_description"] = df["service_description"].astype(str).fillna("")

        # Persist combined column for downstream use and inspection
        texts: List[str] = (df["service_name"] + ": " + df["service_description"]).tolist()
        df["service_full_text"] = texts

        # Compute weighted embeddings from service_name and service_description
        name_texts: List[str] = df["service_name"].tolist()
        desc_texts: List[str] = df["service_description"].tolist()
        name_emb = self._encode_and_normalize(name_texts)
        desc_emb = self._encode_and_normalize(desc_texts)
        # Weighted sum then renormalize
        embeddings = (self.name_weight * name_emb + self.description_weight * desc_emb)
        # Normalize to unit vectors
        import numpy as np  # local import to avoid global shadow
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

        # Build FAISS index
        dim = embeddings.shape[1]
        if self.index_type == "ivf":
            quantizer = self._faiss.IndexFlatIP(dim)
            index = self._faiss.IndexIVFFlat(quantizer, dim, self.nlist, self._faiss.METRIC_INNER_PRODUCT)
            # Train IVF on all vectors (or a sample if extremely large)
            index.train(embeddings)
            index.add(embeddings)
        else:
            index = self._faiss.IndexFlatIP(dim)
            index.add(embeddings)

        # Persist
        self._faiss.write_index(index, self.index_path)

        # Reset index to have positional alignment with FAISS ids
        df = df.reset_index(drop=True)
        df.to_csv(self.metadata_path, index=False)

        self._index = index
        self._metadata_df = df
        _INDEX_CACHE[self.index_path] = (self._index, self._metadata_df)

        # Write meta signature
        try:
            import json
            meta = self._current_meta()
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_index_and_metadata(self) -> None:
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Index or metadata not found. Build the index first: {self.index_path}, {self.metadata_path}"
            )
        # Use cache if present
        cached = _INDEX_CACHE.get(self.index_path)
        if cached is not None:
            self._index, self._metadata_df = cached
            return

        index = self._faiss.read_index(self.index_path)
        df = pd.read_csv(self.metadata_path)
        self._index = index
        self._metadata_df = df.reset_index(drop=True)
        _INDEX_CACHE[self.index_path] = (self._index, self._metadata_df)

    def _current_meta(self) -> Dict[str, object]:
        return {
            "csv_path": os.path.abspath(self.csv_path),
            "model_name": getattr(self, "_model", type("x", (), {})()).__dict__.get("name_or_path", "unknown"),
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "name_weight": self.name_weight,
            "description_weight": self.description_weight,
            "metadata_path": os.path.abspath(self.metadata_path),
            "index_path": os.path.abspath(self.index_path),
        }

    def _is_meta_compatible(self, meta: Dict[str, object]) -> bool:
        try:
            keys = [
                "csv_path", "index_type", "nlist", "nprobe",
                "name_weight", "description_weight", "model_name",
            ]
            current = self._current_meta()
            for k in keys:
                if meta.get(k) != current.get(k):
                    return False
            return True
        except Exception:
            return False

    def _encode_and_normalize(self, texts: List[str]) -> np.ndarray:
        # SentenceTransformer returns numpy array if convert_to_tensor=False
        embeddings = self._model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        embeddings = embeddings.astype(np.float32)
        # Safety normalization to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        return embeddings
