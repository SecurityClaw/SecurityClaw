"""
tests/mock_opensearch.py — In-memory mock of the OpenSearch layer.

Simulates:
  - Document storage / search
  - Anomaly detection findings
  - k-NN vector similarity search (cosine similarity in NumPy)

No real OpenSearch instance required.
"""
from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from core.db_connector import BaseDBConnector


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MockDBConnector(BaseDBConnector):
    """
    In-memory drop-in replacement for OpenSearchConnector.

    Stores documents in nested dicts: self._store[index][doc_id] = doc
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict]] = defaultdict(dict)
        self._anomaly_findings: list[dict] = []

    # ------------------------------------------------------------------
    # BaseDBConnector interface
    # ------------------------------------------------------------------

    def search(self, index: str, query: dict, size: int = 100) -> list[dict]:
        """
        Simplified search: supports range filters on @timestamp and
        full-scan for everything else.
        """
        response = self.search_with_metadata(index, query, size=size)
        return response["results"]

    def search_with_metadata(self, index: str, query: dict, size: int = 100) -> dict[str, Any]:
        """Simplified search with hit-count metadata and basic sort support."""
        docs = list(self._store.get(index, {}).values())
        docs = _apply_query_filter(docs, query)
        total = len(docs)
        docs = _apply_sort(docs, query.get("sort"))
        return {"results": docs[:size], "total": total}

    def index_document(self, index: str, doc_id: str, body: dict) -> dict:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self._store[index][doc_id] = dict(body)
        return {"result": "created", "_id": doc_id, "_index": index}

    def bulk_index(self, index: str, documents: list[dict]) -> dict:
        success = 0
        for doc in documents:
            doc_id = doc.get("_id") or doc.get("id") or str(uuid.uuid4())
            body = {k: v for k, v in doc.items() if k not in ("_id",)}
            self.index_document(index, doc_id, body)
            success += 1
        return {"success": success, "errors": []}

    def get_anomaly_findings(
        self,
        detector_id: str,
        from_epoch_ms: Optional[int] = None,
        size: int = 50,
    ) -> list[dict]:
        results = [
            f for f in self._anomaly_findings
            if f.get("detector_id") == detector_id
        ]
        if from_epoch_ms is not None:
            results = [
                f for f in results
                if f.get("data_end_time", 0) >= from_epoch_ms
            ]
        return results[:size]

    def knn_search(
        self,
        index: str,
        vector: list[float],
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Rank all stored docs by cosine similarity to `vector`."""
        docs = list(self._store.get(index, {}).values())

        if filters:
            category = (
                filters.get("term", {}).get("category")
                if isinstance(filters, dict) else None
            )
            if category:
                docs = [d for d in docs if d.get("category") == category]

        scored = []
        for doc in docs:
            emb = doc.get("embedding")
            if emb and len(emb) == len(vector):
                sim = _cosine_sim(vector, emb)
                scored.append({**doc, "_score": sim})

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:k]

    def ensure_index(
        self,
        index: str,
        mappings: dict,
        settings: Optional[dict] = None,
    ) -> None:
        # No-op for mock: index is created on first write
        pass

    def ensure_vector_index(self, index: str, dims: int = 768) -> None:
        pass  # No-op

    # ------------------------------------------------------------------
    # Test helpers (not part of the interface)
    # ------------------------------------------------------------------

    def seed_anomaly_findings(self, findings: list[dict]) -> None:
        """Inject pre-built anomaly findings for testing."""
        self._anomaly_findings.extend(findings)

    def seed_documents(self, index: str, documents: list[dict]) -> None:
        """Bulk-load documents into an index under a generated ID."""
        for doc in documents:
            doc_id = doc.get("_id") or doc.get("id") or str(uuid.uuid4())
            self._store[index][doc_id] = doc

    def all_documents(self, index: str) -> list[dict]:
        """Return every document stored in an index (for assertions)."""
        return list(self._store.get(index, {}).values())

    def document_count(self, index: str) -> int:
        return len(self._store.get(index, {}))


# ──────────────────────────────────────────────────────────────────────────────
# Query filter helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_query_filter(docs: list[dict], query: dict) -> list[dict]:
    """
    Support a small subset of OpenSearch query DSL:
      - match_all
      - range on @timestamp (epoch_millis)
      - bool.must
      - term
    """
    q = query.get("query", {})
    if not q or "match_all" in q:
        return docs

    if "range" in q:
        return _apply_range(docs, q["range"])

    if "bool" in q:
        must = q["bool"].get("must", [])
        result = docs
        for clause in must:
            result = _apply_query_filter(result, {"query": clause})
        return result

    if "term" in q:
        field, value = next(iter(q["term"].items()))
        return [d for d in docs if _get_nested(d, field) == value]

    return docs


def _apply_range(docs: list[dict], range_clause: dict) -> list[dict]:
    result = docs
    for field, conditions in range_clause.items():
        gte = conditions.get("gte")
        lte = conditions.get("lte")
        filtered = []
        for doc in result:
            val = _get_nested(doc, field)
            if val is None:
                continue
            # Convert ISO strings to epoch ms for comparison
            if isinstance(val, str):
                try:
                    val = _iso_to_epoch_ms(val)
                except ValueError:
                    continue
            if gte is not None and val < gte:
                continue
            if lte is not None and val > lte:
                continue
            filtered.append(doc)
        result = filtered
    return result


def _apply_sort(docs: list[dict], sort_clause: Any) -> list[dict]:
    if not sort_clause:
        return docs

    sorted_docs = list(docs)
    clauses = sort_clause if isinstance(sort_clause, list) else [sort_clause]
    for clause in reversed(clauses):
        if not isinstance(clause, dict) or not clause:
            continue
        field, options = next(iter(clause.items()))
        order = "asc"
        if isinstance(options, dict):
            order = str(options.get("order", "asc")).lower()
        reverse = order == "desc"
        sorted_docs.sort(key=lambda doc: _sort_value(_get_nested(doc, field)), reverse=reverse)
    return sorted_docs


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    if isinstance(value, str):
        try:
            return (0, _iso_to_epoch_ms(value))
        except ValueError:
            return (0, value)
    return (0, value)


def _get_nested(doc: dict, field: str) -> Any:
    """Dot-path access: 'network.bytes' → doc['network']['bytes']."""
    parts = field.split(".")
    node = doc
    for part in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _iso_to_epoch_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)
