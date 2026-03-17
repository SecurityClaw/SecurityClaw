"""
core/db_connector.py — Provider-agnostic database abstraction.

Supports:
  - OpenSearch  (via opensearch-py)
  - Elasticsearch (via elasticsearch-py)

Abstracts:
  - Standard search / index / delete
  - Anomaly Detection findings poll
  - k-NN vector storage and similarity search
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime, timezone

from core.config import Config

logger = logging.getLogger(__name__)


def _short_json(payload: dict, limit: int = 1200) -> str:
    try:
        text = json.dumps(payload, separators=(",", ":"), default=str)
    except Exception:
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class QueryMalformedException(Exception):
    """Raised when OpenSearch query has syntax/parsing errors (400 errors).
    
    The skill can catch this and use LLM to repair the query.
    """
    def __init__(self, index: str, original_query: dict, error_message: str):
        self.index = index
        self.original_query = original_query
        self.error_message = error_message
        super().__init__(f"Query malformed in index {index}: {error_message}")


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseDBConnector(ABC):
    """Common interface every DB backend must implement."""

    @abstractmethod
    def search(self, index: str, query: dict, size: int = 100) -> list[dict]:
        """Execute a search query and return hits as dicts."""

    @abstractmethod
    def search_with_metadata(self, index: str, query: dict, size: int = 100) -> dict[str, Any]:
        """Execute a search query and return hits plus metadata such as total hit count."""

    @abstractmethod
    def aggregate(self, index: str, query: dict) -> dict[str, Any]:
        """Execute an aggregation query and return the raw response."""

    @abstractmethod
    def index_document(self, index: str, doc_id: str, body: dict) -> dict:
        """Index a single document."""

    @abstractmethod
    def bulk_index(self, index: str, documents: list[dict]) -> dict:
        """Bulk-index a list of documents (each must have a '_id' key)."""

    @abstractmethod
    def get_anomaly_findings(
        self,
        detector_id: str,
        from_epoch_ms: Optional[int] = None,
        size: int = 200,
    ) -> list[dict]:
        """Return anomaly detection findings from the DB."""

    @abstractmethod
    def knn_search(
        self,
        index: str,
        vector: list[float],
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """k-NN approximate nearest-neighbor search."""

    @abstractmethod
    def ensure_index(
        self,
        index: str,
        mappings: dict,
        settings: Optional[dict] = None,
    ) -> None:
        """Create index if it does not exist."""


# ──────────────────────────────────────────────────────────────────────────────
# OpenSearch / Elasticsearch implementation
# ──────────────────────────────────────────────────────────────────────────────

class OpenSearchConnector(BaseDBConnector):
    """
    Works against OpenSearch 2.x (and Elasticsearch 8.x with minor
    path differences — governed by `provider` config key).
    """

    def __init__(self, client: Any = None) -> None:
        """
        Pass a pre-built client for testing, or leave None for
        auto-construction from config.yaml.
        """
        self.cfg = Config()
        self._client = client or self._build_client()
        self._provider = self.cfg.get("db", "provider", default="opensearch")
        # Load index configuration
        self.logs_index = self.cfg.get("db", "logs_index", default="securityclaw-logs")
        self.anomaly_index = self.cfg.get("db", "anomaly_index", default="securityclaw-anomalies")
        self.vector_index = self.cfg.get("db", "vector_index", default="securityclaw-vectors")

    def _build_client(self) -> Any:
        provider = self.cfg.get("db", "provider", default="opensearch")
        host = self.cfg.get("db", "host", default="localhost")
        port = int(self.cfg.get("db", "port", default=9200))
        use_ssl = self.cfg.get("db", "use_ssl", default=False)
        verify = self.cfg.get("db", "verify_certs", default=False)
        user = self.cfg.get("db", "username", default="")
        password = self.cfg.get("db", "password", default="")

        conn_args = dict(
            hosts=[{"host": host, "port": port}],
            use_ssl=use_ssl,
            verify_certs=verify,
            ssl_show_warn=False,
        )
        if user and password:
            conn_args["http_auth"] = (user, password)

        if provider == "elasticsearch":
            from elasticsearch import Elasticsearch
            return Elasticsearch(**conn_args)

        from opensearchpy import OpenSearch
        return OpenSearch(**conn_args)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, index: str, query: dict, size: int = 100) -> list[dict]:
        """Execute a search query.
        
        Raises QueryMalformedException if the query has syntax errors.
        The caller (skill) should catch this and use LLM to repair the query.
        """
        response = self.search_with_metadata(index, query, size=size)
        return response.get("results", [])

    def search_with_metadata(self, index: str, query: dict, size: int = 100) -> dict[str, Any]:
        """Execute a search query and return hits plus total hit count."""
        try:
            resp = self._client.search(index=index, body=query, size=size)
            results = []
            for hit in resp["hits"]["hits"]:
                src = dict(hit.get("_source", {}))
                hit_id = hit.get("_id")
                if hit_id is not None and "_id" not in src:
                    src["_id"] = hit_id
                results.append(src)
            total_hits = resp.get("hits", {}).get("total", 0)
            if isinstance(total_hits, dict):
                total_hits = total_hits.get("value", 0)
            return {"results": results, "total": int(total_hits or 0)}
        except Exception as exc:
            error_str = str(exc)
            
            # Check if it's a 400 error (query syntax/parsing error)
            if "400" in error_str or "failed to create query" in error_str or "RequestError" in error_str:
                logger.error("search(%s) failed with malformed query: %s", index, error_str)
                logger.error("search(%s) malformed query payload: %s", index, _short_json(query))
                raise QueryMalformedException(index, query, error_str)
            else:
                # Other errors - log and return empty
                logger.error("search(%s) failed: %s", index, exc)
                return {"results": [], "total": 0}

    def aggregate(self, index: str, query: dict) -> dict[str, Any]:
        """Execute an aggregation query and return the raw provider response."""
        try:
            return self._client.search(index=index, body=query, size=0)
        except Exception as exc:
            logger.error("aggregate(%s) failed: %s", index, exc)
            return {"aggregations": {}, "hits": {"total": {"value": 0}}}

    def get_document(self, index: str, doc_id: str) -> Optional[dict]:
        try:
            resp = self._client.get(index=index, id=doc_id)
            return resp["_source"]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_document(self, index: str, doc_id: str, body: dict) -> dict:
        try:
            return self._client.index(index=index, id=doc_id, body=body, refresh="wait_for")
        except Exception as exc:
            logger.error("index_document(%s, %s) failed: %s", index, doc_id, exc)
            raise

    def bulk_index(self, index: str, documents: list[dict]) -> dict:
        from opensearchpy.helpers import bulk as os_bulk
        from elasticsearch.helpers import bulk as es_bulk

        actions = [
            {
                "_index": index,
                "_id": doc.get("_id", doc.get("id")),
                "_source": {k: v for k, v in doc.items() if k not in ("_id",)},
            }
            for doc in documents
        ]
        try:
            if self._provider == "elasticsearch":
                success, errors = es_bulk(self._client, actions)
            else:
                success, errors = os_bulk(self._client, actions)
            return {"success": success, "errors": errors}
        except Exception as exc:
            logger.error("bulk_index(%s) failed: %s", index, exc)
            raise

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------

    def get_anomaly_findings(
        self,
        detector_id: str,
        from_epoch_ms: Optional[int] = None,
        size: int = 200,
    ) -> list[dict]:
        """
        Query anomaly detection findings.
        
        For OpenSearch: Uses the configured anomaly_index (defaults to OpenSearch's
        built-in .opendistro-anomaly-results* pattern, but can be overridden).
        
        Args:
            detector_id: The AD detector ID
            from_epoch_ms: Optional cursor for incremental polling
            size: Max results to return
        """
        # Use configured anomaly index, or OpenSearch's default pattern
        ad_index = self.anomaly_index
        if ad_index == "securityclaw-anomalies":
            # If using default, check for OpenSearch AD results index
            ad_index = ".opendistro-anomaly-results*"
        
        must_clauses: list[dict] = [
            {"term": {"detector_id": detector_id}}
        ]
        if from_epoch_ms:
            must_clauses.append(
                {"range": {"data_end_time": {"gte": from_epoch_ms}}}
            )

        query = {
            "query": {"bool": {"must": must_clauses}},
            "sort": [{"data_end_time": {"order": "desc"}}],
        }
        return self.search(ad_index, query, size=size)

    # ------------------------------------------------------------------
    # k-NN (vector) search
    # ------------------------------------------------------------------

    def knn_search(
        self,
        index: str,
        vector: list[float],
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        k-NN approximate nearest-neighbor search.
        Handles filters at the query level (not in KNN block) for compatibility with NMSLIB.
        """
        knn_body: dict = {
            "vector": vector,
            "k": k,
        }
        # Note: NMSLIB doesn't support filters inside the KNN block,
        # so we apply filters at the outer query level instead

        query: dict
        if filters:
            # Build a bool query with both KNN and filter
            query = {
                "bool": {
                    "must": {
                        "knn": {"embedding": knn_body}
                    },
                    "filter": filters
                }
            }
        else:
            # Simple KNN query without filters
            query = {"knn": {"embedding": knn_body}}
        
        try:
            resp = self._client.search(index=index, body={"query": query}, size=k)
            return [
                {**hit["_source"], "_score": hit.get("_score")}
                for hit in resp["hits"]["hits"]
            ]
        except Exception as exc:
            logger.error("knn_search(%s) failed: %s", index, exc)
            return []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def ensure_index(
        self,
        index: str,
        mappings: dict,
        settings: Optional[dict] = None,
    ) -> None:
        try:
            if not self._client.indices.exists(index=index):
                body: dict = {"mappings": mappings}
                if settings:
                    body["settings"] = settings
                self._client.indices.create(index=index, body=body)
                logger.info("Created index: %s", index)
        except Exception as exc:
            logger.warning("ensure_index(%s): %s", index, exc)

    # ------------------------------------------------------------------
    # Vector index bootstrap
    # ------------------------------------------------------------------

    def ensure_vector_index(self, index: str, dims: int = 768) -> None:
        """Create a k-NN enabled index for embedding storage.
        
        Default dims=768 is a reasonable default, but callers should
        explicitly pass dims from their LLM provider's embedding_dimension.
        """
        settings = {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        }
        mappings = {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dims,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                    },
                },
                "text": {"type": "text"},
                "category": {"type": "keyword"},
                "source": {"type": "keyword"},
                "timestamp": {"type": "date"},
            }
        }
        self.ensure_index(index, mappings, settings)
