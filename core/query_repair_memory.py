"""
Query Repair Memory: Learn from successful query repairs and apply patterns.

Maintains a persistent cache of:
- Error patterns → successful fixes
- Field type information from the index
- Common query structure issues and their resolutions
"""

import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from core.config import Config

logger = logging.getLogger(__name__)

MEMORY_FILE = Path(__file__).parent.parent / "data" / "query_repair_memory.json"


class QueryRepairMemory:
    """Persistent learning system for query repairs."""
    
    def __init__(self):
        """Initialize the repair memory."""
        cfg = Config()
        self.repairs = {}
        self.field_types = {}
        self.error_patterns = {}
        self.max_repairs = int(cfg.get("memory", "query_repair_max_repairs", default=128) or 128)
        self.max_field_types = int(cfg.get("memory", "query_repair_max_field_types", default=512) or 512)
        self.load()
    
    def load(self):
        """Load memory from disk."""
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE) as f:
                    data = json.load(f)
                    self.repairs = data.get("repairs", {})
                    self.field_types = data.get("field_types", {})
                    self.error_patterns = data.get("error_patterns", {})
                self._compact()
                logger.debug("[QueryRepairMemory] Loaded %d repairs, %d field types", 
                           len(self.repairs), len(self.field_types))
        except Exception as e:
            logger.warning("[QueryRepairMemory] Could not load memory: %s", e)
    
    def save(self):
        """Save memory to disk."""
        try:
            self._compact()
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(MEMORY_FILE, 'w') as f:
                json.dump({
                    "repairs": self.repairs,
                    "field_types": self.field_types,
                    "error_patterns": self.error_patterns,
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning("[QueryRepairMemory] Could not save memory: %s", e)
    
    def record_error_fix(self, error_msg: str, original_query: dict, fixed_query: dict):
        """Record a successful error fix."""
        error_key = _normalize_error(error_msg)
        
        # Store the fix pattern
        self.repairs[error_key] = {
            "error": error_msg,
            "original": original_query,
            "fixed": fixed_query,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info("[QueryRepairMemory] Recorded fix for error: %s", error_key[:50])
        self.save()
    
    def get_known_fix(self, error_msg: str) -> Optional[dict]:
        """Get a known fix for an error."""
        error_key = _normalize_error(error_msg)
        if error_key in self.repairs:
            return self.repairs[error_key]
        return None
    
    def record_field_type(self, field_name: str, field_type: str):
        """Record the type of a field from index mapping."""
        if field_name in self.field_types:
            del self.field_types[field_name]
        self.field_types[field_name] = field_type
        if len(self.field_types) > self.max_field_types:
            overflow = len(self.field_types) - self.max_field_types
            for stale_field in list(self.field_types.keys())[:overflow]:
                del self.field_types[stale_field]
        if len(self.field_types) % 10 == 0:
            logger.debug("[QueryRepairMemory] Learned field: %s → %s", field_name, field_type)
        
        if len(self.field_types) % 50 == 0:
            self.save()
    
    def get_field_type(self, field_name: str) -> Optional[str]:
        """Get the known type of a field."""
        return self.field_types.get(field_name)
    
    def learn_from_mapping(self, mapping: dict):
        """Learn field types from index mapping."""
        def extract_fields(props, prefix=""):
            for field_name, field_info in props.items():
                full_name = f"{prefix}{field_name}" if prefix else field_name
                field_type = field_info.get("type", "unknown")
                self.record_field_type(full_name, field_type)
                
                if "properties" in field_info:
                    extract_fields(field_info["properties"], f"{full_name}.")
        
        if "properties" in mapping:
            extract_fields(mapping["properties"])
        
        logger.info("[QueryRepairMemory] Learned %d fields from mapping", len(self.field_types))
        self.save()

    def _compact(self):
        """Keep on-disk repair memory bounded and recent."""
        if len(self.repairs) > self.max_repairs:
            def _repair_sort_key(item):
                payload = item[1] if isinstance(item[1], dict) else {}
                return payload.get("timestamp", "")

            sorted_repairs = sorted(self.repairs.items(), key=_repair_sort_key, reverse=True)
            self.repairs = dict(sorted_repairs[: self.max_repairs])

        if len(self.field_types) > self.max_field_types:
            kept_items = list(self.field_types.items())[-self.max_field_types :]
            self.field_types = dict(kept_items)


def _normalize_error(error_msg: str) -> str:
    """Normalize error message for comparison (remove variable parts)."""
    msg = (error_msg or "").lower()

    # Keep error family + a short reason slice so unrelated failures don't share one cached fix.
    family = "unknown_error"
    for pattern in ["parsing_exception", "search_phase_execution_exception", "x_content_parse_exception"]:
        if pattern in msg:
            family = pattern
            break

    reason_markers = [
        "for input string",
        "unknown query",
        "no start_object",
        "query malformed",
        "failed to create query",
    ]
    reason = ""
    for marker in reason_markers:
        if marker in msg:
            start = msg.find(marker)
            reason = msg[start:start + 120]
            break

    if not reason:
        reason = msg[:120]

    return f"{family}|{reason}" if reason else family


# Global memory instance
_memory = None


def get_memory() -> QueryRepairMemory:
    """Get the global query repair memory instance."""
    global _memory
    if _memory is None:
        _memory = QueryRepairMemory()
    return _memory
