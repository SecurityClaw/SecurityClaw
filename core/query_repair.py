"""
Intelligent Query Repair with Learning and Retries

Strategies:
1. Check memory for known fix
2. Apply Python-level validation
3. Get LLM to fix with detailed context
4. Retry with increasingly specific instructions
5. Record successful fix in memory
"""

import json
import logging
import time
import re
from typing import Optional, Tuple, Callable
from core.query_repair_memory import get_memory, QueryRepairMemory, _normalize_error

logger = logging.getLogger(__name__)


def _is_time_field(field_name: str) -> bool:
    lowered = str(field_name or "").lower()
    return lowered == "@timestamp" or any(token in lowered for token in ("timestamp", "date", "time"))


def _is_date_like_string(value: str) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return False
    if text in {"now", "now/d", "now/w", "now/m"}:
        return True
    if re.fullmatch(r"now-\d+[hdwm]", text):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)?", text):
        return True
    return False


def _short_json(payload: dict, limit: int = 1200) -> str:
    """Return compact JSON string suitable for logs."""
    try:
        text = json.dumps(payload, separators=(",", ":"), default=str)
    except Exception:
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


class QueryRepairStrategy:
    """Different repair strategies, tried in order."""
    
    @staticmethod
    def apply_python_fix(query: dict) -> dict:
        """Apply Python-level structural fixes."""
        try:
            query = json.loads(json.dumps(query))

            def _walk(node: object, visitor: Callable[[dict], None]) -> None:
                if isinstance(node, dict):
                    visitor(node)
                    for value in list(node.values()):
                        _walk(value, visitor)
                elif isinstance(node, list):
                    for item in node:
                        _walk(item, visitor)

            if "query" not in query:
                return query
            
            query_body = query.get("query", {})
            if not isinstance(query_body, dict):
                return query

            # Fix 0: Normalize misplaced bool clauses at query root
            # e.g. {"query": {"should": [...]}} -> {"query": {"bool": {"should": [...]}}}
            misplaced_clauses = {
                key: query_body.pop(key)
                for key in ["should", "must", "filter", "must_not", "minimum_should_match"]
                if key in query_body
            }
            if misplaced_clauses:
                existing_bool = query_body.get("bool", {})
                if not isinstance(existing_bool, dict):
                    existing_bool = {}
                existing_bool.update(misplaced_clauses)
                query_body["bool"] = existing_bool
                query["query"] = query_body

            bool_query = query.get("query", {}).get("bool", {})
            if not isinstance(bool_query, dict):
                return query

            # Fix 1: Move misplaced size out of bool clauses back to the top level.
            if "size" in bool_query:
                size_value = bool_query.pop("size")
                if "size" not in query and isinstance(size_value, int):
                    query["size"] = size_value
                logger.debug("Python fix: moved bool.size to query root")

            # Fix 2: Remove placeholder timestamp clauses like term(@timestamp=custom).
            placeholder_timestamp_values = {"custom", "any", "none", "null", "timestamp", "date"}
            
            def _strip_bad_timestamp_terms(container: dict) -> None:
                for clause_name in ["must", "should", "filter", "must_not"]:
                    clauses = container.get(clause_name)
                    if not isinstance(clauses, list):
                        continue
                    filtered_clauses = []
                    for clause in clauses:
                        if not isinstance(clause, dict):
                            continue
                        timestamp_value = None
                        if "term" in clause and isinstance(clause["term"], dict):
                            timestamp_value = clause["term"].get("@timestamp")
                        elif "match" in clause and isinstance(clause["match"], dict):
                            timestamp_value = clause["match"].get("@timestamp")
                        elif "range" in clause and isinstance(clause["range"], dict):
                            timestamp_range = clause["range"].get("@timestamp")
                            if isinstance(timestamp_range, dict):
                                for candidate in timestamp_range.values():
                                    if isinstance(candidate, str):
                                        timestamp_value = candidate
                                        break

                        if isinstance(timestamp_value, str) and timestamp_value.strip().lower() in placeholder_timestamp_values:
                            logger.debug("Python fix: removed placeholder @timestamp clause with value %s", timestamp_value)
                            continue
                        filtered_clauses.append(clause)
                    container[clause_name] = filtered_clauses

            _walk(query.get("query", {}), _strip_bad_timestamp_terms)
            
            # Fix 3: Range with string → match
            for clause_type in ["should", "must", "filter"]:
                if clause_type not in bool_query:
                    continue
                
                clauses = bool_query[clause_type]
                if isinstance(clauses, dict):
                    clauses = [clauses]
                
                fixed = []
                for clause in (clauses if isinstance(clauses, list) else [clauses]):
                    if not isinstance(clause, dict):
                        continue
                    
                    # Range with string values → convert to match
                    if "range" in clause and isinstance(clause["range"], dict):
                        for field, cond in clause["range"].items():
                            for op, val in (cond.items() if isinstance(cond, dict) else []):
                                if isinstance(val, str):
                                    if _is_time_field(field) and _is_date_like_string(val):
                                        continue
                                    fixed.append({"match": {field: val}})
                                    logger.debug("Python fix: range(string) → match for %s", field)
                                    clause = None
                                    break
                            if clause is None:
                                break
                    
                    if clause:
                        fixed.append(clause)
                
                if fixed:
                    bool_query[clause_type] = fixed
            
            # Fix 4: Ensure arrays for bool clause query lists
            for clause_type in ["should", "must", "filter"]:
                if clause_type in bool_query and not isinstance(bool_query[clause_type], list):
                    bool_query[clause_type] = [bool_query[clause_type]]
            
            return query
        
        except Exception as e:
            logger.debug("Python fix failed: %s", e)
            return query
    
    @staticmethod
    def apply_llm_fix(query: dict, error_msg: str, llm, attempt: int = 0) -> Optional[dict]:
        """Ask LLM to fix the query with increasingly detailed instructions."""
        
        memory = get_memory()
        
        # Only use cached fix on very first attempt - if it fails, we need new approaches
        if attempt == 0:
            known_fix = memory.get_known_fix(error_msg)
            if known_fix:
                logger.info("[Repair] Using known fix for this error pattern")
                return known_fix["fixed"]
        
        # Build repair prompt based on retry attempt
        # Force different approaches on retries to avoid reusing failed fixes
        if attempt == 0:
            # First attempt: basic fix request
            prompt = _build_repair_prompt_basic(query, error_msg)
        elif attempt <= 2:
            # Second+ attempt: more detailed with field info
            prompt = _build_repair_prompt_detailed(query, error_msg, memory)
        else:
            # Fourth+ attempt: very specific instructions
            prompt = _build_repair_prompt_specific(query, error_msg, attempt)
        
        try:
            logger.info("[Repair] Calling LLM (attempt %d)", attempt)
            response = llm.complete(prompt)
            
            # Extract JSON from response
            fixed = _extract_json_from_llm_response(response)
            if fixed:
                # VALIDATE STRUCTURE - not just JSON syntax
                is_valid, error_reason = _is_valid_query_structure(fixed)
                if is_valid:
                    logger.info("[Repair] LLM produced valid JSON with correct structure")
                    return fixed
                else:
                    logger.warning("[Repair] LLM produced JSON but structure invalid: %s", error_reason)
                    return None
            else:
                logger.warning("[Repair] LLM response had no valid JSON")
                return None
        
        except Exception as e:
            logger.error("[Repair] LLM call failed: %s", e)
            return None


def _build_repair_prompt_basic(query: dict, error_msg: str) -> str:
    """Build a basic repair prompt."""
    return f"""You MUST fix this broken OpenSearch query. Return ONLY valid JSON.

ERROR: {error_msg}

FAILED QUERY:
{json.dumps(query, indent=2)}

RULES:
- 'must', 'should', 'filter' must be ARRAYS inside 'bool' (never at query root)
- Use 'term' for exact matches
- Use 'match' for text searches  
- Use 'range' only for numeric/date values
- Return ONLY JSON - no explanation

Fix it now:"""


def _build_repair_prompt_detailed(query: dict, error_msg: str, memory: QueryRepairMemory) -> str:
    """Build a detailed repair prompt with field type information."""
    
    field_info = "\n".join([
        f"- {field}: {ftype}"
        for field, ftype in list(memory.field_types.items())[:20]
    ])
    
    return f"""You MUST fix this broken OpenSearch query using field information.

ERROR: {error_msg}

KNOWN FIELDS AND TYPES:
{field_info}

FAILED QUERY:
{json.dumps(query, indent=2)}

RULES:
- Match string values with 'match' or 'term' operators
- Use 'range' only for numeric fields (keyword fields use 'term')
- 'should', 'must', 'filter' MUST be arrays inside query.bool
- Never place should/must/filter directly under query
- Check field names against KNOWN FIELDS list above
- Replace unknown fields with correct field names
- Return ONLY valid JSON

Return ONLY JSON:"""


def _build_repair_prompt_specific(query: dict, error_msg: str, attempt: int) -> str:
    """Build a very specific repair prompt for retry attempts with detailed instructions."""
    
    # Extract value from error if it's a "For input string" error
    value_hint = ""
    if "For input string:" in error_msg:
        import re as regex
        match = regex.search(r'For input string: "([^"]*)"', error_msg)
        if match:
            value = match.group(1)
            value_hint = f"\n\nCRITICAL ERROR HINT: The value \"{value}\" is causing field type mismatch errors.\n"
            if value.isdigit() or (value.count('-') == 2):
                value_hint += "- This appears to be a date or numeric value\n"
                value_hint += "- MUST use 'match' or 'multi_match' for text search\n"
                value_hint += "- NEVER use 'range' with string values\n"
            else:
                value_hint += "- This is a string value that needs text search operators\n"
                value_hint += "- Use 'match' or 'multi_match' operators\n"
                value_hint += "- NEVER use 'term' approach which expects exact structure\n"
    
    # CRITICAL: Extract and explain clause-specific errors like "[should] query malformed"
    clause_error = ""
    if "[should]" in error_msg and "no start_object" in error_msg:
        clause_error = """\n\n🔴 CRITICAL: The 'should' clause structure is WRONG.

Your current 'should' clause is INVALID. Look at your query's 'bool' clause.
Find: "should": [ ... ] OR "should": { ... }

FIX: should must be inside query.bool and must be an ARRAY of query objects.
RIGHT:  "query": { "bool": { "should": [ CONDITION1 ] } }     ← one condition
    "query": { "bool": { "should": [ CONDITION1, CONDITION2 ] } }  ← multiple

WRONG:  "query": { "should": [ ... ] }
        "bool": { "should": "string" }
        "bool": { "should": undefined }
"""
    elif "[must]" in error_msg and "no start_object" in error_msg:
        clause_error = "\n\n🔴 CRITICAL: The 'must' clause structure is WRONG. It expects an object or array of objects."
    elif "[filter]" in error_msg and "no start_object" in error_msg:
        clause_error = "\n\n🔴 CRITICAL: The 'filter' clause structure is WRONG. It expects an object or array of objects."
    elif "query malformed" in error_msg or "parse_exception" in error_msg:
        clause_error = "\n\n🔴 CRITICAL: Query structure is BROKEN. Check ALL braces, brackets, and commas."
    
    return f"""RETRY ATTEMPT #{attempt}: This query has FAILED REPEATEDLY.
The LLM keeps generating wrong structures. YOU MUST FIX THE ACTUAL STRUCTURE.

ERROR MESSAGE: {error_msg}{clause_error}

CURRENT FAILED QUERY:
{json.dumps(query, indent=2)}

REQUIRED FIXES (IN THIS EXACT ORDER):
1. CLAUSE STRUCTURE CHECK (MOST IMPORTANT):
    - should/must/filter must be inside query.bool
    - Inside "bool": each clause (should/must/filter) must be ARRAY OF OBJECTS
   - NOT: "should": ""  (empty string)
   - NOT: "should": undefined
    - RIGHT: "must": [{{...}}]

2. If using strings in queries:
   - Use 'match', 'match_phrase', 'multi_match' for text
   - Use 'term' or 'keyword' for exact matching
   - Never mix string values with structural objects

3. Array vs Object Rules:
   - Single condition: "must": {{...}}
   - Multiple conditions: "must": [{{...}}, {{...}}]

4. Syntax check:
   - All {{}} braces balanced
   - All [] brackets balanced
   - All strings quoted with double quotes
   - Proper commas between elements

5. RETURN ONLY VALID JSON - no explanation, no markdown

Focus on: Make sure EVERY 'bool' clause has proper structure.

Return ONLY the corrected JSON query:"""


def _is_valid_query_structure(query: dict) -> Tuple[bool, str]:
    """
    Validate that query structure makes sense for OpenSearch.
    Returns (is_valid, error_reason).
    """
    try:
        q = query.get("query", {})
        if not isinstance(q, dict):
            return (False, "query must be a dict")

        # Clause keys are only valid under query.bool, not directly under query
        misplaced = [k for k in ["should", "must", "filter", "must_not", "minimum_should_match"] if k in q]
        if misplaced:
            return (False, f"misplaced bool clauses at query root: {', '.join(misplaced)}")
        
        bool_clause = q.get("bool", {})
        if not bool_clause:
            # Non-bool query types are valid if they use a known query key
            known_query_types = {
                "match", "match_phrase", "multi_match", "term", "terms", "range",
                "exists", "match_all", "match_none", "query_string", "simple_query_string",
                "wildcard", "regexp", "prefix", "ids", "nested", "script", "dis_max",
                "constant_score", "boosting", "function_score"
            }
            if not any(key in q for key in known_query_types):
                return (False, "query has neither bool nor a recognized query type")
            return (True, "")
        
        if not isinstance(bool_clause, dict):
            return (False, "bool clause must be a dict")
        
        # Check each required clause
        for clause_name in ["should", "must", "filter", "must_not"]:
            if clause_name not in bool_clause:
                continue
            
            clause_val = bool_clause[clause_name]
            
            # Clause must be list (OpenSearch bool query arrays)
            if clause_val is None or clause_val == {} or clause_val == []:
                return (False, f"'{clause_name}' clause is empty or null")
            
            if isinstance(clause_val, list):
                # Array of conditions
                if len(clause_val) == 0:
                    return (False, f"'{clause_name}' array is empty")
                for item in clause_val:
                    if not isinstance(item, dict) or not item:
                        return (False, f"'{clause_name}' array contains invalid item")
            else:
                return (False, f"'{clause_name}' must be a list of query objects, got {type(clause_val).__name__}")
        
        return (True, "")
    except Exception as e:
        return (False, f"Validation error: {str(e)}")


def _extract_json_from_llm_response(response: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling various formats."""
    try:
        # Try direct parse first
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try markdown code blocks
    matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object
    matches = re.findall(r'\{[\s\S]*\}', response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


class IntelligentQueryRepair:
    """Execute query repair with intelligent retries and learning."""
    
    def __init__(self, db, llm):
        """Initialize the repair executor."""
        self.db = db
        self.llm = llm
        self.memory = get_memory()
        self.max_retries = 10  # Increased to 10 for more resilient repair
    
    def repair_and_retry(self, index: str, query: dict, size: int = 200) -> Tuple[bool, Optional[list], str]:
        """
        Repair a query and retry execution.
        
        Returns:
            (success: bool, results: Optional[list], message: str)
        """
        original_query = json.loads(json.dumps(query))  # Deep copy
        current_query = query
        last_error = None
        last_malformed_error = None
        used_repair = False
        repeated_query_failures: dict[tuple[str, str], int] = {}
        seen_llm_fixes: set[str] = set()
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info("[Repair] Attempt %d/%d", attempt + 1, self.max_retries + 1)
                logger.info("[Repair] Query payload (attempt %d): %s", attempt + 1, _short_json(current_query))
                
                # Execute the query
                results = self.db.search(index, current_query, size=size)
                logger.info("[Repair] Success! Got %d results", len(results))

                # Only persist fixes that actually executed successfully.
                if used_repair and last_malformed_error:
                    try:
                        self.memory.record_error_fix(last_malformed_error, original_query, current_query)
                        logger.info("[Repair] Stored successful repair for error pattern")
                    except Exception as memory_exc:
                        logger.warning("[Repair] Failed to persist successful repair: %s", memory_exc)

                return (True, results, "Query successful after repair")
            
            except Exception as exc:
                last_error = str(exc)
                error_msg = _extract_error_message(exc)
                last_malformed_error = error_msg
                failure_signature = (_normalize_error(error_msg), _short_json(current_query, limit=400))
                repeated_query_failures[failure_signature] = repeated_query_failures.get(failure_signature, 0) + 1
                logger.warning("[Repair] Query failed (attempt %d): %s", attempt + 1, error_msg[:100])
                logger.warning("[Repair] Failed query payload (attempt %d): %s", attempt + 1, _short_json(current_query))

                if repeated_query_failures[failure_signature] >= 3:
                    logger.error("[Repair] Aborting repeated identical failure after %d occurrences", repeated_query_failures[failure_signature])
                    return (False, None, f"Repeated identical malformed query failure: {error_msg[:100]}")
                
                if attempt >= self.max_retries:
                    logger.error("[Repair] Max retries reached (%d)", self.max_retries)
                    return (False, None, f"Failed after {self.max_retries} repair attempts: {error_msg[:100]}")
                
                # Try repair strategies
                from core.db_connector import QueryMalformedException
                
                if isinstance(exc, QueryMalformedException):
                    # Strategy 1: Python-level fix
                    logger.info("[Repair] Trying Python-level fix...")
                    fixed = QueryRepairStrategy.apply_python_fix(current_query)
                    if fixed != current_query:
                        current_query = fixed
                        used_repair = True
                        logger.info("[Repair] Applied Python fix, retrying...")
                        logger.info("[Repair] Python-fixed payload: %s", _short_json(current_query))
                        continue
                    
                    # Strategy 2: Try LLM with increasing detail
                    logger.info("[Repair] Python fix didn't help, using LLM (attempt %d)...", attempt)
                    fixed = QueryRepairStrategy.apply_llm_fix(
                        current_query, 
                        error_msg, 
                        self.llm,
                        attempt=attempt
                    )
                    if fixed:
                        fixed_signature = _short_json(fixed, limit=400)
                        if fixed_signature == _short_json(current_query, limit=400):
                            logger.warning("[Repair] LLM returned the same payload again; refusing duplicate retry")
                            continue
                        if fixed_signature in seen_llm_fixes:
                            logger.warning("[Repair] LLM repeated a previously failed repaired payload; refusing duplicate retry")
                            continue
                        seen_llm_fixes.add(fixed_signature)
                        current_query = fixed
                        used_repair = True
                        logger.info("[Repair] Applied LLM fix, retrying...")
                        logger.info("[Repair] LLM-fixed payload: %s", _short_json(current_query))
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    else:
                        # LLM failed, but don't give up - continue loop to retry with different prompt
                        logger.debug("[Repair] LLM fix failed, will retry with more detailed prompt")
                        # Fall through to next iteration to try again
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                else:
                    # Non-malformed error, don't retry
                    logger.error("[Repair] Non-malformed error, not retrying: %s", type(exc).__name__)
                    return (False, None, f"Query execution error: {error_msg[:100]}")
        
        return (False, None, f"Unable to repair query: {last_error[:100] if last_error else 'Unknown error'}")


def _extract_error_message(exc: Exception) -> str:
    """Extract meaningful error message from exception."""
    error_str = str(exc)
    
    # Try to extract OpenSearch error message
    if "'error':" in error_str:
        match = re.search(r"'error':\s*'([^']+)'", error_str)
        if match:
            return match.group(1)
    
    # Try to extract reason
    if "'reason':" in error_str:
        match = re.search(r"'reason':\s*'([^']+)'", error_str)
        if match:
            return match.group(1)
    
    # Return first 200 chars
    return error_str[:200]
