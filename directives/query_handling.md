# Directive: Query Handling

## Purpose
Route user natural language queries to the appropriate retrieval mode and ensure all responses are grounded in retrieved data.

## Decision Logic

### Step 1: Classify the Query
Use `execution/query_classifier.py` to determine query mode:

| Query Pattern | Mode | Example |
|--------------|------|---------|
| Explicit filters (status, assignee, project) | JQL | "In Progress issues assigned to Sarah" |
| Counts or aggregations | JQL | "How many bugs are open?" |
| Conceptual/fuzzy questions | SEMANTIC | "Issues related to authentication" |
| Filters + conceptual | HYBRID | "Critical bugs about performance" |

### Step 2: Execute Retrieval

**JQL Mode:**
```
Call: execution/jql_executor.py
Input: Generated JQL query
Output: Exact matching issues
```

**Semantic Mode:**
```
Call: execution/semantic_search.py
Input: Cleaned query text
Output: Relevant issues ranked by similarity
```

**Hybrid Mode:**
```
1. Call jql_executor.py with filter JQL
2. Rank results using semantic_search.py
Output: Filtered AND ranked issues
```

### Step 3: Apply Permissions
Always call `execution/permission_filter.py` before returning results.
- Filter issues user cannot access
- Redact comments if not permitted
- Never reveal existence of filtered issues

### Step 4: Validate Response
Before returning ANY response, validate with `execution/response_validator.py`:
- Check all mentioned issue keys exist
- Verify counts match retrieved data
- Confirm status/priority claims are accurate

## Error Handling

| Error | Response |
|-------|----------|
| No results | "No issues found matching your query..." |
| Invalid JQL | Fall back to semantic search |
| Permission denied | Return filtered results, note in warnings |
| Validation failure | Use grounded response template |

## Anti-Hallucination Rules

1. **NEVER generate issue keys** - only use keys from retrieved data
2. **NEVER invent statuses** - only use statuses from retrieved issues
3. **NEVER fabricate counts** - always count actual retrieved issues
4. **NEVER guess assignees** - only name people in retrieved data

## Learnings Log

| Date | Issue | Resolution |
|------|-------|------------|
| - | - | - |

*Update this log when edge cases are discovered*
