# Directive: Grounding and Validation

## Purpose
Ensure every AI response contains ONLY information that exists in retrieved data. Prevent all forms of hallucination.

## The Core Principle

> "If it's not in the retrieved data, it doesn't exist."

The LLM layer NEVER generates:
- Issue keys (e.g., FIN-101)
- Issue summaries
- Status values
- Priority values  
- Assignee names
- Counts or statistics
- Dates or timestamps

ALL of these come from deterministic execution scripts.

## Validation Pipeline

```
User Query
    ↓
Query Classification
    ↓
Deterministic Retrieval (JQL or Semantic)
    ↓
Permission Filtering
    ↓
Response Generation (LLM)
    ↓
┌─────────────────────────────┐
│   VALIDATION CHECKPOINT     │
│   execution/response_       │
│   validator.py              │
└─────────────────────────────┘
    ↓
If valid → Return response
If invalid → Use grounded template
```

## What Gets Validated

### 1. Issue Keys
- Extract all patterns like `ABC-123` from response
- Check each key exists in retrieved data
- FAIL if any key is hallucinated

### 2. Counts
- Extract numeric claims ("5 issues", "there are 3")
- Compare against actual `len(retrieved_issues)`
- FAIL if count doesn't match

### 3. Status Claims
- Extract status mentions for specific issues
- Verify against actual status in retrieved data
- FAIL if status is wrong

### 4. Priority Claims
- Same as status validation

### 5. Assignee Claims
- Extract "assigned to X" patterns
- Verify X matches actual assignee
- FAIL if assignee is wrong or fabricated

## Failure Handling

When validation fails:

1. **DO NOT** return the invalid response
2. **DO** log the validation errors
3. **DO** fall back to grounded response template:

```python
grounded = validator.create_grounded_response(issues, 'list')
return grounded.text
```

## Grounded Response Templates

### List Template
```
Found {count} issue(s):
- {key}: {summary} [{status}]
- {key}: {summary} [{status}]
...
```

### Detail Template
```
**{key}**: {summary}
  Status: {status} | Priority: {priority}
  Assignee: {assignee}
  Labels: {labels}
```

### Count Template
```
Found {count} issue(s) matching your query.
```

## Testing Grounding

Run validation tests:
```bash
cd execution
python response_validator.py
```

Expected: All hallucination attempts are caught.

## Learnings Log

| Date | Hallucination Type | How Caught | Prevention Added |
|------|-------------------|------------|------------------|
| - | - | - | - |
