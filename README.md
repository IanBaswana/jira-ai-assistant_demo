# Jira AI Assistant - RAG System with Hallucination Prevention

A production-grade AI assistant for querying Jira issues using natural language, with **zero hallucinations** guaranteed through a deterministic execution architecture.

## 🎯 What This Demonstrates

This project showcases the skills required for building enterprise AI assistants:

- **RAG Architecture**: Retrieval-Augmented Generation with strict grounding
- **Hallucination Prevention**: Multi-layer validation ensures AI never invents data
- **Hybrid Retrieval**: JQL (deterministic) + Semantic Search (conceptual)
- **Permission Enforcement**: User-scoped data access with no leakage
- **Enterprise Patterns**: Testable, auditable, maintainable code

## 🏗️ Architecture: DOE Framework

This system uses a **Directive-Orchestration-Execution** architecture that separates concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    DIRECTIVE LAYER                          │
│  directives/*.md - SOPs defining rules and decision logic   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                        │
│  AI makes decisions: classify query → route → validate      │
│  NEVER generates data, only routes and summarizes           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                          │
│  Deterministic Python scripts - the ONLY source of truth    │
│  • jql_executor.py      - Exact JQL queries                 │
│  • semantic_search.py   - Vector similarity                 │
│  • response_validator.py - Hallucination detection          │
│  • permission_filter.py  - Access control                   │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

> "90% accuracy per step = 59% success over 5 steps."

Traditional RAG lets the LLM do everything, compounding errors. This architecture:
- Pushes all data retrieval into **deterministic code**
- LLM only handles **routing and summarization**
- Every response is **validated before delivery**

## 📁 Project Structure

```
jira-ai-assistant/
├── directives/              # SOPs and decision rules
│   ├── query_handling.md    # How to route queries
│   └── grounding_validation.md  # Anti-hallucination rules
│
├── execution/               # Deterministic Python scripts
│   ├── jql_executor.py      # JQL query execution
│   ├── semantic_search.py   # TF-IDF similarity search
│   ├── query_classifier.py  # Route JQL vs semantic
│   ├── response_validator.py # Hallucination detection
│   ├── permission_filter.py  # User access control
│   └── orchestrator.py      # Main entry point
│
├── tests/                   # Test suite
│   └── test_assistant.py    # 10 functional + 5 privacy tests
│
├── data/                    # Mock Jira data
│   ├── mock_jira_data.json  # Sample issues
│   └── user_permissions.json # Access control rules
│
└── .tmp/                    # Temporary files (gitignored)
```

## 🚀 Quick Start

```bash
# Clone and enter directory
cd jira-ai-assistant

# Install dependencies
pip install -r requirements.txt

# Run the orchestrator demo
cd execution
python orchestrator.py

# Run tests
cd ..
pytest tests/ -v
```

## 🔍 Key Features

### 1. Query Classification

The system automatically determines the best retrieval mode:

| Query Type | Mode | Example |
|-----------|------|---------|
| Structured filters | JQL | "In Progress bugs assigned to Sarah" |
| Conceptual questions | Semantic | "Issues about authentication problems" |
| Mixed | Hybrid | "Critical security issues" |

```python
from execution.query_classifier import QueryClassifier

classifier = QueryClassifier()
result = classifier.classify("Show all critical bugs")
# → mode: JQL, jql_query: "priority = Critical AND type = Bug"
```

### 2. Deterministic JQL Execution

JQL queries return exactly what Jira would return - no hallucinations possible:

```python
from execution.jql_executor import JQLExecutor

executor = JQLExecutor()
result = executor.execute("project = FIN AND status = 'In Progress'")
# → Only real issues matching criteria
```

### 3. Semantic Search (When JQL Can't Apply)

For conceptual queries, semantic search finds relevant issues:

```python
from execution.semantic_search import SemanticSearch

searcher = SemanticSearch()
result = searcher.search("performance problems in trading")
# → Issues ranked by conceptual relevance
```

### 4. Response Validation (Anti-Hallucination)

Every AI response is validated before delivery:

```python
from execution.response_validator import ResponseValidator

validator = ResponseValidator()

# This would FAIL validation - FAKE-999 doesn't exist
response = "The critical issue FAKE-999 needs attention."
validation = validator.validate(response, retrieved_issues)
# → valid: False, hallucinated_keys: ["FAKE-999"]
```

### 5. Permission Filtering

Users only see issues they're authorized to access:

```python
from execution.permission_filter import PermissionFilter

pf = PermissionFilter()
filtered = pf.filter_issues(all_issues, "user-003")
# → Only FIN project issues (user-003 can't see SEC)
```

## 🧪 Test Coverage

### Functional Tests (10)
1. ✅ JQL returns exact matches
2. ✅ No hallucinated issue keys
3. ✅ Counts are accurate
4. ✅ Status values are accurate
5. ✅ Semantic search finds relevant issues
6. ✅ JQL and semantic modes don't contaminate
7. ✅ No results handled gracefully
8. ✅ Labels field is searchable
9. ✅ Components field is searchable
10. ✅ End-to-end query works

### Privacy Tests (5)
1. ✅ Project-level access control
2. ✅ Component-level access control
3. ✅ Label-based restrictions
4. ✅ Comment redaction
5. ✅ Guest/unknown users blocked

## 🔧 Production Enhancements

To make this production-ready, add:

### LLM Integration
```python
# Replace template-based orchestrator with actual LLM
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=SYSTEM_PROMPT,  # Include directives
    tools=[jql_tool, semantic_tool, validate_tool],
    messages=[{"role": "user", "content": query}]
)
```

### Vector Database
```python
# Replace TF-IDF with embeddings + vector DB
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("jira-issues")

# Index issues with embeddings
# Query with semantic similarity
```

### Real Jira Integration
```python
# Replace mock data with Jira REST API
from jira import JIRA

jira = JIRA(server=JIRA_URL, token_auth=API_TOKEN)
issues = jira.search_issues(jql_query)
```

## 📊 Architecture Decisions

### Why Separate JQL and Semantic?

**Problem**: Mixing retrieval modes causes inconsistent results.
- JQL: "Show 5 bugs" → Returns exactly 5 bugs
- Semantic: "Show 5 bugs" → Returns 5 *most relevant* issues (might not all be bugs)

**Solution**: Classify queries and route to ONE mode only. Hybrid mode applies JQL filter FIRST, then semantic ranking.

### Why Validate Responses?

**Problem**: LLMs hallucinate confidently.
- Real issue: FIN-101 (In Progress)
- LLM might say: "FIN-101 is Done" (wrong!)

**Solution**: Response validator checks every claim against source data. If validation fails, fall back to template response.

### Why Permission Filter After Retrieval?

**Problem**: Can't trust the LLM to filter data correctly.

**Solution**: Filter at the data layer, BEFORE the LLM sees it. The LLM can only hallucinate from data it receives.

## 🎓 Skills Demonstrated

- **RAG Systems**: End-to-end retrieval augmented generation
- **LLM Architecture**: Function calling, tool use, orchestration
- **Vector Search**: Semantic similarity (TF-IDF here, embeddings in prod)
- **Data Engineering**: Schema validation, permission systems
- **Testing**: Comprehensive test suite with edge cases
- **Enterprise Patterns**: Audit trails, error handling, validation
- **Python**: Clean, typed, documented code

## 📄 License

MIT License - Use this as a portfolio piece or starting point.

---

Built to demonstrate enterprise AI assistant architecture. See [AGENTS.md](./AGENTS.md) for the underlying DOE framework philosophy.
