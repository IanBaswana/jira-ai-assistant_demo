"""
Query Classifier - Routes queries to appropriate retrieval mode

This module determines whether a user query should be handled by:
1. JQL (deterministic, exact matching)
2. Semantic search (conceptual, fuzzy matching)
3. Hybrid (JQL filter + semantic ranking)

The classifier ensures JQL and semantic modes NEVER contaminate each other.
This prevents the "mixing" problem mentioned in the job posting.
"""

import re
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class QueryMode(Enum):
    JQL = "jql"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CLARIFICATION_NEEDED = "clarification_needed"
    ERROR = "error"


@dataclass
class ClassificationResult:
    """Result of query classification"""
    mode: QueryMode
    confidence: float
    jql_query: Optional[str] = None
    semantic_query: Optional[str] = None
    hybrid_jql_filter: Optional[str] = None
    reasoning: str = ""
    clarification_prompt: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "jql_query": self.jql_query,
            "semantic_query": self.semantic_query,
            "hybrid_jql_filter": self.hybrid_jql_filter,
            "reasoning": self.reasoning,
            "clarification_prompt": self.clarification_prompt
        }


class QueryClassifier:
    """
    Classifies natural language queries into retrieval modes.
    
    JQL Mode triggers:
    - Explicit filters (status, assignee, priority, project, labels)
    - Exact matching requests
    - Counts and aggregations
    - Date-based queries
    
    Semantic Mode triggers:
    - Conceptual questions ("issues related to...")
    - Problem descriptions ("slow", "broken", "authentication problems")
    - Similarity requests ("issues like...")
    - Vague or exploratory queries
    
    Hybrid Mode triggers:
    - Conceptual query with explicit filters
    - "security issues assigned to Sarah"
    """
    
    # JQL field patterns
    JQL_FIELD_PATTERNS = {
        'project': [
            r'\b(?:in|from|for)\s+(?:project\s+)?([A-Z]{2,10})\b',
            r'\bproject\s*[=:]\s*([A-Z]{2,10})\b',
            r'\b([A-Z]{2,10})\s+(?:project|issues?)\b',
        ],
        'status': [
            r'\b(?:status|state)\s*(?:is|=|:)?\s*["\']?(\w+(?:\s+\w+)?)["\']?',
            r'\b(?:in progress|to do|done|blocked|in review)\b',
            r'\b(?:open|closed|resolved|pending)\b',
        ],
        'assignee': [
            r'\b(?:assigned to|assignee\s*(?:is|=|:)?)\s*["\']?([A-Za-z]+(?:\s+[A-Za-z]+)?)["\']?',
            r'\b(?:for|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        ],
        'priority': [
            r'\b(critical|high|medium|low)\s*priority\b',
            r'\bpriority\s*(?:is|=|:)?\s*(critical|high|medium|low)\b',
        ],
        'type': [
            r'\b(bugs?|stories|tasks?|spikes?|epics?)\b',
            r'\btype\s*(?:is|=|:)?\s*(bug|story|task|spike|epic)\b',
        ],
        'labels': [
            r'\blabels?\s*(?:is|=|:|include|contain)?\s*["\']?(\w+)["\']?',
            r'\btagged\s+(?:with\s+)?["\']?(\w+)["\']?',
        ],
        'date': [
            r'\b(?:created|updated)\s*(?:in|during|after|before|since)\b',
            r'\blast\s+(?:week|month|day|year)\b',
            r'\bthis\s+(?:week|month|sprint)\b',
        ],
    }
    
    # Semantic query patterns
    SEMANTIC_PATTERNS = [
        r'\b(?:related to|about|regarding|concerning)\b',
        r'\b(?:similar to|like)\b',
        r'\b(?:issues? with|problems? with)\b',
        r'\b(?:what|which|find|show|get)\s+(?:are|issues?|tickets?)\b',
        r'\b(?:working on|dealing with)\b',
    ]
    
    # Count/aggregate patterns (always JQL)
    COUNT_PATTERNS = [
        r'\bhow many\b',
        r'\bcount\b',
        r'\bnumber of\b',
        r'\btotal\b',
    ]
    
    def __init__(self, valid_values: dict = None):
        """
        Args:
            valid_values: Dict of valid values for each field
                         (used for more accurate classification)
        """
        self.valid_values = valid_values or {
            'statuses': ['To Do', 'In Progress', 'In Review', 'Done', 'Blocked'],
            'priorities': ['Critical', 'High', 'Medium', 'Low'],
            'types': ['Bug', 'Story', 'Task', 'Spike', 'Epic'],
            'projects': ['FIN', 'SEC'],
        }
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a natural language query.
        
        Args:
            query: User's natural language query
            
        Returns:
            ClassificationResult with mode and generated JQL/semantic query
        """
        query_lower = query.lower().strip()
        
        # Check for empty or too short queries
        if len(query_lower) < 3:
            return ClassificationResult(
                mode=QueryMode.CLARIFICATION_NEEDED,
                confidence=1.0,
                reasoning="Query too short",
                clarification_prompt="Could you provide more details about what you're looking for?"
            )
        
        # Detect JQL field mentions
        jql_fields = self._extract_jql_fields(query)
        
        # Detect semantic indicators
        has_semantic_indicators = self._has_semantic_patterns(query_lower)
        
        # Detect count/aggregate requests
        is_count_query = self._is_count_query(query_lower)
        
        # Classification logic
        if is_count_query and jql_fields:
            # Count with filters = JQL
            jql = self._build_jql(jql_fields)
            return ClassificationResult(
                mode=QueryMode.JQL,
                confidence=0.95,
                jql_query=jql,
                reasoning="Count query with explicit filters"
            )
        
        elif jql_fields and not has_semantic_indicators:
            # Pure JQL query
            jql = self._build_jql(jql_fields)
            return ClassificationResult(
                mode=QueryMode.JQL,
                confidence=0.9,
                jql_query=jql,
                reasoning=f"Detected JQL fields: {list(jql_fields.keys())}"
            )
        
        elif has_semantic_indicators and not jql_fields:
            # Pure semantic query
            semantic_query = self._clean_semantic_query(query)
            return ClassificationResult(
                mode=QueryMode.SEMANTIC,
                confidence=0.85,
                semantic_query=semantic_query,
                reasoning="Conceptual/fuzzy query detected"
            )
        
        elif jql_fields and has_semantic_indicators:
            # Hybrid: JQL filter + semantic ranking
            jql_filter = self._build_jql(jql_fields)
            semantic_query = self._extract_semantic_part(query, jql_fields)
            return ClassificationResult(
                mode=QueryMode.HYBRID,
                confidence=0.8,
                hybrid_jql_filter=jql_filter,
                semantic_query=semantic_query,
                reasoning="Both structured filters and conceptual elements detected"
            )
        
        else:
            # Default to semantic for ambiguous queries
            return ClassificationResult(
                mode=QueryMode.SEMANTIC,
                confidence=0.6,
                semantic_query=query,
                reasoning="No clear structure detected, defaulting to semantic search"
            )
    
    def _extract_jql_fields(self, query: str) -> dict:
        """Extract JQL field values from query"""
        fields = {}
        query_lower = query.lower()
        
        # Check for project
        project_matches = re.findall(r'\b([A-Z]{2,10})\b', query)
        for match in project_matches:
            if match in self.valid_values.get('projects', []):
                fields['project'] = match
                break
        
        # Check for status
        for status in self.valid_values.get('statuses', []):
            if status.lower() in query_lower:
                fields['status'] = status
                break
        
        # Check for priority
        for priority in self.valid_values.get('priorities', []):
            if priority.lower() in query_lower:
                fields['priority'] = priority
                break
        
        # Check for type
        type_mapping = {
            'bug': 'Bug', 'bugs': 'Bug',
            'story': 'Story', 'stories': 'Story',
            'task': 'Task', 'tasks': 'Task',
            'spike': 'Spike', 'spikes': 'Spike',
            'epic': 'Epic', 'epics': 'Epic',
        }
        for word, type_val in type_mapping.items():
            if word in query_lower:
                fields['type'] = type_val
                break
        
        # Check for assignee
        assignee_match = re.search(
            r'assigned to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            query,
            re.IGNORECASE
        )
        if assignee_match:
            fields['assignee'] = assignee_match.group(1)
        
        # Check for unassigned
        if 'unassigned' in query_lower:
            fields['assignee_null'] = True
        
        # Check for labels
        label_match = re.search(r'label[s]?\s+(?:is|=|:)?\s*(\w+)', query_lower)
        if label_match:
            fields['labels'] = label_match.group(1)
        
        return fields
    
    def _has_semantic_patterns(self, query: str) -> bool:
        """Check if query has semantic/conceptual indicators"""
        for pattern in self.SEMANTIC_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_count_query(self, query: str) -> bool:
        """Check if query is asking for a count"""
        for pattern in self.COUNT_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _build_jql(self, fields: dict) -> str:
        """Build JQL query from extracted fields"""
        clauses = []
        
        if 'project' in fields:
            clauses.append(f"project = {fields['project']}")
        
        if 'status' in fields:
            clauses.append(f"status = '{fields['status']}'")
        
        if 'priority' in fields:
            clauses.append(f"priority = {fields['priority']}")
        
        if 'type' in fields:
            clauses.append(f"type = {fields['type']}")
        
        if 'assignee' in fields:
            clauses.append(f"assignee = '{fields['assignee']}'")
        
        if fields.get('assignee_null'):
            clauses.append("assignee IS NULL")
        
        if 'labels' in fields:
            clauses.append(f"labels = '{fields['labels']}'")
        
        return ' AND '.join(clauses) if clauses else ""
    
    def _clean_semantic_query(self, query: str) -> str:
        """Clean query for semantic search"""
        # Remove common question words that don't add semantic value
        cleaned = re.sub(r'^(?:what|which|show|find|get|list)\s+', '', query, flags=re.IGNORECASE)
        cleaned = re.sub(r'\?$', '', cleaned)
        return cleaned.strip()
    
    def _extract_semantic_part(self, query: str, jql_fields: dict) -> str:
        """Extract the semantic/conceptual part of a hybrid query"""
        result = query
        
        # Remove JQL field mentions
        for field, value in jql_fields.items():
            if isinstance(value, str):
                result = re.sub(re.escape(value), '', result, flags=re.IGNORECASE)
        
        # Remove common filter phrases
        result = re.sub(r'assigned to\s+\w+(?:\s+\w+)?', '', result, flags=re.IGNORECASE)
        result = re.sub(r'in\s+(?:project\s+)?\w+', '', result, flags=re.IGNORECASE)
        result = re.sub(r'with\s+status\s+\w+', '', result, flags=re.IGNORECASE)
        
        return result.strip()


# Convenience function
def classify_query(query: str, valid_values: dict = None) -> dict:
    """Classify a query and return result as dict"""
    classifier = QueryClassifier(valid_values)
    result = classifier.classify(query)
    return result.to_dict()


if __name__ == "__main__":
    # Test cases
    classifier = QueryClassifier()
    
    print("=== Query Classifier Test Cases ===\n")
    
    test_queries = [
        # Pure JQL queries
        ("Show all bugs in FIN project", "JQL"),
        ("Issues assigned to Sarah Chen", "JQL"),
        ("Critical priority tasks", "JQL"),
        ("How many issues are In Progress?", "JQL"),
        ("Unassigned issues in SEC", "JQL"),
        
        # Pure semantic queries
        ("What are people working on for authentication?", "SEMANTIC"),
        ("Issues related to performance problems", "SEMANTIC"),
        ("Find issues similar to database upgrades", "SEMANTIC"),
        ("What compliance work is happening?", "SEMANTIC"),
        
        # Hybrid queries
        ("Security issues assigned to Sarah", "HYBRID"),
        ("Critical bugs related to trading", "HYBRID"),
        ("FIN project issues about API problems", "HYBRID"),
        
        # Edge cases
        ("hi", "CLARIFICATION"),
        ("Show me everything", "SEMANTIC"),
    ]
    
    for query, expected in test_queries:
        result = classifier.classify(query)
        status = "✓" if result.mode.value.upper().startswith(expected.upper()[:3]) else "✗"
        print(f"{status} Query: \"{query}\"")
        print(f"   Mode: {result.mode.value} (expected: {expected})")
        print(f"   Confidence: {result.confidence}")
        if result.jql_query:
            print(f"   JQL: {result.jql_query}")
        if result.semantic_query:
            print(f"   Semantic: {result.semantic_query}")
        if result.hybrid_jql_filter:
            print(f"   Hybrid JQL: {result.hybrid_jql_filter}")
        print(f"   Reasoning: {result.reasoning}")
        print()
