"""
Semantic Search - Vector-based retrieval for conceptual queries

This module handles queries that CAN'T be expressed as JQL:
- "Issues related to authentication problems"
- "What are people working on for compliance?"
- "Find issues similar to the database upgrade"

CRITICAL: This module still returns ONLY real issues from the data.
It does NOT generate or fabricate any issue information.
The semantic layer finds relevant issues; it doesn't create them.
"""

import json
import re
import math
from typing import Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SemanticResult:
    """Structured result from semantic search"""
    success: bool
    issues: list
    total_count: int
    query: str
    search_mode: str
    relevance_scores: dict  # issue_key -> score
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "issues": self.issues,
            "total_count": self.total_count,
            "query": self.query,
            "search_mode": self.search_mode,
            "relevance_scores": self.relevance_scores,
            "error": self.error
        }


class SemanticSearch:
    """
    Semantic search over Jira issues using TF-IDF similarity.
    
    In production, this would use:
    - Embeddings from OpenAI/Cohere/local models
    - Vector database (Pinecone, Weaviate, Chroma, FAISS)
    - More sophisticated chunking and retrieval
    
    For this mock, we use TF-IDF to demonstrate the architecture.
    The key point: results come from REAL data, never fabricated.
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "mock_jira_data.json"
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.issues = self.data['issues']
        
        # Build search index
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index for all issues"""
        self.documents = {}
        self.doc_tokens = {}
        
        for issue in self.issues:
            # Combine searchable text fields
            text_parts = [
                issue['key'],
                issue['summary'],
                issue.get('description', '') or '',
                issue['status'],
                issue['priority'],
                issue['type'],
                ' '.join(issue.get('labels', [])),
                ' '.join(issue.get('components', [])),
            ]
            
            # Add assignee/reporter names
            if issue.get('assignee'):
                text_parts.append(issue['assignee']['displayName'])
            if issue.get('reporter'):
                text_parts.append(issue['reporter']['displayName'])
            
            # Add comment text
            for comment in issue.get('comments', []):
                text_parts.append(comment.get('body', ''))
            
            full_text = ' '.join(text_parts).lower()
            self.documents[issue['key']] = full_text
            self.doc_tokens[issue['key']] = self._tokenize(full_text)
        
        # Calculate IDF scores
        self._calculate_idf()
    
    def _tokenize(self, text: str) -> list:
        """Simple tokenization - split on non-alphanumeric, filter stopwords"""
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'any',
            'this', 'that', 'these', 'those', 'it', 'its'
        }
        
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def _calculate_idf(self):
        """Calculate inverse document frequency for all terms"""
        self.idf = {}
        num_docs = len(self.documents)
        
        # Count document frequency for each term
        doc_freq = {}
        for tokens in self.doc_tokens.values():
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # Calculate IDF: log(N / df)
        for token, freq in doc_freq.items():
            self.idf[token] = math.log(num_docs / freq)
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> SemanticResult:
        """
        Search for issues semantically similar to the query.
        
        Args:
            query: Natural language query
            top_k: Maximum number of results to return
            min_score: Minimum relevance score (0-1) to include in results
            
        Returns:
            SemanticResult with matching issues and relevance scores
        """
        try:
            query_tokens = self._tokenize(query.lower())
            
            if not query_tokens:
                return SemanticResult(
                    success=True,
                    issues=[],
                    total_count=0,
                    query=query,
                    search_mode="semantic",
                    relevance_scores={},
                    error="Query produced no searchable terms"
                )
            
            # Calculate TF-IDF scores for each document
            scores = {}
            for issue_key, doc_tokens in self.doc_tokens.items():
                score = self._calculate_similarity(query_tokens, doc_tokens)
                if score >= min_score:
                    scores[issue_key] = round(score, 4)
            
            # Sort by score and take top_k
            sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
            
            # Get full issue data for top results
            result_issues = []
            for key in sorted_keys:
                issue = next((i for i in self.issues if i['key'] == key), None)
                if issue:
                    result_issues.append(issue)
            
            return SemanticResult(
                success=True,
                issues=result_issues,
                total_count=len(result_issues),
                query=query,
                search_mode="semantic",
                relevance_scores={k: scores[k] for k in sorted_keys}
            )
            
        except Exception as e:
            return SemanticResult(
                success=False,
                issues=[],
                total_count=0,
                query=query,
                search_mode="semantic",
                relevance_scores={},
                error=f"Semantic search error: {str(e)}"
            )
    
    def _calculate_similarity(self, query_tokens: list, doc_tokens: list) -> float:
        """Calculate TF-IDF cosine similarity between query and document"""
        if not doc_tokens:
            return 0.0
        
        # Build TF vectors
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
        
        doc_tf = {}
        for token in doc_tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1
        
        # Normalize by document length
        for token in doc_tf:
            doc_tf[token] = doc_tf[token] / len(doc_tokens)
        
        # Calculate TF-IDF weighted cosine similarity
        dot_product = 0.0
        query_norm = 0.0
        doc_norm = 0.0
        
        all_tokens = set(query_tf.keys()) | set(doc_tf.keys())
        
        for token in all_tokens:
            q_tfidf = query_tf.get(token, 0) * self.idf.get(token, 0)
            d_tfidf = doc_tf.get(token, 0) * self.idf.get(token, 0)
            
            dot_product += q_tfidf * d_tfidf
            query_norm += q_tfidf ** 2
            doc_norm += d_tfidf ** 2
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))
    
    def find_similar(self, issue_key: str, top_k: int = 3) -> SemanticResult:
        """
        Find issues similar to a given issue.
        
        Args:
            issue_key: The issue to find similar issues for
            top_k: Number of similar issues to return
            
        Returns:
            SemanticResult with similar issues (excluding the source issue)
        """
        if issue_key not in self.documents:
            return SemanticResult(
                success=False,
                issues=[],
                total_count=0,
                query=f"similar to {issue_key}",
                search_mode="similarity",
                relevance_scores={},
                error=f"Issue {issue_key} not found"
            )
        
        # Use the issue's text as the query
        source_text = self.documents[issue_key]
        result = self.search(source_text, top_k=top_k + 1)  # +1 to exclude self
        
        # Filter out the source issue
        result.issues = [i for i in result.issues if i['key'] != issue_key][:top_k]
        result.relevance_scores = {
            k: v for k, v in result.relevance_scores.items() 
            if k != issue_key
        }
        result.total_count = len(result.issues)
        result.query = f"similar to {issue_key}"
        result.search_mode = "similarity"
        
        return result


# Convenience function
def semantic_search(query: str, top_k: int = 5, data_path: str = None) -> dict:
    """Execute semantic search and return results as dict"""
    searcher = SemanticSearch(data_path)
    result = searcher.search(query, top_k)
    return result.to_dict()


if __name__ == "__main__":
    # Test cases
    searcher = SemanticSearch()
    
    print("=== Semantic Search Test Cases ===\n")
    
    # Test 1: Conceptual query
    result = searcher.search("authentication security login")
    print(f"Test 1 - 'authentication security login': {result.total_count} results")
    for issue in result.issues:
        score = result.relevance_scores.get(issue['key'], 0)
        print(f"  [{score:.3f}] {issue['key']}: {issue['summary'][:50]}...")
    
    # Test 2: Problem-oriented query
    result = searcher.search("performance slow database")
    print(f"\nTest 2 - 'performance slow database': {result.total_count} results")
    for issue in result.issues:
        score = result.relevance_scores.get(issue['key'], 0)
        print(f"  [{score:.3f}] {issue['key']}: {issue['summary'][:50]}...")
    
    # Test 3: Compliance related
    result = searcher.search("audit compliance regulatory")
    print(f"\nTest 3 - 'audit compliance regulatory': {result.total_count} results")
    for issue in result.issues:
        score = result.relevance_scores.get(issue['key'], 0)
        print(f"  [{score:.3f}] {issue['key']}: {issue['summary'][:50]}...")
    
    # Test 4: Bug-oriented
    result = searcher.search("bug fix error broken")
    print(f"\nTest 4 - 'bug fix error broken': {result.total_count} results")
    for issue in result.issues:
        score = result.relevance_scores.get(issue['key'], 0)
        print(f"  [{score:.3f}] {issue['key']}: {issue['summary'][:50]}...")
    
    # Test 5: Find similar issues
    result = searcher.find_similar("FIN-101")
    print(f"\nTest 5 - Issues similar to FIN-101: {result.total_count} results")
    for issue in result.issues:
        score = result.relevance_scores.get(issue['key'], 0)
        print(f"  [{score:.3f}] {issue['key']}: {issue['summary'][:50]}...")
