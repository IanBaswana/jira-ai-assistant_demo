"""
Orchestrator - The AI Decision-Making Layer

This is Layer 2 in the DOE (Directive-Orchestration-Execution) architecture.
The orchestrator:
1. Receives natural language queries
2. Classifies them (JQL vs semantic vs hybrid)
3. Calls appropriate execution scripts
4. Applies permission filtering
5. Validates responses before returning
6. Handles errors gracefully

CRITICAL: The orchestrator NEVER generates issue data.
It only routes, filters, and validates data from execution scripts.
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

from query_classifier import QueryClassifier, QueryMode
from jql_executor import JQLExecutor
from semantic_search import SemanticSearch
from response_validator import ResponseValidator
from permission_filter import PermissionFilter


@dataclass
class OrchestratorResponse:
    """Final response from the orchestrator"""
    success: bool
    answer: str
    issues: list
    total_count: int
    query_mode: str
    jql_used: Optional[str] = None
    semantic_query: Optional[str] = None
    validation_passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "answer": self.answer,
            "issues": [{"key": i["key"], "summary": i["summary"], "status": i["status"]} for i in self.issues],
            "total_count": self.total_count,
            "query_mode": self.query_mode,
            "jql_used": self.jql_used,
            "semantic_query": self.semantic_query,
            "validation_passed": self.validation_passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class Orchestrator:
    """
    The AI orchestration layer.
    
    In production, this would be an LLM (Claude, GPT-4) with:
    - Function calling to execute scripts
    - System prompt with directives
    - Response streaming
    
    For this mock, we simulate the orchestration logic to demonstrate
    the architecture without requiring API calls.
    """
    
    def __init__(self, data_path: str = None, permissions_path: str = None):
        base_path = Path(__file__).parent.parent / "data"
        
        if data_path is None:
            data_path = base_path / "mock_jira_data.json"
        if permissions_path is None:
            permissions_path = base_path / "user_permissions.json"
        
        self.classifier = QueryClassifier()
        self.jql_executor = JQLExecutor(data_path)
        self.semantic_search = SemanticSearch(data_path)
        self.validator = ResponseValidator(data_path)
        self.permission_filter = PermissionFilter(permissions_path)
    
    def process_query(self, query: str, user_id: str) -> OrchestratorResponse:
        """
        Process a natural language query end-to-end.
        
        This is the main entry point. It:
        1. Classifies the query
        2. Routes to appropriate retrieval
        3. Filters by permissions
        4. Generates and validates response
        5. Returns structured result
        
        Args:
            query: Natural language query from user
            user_id: ID of the user making the query (for permissions)
            
        Returns:
            OrchestratorResponse with answer and metadata
        """
        debug_info = {}
        
        # Step 1: Classify the query
        classification = self.classifier.classify(query)
        debug_info['classification'] = classification.to_dict()
        
        # Step 2: Handle clarification needed
        if classification.mode == QueryMode.CLARIFICATION_NEEDED:
            return OrchestratorResponse(
                success=True,
                answer=classification.clarification_prompt or "Could you please provide more details?",
                issues=[],
                total_count=0,
                query_mode="clarification",
                debug_info=debug_info
            )
        
        # Step 3: Execute retrieval based on mode
        try:
            if classification.mode == QueryMode.JQL:
                issues, jql_used = self._execute_jql(classification.jql_query)
                debug_info['jql_result_count'] = len(issues)
                
            elif classification.mode == QueryMode.SEMANTIC:
                issues = self._execute_semantic(classification.semantic_query)
                jql_used = None
                debug_info['semantic_result_count'] = len(issues)
                
            elif classification.mode == QueryMode.HYBRID:
                issues = self._execute_hybrid(
                    classification.hybrid_jql_filter,
                    classification.semantic_query
                )
                jql_used = classification.hybrid_jql_filter
                debug_info['hybrid_result_count'] = len(issues)
                
            else:
                return OrchestratorResponse(
                    success=False,
                    answer="I couldn't understand your query. Please try rephrasing.",
                    issues=[],
                    total_count=0,
                    query_mode="error",
                    errors=["Unknown query mode"],
                    debug_info=debug_info
                )
                
        except Exception as e:
            return OrchestratorResponse(
                success=False,
                answer=f"An error occurred while searching: {str(e)}",
                issues=[],
                total_count=0,
                query_mode=classification.mode.value,
                errors=[str(e)],
                debug_info=debug_info
            )
        
        # Step 4: Apply permission filtering
        filter_result = self.permission_filter.filter_issues(issues, user_id)
        filtered_issues = filter_result.allowed_issues
        debug_info['permission_filter'] = {
            'before': len(issues),
            'after': len(filtered_issues),
            'filtered': filter_result.filtered_count
        }
        
        # Step 5: Generate grounded response
        answer = self._generate_answer(query, filtered_issues, classification.mode)
        
        # Step 6: Validate response (double-check no hallucinations)
        validation = self.validator.validate(answer, filtered_issues)
        debug_info['validation'] = {
            'passed': validation.valid,
            'errors': validation.errors,
            'warnings': validation.warnings
        }
        
        # Step 7: Handle validation failure
        if not validation.valid:
            # Fall back to grounded response
            grounded = self.validator.create_grounded_response(filtered_issues, 'list')
            answer = grounded.text
            debug_info['fallback_to_grounded'] = True
        
        # Step 8: Build final response
        warnings = []
        if filter_result.filtered_count > 0:
            warnings.append(f"{filter_result.filtered_count} issue(s) hidden due to permissions")
        
        return OrchestratorResponse(
            success=True,
            answer=answer,
            issues=filtered_issues,
            total_count=len(filtered_issues),
            query_mode=classification.mode.value,
            jql_used=jql_used,
            semantic_query=classification.semantic_query,
            validation_passed=validation.valid,
            warnings=warnings,
            debug_info=debug_info
        )
    
    def _execute_jql(self, jql: str) -> tuple:
        """Execute JQL query and return issues"""
        result = self.jql_executor.execute(jql)
        if not result.success:
            raise Exception(f"JQL execution failed: {result.error}")
        return result.issues, jql
    
    def _execute_semantic(self, query: str, top_k: int = 10) -> list:
        """Execute semantic search and return issues"""
        result = self.semantic_search.search(query, top_k=top_k)
        if not result.success:
            raise Exception(f"Semantic search failed: {result.error}")
        return result.issues
    
    def _execute_hybrid(self, jql_filter: str, semantic_query: str) -> list:
        """Execute hybrid search: JQL filter + semantic ranking"""
        # First, get JQL results
        jql_result = self.jql_executor.execute(jql_filter)
        if not jql_result.success:
            raise Exception(f"JQL filter failed: {jql_result.error}")
        
        jql_issues = jql_result.issues
        
        if not jql_issues:
            return []
        
        # Then, rank by semantic similarity
        # Create a mini search index with just the JQL results
        semantic_result = self.semantic_search.search(semantic_query, top_k=len(jql_issues))
        
        # Filter to only include issues that passed JQL
        jql_keys = {i['key'] for i in jql_issues}
        ranked_issues = [i for i in semantic_result.issues if i['key'] in jql_keys]
        
        # Add any JQL issues not in semantic results (low relevance but match filter)
        ranked_keys = {i['key'] for i in ranked_issues}
        for issue in jql_issues:
            if issue['key'] not in ranked_keys:
                ranked_issues.append(issue)
        
        return ranked_issues
    
    def _generate_answer(self, query: str, issues: list, mode: QueryMode) -> str:
        """
        Generate a natural language answer from retrieved issues.
        
        In production, this would be an LLM call with:
        - The query
        - The retrieved issues (as context)
        - Instructions to ONLY use data from the context
        
        For this mock, we use templates to demonstrate the architecture.
        """
        if not issues:
            return self._generate_no_results_answer(query)
        
        # Detect query intent
        query_lower = query.lower()
        
        # Count queries
        if any(word in query_lower for word in ['how many', 'count', 'number of', 'total']):
            return f"There are {len(issues)} issue(s) matching your query."
        
        # List queries
        if len(issues) <= 5:
            lines = [f"Found {len(issues)} issue(s):"]
            for issue in issues:
                assignee = issue['assignee']['displayName'] if issue.get('assignee') else 'Unassigned'
                lines.append(
                    f"• **{issue['key']}**: {issue['summary']} "
                    f"[{issue['status']}] - {assignee}"
                )
            return '\n'.join(lines)
        
        # Summary for larger result sets
        status_counts = {}
        for issue in issues:
            status = issue['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        status_summary = ', '.join(f"{count} {status}" for status, count in status_counts.items())
        
        lines = [f"Found {len(issues)} issue(s) ({status_summary}):"]
        
        # Show first 5
        for issue in issues[:5]:
            lines.append(f"• **{issue['key']}**: {issue['summary']} [{issue['status']}]")
        
        if len(issues) > 5:
            lines.append(f"... and {len(issues) - 5} more")
        
        return '\n'.join(lines)
    
    def _generate_no_results_answer(self, query: str) -> str:
        """Generate helpful response when no results found"""
        return (
            "No issues found matching your query. This could mean:\n"
            "• No issues match your criteria\n"
            "• You may not have permission to view matching issues\n"
            "• Try broadening your search terms"
        )


# Convenience function for direct use
def process_query(query: str, user_id: str = "user-001") -> dict:
    """Process a query and return results as dict"""
    orchestrator = Orchestrator()
    result = orchestrator.process_query(query, user_id)
    return result.to_dict()


if __name__ == "__main__":
    # Test cases
    orchestrator = Orchestrator()
    
    print("=" * 60)
    print("ORCHESTRATOR END-TO-END TEST CASES")
    print("=" * 60)
    
    test_cases = [
        # JQL queries
        ("user-001", "Show all In Progress issues in FIN"),
        ("user-001", "How many critical priority issues are there?"),
        ("user-001", "Issues assigned to Sarah Chen"),
        
        # Semantic queries
        ("user-001", "What issues are related to authentication?"),
        ("user-001", "Find issues about performance problems"),
        
        # Hybrid queries
        ("user-001", "Critical bugs related to security"),
        
        # Permission tests
        ("user-003", "Show all issues in SEC project"),  # Should be denied
        ("user-008", "Show all FIN issues"),  # Should only see Frontend components
        ("guest-user", "Show all issues"),  # Should see nothing
        
        # Edge cases
        ("user-001", ""),  # Empty query
        ("user-001", "Issues with status Banana"),  # Invalid status
    ]
    
    for user_id, query in test_cases:
        print(f"\n{'─' * 60}")
        print(f"User: {user_id}")
        print(f"Query: \"{query}\"")
        print(f"{'─' * 60}")
        
        result = orchestrator.process_query(query, user_id)
        
        print(f"Mode: {result.query_mode}")
        print(f"Success: {result.success}")
        print(f"Issues found: {result.total_count}")
        
        if result.jql_used:
            print(f"JQL: {result.jql_used}")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        print(f"\nAnswer:")
        print(result.answer)
