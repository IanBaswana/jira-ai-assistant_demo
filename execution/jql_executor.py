"""
JQL Executor - Deterministic Jira Query Language execution

This script handles all JQL-based retrieval. It returns ONLY what exists in the data.
The LLM orchestration layer should NEVER generate issue data - only this script returns issues.

Key principle: If it's not in the query results, it doesn't exist.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class JQLResult:
    """Structured result from JQL execution - ensures type safety"""
    success: bool
    issues: list
    total_count: int
    jql_query: str
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "issues": self.issues,
            "total_count": self.total_count,
            "jql_query": self.jql_query,
            "error": self.error
        }


class JQLExecutor:
    """
    Executes JQL queries against Jira data.
    
    In production, this would call the Jira REST API.
    For this mock, we simulate JQL parsing and filtering against local JSON data.
    
    CRITICAL: This class is the ONLY source of truth for issue data.
    The LLM must never generate issue keys, statuses, or counts - only this class.
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "mock_jira_data.json"
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.issues = self.data['issues']
        self.valid_statuses = set(self.data['statuses'])
        self.valid_priorities = set(self.data['priorities'])
        self.valid_types = set(self.data['issue_types'])
        self.valid_projects = {p['key'] for p in self.data['projects']}
    
    def execute(self, jql: str) -> JQLResult:
        """
        Execute a JQL query and return structured results.
        
        Args:
            jql: JQL query string (e.g., "project = FIN AND status = 'In Progress'")
            
        Returns:
            JQLResult with matching issues or error details
        """
        try:
            # Parse and validate JQL
            parsed = self._parse_jql(jql)
            if parsed.get('error'):
                return JQLResult(
                    success=False,
                    issues=[],
                    total_count=0,
                    jql_query=jql,
                    error=parsed['error']
                )
            
            # Filter issues based on parsed conditions
            filtered_issues = self._filter_issues(parsed['conditions'])
            
            # Apply ordering if specified
            if parsed.get('order_by'):
                filtered_issues = self._apply_ordering(filtered_issues, parsed['order_by'])
            
            return JQLResult(
                success=True,
                issues=filtered_issues,
                total_count=len(filtered_issues),
                jql_query=jql
            )
            
        except Exception as e:
            return JQLResult(
                success=False,
                issues=[],
                total_count=0,
                jql_query=jql,
                error=f"JQL execution error: {str(e)}"
            )
    
    def _parse_jql(self, jql: str) -> dict:
        """
        Parse JQL into structured conditions.
        
        Supported operators: =, !=, IN, NOT IN, ~, IS, IS NOT
        Supported fields: project, status, priority, type, assignee, reporter, labels, components, created, updated
        Supported connectors: AND, OR
        """
        conditions = []
        order_by = None
        
        # Extract ORDER BY clause
        order_match = re.search(r'\s+ORDER\s+BY\s+(\w+)\s*(ASC|DESC)?', jql, re.IGNORECASE)
        if order_match:
            order_by = {
                'field': order_match.group(1).lower(),
                'direction': (order_match.group(2) or 'ASC').upper()
            }
            jql = jql[:order_match.start()]
        
        # Simple JQL parser - handles common patterns
        # In production, use a proper JQL parser library
        
        # Split by AND/OR (simplified - doesn't handle nested parentheses)
        clauses = re.split(r'\s+AND\s+', jql, flags=re.IGNORECASE)
        
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
                
            condition = self._parse_clause(clause)
            if condition.get('error'):
                return {'error': condition['error']}
            conditions.append(condition)
        
        return {'conditions': conditions, 'order_by': order_by}
    
    def _parse_clause(self, clause: str) -> dict:
        """Parse a single JQL clause into a condition dict"""
        
        # Handle IN operator: field IN (value1, value2)
        in_match = re.match(r'(\w+)\s+(NOT\s+)?IN\s*\(([^)]+)\)', clause, re.IGNORECASE)
        if in_match:
            field = in_match.group(1).lower()
            negated = bool(in_match.group(2))
            values = [v.strip().strip('"\'') for v in in_match.group(3).split(',')]
            return {'field': field, 'operator': 'NOT IN' if negated else 'IN', 'values': values}
        
        # Handle IS NULL / IS NOT NULL
        null_match = re.match(r'(\w+)\s+IS\s+(NOT\s+)?NULL', clause, re.IGNORECASE)
        if null_match:
            field = null_match.group(1).lower()
            negated = bool(null_match.group(2))
            return {'field': field, 'operator': 'IS NOT NULL' if negated else 'IS NULL'}
        
        # Handle IS EMPTY / IS NOT EMPTY
        empty_match = re.match(r'(\w+)\s+IS\s+(NOT\s+)?EMPTY', clause, re.IGNORECASE)
        if empty_match:
            field = empty_match.group(1).lower()
            negated = bool(empty_match.group(2))
            return {'field': field, 'operator': 'IS NOT EMPTY' if negated else 'IS EMPTY'}
        
        # Handle contains operator: field ~ "value"
        contains_match = re.match(r'(\w+)\s+~\s+["\']?([^"\']+)["\']?', clause, re.IGNORECASE)
        if contains_match:
            field = contains_match.group(1).lower()
            value = contains_match.group(2).strip()
            return {'field': field, 'operator': '~', 'value': value}
        
        # Handle comparison operators: =, !=, >, <, >=, <=
        comp_match = re.match(r'(\w+)\s*(!=|>=|<=|=|>|<)\s*["\']?([^"\']+)["\']?', clause, re.IGNORECASE)
        if comp_match:
            field = comp_match.group(1).lower()
            operator = comp_match.group(2)
            value = comp_match.group(3).strip()
            return {'field': field, 'operator': operator, 'value': value}
        
        return {'error': f"Unable to parse JQL clause: {clause}"}
    
    def _filter_issues(self, conditions: list) -> list:
        """Filter issues based on parsed conditions"""
        result = self.issues
        
        for condition in conditions:
            result = [issue for issue in result if self._matches_condition(issue, condition)]
        
        return result
    
    def _matches_condition(self, issue: dict, condition: dict) -> bool:
        """Check if an issue matches a single condition"""
        field = condition['field']
        operator = condition['operator']
        
        # Map JQL fields to issue fields
        field_mapping = {
            'project': lambda i: i['key'].split('-')[0],
            'status': lambda i: i['status'],
            'priority': lambda i: i['priority'],
            'type': lambda i: i['type'],
            'issuetype': lambda i: i['type'],
            'assignee': lambda i: i['assignee']['displayName'] if i['assignee'] else None,
            'reporter': lambda i: i['reporter']['displayName'] if i['reporter'] else None,
            'labels': lambda i: i.get('labels', []),
            'components': lambda i: [c for c in i.get('components', [])],
            'summary': lambda i: i['summary'],
            'description': lambda i: i.get('description', ''),
            'created': lambda i: i['created'],
            'updated': lambda i: i['updated'],
            'resolution': lambda i: i.get('resolution'),
            'sprint': lambda i: i.get('sprint'),
            'key': lambda i: i['key']
        }
        
        if field not in field_mapping:
            return True  # Unknown field, don't filter
        
        issue_value = field_mapping[field](issue)
        
        # Handle different operators
        if operator == 'IS NULL':
            return issue_value is None
        elif operator == 'IS NOT NULL':
            return issue_value is not None
        elif operator == 'IS EMPTY':
            return issue_value is None or issue_value == [] or issue_value == ''
        elif operator == 'IS NOT EMPTY':
            return issue_value is not None and issue_value != [] and issue_value != ''
        elif operator == 'IN':
            if isinstance(issue_value, list):
                return any(v in condition['values'] for v in issue_value)
            return str(issue_value) in condition['values']
        elif operator == 'NOT IN':
            if isinstance(issue_value, list):
                return not any(v in condition['values'] for v in issue_value)
            return str(issue_value) not in condition['values']
        elif operator == '~':
            # Contains operator - case insensitive search
            search_value = condition['value'].lower()
            if isinstance(issue_value, list):
                return any(search_value in str(v).lower() for v in issue_value)
            return search_value in str(issue_value or '').lower()
        elif operator == '=':
            if isinstance(issue_value, list):
                return condition['value'] in issue_value
            return str(issue_value).lower() == condition['value'].lower()
        elif operator == '!=':
            if isinstance(issue_value, list):
                return condition['value'] not in issue_value
            return str(issue_value).lower() != condition['value'].lower()
        elif operator in ('>', '<', '>=', '<='):
            # Date comparison
            if field in ('created', 'updated'):
                issue_date = datetime.fromisoformat(issue_value.replace('Z', '+00:00'))
                compare_date = self._parse_date_value(condition['value'])
                if compare_date is None:
                    return True
                if operator == '>':
                    return issue_date > compare_date
                elif operator == '<':
                    return issue_date < compare_date
                elif operator == '>=':
                    return issue_date >= compare_date
                elif operator == '<=':
                    return issue_date <= compare_date
        
        return True
    
    def _parse_date_value(self, value: str) -> Optional[datetime]:
        """Parse date values including relative dates like -7d"""
        value = value.strip()
        
        # Relative date: -7d, -1w, etc.
        relative_match = re.match(r'-(\d+)([dwmh])', value)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            now = datetime.now()
            if unit == 'd':
                return now - timedelta(days=amount)
            elif unit == 'w':
                return now - timedelta(weeks=amount)
            elif unit == 'h':
                return now - timedelta(hours=amount)
        
        # Absolute date
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y']:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        return None
    
    def _apply_ordering(self, issues: list, order_by: dict) -> list:
        """Apply ORDER BY to results"""
        field = order_by['field']
        reverse = order_by['direction'] == 'DESC'
        
        def get_sort_key(issue):
            if field == 'created':
                return issue.get('created', '')
            elif field == 'updated':
                return issue.get('updated', '')
            elif field == 'priority':
                priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
                return priority_order.get(issue.get('priority'), 99)
            elif field == 'key':
                return issue.get('key', '')
            elif field == 'status':
                return issue.get('status', '')
            return ''
        
        return sorted(issues, key=get_sort_key, reverse=reverse)
    
    def get_valid_values(self, field: str) -> list:
        """
        Return valid values for a field - used for validation and autocomplete.
        This prevents the LLM from inventing invalid statuses, priorities, etc.
        """
        if field == 'status':
            return list(self.valid_statuses)
        elif field == 'priority':
            return list(self.valid_priorities)
        elif field == 'type':
            return list(self.valid_types)
        elif field == 'project':
            return list(self.valid_projects)
        elif field == 'assignee':
            return list(set(
                i['assignee']['displayName'] 
                for i in self.issues 
                if i['assignee']
            ))
        elif field == 'labels':
            labels = set()
            for i in self.issues:
                labels.update(i.get('labels', []))
            return list(labels)
        elif field == 'components':
            components = set()
            for i in self.issues:
                components.update(i.get('components', []))
            return list(components)
        return []


# Convenience function for direct execution
def execute_jql(jql: str, data_path: str = None) -> dict:
    """Execute JQL and return results as dict"""
    executor = JQLExecutor(data_path)
    result = executor.execute(jql)
    return result.to_dict()


if __name__ == "__main__":
    # Test cases
    executor = JQLExecutor()
    
    print("=== JQL Executor Test Cases ===\n")
    
    # Test 1: Simple project filter
    result = executor.execute("project = FIN")
    print(f"Test 1 - project = FIN: {result.total_count} issues found")
    
    # Test 2: Status filter
    result = executor.execute("project = FIN AND status = 'In Progress'")
    print(f"Test 2 - In Progress issues: {result.total_count} issues")
    for issue in result.issues:
        print(f"  - {issue['key']}: {issue['summary'][:50]}...")
    
    # Test 3: Priority filter
    result = executor.execute("priority = Critical")
    print(f"\nTest 3 - Critical priority: {result.total_count} issues")
    
    # Test 4: Assignee filter
    result = executor.execute("assignee = 'Sarah Chen'")
    print(f"\nTest 4 - Assigned to Sarah Chen: {result.total_count} issues")
    
    # Test 5: Labels filter
    result = executor.execute("labels IN (security, compliance)")
    print(f"\nTest 5 - Security or compliance labels: {result.total_count} issues")
    
    # Test 6: Unassigned issues
    result = executor.execute("assignee IS NULL")
    print(f"\nTest 6 - Unassigned issues: {result.total_count} issues")
    
    # Test 7: Text search
    result = executor.execute("summary ~ 'authentication'")
    print(f"\nTest 7 - Summary contains 'authentication': {result.total_count} issues")
    
    # Test 8: Order by
    result = executor.execute("project = FIN ORDER BY priority ASC")
    print(f"\nTest 8 - Ordered by priority: {[i['key'] + ':' + i['priority'] for i in result.issues[:3]]}")
    
    print("\n=== Valid Values (for validation) ===")
    print(f"Statuses: {executor.get_valid_values('status')}")
    print(f"Priorities: {executor.get_valid_values('priority')}")
