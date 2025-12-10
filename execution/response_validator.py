"""
Response Validator - Grounding and Hallucination Prevention

This is the CRITICAL module that ensures AI responses contain ONLY data
that exists in the retrieved results. It's the final gate before a response
reaches the user.

Key principle: If the AI claims something that isn't in the source data,
the validator catches it and returns an error.
"""

import json
import re
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of response validation"""
    valid: bool
    original_response: str
    validated_response: Optional[str] = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    hallucinated_keys: list = field(default_factory=list)
    hallucinated_values: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "original_response": self.original_response,
            "validated_response": self.validated_response,
            "errors": self.errors,
            "warnings": self.warnings,
            "hallucinated_keys": self.hallucinated_keys,
            "hallucinated_values": self.hallucinated_values
        }


@dataclass 
class GroundedResponse:
    """A response that has been verified against source data"""
    text: str
    source_issues: list
    citations: dict  # Maps claims to source issue keys
    confidence: float


class ResponseValidator:
    """
    Validates AI responses against retrieved source data.
    
    This module:
    1. Extracts all claims from the AI response
    2. Verifies each claim against the source data
    3. Flags hallucinated issue keys, statuses, counts, etc.
    4. Returns validation results with specific error details
    
    The validator doesn't "fix" responses - it gates them.
    If validation fails, the orchestrator must regenerate.
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "mock_jira_data.json"
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Build validation sets from source data
        self.valid_issue_keys = {issue['key'] for issue in self.data['issues']}
        self.valid_statuses = set(self.data['statuses'])
        self.valid_priorities = set(self.data['priorities'])
        self.valid_types = set(self.data['issue_types'])
        
        # Build issue lookup for detailed validation
        self.issues_by_key = {issue['key']: issue for issue in self.data['issues']}
    
    def validate(self, response: str, retrieved_issues: list) -> ValidationResult:
        """
        Validate an AI response against retrieved issue data.
        
        Args:
            response: The AI-generated response text
            retrieved_issues: List of issues that were retrieved for this query
            
        Returns:
            ValidationResult with detailed validation outcome
        """
        errors = []
        warnings = []
        hallucinated_keys = []
        hallucinated_values = {}
        
        # Build set of valid keys for this response (from retrieved data)
        valid_keys_for_response = {issue['key'] for issue in retrieved_issues}
        
        # Extract and validate issue keys mentioned in response
        mentioned_keys = self._extract_issue_keys(response)
        for key in mentioned_keys:
            if key not in self.valid_issue_keys:
                hallucinated_keys.append(key)
                errors.append(f"Hallucinated issue key: {key} does not exist")
            elif key not in valid_keys_for_response:
                # Key exists but wasn't in retrieved data - might be leaking info
                warnings.append(f"Issue {key} mentioned but not in retrieved results")
        
        # Validate counts if present
        count_validation = self._validate_counts(response, retrieved_issues)
        if count_validation['errors']:
            errors.extend(count_validation['errors'])
            hallucinated_values['counts'] = count_validation['invalid_counts']
        
        # Validate status claims
        status_validation = self._validate_status_claims(response, retrieved_issues)
        if status_validation['errors']:
            errors.extend(status_validation['errors'])
            hallucinated_values['statuses'] = status_validation['invalid_statuses']
        
        # Validate priority claims
        priority_validation = self._validate_priority_claims(response, retrieved_issues)
        if priority_validation['errors']:
            errors.extend(priority_validation['errors'])
            hallucinated_values['priorities'] = priority_validation['invalid_priorities']
        
        # Validate assignee claims
        assignee_validation = self._validate_assignee_claims(response, retrieved_issues)
        if assignee_validation['errors']:
            errors.extend(assignee_validation['errors'])
            hallucinated_values['assignees'] = assignee_validation['invalid_assignees']
        
        is_valid = len(errors) == 0 and len(hallucinated_keys) == 0
        
        return ValidationResult(
            valid=is_valid,
            original_response=response,
            validated_response=response if is_valid else None,
            errors=errors,
            warnings=warnings,
            hallucinated_keys=hallucinated_keys,
            hallucinated_values=hallucinated_values
        )
    
    def _extract_issue_keys(self, text: str) -> list:
        """Extract all issue keys mentioned in text (e.g., FIN-101, SEC-201)"""
        pattern = r'\b([A-Z]{2,10}-\d{1,6})\b'
        return list(set(re.findall(pattern, text)))
    
    def _validate_counts(self, response: str, retrieved_issues: list) -> dict:
        """Validate any numeric counts mentioned in the response"""
        errors = []
        invalid_counts = []
        actual_count = len(retrieved_issues)
        
        # Look for count patterns like "5 issues", "there are 3", etc.
        count_patterns = [
            r'(\d+)\s+issues?\b',
            r'(\d+)\s+tickets?\b',
            r'there\s+are\s+(\d+)',
            r'found\s+(\d+)',
            r'total\s+of\s+(\d+)',
        ]
        
        for pattern in count_patterns:
            matches = re.findall(pattern, response.lower())
            for match in matches:
                claimed_count = int(match)
                if claimed_count != actual_count:
                    errors.append(
                        f"Count mismatch: response claims {claimed_count} but retrieved {actual_count}"
                    )
                    invalid_counts.append({
                        'claimed': claimed_count,
                        'actual': actual_count
                    })
        
        return {'errors': errors, 'invalid_counts': invalid_counts}
    
    def _validate_status_claims(self, response: str, retrieved_issues: list) -> dict:
        """Validate status claims match the retrieved issues"""
        errors = []
        invalid_statuses = []
        
        # Build map of issue key to status from retrieved data
        status_map = {issue['key']: issue['status'] for issue in retrieved_issues}
        
        # Look for status claims like "FIN-101 is In Progress"
        for key in self._extract_issue_keys(response):
            if key not in status_map:
                continue
            
            actual_status = status_map[key]
            
            # Check if the response makes a status claim for this key
            for status in self.valid_statuses:
                pattern = rf'{re.escape(key)}.*?\b{re.escape(status)}\b'
                if re.search(pattern, response, re.IGNORECASE):
                    if status.lower() != actual_status.lower():
                        errors.append(
                            f"Status mismatch for {key}: claimed '{status}' but actual is '{actual_status}'"
                        )
                        invalid_statuses.append({
                            'key': key,
                            'claimed': status,
                            'actual': actual_status
                        })
        
        return {'errors': errors, 'invalid_statuses': invalid_statuses}
    
    def _validate_priority_claims(self, response: str, retrieved_issues: list) -> dict:
        """Validate priority claims match the retrieved issues"""
        errors = []
        invalid_priorities = []
        
        priority_map = {issue['key']: issue['priority'] for issue in retrieved_issues}
        
        for key in self._extract_issue_keys(response):
            if key not in priority_map:
                continue
            
            actual_priority = priority_map[key]
            
            for priority in self.valid_priorities:
                pattern = rf'{re.escape(key)}.*?\b{re.escape(priority)}\b(?:\s+priority)?'
                if re.search(pattern, response, re.IGNORECASE):
                    if priority.lower() != actual_priority.lower():
                        errors.append(
                            f"Priority mismatch for {key}: claimed '{priority}' but actual is '{actual_priority}'"
                        )
                        invalid_priorities.append({
                            'key': key,
                            'claimed': priority,
                            'actual': actual_priority
                        })
        
        return {'errors': errors, 'invalid_priorities': invalid_priorities}
    
    def _validate_assignee_claims(self, response: str, retrieved_issues: list) -> dict:
        """Validate assignee claims match the retrieved issues"""
        errors = []
        invalid_assignees = []
        
        assignee_map = {}
        for issue in retrieved_issues:
            if issue.get('assignee'):
                assignee_map[issue['key']] = issue['assignee']['displayName']
            else:
                assignee_map[issue['key']] = None
        
        for key in self._extract_issue_keys(response):
            if key not in assignee_map:
                continue
            
            actual_assignee = assignee_map[key]
            
            # Check for "assigned to X" patterns
            assign_pattern = rf'{re.escape(key)}.*?assigned\s+to\s+([A-Za-z\s]+?)(?:\.|,|$)'
            match = re.search(assign_pattern, response, re.IGNORECASE)
            if match:
                claimed_assignee = match.group(1).strip()
                if actual_assignee is None:
                    errors.append(
                        f"Assignee mismatch for {key}: claimed '{claimed_assignee}' but issue is unassigned"
                    )
                    invalid_assignees.append({
                        'key': key,
                        'claimed': claimed_assignee,
                        'actual': 'Unassigned'
                    })
                elif claimed_assignee.lower() not in actual_assignee.lower():
                    errors.append(
                        f"Assignee mismatch for {key}: claimed '{claimed_assignee}' but actual is '{actual_assignee}'"
                    )
                    invalid_assignees.append({
                        'key': key,
                        'claimed': claimed_assignee,
                        'actual': actual_assignee
                    })
        
        return {'errors': errors, 'invalid_assignees': invalid_assignees}
    
    def create_grounded_response(
        self, 
        issues: list, 
        query_type: str,
        include_details: bool = True
    ) -> GroundedResponse:
        """
        Create a pre-grounded response from retrieved issues.
        
        This is an alternative approach: instead of validating LLM output,
        we provide the LLM with a structured template it must follow.
        
        Args:
            issues: Retrieved issues to summarize
            query_type: Type of query (list, count, detail, etc.)
            include_details: Whether to include full issue details
            
        Returns:
            GroundedResponse that's guaranteed to be accurate
        """
        if not issues:
            return GroundedResponse(
                text="No issues found matching your query.",
                source_issues=[],
                citations={},
                confidence=1.0
            )
        
        citations = {}
        
        if query_type == 'count':
            text = f"Found {len(issues)} issue(s) matching your query."
            citations['count'] = [i['key'] for i in issues]
            
        elif query_type == 'list':
            lines = [f"Found {len(issues)} issue(s):"]
            for issue in issues:
                line = f"- {issue['key']}: {issue['summary']} [{issue['status']}]"
                lines.append(line)
                citations[issue['key']] = issue['key']
            text = '\n'.join(lines)
            
        elif query_type == 'detail' and include_details:
            lines = []
            for issue in issues:
                lines.append(f"**{issue['key']}**: {issue['summary']}")
                lines.append(f"  Status: {issue['status']} | Priority: {issue['priority']}")
                if issue.get('assignee'):
                    lines.append(f"  Assignee: {issue['assignee']['displayName']}")
                else:
                    lines.append(f"  Assignee: Unassigned")
                if issue.get('labels'):
                    lines.append(f"  Labels: {', '.join(issue['labels'])}")
                lines.append("")
                citations[issue['key']] = issue['key']
            text = '\n'.join(lines)
            
        else:
            # Default: simple list
            keys = [i['key'] for i in issues]
            text = f"Issues: {', '.join(keys)}"
            for key in keys:
                citations[key] = key
        
        return GroundedResponse(
            text=text,
            source_issues=issues,
            citations=citations,
            confidence=1.0
        )


# Convenience functions
def validate_response(response: str, retrieved_issues: list, data_path: str = None) -> dict:
    """Validate a response against retrieved issues"""
    validator = ResponseValidator(data_path)
    result = validator.validate(response, retrieved_issues)
    return result.to_dict()


def create_grounded_response(issues: list, query_type: str = 'list', data_path: str = None) -> dict:
    """Create a pre-grounded response from issues"""
    validator = ResponseValidator(data_path)
    result = validator.create_grounded_response(issues, query_type)
    return {
        'text': result.text,
        'source_issues': [i['key'] for i in result.source_issues],
        'citations': result.citations,
        'confidence': result.confidence
    }


if __name__ == "__main__":
    # Test cases
    from jql_executor import JQLExecutor
    
    executor = JQLExecutor()
    validator = ResponseValidator()
    
    print("=== Response Validator Test Cases ===\n")
    
    # Get some test data
    result = executor.execute("project = FIN AND status = 'In Progress'")
    retrieved = result.issues
    
    print(f"Retrieved {len(retrieved)} issues for testing\n")
    
    # Test 1: Valid response
    valid_response = f"There are {len(retrieved)} issues in progress."
    validation = validator.validate(valid_response, retrieved)
    print(f"Test 1 - Valid response: {validation.valid}")
    if validation.errors:
        print(f"  Errors: {validation.errors}")
    
    # Test 2: Hallucinated issue key
    hallucinated_key_response = "The issue FAKE-999 is critical."
    validation = validator.validate(hallucinated_key_response, retrieved)
    print(f"\nTest 2 - Hallucinated key: {validation.valid}")
    print(f"  Hallucinated keys: {validation.hallucinated_keys}")
    print(f"  Errors: {validation.errors}")
    
    # Test 3: Wrong count
    wrong_count_response = "There are 999 issues in progress."
    validation = validator.validate(wrong_count_response, retrieved)
    print(f"\nTest 3 - Wrong count: {validation.valid}")
    print(f"  Errors: {validation.errors}")
    
    # Test 4: Wrong status for real key
    if retrieved:
        real_key = retrieved[0]['key']
        real_status = retrieved[0]['status']
        # Claim the opposite status
        fake_status = "Done" if real_status != "Done" else "To Do"
        wrong_status_response = f"{real_key} is currently {fake_status}."
        validation = validator.validate(wrong_status_response, retrieved)
        print(f"\nTest 4 - Wrong status: {validation.valid}")
        print(f"  Errors: {validation.errors}")
    
    # Test 5: Create grounded response
    print("\n=== Grounded Response Generation ===\n")
    grounded = validator.create_grounded_response(retrieved, 'list')
    print("List format:")
    print(grounded.text)
    
    grounded = validator.create_grounded_response(retrieved, 'detail')
    print("\nDetail format:")
    print(grounded.text)
