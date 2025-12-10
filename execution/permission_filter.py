"""
Permission Filter - User-scoped data access control

This module ensures users only see issues they're authorized to access.
It's applied AFTER retrieval but BEFORE the LLM sees the data.

Key principle: The LLM can only hallucinate from data it receives.
If we filter data before it reaches the LLM, we prevent data leakage.
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterResult:
    """Result of permission filtering"""
    allowed_issues: list
    filtered_count: int
    total_before_filter: int
    user_id: str
    filter_reasons: dict = field(default_factory=dict)  # issue_key -> reason
    
    def to_dict(self) -> dict:
        return {
            "allowed_issues": self.allowed_issues,
            "filtered_count": self.filtered_count,
            "total_before_filter": self.total_before_filter,
            "user_id": self.user_id,
            "filter_reasons": self.filter_reasons
        }


class PermissionFilter:
    """
    Filters issues based on user permissions.
    
    Permission rules:
    1. Users can only see issues from projects they have access to
    2. If can_view_all_issues is False, only see issues with allowed components
    3. Issues with restricted labels are hidden
    4. Comments are stripped if can_view_comments is False
    
    This is a simplified model. Production systems would integrate with:
    - Jira's actual permission scheme
    - LDAP/AD group memberships
    - Row-level security in the database
    """
    
    def __init__(self, permissions_path: str = None):
        if permissions_path is None:
            permissions_path = Path(__file__).parent.parent / "data" / "user_permissions.json"
        
        with open(permissions_path, 'r') as f:
            self.permissions_data = json.load(f)
        
        self.permissions = self.permissions_data['permissions']
    
    def filter_issues(self, issues: list, user_id: str) -> FilterResult:
        """
        Filter issues based on user permissions.
        
        Args:
            issues: List of issues to filter
            user_id: The user requesting access
            
        Returns:
            FilterResult with allowed issues and filter details
        """
        if user_id not in self.permissions:
            # Unknown user gets no access
            return FilterResult(
                allowed_issues=[],
                filtered_count=len(issues),
                total_before_filter=len(issues),
                user_id=user_id,
                filter_reasons={i['key']: 'unknown_user' for i in issues}
            )
        
        user_perms = self.permissions[user_id]
        allowed_projects = set(user_perms.get('projects', []))
        can_view_all = user_perms.get('can_view_all_issues', False)
        viewable_components = set(user_perms.get('viewable_components', []))
        restricted_labels = set(user_perms.get('restricted_labels', []))
        can_view_comments = user_perms.get('can_view_comments', True)
        
        allowed_issues = []
        filter_reasons = {}
        
        for issue in issues:
            # Check 1: Project access
            project = issue['key'].split('-')[0]
            if project not in allowed_projects:
                filter_reasons[issue['key']] = f'no_project_access:{project}'
                continue
            
            # Check 2: Component access (if not full access)
            if not can_view_all:
                issue_components = set(issue.get('components', []))
                if not issue_components.intersection(viewable_components):
                    filter_reasons[issue['key']] = 'no_component_access'
                    continue
            
            # Check 3: Restricted labels
            issue_labels = set(issue.get('labels', []))
            if restricted_labels == {'*'}:
                # Wildcard - all labels restricted
                if issue_labels:
                    filter_reasons[issue['key']] = 'all_labels_restricted'
                    continue
            elif issue_labels.intersection(restricted_labels):
                blocked_labels = issue_labels.intersection(restricted_labels)
                filter_reasons[issue['key']] = f'restricted_labels:{",".join(blocked_labels)}'
                continue
            
            # Issue passed all checks - include it
            filtered_issue = issue.copy()
            
            # Strip comments if not allowed
            if not can_view_comments:
                filtered_issue['comments'] = []
                filtered_issue['_comments_redacted'] = True
            
            allowed_issues.append(filtered_issue)
        
        return FilterResult(
            allowed_issues=allowed_issues,
            filtered_count=len(issues) - len(allowed_issues),
            total_before_filter=len(issues),
            user_id=user_id,
            filter_reasons=filter_reasons
        )
    
    def get_user_permissions(self, user_id: str) -> dict:
        """Get permission details for a user"""
        if user_id not in self.permissions:
            return {
                'user_id': user_id,
                'exists': False,
                'projects': [],
                'access_level': 'none'
            }
        
        perms = self.permissions[user_id]
        return {
            'user_id': user_id,
            'exists': True,
            'projects': perms.get('projects', []),
            'can_view_all_issues': perms.get('can_view_all_issues', False),
            'can_view_comments': perms.get('can_view_comments', True),
            'restricted_labels': perms.get('restricted_labels', [])
        }
    
    def check_access(self, issue: dict, user_id: str) -> dict:
        """
        Check if a specific user can access a specific issue.
        Returns detailed access decision.
        """
        result = self.filter_issues([issue], user_id)
        
        return {
            'allowed': len(result.allowed_issues) > 0,
            'reason': result.filter_reasons.get(issue['key'], 'allowed'),
            'user_id': user_id,
            'issue_key': issue['key']
        }


# Convenience function
def filter_for_user(issues: list, user_id: str, permissions_path: str = None) -> dict:
    """Filter issues for a specific user"""
    pf = PermissionFilter(permissions_path)
    result = pf.filter_issues(issues, user_id)
    return result.to_dict()


if __name__ == "__main__":
    # Test cases
    from jql_executor import JQLExecutor
    
    executor = JQLExecutor()
    pf = PermissionFilter()
    
    print("=== Permission Filter Test Cases ===\n")
    
    # Get all issues
    all_issues_result = executor.execute("project IN (FIN, SEC)")
    all_issues = all_issues_result.issues
    print(f"Total issues in system: {len(all_issues)}\n")
    
    # Test 1: Full access user (Sarah Chen - user-001)
    result = pf.filter_issues(all_issues, "user-001")
    print(f"Test 1 - user-001 (full access):")
    print(f"  Can see: {len(result.allowed_issues)} issues")
    print(f"  Filtered: {result.filtered_count} issues")
    
    # Test 2: Limited project access (Alex Rivera - user-003, no SEC access)
    result = pf.filter_issues(all_issues, "user-003")
    print(f"\nTest 2 - user-003 (FIN only, no compliance labels):")
    print(f"  Can see: {len(result.allowed_issues)} issues")
    print(f"  Filtered: {result.filtered_count} issues")
    if result.filter_reasons:
        print(f"  Filter reasons sample: {dict(list(result.filter_reasons.items())[:3])}")
    
    # Test 3: Component-restricted user (Emma Davis - user-008)
    result = pf.filter_issues(all_issues, "user-008")
    print(f"\nTest 3 - user-008 (Frontend components only):")
    print(f"  Can see: {len(result.allowed_issues)} issues")
    print(f"  Visible issues: {[i['key'] for i in result.allowed_issues]}")
    
    # Test 4: No comment access (Chris Martinez - user-010)
    result = pf.filter_issues(all_issues, "user-010")
    print(f"\nTest 4 - user-010 (no comment access):")
    print(f"  Can see: {len(result.allowed_issues)} issues")
    if result.allowed_issues:
        sample = result.allowed_issues[0]
        print(f"  Comments visible: {not sample.get('_comments_redacted', False)}")
    
    # Test 5: Guest user (no access)
    result = pf.filter_issues(all_issues, "guest-user")
    print(f"\nTest 5 - guest-user (no access):")
    print(f"  Can see: {len(result.allowed_issues)} issues")
    print(f"  Filtered: {result.filtered_count} issues")
    
    # Test 6: Unknown user
    result = pf.filter_issues(all_issues, "unknown-person")
    print(f"\nTest 6 - unknown-person:")
    print(f"  Can see: {len(result.allowed_issues)} issues")
    
    # Test 7: Check specific access
    print("\n=== Specific Access Checks ===")
    sec_issue = next((i for i in all_issues if i['key'].startswith('SEC-')), None)
    if sec_issue:
        access = pf.check_access(sec_issue, "user-003")
        print(f"Can user-003 access {sec_issue['key']}? {access['allowed']} ({access['reason']})")
