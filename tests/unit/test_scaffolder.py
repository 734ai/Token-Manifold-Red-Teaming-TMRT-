"""
Unit tests for role scaffolding component.
"""

import pytest
import json
from datetime import datetime
from src.tmrt.scaffolder import RoleScaffolder


class TestRoleScaffolder:
    """Test role scaffolding functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scaffolder = RoleScaffolder(seed=42)
        
    def test_generate_email_scaffold(self):
        """Test email scaffold generation."""
        email = self.scaffolder.generate_email_scaffold(
            sender_role="authority",
            urgency_level="high",
            include_metadata=True
        )
        
        # Should contain email components
        assert "From:" in email
        assert "To:" in email
        assert "Subject:" in email
        assert "@" in email  # Email address
        
        # Should contain urgency indicators for high urgency
        urgency_indicators = ["URGENT", "CRITICAL", "IMMEDIATE", "HIGH PRIORITY"]
        has_urgency = any(indicator in email.upper() for indicator in urgency_indicators)
        assert has_urgency
        
    def test_generate_chat_transcript(self):
        """Test chat transcript generation."""
        chat = self.scaffolder.generate_chat_transcript(
            num_messages=5,
            include_timestamps=True,
            build_rapport=True
        )
        
        # Should contain chat elements
        assert "===" in chat  # Header
        assert "<" in chat and ">" in chat  # Message format
        assert "[" in chat and "]" in chat  # Timestamps
        
        # Should have expected number of messages
        message_count = chat.count("<")
        assert message_count == 5
        
    def test_generate_log_format_scaffold(self):
        """Test log format scaffold generation."""
        log = self.scaffolder.generate_log_format_scaffold(
            log_type="security",
            num_entries=3
        )
        
        # Should contain log elements
        assert "LOG FILE" in log.upper()
        assert "[" in log  # Timestamps
        assert "INFO" in log or "WARN" in log or "ERROR" in log  # Log levels
        
        # Should have expected number of entries
        entry_count = log.count("[20")  # Timestamp format [2025-
        assert entry_count >= 3
        
    def test_generate_json_metadata_scaffold(self):
        """Test JSON metadata scaffold generation."""
        json_scaffold = self.scaffolder.generate_json_metadata_scaffold(
            schema_type="api_request",
            include_auth=True
        )
        
        # Should be valid JSON
        parsed = json.loads(json_scaffold)
        assert isinstance(parsed, dict)
        
        # Should contain expected fields
        assert "request_id" in parsed
        assert "timestamp" in parsed
        assert "auth" in parsed  # Since include_auth=True
        
    def test_generate_multi_role_scenario(self):
        """Test multi-role scenario generation."""
        scenario = self.scaffolder.generate_multi_role_scenario(
            scenario_type="incident_response",
            num_participants=3
        )
        
        # Should contain scenario elements
        assert "Scenario" in scenario
        assert "Participants" in scenario
        assert "Context:" in scenario
        
        # Should have participant dialogue
        assert ":" in scenario  # Speaker identification
        
    def test_create_authority_chain_scaffold(self):
        """Test authority chain scaffold generation."""
        chain = self.scaffolder.create_authority_chain_scaffold(
            chain_length=3,
            final_authority="CEO"
        )
        
        # Should contain chain elements
        assert "Authorization Chain" in chain
        assert "Level" in chain
        assert "APPROVED" in chain or "PENDING" in chain
        assert "CEO" in chain
        
    def test_deterministic_with_seed(self):
        """Test deterministic results with same seed."""
        scaffolder1 = RoleScaffolder(seed=456)
        scaffolder2 = RoleScaffolder(seed=456)
        
        email1 = scaffolder1.generate_email_scaffold("technical", "medium")
        email2 = scaffolder2.generate_email_scaffold("technical", "medium")
        
        # Should be identical with same seed
        assert email1 == email2
        
    def test_different_sender_roles(self):
        """Test different sender role types."""
        roles = ["authority", "technical", "business", "external"]
        
        for role in roles:
            email = self.scaffolder.generate_email_scaffold(sender_role=role)
            # Should not crash and should contain email structure
            assert "@" in email
            assert "From:" in email
            
    def test_different_urgency_levels(self):
        """Test different urgency levels."""
        urgency_levels = ["low", "medium", "high", "critical"]
        
        for level in urgency_levels:
            email = self.scaffolder.generate_email_scaffold(urgency_level=level)
            # Should not crash and should contain email structure
            assert "@" in email
            assert "Subject:" in email


if __name__ == "__main__":
    pytest.main([__file__])
