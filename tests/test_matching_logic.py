import pytest
from unittest.mock import MagicMock
from src.matching_system import ProjectMatcher
from src.models import ProjectBrief

@pytest.fixture
def mock_matcher():
    matcher = ProjectMatcher()
    matcher.collection = MagicMock()
    return matcher

def test_basic_matching(mock_matcher):
    test_brief = ProjectBrief(
        title="Test Project",
        description="Test Description",
        required_skills=["Python"],
        domain="Test"
    )
    
    mock_matcher.collection.query.return_value = {
        "ids": [["expert1"]],
        "metadatas": [[{"skills": ["Python"]}]]
    }
    
    matches = mock_matcher.find_matches(test_brief)
    assert len(matches) > 0
    assert "Python" in matches[0].skills