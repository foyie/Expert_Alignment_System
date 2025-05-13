# tests/test_matching.py
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.matching_system import ProjectMatcher
from src.models import ProjectBrief

TEST_CASES = [
    {
        "input": ProjectBrief(
            title="Medical Chatbot",
            description="HIPAA-compliant patient triage system",
            required_skills=["NLP", "Security"],
            domain="Healthcare"
        ),
        "expected": ["MedBot AI v3.2", "Dr. Security Expert"]
    },
    {
        "input": ProjectBrief(
            title="Retail Analytics",
            description="Computer vision for foot traffic analysis",
            required_skills=["CV", "Data Visualization"],
            domain="Retail"
        ),
        "expected": ["VisionAnalytics AI v1.5"]
    }
]

@pytest.fixture
def matcher():
    return ProjectMatcher()

def test_matching(matcher):
    for case in TEST_CASES:
        matches = matcher.find_matches(case["input"])
        assert any(expert.name in case["expected"] for expert in matches[:3])