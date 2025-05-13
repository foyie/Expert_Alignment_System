import pytest
from src.utils import load_briefs, load_experts
from src.models import ProjectBrief, Expert

def test_brief_loading():
    briefs = load_briefs("data/briefs.json")
    assert len(briefs) > 0
    assert isinstance(briefs[0], ProjectBrief)

def test_expert_loading():
    experts = load_experts("data/experts.json")
    assert len(experts) > 0
    assert isinstance(experts[0], Expert)

def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        load_briefs("nonexistent.json")
