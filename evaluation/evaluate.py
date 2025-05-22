

from typing import List, Dict
import numpy as np
from sklearn.metrics import ndcg_score
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import ProjectBrief

def precision_at_k(expected: List[str], retrieved: List[str], k: int) -> float:
    """Calculate precision@k metric"""
    relevant = set(expected)
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / k

def mean_reciprocal_rank(expected: List[str], retrieved: List[str]) -> float:
    """Calculate MRR metric"""
    for i, item in enumerate(retrieved, 1):
        if item in expected:
            return 1.0 / i
    return 0.0

from sklearn.metrics import ndcg_score
import numpy as np

def calculate_ndcg(expected: List[str], retrieved: List[str], k: int) -> float:
    """Calculate NDCG@k with proper array construction"""
    # Create actual relevance array
    actual = np.array([
        1.0 if doc in expected else 0.0 
        for doc in retrieved[:k]  # First k retrieved docs
    ] + [0.0] * (k - len(retrieved[:k]))  # Pad with zeros if needed
    )[:k]  # Ensure exact length k

    # Create ideal relevance array
    ideal = np.array([1.0] * min(len(expected), k) + [0.0] * (k - len(expected)))[:k]
    
    return ndcg_score([ideal], [actual])
    return ndcg_score([ideal], [actual])


def evaluate_all(test_cases: List[Dict], matcher, k: int = 3) -> Dict:
    """Run full evaluation suite"""
    metrics = {
        "precision@k": [],
        "mrr": [],
        "ndcg": []
    }
    
    for case in test_cases:
        # Load brief from test case
        brief_data = case["project"]
        brief = ProjectBrief(
            title=brief_data["title"],
            description=brief_data["description"],
            required_skills=brief_data["required_skills"],
            domain=brief_data["domain"]
        )
        
        # Get matches and expected results
        matches = [m["name"] for m in matcher.find_matches(brief)]
        expected = case["expected_matches"]
        
        # Calculate metrics
        metrics["precision@k"].append(precision_at_k(expected, matches, k))
        metrics["mrr"].append(mean_reciprocal_rank(expected, matches))
        metrics["ndcg"].append(calculate_ndcg(expected, matches, k))
    
    return {
        "precision@k": np.mean(metrics["precision@k"]),
        "mrr": np.mean(metrics["mrr"]),
        "ndcg": np.mean(metrics["ndcg"]),
        "num_cases": len(test_cases),
        "k": k
    }