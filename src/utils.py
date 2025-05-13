import json
from pathlib import Path
from typing import List
from .models import ProjectBrief, Expert

def load_briefs(file_path: str) -> List[ProjectBrief]:
    """Load project briefs from JSON file"""
    return _load_data(file_path, ProjectBrief)

def load_experts(file_path: str) -> List[Expert]:
    """Load expert profiles from JSON file"""
    return _load_data(file_path, Expert)

def _load_data(file_path: str, model_class):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with path.open('r') as f:
        data = json.load(f)
    
    return [model_class(**item) for item in data]

def save_evaluation(results: dict, output_path: str):
    """Save evaluation results to markdown"""
    with open(output_path, 'w') as f:
        f.write("# Evaluation Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for k, v in results.items():
            f.write(f"| {k} | {v:.2f} |\n")