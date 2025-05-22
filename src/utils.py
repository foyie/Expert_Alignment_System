import json
from typing import List
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import ProjectBrief, Expert
from src.chroma_setup import get_chroma_collection
from src.matching_system import ProjectMatcher



def load_data(file_path: str) -> List[dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_briefs(file_path: str) -> List[ProjectBrief]:
    """Load and validate project briefs"""
    return [ProjectBrief(**data) for data in load_data(file_path)]

def load_experts(file_path: str) -> List[Expert]:
    """Load and validate expert profiles"""
    return [Expert(**data) for data in load_data(file_path)]

def initialize_chroma(experts: List[Expert]):
    """Initialize ChromaDB with expert data"""
    collection = get_chroma_collection()
    
    collection.add(
        documents=[f"{e.name}: {', '.join(e.skills)} | {', '.join(e.expertise)}" 
                   for e in experts],
        metadatas=[{"type": e.type, "skills": e.skills, "expertise": e.expertise}
                   for e in experts],
        ids=[e.name for e in experts]
    )

# def initialize_system():
    # """Initialize all components with Ollama check"""
    # from .ollama_setup import pull_model_if_needed
    # from .config import OLLAMA_MODEL
    
    # # Ensure Ollama model is available
    # pull_model_if_needed(OLLAMA_MODEL)
    
    # # Rest of initialization
    # experts = load_experts("data/experts.json")
    # initialize_chroma(experts)
    # return ProjectMatcher()

def initialize_chroma(experts):
    """Initialize ChromaDB with expert data"""
    collection = get_chroma_collection()
    
    collection.add(
        documents=[f"{e.name}: {', '.join(e.skills)} | {', '.join(e.expertise)}" 
                   for e in experts],
        metadatas=[{
            "type": e.type,
            "skills": ", ".join(e.skills),  # Convert list to string
            "expertise": ", ".join(e.expertise),  # Convert list to string
            "experience": str(e.experience)
        } for e in experts],
        ids=[e.name for e in experts]
    )
