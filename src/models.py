
    
from pydantic import BaseModel

class ProjectBrief(BaseModel):
    """Data model for project briefs"""
    title: str
    description: str
    required_skills: list[str]
    domain: str

class Expert(BaseModel):
    """Data model for expert/agent profiles"""
    name: str
    type: str  # "AI" or "Human"
    skills: list[str]
    experience: int
    expertise: list[str]