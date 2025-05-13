from pydantic import BaseModel

class ProjectBrief(BaseModel):
    title: str
    description: str
    required_skills: list[str]
    domain: str

class Expert(BaseModel):
    name: str
    type: str  # "AI" or "Human"
    skills: list[str]
    experience: int
    expertise: list[str]