from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import json

class MatchEvaluator:
    """LLM-based match quality evaluation using Ollama"""
    
    def __init__(self, model_name: str = "llama2"):
        self.llm = Ollama(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(
            """[INST] Evaluate how well this expert matches the project:
            
            Project: {project}
            Expert: {expert}
            
            Consider skills ({skills_weight}%), domain fit ({domain_weight}%), 
            and experience ({exp_weight}%). Return JSON format:
            {{"score": 0-10, "reason": "..."}} [/INST]"""
        )
        
    def evaluate_match(self, project: dict, expert: dict, 
                      skills_weight: int = 40, 
                      domain_weight: int = 40,
                      exp_weight: int = 20) -> dict:
        chain = self.prompt | self.llm
        response = chain.invoke({
            "project": json.dumps(project),
            "expert": json.dumps(expert),
            "skills_weight": skills_weight,
            "domain_weight": domain_weight,
            "exp_weight": exp_weight
        })
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"score": 0, "reason": "Invalid response format"}
        
 