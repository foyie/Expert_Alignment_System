
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from .models import ProjectBrief
from .chroma_setup import get_chroma_collection
from .config import OLLAMA_MODEL, OLLAMA_BASE_URL

class ProjectMatcher:
    """Core matching system with RAG pipeline using Ollama"""
    
    def __init__(self):
        self.collection = get_chroma_collection()
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.3,
            format="json"  # Ensure JSON output
        )
        self.min_confidence = 0.65

    def find_matches(self, brief: ProjectBrief, top_k: int = 5) -> List[Dict]:
        """Main matching pipeline"""
        # First-stage: Vector similarity search
        vector_results = self._vector_search(brief)
        
        # Second-stage: LLM reranking
        reranked_results = self._rerank_with_llm(brief, vector_results)
        
        return self._format_results(reranked_results[:top_k])

    def _vector_search(self, brief: ProjectBrief) -> List[Dict]:
        """Perform vector database query"""
        query_text = f"{brief.title}: {brief.description}"
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=20,
            where={"domain": {"$in": [brief.domain]}},
            include=["metadatas", "distances"]
        )
        
        return self._process_vector_results(results)

    def _process_vector_results(self, results: Dict) -> List[Dict]:
        """Process raw ChromaDB results"""
        processed = []
        for name, meta, distance in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            processed.append({
                "name": name,
                "type": meta["type"],
                "skills": meta["skills"],
                "expertise": meta["expertise"],
                "experience": meta.get("experience", 0),
                "distance": distance
            })
        return processed

    def _rerank_with_llm(self, brief: ProjectBrief, candidates: List[Dict]) -> List[Dict]:
        """Refine results using local LLM via Ollama"""
        prompt_template = ChatPromptTemplate.from_template(
            """<s>[INST] Analyze these candidates for a project brief:
            
            Project Title: {title}
            Description: {description}
            Required Skills: {skills}
            
            Candidates:
            {candidates}
            
            Score each candidate 1-10 based on:
            - Skill match (30%)
            - Domain expertise (40%)
            - Experience relevance (30%)
            
            Return ONLY a JSON array with format:
            [{{"name": "...", "score": ..., "reason": "..."}}] [/INST]</s>"""
        )
        
        chain = prompt_template | self.llm
        response = chain.invoke({
            "title": brief.title,
            "description": brief.description,
            "skills": ", ".join(brief.required_skills),
            "candidates": "\n".join(
                [f"- {c['name']}: Skills: {', '.join(c['skills'])}, Expertise: {', '.join(c['expertise'])}" 
                 for c in candidates]
            )
        })
        
        return self._parse_llm_response(response)

    # def _parse_llm_response(self, response: str) -> List[Dict]:
    #     """Parse LLM response with enhanced error handling"""
    #     import json
    #     try:
    #         # Clean response for JSON parsing
    #         cleaned = response.replace("```json", "").replace("```", "").strip()
    #         return json.loads(cleaned)
    #     except json.JSONDecodeError as e:
    #         print(f"Failed to parse LLM response: {e}")
    #         print(f"Raw response: {response}")
    #         return []
    #     except Exception as e:
    #         print(f"Unexpected error: {e}")
    #         return []

    # def _format_results(self, results: List[Dict]) -> List[Dict]:
    #     """Format final output"""
    #     return sorted(
    #         results,
    #         key=lambda x: x.get("score", 0),
    #         reverse=True
    #     )
    # src/matching_system.py (updated)
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM response with enhanced error handling"""
        import json
        try:
            # Clean response for JSON parsing
            cleaned = response.replace("```json", "").replace("```", "").strip()
            
            # Handle both list and single object responses
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return [parsed]  # Wrap single result in list
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Raw response: {response}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """Format final output with safety checks"""
        valid_results = []
        for res in results:
            if isinstance(res, dict) and "name" in res and "score" in res:
                valid_results.append({
                    "name": res["name"],
                    "type": res.get("type", "Unknown"),
                    "score": float(res["score"]),
                    "reason": res.get("reason", "No reason provided")
                })
        return sorted(valid_results, key=lambda x: x["score"], reverse=True)