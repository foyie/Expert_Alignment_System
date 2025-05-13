from typing import List
# from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ProjectBrief, Expert
from src.chroma_setup import get_chroma_collection
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.utils import load_briefs
from src.utils import load_experts


import time

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

class ProjectMatcher:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from langchain.llms import Ollama
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = Ollama(model="mistral")
        self.collection = get_chroma_collection()

    def _create_prompt(self, brief, results):
        context_snippets = [doc['metadata']['content'] for doc in results['documents'][0]]

        prompt = f"""
            You are an intelligent project matcher. Based on the following project brief and available agents/experts, identify the best matches.

            Project Title: {brief.title}
            Description: {brief.description}
            Required Skills: {', '.join(brief.required_skills)}
            Domain: {brief.domain}

            Available Agents/Experts:
            {chr(10).join(f"- {snippet}" for snippet in context_snippets)}

            Please return the top {len(context_snippets)} most relevant experts or agents for this project, along with a brief justification for each.
            """
        return prompt

    def find_matches(self, brief, top_k: int = 5):
        query_embedding = self.embed_model.encode(brief.description)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2
        )

        prompt = self._create_prompt(brief, results)
        return self.llm.invoke(prompt)
