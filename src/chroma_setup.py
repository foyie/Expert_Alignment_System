import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    
# src/chroma_setup.py
import chromadb
from chromadb.utils import embedding_functions
from src.config import EMBEDDING_MODEL_NAME, CHROMA_DB_PATH

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

def get_chroma_collection():
    """Initialize and return ChromaDB collection with persistent storage"""
    
    # Initialize Chroma client with persistent storage
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Create embedding function using Sentence Transformers
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Get or create the collection
    collection = client.get_or_create_collection(
        name="expert_profiles",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
    )
    
    return collection

def update_chroma_collection(experts: list):
    """Update collection with latest expert profiles"""
    collection = get_chroma_collection()
    
    # Generate document representations
    documents = [
        f"Name: {e.name}\nType: {e.type}\nSkills: {', '.join(e.skills)}\nExpertise: {', '.join(e.expertise)}"
        for e in experts
    ]
    
    # Upsert data in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = [e.name for e in experts[i:i+batch_size]]
        batch_metadata = [
            {
                "type": e.type,
                "skills": e.skills,
                "expertise": e.expertise,
                "experience": e.experience
            }
            for e in experts[i:i+batch_size]
        ]
        
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_metadata,
            ids=batch_ids
        )