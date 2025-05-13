import chromadb
from chromadb.utils import embedding_functions

def get_chroma_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="experts",
        embedding_function=sentence_transformer_ef
    )