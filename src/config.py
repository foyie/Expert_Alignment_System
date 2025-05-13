from pydantic import BaseSettings

class Settings(BaseSettings):
    chroma_db_path: str = "chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4-turbo"
    test_data_path: str = "data/test_cases.json"
    
    class Config:
        env_file = ".env"

settings = Settings()