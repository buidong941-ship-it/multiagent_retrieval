from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    milvus_uri: str = "http://localhost:19530"
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    threshold: float = 0.015

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
