from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cors_origins: str = "http://localhost:5173"

    # Google Drive
    google_drive_folder_id: str = ""
    google_credentials_file: str = "credentials.json"
    google_token_file: str = "token.json"

    # Embeddings / vector store
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "./data/chroma_db"
    sync_state_file: str = "./data/sync_state.json"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4
    # Chroma cosine distance: 0 = identical, 2 = opposite. Chunks farther than this are dropped.
    max_distance: float = 0.75

    # LLM
    llm_provider: str = "modal"  # modal | openai | groq | gemini | custom
    llm_model: str = "Qwen/Qwen3.8-Max-VL-Thinking"
    llm_base_url: str = ""  # overrides the provider's default URL
    llm_api_key: str = ""  # used by openai / groq / gemini / custom
    llm_timeout_seconds: float = 120.0
    llm_max_tokens: int = 2048
    modal_proxy_token_id: str = ""
    modal_proxy_token_secret: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
