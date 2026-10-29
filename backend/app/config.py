from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cors_origins: str = "http://localhost:5173"

    # Google Drive
    google_drive_folder_id: str = ""
    google_credentials_file: str = "credentials.json"
    google_token_file: str = "token.json"

    # Embeddings / vector store
    # "local" (Sentence Transformers) or "openai". Empty = openai if OPENAI_EMBEDDING_MODEL is set, else local.
    embedding_provider: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = ""
    chroma_dir: str = "./data/chroma_db"
    sync_state_file: str = "./data/sync_state.json"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4
    # Chroma cosine distance: 0 = identical, 2 = opposite. Chunks farther than this are dropped.
    max_distance: float = 0.75
    # Earlier messages (user + assistant) sent to the LLM for follow-up questions. 0 = no history.
    chat_history_max_messages: int = 6

    # LLM
    llm_provider: str = "modal"  # modal | openai | groq | gemini | custom
    llm_model: str = "Qwen/Qwen3.8-Max-VL-Thinking"
    llm_base_url: str = ""  # overrides the provider's default URL
    llm_api_key: str = ""  # used by groq / gemini / custom (and openai if OPENAI_API_KEY is empty)
    openai_api_key: str = ""  # used by the openai provider and OpenAI embeddings
    llm_timeout_seconds: float = 120.0
    llm_max_tokens: int = 2048
    modal_proxy_token_id: str = ""
    modal_proxy_token_secret: str = ""

    # Small talk (greetings, thanks, goodbyes) answered without retrieval or the LLM
    # Name used in replies. When empty, replies say "this project".
    project_name: str = ""
    conversational_replies_enabled: bool = True
    semantic_small_talk_enabled: bool = True
    semantic_small_talk_threshold: float = 0.65  # calibrated for all-MiniLM-L6-v2 and text-embedding-3-small
    semantic_small_talk_margin: float = 0.10
    semantic_small_talk_max_words: int = 6
    # Abusive-only messages get a polite reply (naming project_name); abuse mixed into a question is removed
    abusive_language_filter_enabled: bool = True

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider:
            return self.embedding_provider.lower()
        return "openai" if self.openai_embedding_model else "local"

    @property
    def embedding_id(self) -> str:
        """Identifies the embedding space; vectors from different ids must never be mixed."""
        if self.resolved_embedding_provider == "openai":
            return f"openai:{self.openai_embedding_model}"
        return f"local:{self.embedding_model}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
