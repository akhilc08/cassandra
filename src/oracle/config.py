"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Oracle application settings."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "oracle_dev_password"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # NewsAPI
    newsapi_key: str = ""

    # Polymarket
    polymarket_api_base: str = "https://gamma-api.polymarket.com"

    # Claude API
    anthropic_api_key: str = ""

    # Twitter/X
    twitter_bearer_token: str = ""

    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""

    # Government APIs
    congress_api_key: str = ""
    courtlistener_api_key: str = ""

    # Audio/Whisper
    whisper_model_size: str = "base"
    audio_sources_config: str = ""

    # Ingestion
    news_poll_interval_seconds: int = 900  # 15 min
    market_poll_interval_seconds: int = 60

    # Forecaster credentials. Declared without the ORACLE_ prefix because the
    # OpenAI SDK and GitHub Actions both use the bare OPENAI_API_KEY name; the
    # forecaster reads os.environ directly, this entry just stops the .env file
    # from tripping the extra-inputs guard.
    openai_api_key: str = ""

    model_config = {"env_prefix": "ORACLE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
