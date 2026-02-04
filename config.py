"""Configuration settings for the RAG system."""

import os

# MinIO Settings
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Local Storage
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# Embedding Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")

# Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval Settings
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))

# Keyword Boosting Settings
KEYWORD_BOOST_WEIGHT = float(os.getenv("KEYWORD_BOOST_WEIGHT", "0.1"))
MAX_KEYWORD_BOOST = float(os.getenv("MAX_KEYWORD_BOOST", "0.3"))
