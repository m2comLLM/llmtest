"""LlamaIndex 임베딩 설정 - 한국어 특화."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config

# 싱글톤 인스턴스
_embed_model: HuggingFaceEmbedding | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Get the Korean embedding model (singleton)."""
    global _embed_model

    if _embed_model is None:
        print("[초기화] 임베딩 모델 로딩 중...")
        _embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL,
            device="cuda",
        )
        print("[초기화] 임베딩 모델 로딩 완료")

    return _embed_model
