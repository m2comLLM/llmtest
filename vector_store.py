"""LlamaIndex VectorStoreIndex 관리 - ChromaDB 기반."""

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from embeddings import get_embed_model

# 싱글톤 인스턴스
_vector_store: ChromaVectorStore | None = None
_index: VectorStoreIndex | None = None
_chroma_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Get ChromaDB client (singleton)."""
    global _chroma_client

    if _chroma_client is None:
        # WSL2 호환성 문제로 인메모리 모드 사용
        _chroma_client = chromadb.EphemeralClient()

    return _chroma_client


def get_vector_store() -> ChromaVectorStore:
    """Get ChromaDB vector store (singleton)."""
    global _vector_store

    if _vector_store is None:
        print("[초기화] ChromaDB 연결 중...")
        client = get_chroma_client()
        chroma_collection = client.get_or_create_collection("documents")
        _vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print("[초기화] ChromaDB 연결 완료 (인메모리 모드)")

    return _vector_store


def get_index() -> VectorStoreIndex:
    """Get or create the VectorStoreIndex (singleton)."""
    global _index

    if _index is None:
        print("[초기화] VectorStoreIndex 생성 중...")
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = get_embed_model()

        _index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model,
        )
        print("[초기화] VectorStoreIndex 생성 완료")

    return _index


def add_documents(nodes: list[TextNode]) -> None:
    """Add documents to the index."""
    if not nodes:
        return

    index = get_index()
    index.insert_nodes(nodes)
    print(f"[인덱싱] {len(nodes)}개 문서 추가 완료")


def get_all_by_filter(filters: dict | None) -> list[TextNode]:
    """Get ALL documents matching the filter (no similarity limit)."""
    client = get_chroma_client()
    collection = client.get_or_create_collection("documents")

    # ChromaDB에서 필터에 맞는 모든 문서 조회
    # 빈 딕셔너리는 ChromaDB에서 오류 발생하므로 None으로 처리
    if filters:
        results = collection.get(
            where=filters,
            include=["documents", "metadatas"],
        )
    else:
        # 필터 없이 전체 조회
        results = collection.get(
            include=["documents", "metadatas"],
        )

    nodes = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"]):
            text = results["documents"][i] if results["documents"] else ""
            metadata = results["metadatas"][i] if results["metadatas"] else {}

            node = TextNode(
                text=text,
                id_=doc_id,
                metadata=metadata,
            )
            nodes.append(node)

    return nodes


def clear_store() -> None:
    """Clear the vector store and reset index."""
    global _vector_store, _index, _chroma_client

    if _chroma_client is not None:
        try:
            _chroma_client.reset()
        except Exception:
            pass
        _chroma_client = None

    _vector_store = None
    _index = None
    print("[초기화] 벡터 스토어 초기화 완료")
