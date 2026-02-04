"ChatEXAONE" "Gradio web application for the LlamaIndex RAG system." ""

import config
import gradio as gr
from document_loader import load_documents_from_dir
from embeddings import get_embed_model
from minio_client import get_client, sync_documents
from rag_chain import chat as rag_chat
from rag_chain import get_llm, reset_chat_engine
from vector_store import add_documents, clear_store, get_index


def preload_models():
    """앱 시작 시 모든 모델을 미리 로딩."""
    print("=" * 50)
    print("LlamaIndex RAG 시스템 초기화 중...")
    print("=" * 50)
    get_embed_model()  # 임베딩 모델 로딩
    get_index()  # VectorStoreIndex 생성
    get_llm()  # LLM 연결
    print("=" * 50)
    print("초기화 완료!")
    print("=" * 50)


def initialize_system():
    """Initialize the RAG system by syncing and indexing documents."""
    status_messages = []

    # Sync documents from MinIO
    try:
        client = get_client()
        downloaded = sync_documents(client)
        status_messages.append(f"MinIO에서 {len(downloaded)}개 파일 동기화 완료")
    except Exception as e:
        status_messages.append(f"MinIO 동기화 실패: {e}")
        status_messages.append("로컬 docs/ 폴더의 파일을 사용합니다")

    # Load documents (TextNodes)
    nodes = load_documents_from_dir()
    if not nodes:
        return "문서를 찾을 수 없습니다. docs/ 폴더에 .md, .csv 또는 .jsonl 파일을 추가하세요."

    status_messages.append(f"{len(nodes)}개 문서 로드 완료")

    # Clear and re-index
    clear_store()
    add_documents(nodes)
    status_messages.append("VectorStoreIndex 인덱싱 완료")

    return "\n".join(status_messages)


def chat(message: str, history: list) -> str:
    """Process a chat message and return the response."""
    if not message.strip():
        return "질문을 입력해주세요."

    try:
        response = rag_chat(message)
        return response
    except Exception as e:
        return f"오류가 발생했습니다: {e}"


def reset_conversation():
    """Reset the conversation history."""
    reset_chat_engine()
    return "대화 기록이 초기화되었습니다."


def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    with gr.Blocks(title="사내 RAG 시스템 (LlamaIndex)") as app:
        gr.Markdown("# 🏢 사내 문서 Q&A 시스템")
        gr.Markdown("LlamaIndex 기반 RAG 시스템 - 메타데이터 필터링 지원")

        with gr.Row():
            init_btn = gr.Button("📚 문서 동기화 및 인덱싱", variant="primary")
            reset_btn = gr.Button("🔄 대화 초기화", variant="secondary")

        init_status = gr.Textbox(label="상태", lines=4, interactive=False)

        init_btn.click(fn=initialize_system, outputs=init_status)
        reset_btn.click(fn=reset_conversation, outputs=init_status)

        gr.Markdown("---")

        chatbot = gr.ChatInterface(
            fn=chat,
            title="💬 질문하기",
            description="연도, 월, 카테고리(심포지엄/워크숍/스쿨 등) 필터링을 지원합니다.",
            examples=[
                "2025년 심포지엄 목록을 알려줘",
                "2025년 4월 행사 알려줘",
                "양재 aT센터에서 하는 행사는?",
            ],
        )

    return app


if __name__ == "__main__":
    preload_models()
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
