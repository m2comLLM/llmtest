# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Lightweight on-premise Korean RAG (Retrieval-Augmented Generation) system for
internal Q&A without external internet access. Optimized for constrained
hardware (NVIDIA GTX 1660 Super, 6GB VRAM).

## Tech Stack

| Component     | Technology                  | Notes                          |
| ------------- | --------------------------- | ------------------------------ |
| LLM Engine    | Ollama                      | GGUF quantization support      |
| Model         | Gemma-2-2B                  | Korean-capable small model     |
| Embedding     | jhgan/ko-sroberta-multitask | Korean sentence similarity     |
| Vector DB     | ChromaDB                    | Local file-based storage       |
| Storage       | MinIO                       | S3-compatible document storage |
| Orchestration | LangChain                   | Pipeline management            |

## Architecture

**Data Flow:**

1. Admin uploads `.md`, `.csv` files to MinIO bucket
2. App syncs files from MinIO on startup
3. Text chunking → Embedding → ChromaDB indexing
4. User query → Retrieve relevant documents
5. Inject context into LLM → Generate Korean response

## Development Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```

## Code Structure

| File                 | Description                     |
| -------------------- | ------------------------------- |
| `config.py`          | Environment configuration       |
| `minio_client.py`    | MinIO document sync             |
| `document_loader.py` | Document loading & chunking     |
| `embeddings.py`      | Korean sentence embeddings      |
| `vector_store.py`    | ChromaDB operations             |
| `rag_chain.py`       | LangChain RAG chain with Ollama |
| `app.py`             | Gradio web UI (entry point)     |

## Infrastructure Setup

**MinIO (Docker):**

```bash
docker run -d -p 9000:9000 -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  quay.io/minio/minio server /data --console-address ":9001"
```

**Ollama:**

```bash
# Install Ollama and pull the model
ollama pull gemma2:2b
```

**Hardware Requirements:**

- GPU: NVIDIA GTX 1660 Super (6GB VRAM) or better
- RAM: 16GB recommended
- Storage: SSD recommended
