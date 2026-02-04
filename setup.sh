#!/bin/bash
set -e

echo "=== Korean RAG System Setup ==="

# 1. Python 가상환경 생성 및 활성화
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists"
fi

source venv/bin/activate
echo "Virtual environment activated"

# 2. Python 의존성 설치
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Ollama 설치 확인
echo "[3/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed"
fi

# 4. LLM 모델 다운로드
echo "[4/4] Pulling EXAONE 3.5 32B model..."
ollama pull exaone3.5:32b

echo ""
echo "=== Setup Complete ==="
echo "Run the app with: source venv/bin/activate && python app.py"
