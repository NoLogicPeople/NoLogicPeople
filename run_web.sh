#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_INIT_AT_FORK=FALSE

# Unbuffered Python logs so LLM prints appear immediately
export PYTHONUNBUFFERED=1

# Load local env if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

# Local LLM (llama.cpp server) defaults; override via .env or shell
# Use OpenAI-compatible server like llama.cpp's server or llama-cpp-python's server
export LLAMACPP_SERVER_URL="${LLAMACPP_SERVER_URL:-http://localhost:11434/v1}"
export LLAMACPP_MODEL="${LLAMACPP_MODEL:-default}"
export LLAMACPP_TIMEOUT="${LLAMACPP_TIMEOUT:-12}"
export LLAMACPP_AUTOSTART="${LLAMACPP_AUTOSTART:-1}"
export LLAMACPP_N_CTX="${LLAMACPP_N_CTX:-4096}"
export LLAMACPP_N_THREADS="${LLAMACPP_N_THREADS:-4}"

# Determine Python interpreter (stick to one for installs and running)
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

# Autostart a local llama.cpp server if not reachable and AUTOSTART=1
if [ "${LLAMACPP_AUTOSTART}" = "1" ]; then
  BASE_URL="${LLAMACPP_SERVER_URL}"
  # quick health check
  if ! curl -sS -m 2 "${BASE_URL%/}/models" >/dev/null 2>&1; then
    echo "[run] No llama.cpp server detected at ${BASE_URL}. Autostarting..."
    # Ensure a GGUF model exists (downloads default TinyLlama if missing)
    MODEL_PATH=$("${PYTHON_BIN}" - <<'PY'
from app.tools.llm import _ensure_llamacpp_model
p=_ensure_llamacpp_model()
print(p or "")
PY
)
    if [ -z "${MODEL_PATH}" ]; then
      echo "[run] Failed to ensure GGUF model. Exiting." >&2
      exit 1
    fi
    echo "[run] Starting llama-cpp-python server with model: ${MODEL_PATH}"
    # Start server in background
    "${PYTHON_BIN}" -m llama_cpp.server \
      --model "${MODEL_PATH}" \
      --model-alias "${LLAMACPP_MODEL}" \
      --host 127.0.0.1 \
      --port 11434 \
      --n_ctx "${LLAMACPP_N_CTX}" \
      --n_threads "${LLAMACPP_N_THREADS}" \
      >/dev/null 2>&1 &
    SERVER_PID=$!
    echo ${SERVER_PID} > .llama_cpp_server.pid || true
    # Wait for readiness (max ~20s)
    ATTEMPTS=40
    until curl -sS -m 1 "${BASE_URL%/}/models" >/dev/null 2>&1 || [ $ATTEMPTS -le 0 ]; do
      sleep 0.5
      ATTEMPTS=$((ATTEMPTS-1))
    done
    if ! curl -sS -m 2 "${BASE_URL%/}/models" >/dev/null 2>&1; then
      echo "[run] llama.cpp server failed to start." >&2
      # Do not exit; app can still run with OpenAI fallback or local bindings
    else
      echo "[run] llama.cpp server is up at ${BASE_URL}"
    fi
  fi
fi

# Optional: OpenAI fallback (only used if OPENAI_API_KEY is set)
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# Ensure app is importable and dependencies present
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
"${PYTHON_BIN}" - <<'PY'
import sys
to_check = [
    ('langchain', 'langchain'),
    ('langchain_community', 'langchain-community'),
    ('langchain_openai', 'langchain-openai'),
    ('pydantic', 'pydantic'),
    ('llama_cpp', 'llama-cpp-python'),
    ('fastapi', 'fastapi'),
    ('uvicorn', 'uvicorn'),
    ('sentence_transformers', 'sentence-transformers'),
    ('faiss', 'faiss-cpu'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
]
to_install = []
for mod, pkg in to_check:
    try:
        __import__(mod)
    except Exception:
        to_install.append(pkg)
if to_install:
    print('[run] Installing missing deps (pip):', ' '.join(to_install))
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + to_install)
PY

# Start web server with the same Python interpreter
exec "${PYTHON_BIN}" -m uvicorn app.web.server:app --host 127.0.0.1 --port 8000 --reload