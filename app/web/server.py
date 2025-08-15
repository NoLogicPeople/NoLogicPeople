import os
from typing import Dict, List, Optional

# Tame thread counts and OpenMP before heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("FAISS_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.web.session import SessionStore

DATA_PATH_DEFAULT = os.path.join("rag", "scraped_items_v2_tripled.csv")

app = FastAPI(title="e-Devlet Chat Web")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


store = SessionStore(data_path=DATA_PATH_DEFAULT)


@app.get("/api/session/new")
def new_session() -> Dict[str, object]:
    sess = store.get_or_create(session_id=None)
    return {"session_id": sess.session_id, "messages": sess.messages}


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> Dict[str, object]:
    sess = store.get_or_create(session_id=session_id)
    return {"session_id": session_id, "messages": sess.messages}


@app.get("/api/sessions")
def list_sessions() -> Dict[str, object]:
    return {"sessions": store.list_saved_sessions()}


@app.post("/api/session/{session_id}/message")
async def send_message(session_id: str, request: Request) -> Dict[str, object]:
    data = await request.json()
    text: Optional[str] = data.get("text")
    if text is None:
        raise HTTPException(status_code=400, detail="Missing 'text'")
    sess = store.get_or_create(session_id=session_id)
    reply = sess.handle_message(text)
    return {"session_id": session_id, "reply": reply, "messages": sess.messages}


# Minimal HTML page
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>e-Devlet Chat</title>
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <div class="container layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>Geçmiş</h2>
        <button id="new-session" class="small">Yeni</button>
      </div>
      <ul id="history" class="history"></ul>
    </aside>
    <section class="content">
      <header>
        <h1>e-Devlet Chat</h1>
        <div style="display:flex; align-items:center; gap:8px;">
          <button id="tts-toggle" class="small" aria-pressed="true" title="Cevapları sesli oku">🔊</button>
          <span id="session-label"></span>
        </div>
      </header>
      <main id="chat"></main>
      <footer>
        <form id="chat-form">
          <input id="chat-input" type="text" placeholder="Mesajınızı yazın..." autocomplete="off" />
          <button type="button" id="voice-btn" aria-pressed="false" title="Sesle konuş">🎤</button>
          <button type="submit">Gönder</button>
        </form>
      </footer>
    </section>
  </div>
  <script src="/static/app.js"></script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)
