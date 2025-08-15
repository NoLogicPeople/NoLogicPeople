import os
from typing import Optional
import json

try:
    # OpenAI Python SDK v1+
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    # Optional direct local inference via llama.cpp python bindings
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


def _fallback_smalltalk(user_text: str, language: str = "tr") -> str:
    text = (user_text or "").strip().lower()
    if any(w in text for w in ["merhaba", "selam", "günaydın", "iyi akşamlar", "iyi gunler", "iyi günler", "gunaydin"]):
        return "Merhaba! Nasıl yardımcı olabilirim?"
    if any(w in text for w in ["nasılsın", "nasilsin", "ne haber", "naber"]):
        return "Teşekkürler, iyiyim. Size nasıl yardımcı olabilirim?"
    return "Merhaba! Size nasıl yardımcı olabilirim?"


def _sanitize(text: str) -> str:
    try:
        import re
        return re.sub(r"\b\d{11}\b", "***********", text or "")
    except Exception:
        return text or ""


def generate_smalltalk_opening(user_text: str, language: str = "tr") -> str:
    """Generate a brief, goal-oriented small talk opening via LLM.

    Preference order:
    1) Local llama.cpp server (OpenAI-compatible API) if available
    2) OpenAI API if configured
    3) Rule-based fallback
    """
    # 1) Try local llama.cpp server (OpenAI-compatible API)
    server_url = os.getenv("LLAMACPP_SERVER_URL", "http://localhost:11434/v1")
    server_model = os.getenv("LLAMACPP_MODEL", "default")
    server_timeout = float(os.getenv("LLAMACPP_TIMEOUT", "12"))
    if requests is not None and server_url:
        try:
            chat_url = f"{server_url.rstrip('/')}/chat/completions"
            payload = {
                "model": server_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Türkçe konuşan, e-Devlet hizmetlerinde yardımcı olan kısa ve öz bir asistansın."
                            " Nazikçe tek cümlede selam ver ve nasıl yardımcı olabileceğini sor."
                            " Konudan sapma, gereksiz bilgi verme."
                        ),
                    },
                    {"role": "user", "content": (user_text or "Kullanıcı sohbete yeni başladı.")},
                ],
                "temperature": 0.2,
                "max_tokens": 64,
                "stream": False,
            }
            print(f"[LLM] Smalltalk via llama.cpp server url='{server_url}' model='{server_model}' input='{_sanitize(user_text or '')}'")
            r = requests.post(chat_url, json=payload, headers={"Content-Type": "application/json"}, timeout=server_timeout)
            if r.ok:
                data = r.json()
                msg = ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "").strip()
                if msg:
                    if len(msg.split()) > 30:
                        msg = msg.split(".")[0].strip() + "."
                    print(f"[LLM] Smalltalk llama.cpp response='{_sanitize(msg)[:120]}'")
                    return msg
            else:
                print(f"[LLM] Smalltalk llama.cpp HTTP {r.status_code}: {_sanitize(r.text)[:200]}")
        except Exception as exc:
            print(f"[LLM] Smalltalk llama.cpp request failed: {exc}")

    # 1b) Try direct local llama.cpp (no server) if available
    if Llama is not None:
        try:
            model_path = _ensure_llamacpp_model()
            if model_path:
                llm = _get_local_llama(model_path)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Türkçe konuşan, e-Devlet hizmetlerinde yardımcı olan kısa ve öz bir asistansın."
                            " Nazikçe tek cümlede selam ver ve nasıl yardımcı olabileceğini sor."
                            " Konudan sapma, gereksiz bilgi verme."
                        ),
                    },
                    {"role": "user", "content": (user_text or "Kullanıcı sohbete yeni başladı.")},
                ]
                out = llm.create_chat_completion(messages=messages, temperature=0.2, max_tokens=64)
                msg = (((out.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or "").strip()
                if msg:
                    if len(msg.split()) > 30:
                        msg = msg.split(".")[0].strip() + "."
                    print(f"[LLM] Smalltalk llama.cpp local response='{_sanitize(msg)[:120]}'")
                    return msg
        except Exception as exc:
            print(f"[LLM] Smalltalk llama.cpp local failed: {exc}")

    # 2) Try OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is not None and api_key:
        try:
            client = OpenAI(api_key=api_key)
            system = (
                "You are a concise Turkish assistant for e-Devlet services."
                " Greet politely in one short sentence and ask how you can help."
                " Keep it on-task, do not add extra information, marketing, or multi-sentence chatter."
            )
            user = (user_text or "").strip()
            print(f"[LLM] Smalltalk via OpenAI model='{os.getenv('OPENAI_MODEL','gpt-4o-mini')}' input='{_sanitize(user) or ''}'")
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                max_tokens=50,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user or "Kullanıcı sohbete yeni başladı."},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            if len(content.split()) > 30:
                content = content.split(".")[0].strip() + "."
            print(f"[LLM] Smalltalk OpenAI response='{_sanitize(content)[:120]}'")
            return content or _fallback_smalltalk(user_text, language)
        except Exception:
            print("[LLM] Smalltalk OpenAI request failed; falling back")
            pass

    # 3) Fallback
    print("[LLM] Smalltalk fallback (rule-based) used")
    return _fallback_smalltalk(user_text, language)


__all__ = ["generate_smalltalk_opening"]


# -----------------------------
# Local llama.cpp helpers
# -----------------------------
_LLAMA_SINGLETON: Optional[object] = None

def _ensure_llamacpp_model() -> Optional[str]:
    """Ensure a GGUF model exists locally. Download if LLAMACPP_MODEL_URL is set.

    Returns the model path if available, else None.
    """
    import os
    from pathlib import Path
    import requests as _requests  # type: ignore

    model_path = os.getenv("LLAMACPP_MODEL_PATH")
    if not model_path:
        # Default small chat model (TinyLlama 1.1B Chat Q4_K_M)
        model_path = os.path.join("models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    p = Path(model_path)
    if p.exists():
        return str(p)

    url = os.getenv("LLAMACPP_MODEL_URL", "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[LLM] Downloading GGUF model from {url} -> {p}")
        with _requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(p, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("[LLM] Model downloaded")
        return str(p)
    except Exception as exc:
        print(f"[LLM] Model download failed: {exc}")
        return None


def _get_local_llama(model_path: str):
    global _LLAMA_SINGLETON
    if _LLAMA_SINGLETON is not None:
        return _LLAMA_SINGLETON  # type: ignore
    n_ctx = int(os.getenv("LLAMACPP_N_CTX", "4096"))
    n_threads = int(os.getenv("LLAMACPP_N_THREADS", "4"))
    _LLAMA_SINGLETON = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
    return _LLAMA_SINGLETON
