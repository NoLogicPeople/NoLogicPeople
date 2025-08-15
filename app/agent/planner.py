import os
import json
from typing import Any, Dict, List, Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


def _call_llamacpp_json(messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        server_url: Optional[str] = None,
                        timeout_sec: float = 6.0) -> Optional[Dict[str, Any]]:
    if requests is None:
        return None
    base = (server_url or os.getenv('LLAMACPP_SERVER_URL', 'http://localhost:11434/v1')).rstrip('/')
    url = f"{base}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model or os.getenv("LLAMACPP_MODEL", "default"),
        "messages": messages,
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 128,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout_sec)
        if not r.ok:
            print(f"[LLM] Planner llama.cpp HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        content = (((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or "").strip()
        if not content:
            return None
        try:
            plan = json.loads(content)
        except Exception:
            print(f"[LLM] Planner non-JSON content: {content[:200]}")
            return None
        print(f"[LLM] Planner result: {str(plan)[:200]}")
        return plan
    except Exception:
        print("[LLM] Planner llama.cpp request failed")
        return None


def plan_next(user_text: str,
              state: str,
              available_services: Optional[List[Dict[str, Any]]] = None,
              selected_service: Optional[Dict[str, Any]] = None,
              language: str = "tr") -> Dict[str, Any]:
    """Ask a local llama.cpp model (OpenAI API compatible) to choose next intent/tool.

    Returns a dict with keys like:
      {"intent": "select_service"|"request_info"|"choose_action"|"switch_topic"|"cancel"|"resume"|"consent_yes"|"consent_no",
       "target_index": 0-based index when relevant,
       "action_keyword": optional string}
    """
    services = available_services or []
    service_names = [s.get("service_name", "") for s in services]

    messages = [
        {
            "role": "system",
            "content": (
                "Sen bir konuşma planlayıcısısın. Sadece JSON döndür. Şema: "
                "{\"intent\": one_of['select_service','request_info','choose_action','switch_topic','cancel','resume','consent_yes','consent_no'], "
                "\"target_index\": optional int (0-tabanlı), \"action_keyword\": optional string}. "
                "Bağlam: state = '" + state + "'. Hizmetler: " + ", ".join(service_names) + ". "
                "Kısa ve net karar ver. Açıklama yazma."
            ),
        },
        {
            "role": "user",
            "content": (user_text or "").strip(),
        },
    ]

    result = _call_llamacpp_json(messages)
    if isinstance(result, dict) and result.get("intent"):
        return result

    # Fallback heuristic if LLM not available
    lw = (user_text or "").lower()
    if any(k in lw for k in {"iptal", "vazgeç", "vazgec", "cancel"}):
        return {"intent": "cancel"}
    if any(k in lw for k in {"devam", "resume", "kaldığımız yerden", "kaldigimiz yerden"}):
        return {"intent": "resume"}
    if any(k in lw for k in {"konu değiştir", "konu degistir", "yeni konu", "baska konu", "farklı bir konu", "farkli bir konu"}):
        return {"intent": "switch_topic"}
    # Try to detect info request pattern like 'bilgi 2'
    import re
    m = re.search(r"bilgi\s*(\d+)", lw)
    if m:
        try:
            idx = int(m.group(1)) - 1
            return {"intent": "request_info", "target_index": idx}
        except Exception:
            pass
    return {"intent": "unknown"}


__all__ = ["plan_next"]
