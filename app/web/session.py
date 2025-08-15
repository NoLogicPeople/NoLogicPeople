import os
import uuid
from typing import Any, Dict, List, Optional

from app.memory.memory import ConversationMemory
from app.security.guard import Guard
from app.tools.identity import is_valid_tckn
from app.tools.services import execute_service, list_service_actions
from app.tools.smalltalk import smalltalk_opening
from app.tools.rag_tool import RagClient
from app.agent.lang_agent import build_agent


class ChatSession:
    """Stateful chat session that mirrors the CLI orchestrator flow, step-by-step.

    States:
      - start: expecting initial query
      - awaiting_selection: waiting for a service selection
      - awaiting_action: waiting for an operation/action for the selected service
      - awaiting_tckn: waiting for 11-digit TCKN
      - awaiting_consent: waiting for yes/no
    """

    def __init__(
        self,
        session_id: str,
        data_path: str,
        storage_dir: str = "./web_sessions",
        top_k: int = 3,
        language: str = "tr",
    ) -> None:
        self.session_id = session_id
        self.state: str = "start"
        self.language = language
        self.top_k = top_k

        os.makedirs(storage_dir, exist_ok=True)
        memory_path = os.path.join(storage_dir, f"{session_id}.json")
        self.memory = ConversationMemory(storage_path=memory_path)
        self.guard = Guard()
        self.rag = RagClient(data_path=data_path)
        self.agent = build_agent(data_path=data_path)

        self.available_services: List[Dict[str, str]] = []
        self.selected_service: Optional[Dict[str, str]] = None
        self.tckn_last4: Optional[str] = None
        self.selected_action: Optional[Dict[str, str]] = None
        # Allow suspending a flow on context change and resuming later
        self.suspended_flow: Optional[Dict[str, object]] = None

        # Keep a simple chat transcript: list of {role, content}
        self.messages: List[Dict[str, str]] = []
        # Reconstruct prior transcript from memory events if present
        for ev in getattr(self.memory, "events", []):
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == "query" and isinstance(ev.get("text"), str):
                self.messages.append({"role": "user", "content": ev["text"]})
            elif ev.get("type") == "response" and isinstance(ev.get("text"), str):
                self.messages.append({"role": "assistant", "content": ev["text"]})
        # Mark greeted if we have any assistant message already
        self.has_greeted: bool = any(m.get("role") == "assistant" for m in self.messages)

    def _append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Persist high-level events too
        if role == "user":
            self.memory.add_event({"type": "query", "text": content})
        else:
            self.memory.add_event({"type": "response", "text": content})

    def handle_message(self, text: str) -> str:
        user_text = (text or "").strip()
        self._append("user", user_text)

        # Step 1: safety checks always
        safe, reason = self.guard.check_prompt_safety(user_text)
        if not safe and self.state == "start":
            reply = f"Bu istek güvenlik politikalarımıza uygun değil: {reason}"
            self._append("assistant", reply)
            return reply

        # Global intents: resume/cancel/context switch keywords
        lowered = user_text.lower()
        resume_keywords = {"devam", "devam et", "kaldığımız yerden", "kaldigimiz yerden", "resume", "continue"}
        cancel_keywords = {"iptal", "vazgeç", "vazgec", "cancel", "abort", "durdur"}
        switch_keywords = {"konu değiştir", "konu degistir", "başka konu", "baska konu", "yeni konu", "farklı bir konu", "farkli bir konu"}

        if any(k in lowered for k in cancel_keywords) and self.state != "start":
            # Cancel current flow and reset
            self._record_event({"type": "flow_cancelled", "state": self.state})
            self._reset_flow()
            reply = "Mevcut işlemi iptal ettim. Yeni konunuz nedir?"
            self._append("assistant", reply)
            return reply

        if any(k in lowered for k in resume_keywords) and self.suspended_flow is not None:
            # Resume previously suspended flow
            self._resume_suspended_flow()
            reply = self._prompt_for_current_state()
            self._append("assistant", reply)
            return reply

        explicit_switch = any(k in lowered for k in switch_keywords)

        # State machine
        if self.state == "start":
            # First, small talk opening (concise), then recommend services
            opening = smalltalk_opening(user_text, language=self.language) if not self.has_greeted else ""
            try:
                services = self.rag.recommend_services(query=user_text, top_k=self.top_k)
            except Exception as exc:
                reply = (f"{opening}\n" if opening else "") + f"Bilgi kaynağına erişirken bir hata oluştu: {exc}"
                self._append("assistant", reply)
                if opening:
                    self.has_greeted = True
                return reply

            if not services:
                reply = (f"{opening}\n" if opening else "") + "İlgili bir hizmet bulamadım. Lütfen daha farklı ifade edin."
                self._append("assistant", reply)
                if opening:
                    self.has_greeted = True
                return reply

            self.available_services = services
            options_text = "\n".join([f"{i+1}. {s['service_name']}" for i, s in enumerate(services)])
            reply = (
                (f"{opening}\n" if opening else "") +
                "İlgili olabilecek hizmetler:\n" + options_text +
                "\nBir hizmet hakkında bilgi isterseniz 'bilgi 2' gibi yazabilirsiniz."
            )
            self.state = "awaiting_selection"
            self._append("assistant", reply)
            if opening:
                self.has_greeted = True
            return reply

        if self.state == "awaiting_selection":
            if explicit_switch:
                self._suspend_current_flow(reason="context_switch")
                return self._handle_new_topic(user_text)
            # LangChain agent can answer info requests or suggest next steps
            try:
                agent_reply = self.agent.invoke({
                    "input": f"Kullanıcı mesaji: '{user_text}'. Mevcut hizmetler: {[s['service_name'] for s in self.available_services]}."
                })
                text = str(agent_reply.get("output", "")).strip() if isinstance(agent_reply, dict) else str(agent_reply).strip()
                if text:
                    self._append("assistant", text)
                    return text
            except Exception:
                pass
            selected_index: int = -1
            # Try number
            try:
                selected_index = int(user_text) - 1
            except Exception:
                # Try name contains
                lowered = user_text.lower()
                for i, s in enumerate(self.available_services):
                    if s.get("service_name", "").lower() in lowered:
                        selected_index = i
                        break

            # Info-only request: e.g., 'bilgi 2' or '2 hakkında bilgi'
            if selected_index < 0:
                import re
                m = re.search(r"bilgi\s*(\d+)|^(\d+)\s*hakkında\s*bilgi", lowered)
                idx = None
                if m:
                    idx = m.group(1) or m.group(2)
                if idx is not None:
                    try:
                        i = int(idx) - 1
                        if 0 <= i < len(self.available_services):
                            svc = self.available_services[i]
                            actions = list_service_actions(svc)
                            actions_text = "\n".join([f"- {a['label']}" for a in actions])
                            info = (
                                f"{svc['service_name']} hakkında bilgi:\n" +
                                (svc.get('service_description') or svc.get('service_full_text',''))[:400] +
                                "\nOlası işlemler:\n" + actions_text +
                                "\nKullanmak isterseniz lütfen belirtin (ör. '2 ile belgeyi doğrula')."
                            )
                            self._append("assistant", info)
                            return info
                    except Exception:
                        pass
            if selected_index < 0 or selected_index >= len(self.available_services):
                reply = "Geçerli bir seçim algılayamadım. Lütfen listedeki numarayı yazın veya 'bilgi 2' yazın."
                self._append("assistant", reply)
                return reply

            self.selected_service = self.available_services[selected_index]
            self.memory.add_event({"type": "service_selected", "service": self.selected_service})

            # Ask for desired action first (informational step)
            actions = list_service_actions(self.selected_service)
            actions_text = "\n".join([f"{i+1}. {a['label']}" for i, a in enumerate(actions)])
            reply = (
                f"{self.selected_service['service_name']} için hangi işlemi yapmak istersiniz?\n" +
                actions_text +
                "\nSadece bilgi almak isterseniz 'bilgi' yazabilirsiniz."
            )
            self.state = "awaiting_action"
            self._append("assistant", reply)
            return reply

        if self.state == "awaiting_action":
            if explicit_switch or self._looks_like_new_topic(user_text):
                self._suspend_current_flow(reason="context_switch")
                return self._handle_new_topic(user_text)
            # Agent can propose an action
            try:
                agent_reply = self.agent.invoke({
                    "input": f"Seçili hizmet: '{(self.selected_service or {}).get('service_name','')}'. Kullanıcı mesajı: '{user_text}'."
                })
                text = str(agent_reply.get("output", "")).strip() if isinstance(agent_reply, dict) else str(agent_reply).strip()
                if text and "kimlik" in text.lower():
                    # assume agent decided action and asked for TCKN
                    self.state = "awaiting_tckn"
                    self._append("assistant", text)
                    return text
            except Exception:
                pass
            # Info-only request
            lw = lowered
            if any(k in lw for k in {"bilgi", "detay", "açıklama", "aciklama"}):
                svc = self.selected_service or {}
                actions = list_service_actions(svc)
                actions_text = "\n".join([f"- {a['label']}" for a in actions])
                info = (
                    f"{svc.get('service_name','')} hakkında kısa bilgi:\n" +
                    (svc.get('service_description') or svc.get('service_full_text',''))[:400] +
                    "\nOlası işlemler:\n" + actions_text +
                    "\nDevam etmek isterseniz lütfen işlem numarasını veya adını belirtin."
                )
                self._append("assistant", info)
                return info
            # Try to map to action index or keyword
            actions = list_service_actions(self.selected_service or {})
            # number
            try:
                idx = int(user_text.strip()) - 1
                if 0 <= idx < len(actions):
                    self.selected_action = actions[idx]
                else:
                    idx = -1
            except Exception:
                idx = -1
            if idx == -1:
                # keyword match
                for a in actions:
                    if a["action"] in lowered or a["label"].split()[0].lower() in lowered:
                        self.selected_action = a
                        break
            if not self.selected_action:
                reply = "Geçerli bir işlem algılayamadım. Lütfen listedeki numarayı veya adını yazın."
                self._append("assistant", reply)
                return reply
            # Proceed to identity for execution
            reply = (
                f"Seçilen işlem: {self.selected_action['label']}. "
                "Kimlik doğrulaması gerekiyor: Lütfen 11 haneli T.C. kimlik numaranızı yazın."
            )
            self.state = "awaiting_tckn"
            self._append("assistant", reply)
            return reply

        if self.state == "awaiting_tckn":
            if explicit_switch or self._looks_like_new_topic(user_text):
                self._suspend_current_flow(reason="context_switch")
                return self._handle_new_topic(user_text)
            digits = "".join(ch for ch in user_text if ch.isdigit())
            if not is_valid_tckn(digits):
                reply = "Geçersiz T.C. kimlik numarası girdiniz. Lütfen 11 haneli doğru numarayı yazın."
                self._append("assistant", reply)
                return reply

            self.tckn_last4 = digits[-4:]
            self.memory.add_event({"type": "identity_verified", "tckn": self.tckn_last4})

            action_label = (self.selected_action or {}).get('label', '')
            reply = ("Bu işlemi başlatmak istiyor musunuz? 'Evet' veya 'Hayır' yazın." if not action_label else 
                     f"'{action_label}' işlemini başlatmak istiyor musunuz? 'Evet' veya 'Hayır' yazın.")
            self.state = "awaiting_consent"
            self._append("assistant", reply)
            return reply

        if self.state == "awaiting_consent":
            lowered = user_text.lower()
            if explicit_switch or self._looks_like_new_topic(user_text):
                self._suspend_current_flow(reason="context_switch")
                return self._handle_new_topic(user_text)
            if lowered in {"evet", "e", "yes", "y"}:
                # Execute
                result = execute_service(
                    service_name=self.selected_service.get("service_name", ""),
                    payload={
                        "tckn_last4": self.tckn_last4,
                        "service": self.selected_service,
                        "action": (self.selected_action or {}).get("action"),
                        "action_label": (self.selected_action or {}).get("label"),
                    },
                )
                if result.get("status") == "ok":
                    summary = (self.selected_service or {}).get("service_full_text", "")[:240]
                    reply = (
                        f"'{self.selected_service.get('service_name','')}' için '{(self.selected_action or {}).get('label','')}' talebinizi aldım ve işlemi simüle ederek tamamladım. "
                        f"Özet bilgi: {summary} Başka bir konuda yardımcı olmamı ister misiniz?"
                    )
                else:
                    reply = "İşlem sırasında bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

                # Reset to new turn
                self.state = "start"
                self.available_services = []
                self.selected_service = None
                self.tckn_last4 = None
                self.selected_action = None
                self._append("assistant", reply)
                return reply
            else:
                reply = "Onay verilmedi. İşlem iptal edildi. Başka bir konuda yardımcı olabilir miyim?"
                # Reset
                self.state = "start"
                self.available_services = []
                self.selected_service = None
                self.tckn_last4 = None
                self.selected_action = None
                self._append("assistant", reply)
                return reply

        # Fallback
        reply = "Anlayamadım. Lütfen yeniden ifade eder misiniz?"
        self._append("assistant", reply)
        return reply

    # --- Helpers for context switching and flow control ---
    def _record_event(self, event: Dict[str, object]) -> None:
        try:
            self.memory.add_event(event)
        except Exception:
            pass

    def _reset_flow(self) -> None:
        self.state = "start"
        self.available_services = []
        self.selected_service = None
        self.tckn_last4 = None
        self.selected_action = None

    def _suspend_current_flow(self, reason: str = "context_switch") -> None:
        self.suspended_flow = {
            "state": self.state,
            "available_services": self.available_services,
            "selected_service": self.selected_service,
            "tckn_last4": self.tckn_last4,
            "selected_action": self.selected_action,
        }
        self._record_event({"type": "flow_suspended", "reason": reason, "snapshot": {
            "state": self.state,
            "selected_service": (self.selected_service or {}).get("service_name", ""),
            "selected_action": (self.selected_action or {}).get("action", ""),
        }})
        self._reset_flow()
        # Inform user
        self._append("assistant", "Mevcut işlemi askıya aldım. Yeni konunuza geçiyorum.")

    def _resume_suspended_flow(self) -> None:
        snap = self.suspended_flow or {}
        self.state = snap.get("state", "start")  # type: ignore[assignment]
        self.available_services = snap.get("available_services", [])  # type: ignore[assignment]
        self.selected_service = snap.get("selected_service")  # type: ignore[assignment]
        self.tckn_last4 = snap.get("tckn_last4")  # type: ignore[assignment]
        self.selected_action = snap.get("selected_action")  # type: ignore[assignment]
        self.suspended_flow = None
        self._record_event({"type": "flow_resumed"})

    def _prompt_for_current_state(self) -> str:
        if self.state == "awaiting_selection":
            options_text = "\n".join([f"{i+1}. {s['service_name']}" for i, s in enumerate(self.available_services)])
            return "Kaldığımız yerden devam edelim. Lütfen seçim yapın:\n" + options_text
        if self.state == "awaiting_action":
            actions = list_service_actions(self.selected_service or {})
            actions_text = "\n".join([f"{i+1}. {a['label']}" for i, a in enumerate(actions)])
            return "Kaldığımız yerden devam: Lütfen işlem seçin.\n" + actions_text
        if self.state == "awaiting_tckn":
            return "Kaldığımız yerden devam: Lütfen 11 haneli T.C. kimlik numaranızı yazın."
        if self.state == "awaiting_consent":
            return "Kaldığımız yerden devam: Bu işlemi başlatmak istiyor musunuz? 'Evet' veya 'Hayır' yazın."
        return "Kaldığımız yerden devam edebiliriz. Nasıl yardımcı olabilirim?"

    def _looks_like_new_topic(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        # If text contains letters and spaces and is not a simple yes/no or number-like, treat as new topic
        yes_no = {"evet", "e", "hayır", "hayir", "h", "yes", "y", "no", "n"}
        if t in yes_no:
            return False
        # Non-digit content indicates a probable topic/question
        has_letters = any(ch.isalpha() for ch in t)
        is_number_like = t.isdigit() or all(ch.isdigit() or ch in {" ", "-"} for ch in t)
        return has_letters and not is_number_like and len(t) >= 3

    def _handle_new_topic(self, user_text: str) -> str:
        try:
            services = self.rag.recommend_services(query=user_text, top_k=self.top_k)
        except Exception as exc:
            reply = f"Bilgi kaynağına erişirken bir hata oluştu: {exc}"
            self._append("assistant", reply)
            return reply
        if not services:
            reply = "İlgili bir hizmet bulamadım. Lütfen daha farklı ifade edin."
            self._append("assistant", reply)
            return reply
        self.available_services = services
        options_text = "\n".join([f"{i+1}. {s['service_name']}" for i, s in enumerate(services)])
        reply = "Yeni konunuz için aşağıdaki hizmetlerden hangisini istiyorsunuz? Numarayı yazın veya adını belirtin.\n" + options_text
        self.state = "awaiting_selection"
        self._append("assistant", reply)
        return reply


class SessionStore:
    """Manages ChatSession instances and persistence locations per session id."""

    def __init__(self, data_path: str, storage_dir: str = "./web_sessions") -> None:
        self.data_path = data_path
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: Optional[str]) -> ChatSession:
        if not session_id:
            session_id = uuid.uuid4().hex
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(
                session_id=session_id,
                data_path=self.data_path,
                storage_dir=self.storage_dir,
            )
        return self._sessions[session_id]

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        session = self.get_or_create(session_id)
        return session.messages

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def list_saved_session_ids(self) -> List[str]:
        ids: List[str] = []
        try:
            for name in os.listdir(self.storage_dir):
                if name.endswith('.json'):
                    ids.append(name[:-5])
        except Exception:
            pass
        # Recently modified first
        try:
            ids.sort(key=lambda i: os.path.getmtime(os.path.join(self.storage_dir, f"{i}.json")), reverse=True)
        except Exception:
            pass
        return ids

    def list_saved_sessions(self) -> List[Dict[str, str]]:
        """Return metadata for saved sessions: id, title, last_message, updated_at."""
        sessions: List[Dict[str, str]] = []
        for sid in self.list_saved_session_ids():
            path = os.path.join(self.storage_dir, f"{sid}.json")
            title = f"Oturum {sid[:8]}"
            last_message = ""
            updated_at = ""
            try:
                # Inspect memory events to build title/preview
                from datetime import datetime
                with open(path, "r", encoding="utf-8") as f:
                    import json
                    events = json.load(f)
                # Title: first user query
                for ev in events:
                    if isinstance(ev, dict) and ev.get("type") == "query" and isinstance(ev.get("text"), str):
                        title = ev["text"][:60] or title
                        break
                # Last message preview: prefer assistant
                for ev in reversed(events):
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") in {"response", "query"} and isinstance(ev.get("text"), str):
                        last_message = ev["text"][:80]
                        break
                # Timestamp
                ts = os.path.getmtime(path)
                updated_at = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            sessions.append({
                "id": sid,
                "title": title,
                "last_message": last_message,
                "updated_at": updated_at,
            })
        return sessions
