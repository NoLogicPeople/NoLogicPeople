import os
import sys
from typing import List, Dict, Optional

from app.memory.memory import ConversationMemory
from app.security.guard import Guard
from app.tools.identity import is_valid_tckn
from app.tools.services import execute_service
from app.tools.rag_tool import RagClient
from app.tools.consent import Consent
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech


class Orchestrator:
    def __init__(
        self,
        data_path: str,
        memory_path: str = "./memory.json",
        language: str = "tr",
        top_k: int = 3,
    ) -> None:
        # Ensure rag is importable by rag_tool
        self.rag = RagClient(data_path=data_path)
        self.guard = Guard()
        self.memory = ConversationMemory(storage_path=memory_path)
        self.stt = SpeechToText(language=language)
        self.tts = TextToSpeech(language=language)
        self.consent = Consent(stt=self.stt, tts=self.tts, language=language)
        self.language = language
        self.top_k = top_k

    def say_and_print(self, text: str) -> None:
        print(f"\n[YANIT]\n{text}\n")
        try:
            self.tts.speak(text)
        except Exception as exc:
            print(f"[Uyarı] TTS başarısız: {exc}")

    def listen(self, prompt: Optional[str] = None, seconds: int = 100000) -> str:
        if prompt:
            self.say_and_print(prompt)
        try:
            text = self.stt.from_microphone(seconds=seconds)
            if text:
                print(f"[Anlaşılan] {text}")
                return text
        except Exception as exc:
            print(f"[Uyarı] Mikrofon/STT başarısız: {exc}")
        # Fallback to text input
        return input("Lütfen metin olarak girin: ").strip()

    def run_once(self) -> None:
        # Giriş: kullanıcının isteğini dinle
        query = self.listen("Hangi e-Devlet hizmetiyle ilgili yardımcı olmamı istersiniz?")
        if not query:
            self.say_and_print("Sizi anlayamadım. Lütfen tekrar deneyin.")
            return

        # Güvenlik
        safe, reason = self.guard.check_prompt_safety(query)
        if not safe:
            self.say_and_print(f"Bu istek güvenlik politikalarımıza uygun değil: {reason}")
            self.memory.add_event({"type": "blocked", "reason": reason, "query": query})
            return

        self.memory.add_event({"type": "query", "text": query})

        # RAG: servis önerileri
        try:
            services = self.rag.recommend_services(query=query, top_k=self.top_k)
        except Exception as exc:
            self.say_and_print(f"Bilgi kaynağına erişirken bir hata oluştu: {exc}")
            return

        if not services:
            self.say_and_print("İlgili bir hizmet bulamadım. Daha farklı ifade edebilir misiniz?")
            return

        # Önerileri sun ve seçim al
        options_text = "\n".join([f"{i+1}. {s['service_name']}" for i, s in enumerate(services)])
        selection_prompt = (
            "Aşağıdaki hizmetlerden hangisini istiyorsunuz? Numarayı söyleyin veya yazın.\n" + options_text
        )
        selection_raw = self.listen(selection_prompt, seconds=5)
        selected_index: int = -1
        try:
            selected_index = int(selection_raw.strip()) - 1
        except Exception:
            # Basit eşleşme ile dene
            lowered = selection_raw.strip().lower()
            for i, s in enumerate(services):
                if s["service_name"].lower() in lowered:
                    selected_index = i
                    break

        if selected_index < 0 or selected_index >= len(services):
            self.say_and_print("Geçerli bir seçim algılayamadım. İşlemi iptal ediyorum.")
            return

        selected_service = services[selected_index]
        self.memory.add_event({"type": "service_selected", "service": selected_service})

        # Kimlik doğrulama: TCKN iste
        tckn = self.listen(
            f"{selected_service['service_name']} için kimlik doğrulaması gerekiyor. Lütfen 11 haneli T.C. kimlik numaranızı söyleyin veya yazın.",
            seconds=6,
        )
        tckn_digits = "".join(ch for ch in tckn if ch.isdigit())
        if not is_valid_tckn(tckn_digits):
            self.say_and_print("Geçersiz T.C. kimlik numarası girdiniz. İşlemi iptal ediyorum.")
            self.memory.add_event({"type": "identity_failed", "tckn": tckn_digits})
            return

        self.memory.add_event({"type": "identity_verified", "tckn": tckn_digits[-4:]})

        # Onay
        if not self.consent.ask("Bu işlemi başlatmak istiyor musunuz? 'Evet' veya 'Hayır' deyin."):
            self.say_and_print("Onay verilmedi. İşlem iptal edildi.")
            self.memory.add_event({"type": "consent_denied"})
            return

        self.memory.add_event({"type": "consent_granted"})

        # Mock servis yürütme
        result = execute_service(
            service_name=selected_service["service_name"],
            payload={
                "query": query,
                "tckn_last4": tckn_digits[-4:],
            },
        )

        # Yanıt oluştur
        response_text = (
            f"'{selected_service['service_name']}' için talebinizi aldım ve işlemi simüle ederek tamamladım. "
            f"Özet bilgi: {selected_service.get('service_full_text', '')[:240]}"
        )
        if result.get("status") == "ok":
            response_text += " Başka bir konuda yardımcı olmamı ister misiniz?"
        else:
            response_text = "İşlem sırasında bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

        self.memory.add_event({"type": "response", "text": response_text})
        self.say_and_print(response_text)


__all__ = ["Orchestrator"]
