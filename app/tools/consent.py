from typing import Optional


class Consent:
    def __init__(self, stt, tts, language: str = "tr") -> None:
        self.stt = stt
        self.tts = tts
        self.language = language

    def ask(self, prompt: str = "Onaylıyor musunuz?", seconds: int = 4) -> bool:
        try:
            self.tts.speak(prompt)
        except Exception:
            pass
        try:
            reply = self.stt.from_microphone(seconds=seconds).strip().lower()
        except Exception:
            reply = input(f"{prompt} ").strip().lower()
        normalized = reply.replace("ı", "i").replace("İ", "i")
        return any(x in normalized for x in ["evet", "onay", "kabul", "olur", "tamam", "yes"]) and not any(
            x in normalized for x in ["hayir", "hayır", "no", "iptal", "vazgec", "vazgeç"]
        )
