import os
from typing import Optional

try:
    import speech_recognition as sr
except Exception:
    sr = None


class SpeechToText:
    def __init__(self, language: str = "tr") -> None:
        self.language = language
        self.recognizer = sr.Recognizer() if sr else None

    def from_microphone(self, seconds: int = 6) -> str:
        # Fallback if lib not available
        if self.recognizer is None or sr is None:
            return input("(STT kapalı) Lütfen metin girin: ")
        with sr.Microphone() as source:
            print("111")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, phrase_time_limit=seconds)
        try:
            print("222")
            # Use Google Web Speech API for Turkish (no key needed for simple usage)
            return self.recognizer.recognize_google(audio, language="tr-TR")
        except Exception:
            print("333")
            return ""
