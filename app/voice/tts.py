from typing import Optional

try:
    from gtts import gTTS
    import tempfile
    import os
    import platform
    import subprocess
except Exception:
    gTTS = None


class TextToSpeech:
    def __init__(self, language: str = "tr") -> None:
        self.language = language

    def speak(self, text: str) -> None:
        if gTTS is None:
            # Fallback: no TTS available
            return
        tts = gTTS(text=text, lang="tr")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        # Try to play audio cross-platform
        try:
            if platform.system() == "Darwin":
                # Use afplay on macOS
                subprocess.run(["afplay", temp_path], check=False)
            elif platform.system() == "Windows":
                os.startfile(temp_path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["mpg123", temp_path], check=False)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
