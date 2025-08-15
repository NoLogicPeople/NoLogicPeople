import json
import os
from typing import Any, Dict, List


class ConversationMemory:
    def __init__(self, storage_path: str = "./memory.json") -> None:
        self.storage_path = storage_path
        self.events: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.events = json.load(f)
            except Exception:
                self.events = []

    def add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.events, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def last(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.events[-n:]


__all__ = ["ConversationMemory"]
