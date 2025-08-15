from typing import Dict, Any, List


def list_service_actions(service: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return possible actions for a service (informational tool).

    Example actions: obtain, verify, update. This is a mock list; in real systems,
    this would be data-driven per service.
    """
    name = (service.get("service_name") or "").lower()
    actions = [
        {"action": "al", "label": "Belgeyi alma"},
        {"action": "dogrula", "label": "Belgeyi doğrulama"},
        {"action": "guncelle", "label": "Belgeyi güncelleme"},
    ]
    # Could tailor per service name in future
    return actions


def execute_service(service_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mock executor for all services.

    In real deployment, dispatch based on `service_name` and perform API calls.
    Here we only simulate with a success response.
    """
    return {
        "status": "ok",
        "service": service_name,
        "payload": payload,
        "message": f"Servis '{service_name}' simüle edildi."
    }


__all__ = ["execute_service", "list_service_actions"]
