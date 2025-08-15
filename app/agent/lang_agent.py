from typing import Any, Dict, Optional, List, Type
import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr

# Tools: wrap existing functions
from app.tools.identity import is_valid_tckn
from app.tools.services import execute_service, list_service_actions
from app.tools.rag_tool import RagClient
from app.tools.smalltalk import smalltalk_opening


class ValidateTCKNInput(BaseModel):
    tckn: str = Field(..., description="11 haneli T.C. kimlik numarası")


class ValidateTCKNTool(BaseTool):
    name: str = "validate_tckn"
    description: str = "Validate a Turkish TCKN (11-digit). Input: the number as string. Returns true/false."
    args_schema: Type[ValidateTCKNInput] = ValidateTCKNInput

    def _run(self, tckn: str, **kwargs) -> bool:  # type: ignore[override]
        return bool(is_valid_tckn((tckn or "").strip()))

    async def _arun(self, tckn: str, **kwargs) -> bool:  # type: ignore[override]
        return self._run(tckn)


class ListServiceActionsInput(BaseModel):
    service: Dict[str, Any] = Field(..., description="Hizmet nesnesi (JSON)")


class ListServiceActionsTool(BaseTool):
    name: str = "list_service_actions"
    description: str = "List available actions for a given service. Input: JSON with {service}. Returns a list of actions."
    args_schema: Type[ListServiceActionsInput] = ListServiceActionsInput

    def _run(self, service: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:  # type: ignore[override]
        return list_service_actions(service or {})

    async def _arun(self, service: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:  # type: ignore[override]
        return self._run(service)


class ExecuteServiceInput(BaseModel):
    service_name: str = Field("", description="Hizmet adı")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Yürütme ayrıntıları")


class ExecuteServiceTool(BaseTool):
    name: str = "execute_service"
    description: str = (
        "Execute a selected service action. Input JSON keys: service_name, tckn_last4, service, action, action_label."
    )
    args_schema: Type[ExecuteServiceInput] = ExecuteServiceInput

    def _run(self, service_name: str = "", payload: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        if payload is None:
            payload = {}
        return execute_service(
            service_name=service_name or (payload or {}).get("service_name", ""),
            payload=payload,
        )

    async def _arun(self, service_name: str = "", payload: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return self._run(service_name=service_name, payload=payload)


class RagRetrieveInput(BaseModel):
    query: str = Field(..., description="Kullanıcı sorgusu")
    top_k: int = Field(3, description="Kaç sonuç dönsün")


class RagRetrieveTool(BaseTool):
    name: str = "rag_retrieve"
    description: str = (
        "Retrieve top-k services related to a user query. Input JSON: {query: str, top_k?: int}."
    )
    args_schema: Type[RagRetrieveInput] = RagRetrieveInput

    _rag: RagClient = PrivateAttr()
    _top_k_default: int = PrivateAttr(default=3)

    def __init__(self, data_path: str, top_k_default: int = 3) -> None:
        super().__init__()
        self._rag = RagClient(data_path=data_path)
        self._top_k_default = int(top_k_default)

    def _run(self, query: str, top_k: int = None, **kwargs) -> List[Dict[str, Any]]:  # type: ignore[override]
        q = query or ""
        k = int(top_k or self._top_k_default)
        return self._rag.recommend_services(query=q, top_k=k)

    async def _arun(self, query: str, top_k: int = None, **kwargs) -> List[Dict[str, Any]]:  # type: ignore[override]
        return self._run(query=query, top_k=top_k)


def _build_llm() -> ChatOpenAI:
    # Use llama.cpp OpenAI-compatible server if present; else OpenAI fallback
    base_url = os.getenv("LLAMACPP_SERVER_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "sk-no-key")
    model = os.getenv("LLAMACPP_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    params: Dict[str, Any] = {"model": model, "temperature": 0.2}
    if base_url:
        params["base_url"] = base_url
    return ChatOpenAI(**params, api_key=api_key)


def build_agent(data_path: str) -> AgentExecutor:
    tools: List[BaseTool] = [
        ValidateTCKNTool(),
        ListServiceActionsTool(),
        ExecuteServiceTool(),
        RagRetrieveTool(data_path=data_path),
    ]

    system = (
        "You are a concise Turkish assistant for e-Devlet services. "
        "You have tools to validate TCKN, retrieve services, list actions, and execute actions. "
        "Always keep responses short. Ask clarifying questions when needed."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    llm = _build_llm()
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)
