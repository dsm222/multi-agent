from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentState(BaseModel):
    """Unified state object for multi-agent orchestration rounds."""

    task_id: str
    task_text: str
    benchmark_name: str

    round_id: int = 0
    round_goal: str = ""
    active_agents: List[str] = Field(default_factory=list)

    messages: List[Dict[str, Any]] = Field(default_factory=list)
    blackboard: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    router_edges: List[Dict[str, Any]] = Field(default_factory=list)

    candidate_answers: List[Dict[str, Any]] = Field(default_factory=list)
    vote_result: Dict[str, Any] = Field(default_factory=dict)
    verifier_result: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    termination_reason: str = ""

    token_usage: Dict[str, int] = Field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    trace_events: List[Dict[str, Any]] = Field(default_factory=list)

    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @classmethod
    def bootstrap(cls, task_id: str, task_text: str, benchmark_name: str) -> "AgentState":
        return cls(task_id=task_id, task_text=task_text, benchmark_name=benchmark_name)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def set_round(self, round_id: int, round_goal: str, active_agents: List[str]) -> None:
        self.round_id = round_id
        self.round_goal = round_goal
        self.active_agents = list(active_agents)
        self.touch()

    def append_message(self, sender: str, content: str, receiver: Optional[str] = None) -> None:
        self.messages.append(
            {
                "timestamp": utc_now_iso(),
                "round_id": self.round_id,
                "sender": sender,
                "receiver": receiver,
                "content": content,
            }
        )
        self.touch()

    def add_claim(self, claim: Dict[str, Any]) -> None:
        self.claims.append(claim)
        self.touch()

    def add_evidence(self, evidence_item: Dict[str, Any]) -> None:
        self.evidence.append(evidence_item)
        self.touch()

    def add_router_edge(self, edge: Dict[str, Any]) -> None:
        self.router_edges.append(edge)
        self.touch()

    def add_candidate_answer(self, candidate: Dict[str, Any]) -> None:
        self.candidate_answers.append(candidate)
        self.touch()

    def set_vote_result(self, result: Dict[str, Any]) -> None:
        self.vote_result = result
        self.touch()

    def set_verifier_result(self, result: Dict[str, Any]) -> None:
        self.verifier_result = result
        self.touch()

    def set_final_answer(self, answer: str, termination_reason: str) -> None:
        self.final_answer = answer
        self.termination_reason = termination_reason
        self.touch()

    def set_blackboard_records(self, records: List[Dict[str, Any]]) -> None:
        self.blackboard = records
        self.touch()

    def set_trace_events(self, events: List[Dict[str, Any]]) -> None:
        self.trace_events = events
        self.touch()

    def to_record(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

