from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from core.blackboard import BlackboardStore
from core.state import AgentState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceEvent(BaseModel):
    """Minimal trace event schema used for replay and failure analysis."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=utc_now_iso)
    round_id: int
    node_name: str
    input_summary: str
    output_summary: str
    tool_name: Optional[str] = None
    cost: float = 0.0
    latency_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceRecorder(BaseModel):
    """In-memory event collector with JSONL read/write support."""

    events: List[TraceEvent] = Field(default_factory=list)

    def add_event(
        self,
        *,
        round_id: int,
        node_name: str,
        input_summary: str,
        output_summary: str,
        tool_name: Optional[str] = None,
        cost: float = 0.0,
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            round_id=round_id,
            node_name=node_name,
            input_summary=input_summary,
            output_summary=output_summary,
            tool_name=tool_name,
            cost=cost,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def as_records(self) -> List[Dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self.events]

    def write_jsonl(self, output_path: Path, state_snapshot: Optional[Dict[str, Any]] = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for event in self.as_records():
                f.write(json.dumps({"record_type": "trace_event", **event}, ensure_ascii=False))
                f.write("\n")
            if state_snapshot is not None:
                f.write(
                    json.dumps(
                        {
                            "record_type": "state_snapshot",
                            "timestamp": utc_now_iso(),
                            "state": state_snapshot,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
        return output_path

    @classmethod
    def read_jsonl(cls, input_path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def run_empty_round_jsonl(output_path: Path) -> Path:
    """
    Execute a one-round empty flow and persist complete JSONL records.
    """

    state = AgentState.bootstrap(
        task_id="stage3-empty-round",
        task_text="No-op smoke flow for schema validation.",
        benchmark_name="internal_smoke",
    )
    state.set_round(round_id=1, round_goal="Validate state/blackboard/trace schema.", active_agents=["manager"])

    blackboard = BlackboardStore()
    subgoal = blackboard.add_item(
        item_type="subgoal",
        content="Run one empty round without tool usage.",
        source_agent="manager",
        round_id=1,
        confidence=1.0,
        status="verified",
    )
    blackboard.add_item(
        item_type="result",
        content="Empty round completed.",
        source_agent="manager",
        round_id=1,
        confidence=1.0,
        status="verified",
        parent_id=subgoal.item_id,
    )
    state.set_blackboard_records(blackboard.as_records())

    state.add_claim(
        {
            "claim_id": str(uuid4()),
            "round_id": 1,
            "agent_id": "manager",
            "text": "A no-op round can be traced and replayed.",
            "confidence": 1.0,
        }
    )
    state.add_evidence(
        {
            "evidence_id": str(uuid4()),
            "round_id": 1,
            "agent_id": "manager",
            "text": "All required schema fields are filled.",
            "confidence": 1.0,
        }
    )
    state.add_candidate_answer(
        {
            "agent_id": "manager",
            "answer": "EMPTY_ROUND_OK",
            "confidence": 1.0,
            "round_id": 1,
        }
    )
    state.set_vote_result(
        {
            "method": "majority_vote",
            "winner": "EMPTY_ROUND_OK",
            "votes": {"EMPTY_ROUND_OK": 1},
            "round_id": 1,
        }
    )
    state.set_verifier_result(
        {
            "status": "pass",
            "conflicts": [],
            "notes": "No contradiction detected in empty round.",
            "round_id": 1,
        }
    )
    state.set_final_answer(answer="EMPTY_ROUND_OK", termination_reason="empty_round_completed")

    recorder = TraceRecorder()
    recorder.add_event(
        round_id=1,
        node_name="round_start",
        input_summary="Initialize state, blackboard, and trace containers.",
        output_summary="Round context initialized.",
        latency_ms=3,
    )
    recorder.add_event(
        round_id=1,
        node_name="consensus",
        input_summary="One candidate answer from manager.",
        output_summary="majority_vote winner=EMPTY_ROUND_OK.",
        latency_ms=2,
    )
    recorder.add_event(
        round_id=1,
        node_name="verifier",
        input_summary="Vote winner and structured records.",
        output_summary="Verification passed with no conflicts.",
        latency_ms=2,
    )
    recorder.add_event(
        round_id=1,
        node_name="round_end",
        input_summary="State finalized.",
        output_summary="Empty round complete.",
        latency_ms=1,
    )
    state.set_trace_events(recorder.as_records())

    return recorder.write_jsonl(output_path=output_path, state_snapshot=state.to_record())

