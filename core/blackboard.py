from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


BlackboardItemType = Literal["fact", "subgoal", "claim", "evidence", "action", "result"]
BlackboardItemStatus = Literal["proposed", "verified", "rejected", "resolved", "archived"]


class BlackboardItem(BaseModel):
    """Minimal blackboard schema for structured collaboration memory."""

    item_id: str = Field(default_factory=lambda: str(uuid4()))
    type: BlackboardItemType
    content: str
    source_agent: str
    round_id: int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: BlackboardItemStatus = "proposed"
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def mark_status(self, status: BlackboardItemStatus) -> None:
        self.status = status
        self.updated_at = utc_now_iso()


class BlackboardStore(BaseModel):
    """In-memory blackboard wrapper with export helpers."""

    items: List[BlackboardItem] = Field(default_factory=list)

    def add_item(
        self,
        *,
        item_type: BlackboardItemType,
        content: str,
        source_agent: str,
        round_id: int,
        confidence: float = 0.5,
        status: BlackboardItemStatus = "proposed",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BlackboardItem:
        item = BlackboardItem(
            type=item_type,
            content=content,
            source_agent=source_agent,
            round_id=round_id,
            confidence=confidence,
            status=status,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self.items.append(item)
        return item

    def get_item(self, item_id: str) -> Optional[BlackboardItem]:
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def update_item_status(self, item_id: str, status: BlackboardItemStatus) -> bool:
        item = self.get_item(item_id)
        if item is None:
            return False
        item.mark_status(status)
        return True

    def list_by_round(self, round_id: int) -> List[BlackboardItem]:
        return [item for item in self.items if item.round_id == round_id]

    def as_records(self) -> List[Dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.items]

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> "BlackboardStore":
        store = cls()
        store.items = [BlackboardItem(**record) for record in records]
        return store

