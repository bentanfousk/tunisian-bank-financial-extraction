"""Stable API contracts; frozen research schemas remain in schemas/."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


JobStage = Literal[
    "queued", "loading_pdf", "extracting_pages", "embedding", "indexing",
    "retrieving", "building_context", "generating", "retrying", "validating",
    "completed", "failed",
]


class ExtractionAccepted(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobEvent(BaseModel):
    status: JobStage
    message: str
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class JobState(BaseModel):
    job_id: str
    status: JobStage
    filename: str
    bank: str
    year: int
    page_count: int
    result: dict[str, Any] | None = None
    error: str | None = None
    events: list[JobEvent]
