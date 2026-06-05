"""Pydantic request/response models."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    project_number: str = Field(..., min_length=1)
    document_type: Optional[str] = None
    originator: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    template_id: Optional[int] = None
    guiding_prompt: Optional[str] = None
    is_revision: bool = False
    prior_run_id: Optional[str] = None
    reference_mode: str = "fresh"          # existing | fresh | both
    reference_project: Optional[str] = None


class TemplateIn(BaseModel):
    name: str = Field(..., min_length=1)
    instructions: str = Field(..., min_length=1)
    description: Optional[str] = None
    default_doc_type: Optional[str] = None
    severity_scheme: Optional[str] = None
    categories: Optional[list[str]] = None


class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    instructions: Optional[str] = None
    description: Optional[str] = None
    default_doc_type: Optional[str] = None
    severity_scheme: Optional[str] = None
    categories: Optional[list[str]] = None
    archived: Optional[bool] = None


class RunOut(BaseModel):
    id: str
    status: str
    stage: str = ""
    project_number: str
    document_type: Optional[str] = None
    originator: Optional[str] = None
    is_revision: bool = False
    reference_mode: Optional[str] = None
    ocrrag_project: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    uploads: list[dict] = Field(default_factory=list)
