"""Shared schema base classes."""

from pydantic import BaseModel, ConfigDict


class ResumeAgentModel(BaseModel):
    """Strict base model shared by all domain schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
    )
