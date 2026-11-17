"""Validation result types."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    ERROR = "error"      # blocks execution
    WARNING = "warning"  # allowed, but surfaced/logged


class Violation(BaseModel):
    code: str                      # machine-readable, e.g. 'target_not_reachable'
    severity: Severity
    message: str
    step_index: Optional[int] = None
    subject: Optional[str] = None  # object id the violation concerns


class ValidationResult(BaseModel):
    ok: bool = True                # True iff no ERROR violations
    violations: List[Violation] = Field(default_factory=list)
    checked: List[str] = Field(default_factory=list)  # names of checks run

    def add(self, v: Violation) -> None:
        self.violations.append(v)
        if v.severity == Severity.ERROR:
            self.ok = False

    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.ERROR]

    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]

    def summary(self) -> str:
        if self.ok and not self.violations:
            return "ok (no violations)"
        parts = [f"{len(self.errors())} error(s)", f"{len(self.warnings())} warning(s)"]
        return ("OK, " if self.ok else "BLOCKED, ") + ", ".join(parts)
