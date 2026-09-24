"""
In-repo operator-support audit against the vendored supported_operators.json.

Self-contained (no dependency on the sima-model-surgery skill) so it runs in the
Nx cloud. Platform target is Modalix, so bfloat16 is a valid support policy.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import onnx

log = logging.getLogger("model_surgery.audit")

_DB_PATH = Path(__file__).resolve().parent / "data" / "supported_operators.json"

# Graph-only ops that are folded/handled at compile time — not real MLA compute
# ops, so absence from the support DB is not a compatibility concern.
_IGNORE_OPS = {"Constant", "Identity"}


@dataclass
class AuditReport:
    dtype: str
    release: str | None
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsupported and not self.unknown

    def summary(self) -> str:
        return (f"ops audit [{self.dtype}] release={self.release}: "
                f"{len(self.supported)} supported, "
                f"{len(self.unsupported)} unsupported, {len(self.unknown)} unknown")


def _load_db() -> dict:
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"vendored support DB missing: {_DB_PATH}")
    return json.loads(_DB_PATH.read_text(encoding="utf-8"))


def _supported(entry: dict | None, dtype: str) -> bool:
    if entry is None:
        return False
    if dtype == "any":
        return any(str(entry.get(k, "")).strip().upper() == "Y" for k in ("int8", "bfloat16"))
    return str(entry.get(dtype, "")).strip().upper() == "Y"


def audit_model(model: onnx.ModelProto | str, dtype: str = "bfloat16") -> AuditReport:
    """Audit op types in a model (path or proto) against the support DB."""
    if isinstance(model, str):
        model = onnx.load(model)
    db = _load_db()
    ops = db.get("operators", {})

    counts: Counter = Counter(n.op_type for n in model.graph.node)
    report = AuditReport(dtype=dtype, release=db.get("release"))
    for op in sorted(counts):
        if op in _IGNORE_OPS:
            continue
        entry = ops.get(op)
        if entry is None:
            report.unknown.append(op)
        elif _supported(entry, dtype):
            report.supported.append(op)
        else:
            report.unsupported.append(op)
    log.info(report.summary())
    if report.unsupported:
        log.warning("unsupported ops (%s): %s", dtype, ", ".join(report.unsupported))
    if report.unknown:
        log.warning("unknown ops (not in DB): %s", ", ".join(report.unknown))
    return report
