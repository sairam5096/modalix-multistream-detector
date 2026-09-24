"""Surgeon base class + a tiny name->class registry (metaclass auto-registration)."""

from __future__ import annotations

import logging

import onnx

from ..identify import YoloIdentity

log = logging.getLogger("model_surgery.surgeon")

registry: dict[str, type["SurgeonBase"]] = {}


class _SurgeonMeta(type):
    def __new__(mcls, name, bases, ns):
        cls = super().__new__(mcls, name, bases, ns)
        key = ns.get("name")
        if key:
            registry[key] = cls
        return cls


class SurgeonBase(metaclass=_SurgeonMeta):
    """Rewrite a YOLO detection head into the box-decoder contract."""

    name: str | None = None

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        """Edit `model` in place (or return a new one) and return the result."""
        raise NotImplementedError(f"surgeon '{self.name}' has no do_surgery")


def get_surgeon(key: str) -> type[SurgeonBase] | None:
    return registry.get(key)
