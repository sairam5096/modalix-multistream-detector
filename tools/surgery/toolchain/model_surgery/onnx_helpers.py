"""
Self-contained ONNX graph-editing primitives used by the surgeons.

Intentionally a small, original subset (no dependency on the SDK's
`sima_utils.onnx` nor the reference tool's proprietary helpers). Everything here
is a thin wrapper over `onnx`/`onnx.helper`. `onnxsim` is used opportunistically
if installed, otherwise we fall back to shape-inference + checker only.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import onnx
import onnx.version_converter
from onnx import helper, numpy_helper

log = logging.getLogger("model_surgery.onnx")

# IR / opset accepted by the SiMa Model SDK front-end.
_ONNX_IR_VERSION = 8
_ONNX_OPSET_VERSION = 17


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #
def _coerce_version(model: onnx.ModelProto) -> onnx.ModelProto:
    model.ir_version = _ONNX_IR_VERSION
    try:
        return onnx.version_converter.convert_version(model, _ONNX_OPSET_VERSION)
    except Exception as exc:  # opset already fine, or converter can't help
        log.debug("version_converter skipped: %s", exc)
        return model


def _try_simplify(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        from onnxsim import simplify  # type: ignore

        model_opt, ok = simplify(model)
        if ok:
            return model_opt
        log.warning("onnxsim could not validate the simplified model; using original")
    except ImportError:
        log.warning("onnxsim not installed — skipping simplify (surgery still works)")
    except Exception as exc:
        log.warning("onnxsim failed (%s) — skipping simplify", exc)
    return model


def load_model(path: str, simplify: bool = True) -> onnx.ModelProto:
    model = onnx.load(path)
    model = _coerce_version(model)
    if simplify:
        model = _try_simplify(model)
    return model


def _reconcile_output_dtypes(model: onnx.ModelProto) -> None:
    """Set each graph output's elem_type to its producer's inferred dtype.

    `add_output` declares FLOAT, but a surgeon that re-exposes an existing tensor
    (e.g. an fp16 head conv in v5/v7) needs the output dtype to match — otherwise
    onnx.checker / onnxruntime reject the graph. Runs after shape inference.
    """
    inferred = {v.name: v.type.tensor_type.elem_type for v in model.graph.value_info}
    for o in model.graph.output:
        t = inferred.get(o.name)
        if t and t != onnx.TensorProto.UNDEFINED and o.type.tensor_type.elem_type != t:
            o.type.tensor_type.elem_type = t


def keep_reachable_from_outputs(model: onnx.ModelProto) -> int:
    """
    Dead-node elimination: keep only nodes that (transitively) feed a graph
    output. Lets a surgeon re-point the graph outputs and drop an entire stale
    tail (e.g. YOLO26's TopK/Gather E2E block) without enumerating it.
    """
    producer: dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for o in n.output:
            producer[o] = n

    keep: set[int] = set()
    seen_t: set[str] = set()
    stack = [o.name for o in model.graph.output]
    while stack:
        t = stack.pop()
        if t in seen_t:
            continue
        seen_t.add(t)
        n = producer.get(t)
        if n is None or id(n) in keep:
            continue
        keep.add(id(n))
        stack.extend(inp for inp in n.input if inp)

    removed = 0
    for n in list(model.graph.node):
        if id(n) not in keep:
            model.graph.node.remove(n)
            removed += 1
    if removed:
        log.info("pruned %d dead node(s) not feeding any output", removed)
    return removed


def prune_unused_initializers(model: onnx.ModelProto) -> int:
    """Drop initializers no node consumes (leftovers from removed subgraphs)."""
    used = {inp for node in model.graph.node for inp in node.input}
    used.update(o.name for o in model.graph.output)
    dead = [i for i in list(model.graph.initializer) if i.name not in used]
    for i in dead:
        model.graph.initializer.remove(i)
    if dead:
        log.info("pruned %d unused initializer(s)", len(dead))
    return len(dead)


def save_model(model: onnx.ModelProto, path: str, simplify: bool = True) -> None:
    model = _coerce_version(model)
    # Drop any stale subgraph the surgeon detached from the outputs, then tidy.
    keep_reachable_from_outputs(model)
    topo_sort(model)
    prune_unused_initializers(model)
    if simplify:
        model = _try_simplify(model)
    try:
        model = onnx.shape_inference.infer_shapes(model)
        _reconcile_output_dtypes(model)   # fp16 models: match declared output dtype to producer
    except Exception as exc:
        log.warning("shape inference failed: %s", exc)
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        log.warning("onnx.checker reported: %s", exc)
    onnx.save(model, path)
    log.info("saved surgered model -> %s", path)


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
def make_node(**kwargs) -> onnx.NodeProto:
    return helper.make_node(**kwargs)


def find_node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    for n in model.graph.node:
        if n.name == name:
            return n
    raise KeyError(f"node not found: {name}")


def has_node(model: onnx.ModelProto, name: str) -> bool:
    return any(n.name == name for n in model.graph.node)


def find_node_by_output(model: onnx.ModelProto, output: str) -> onnx.NodeProto:
    for n in model.graph.node:
        if output in n.output:
            return n
    raise KeyError(f"no node produces output: {output}")


def insert_after(model: onnx.ModelProto, ref: onnx.NodeProto, new: onnx.NodeProto) -> onnx.NodeProto:
    """Insert `new` immediately after `ref` in graph order (no rewiring)."""
    nodes = model.graph.node
    for i, x in enumerate(nodes):
        if x.name == ref.name:
            nodes.insert(i + 1, new)
            return new
    nodes.append(new)
    return new


def remove_node(model: onnx.ModelProto, name: str, tolerant: bool = True) -> bool:
    """Delete a node by name (no reconnection). Returns True if removed."""
    for n in list(model.graph.node):
        if n.name == name:
            model.graph.node.remove(n)
            return True
    if not tolerant:
        raise KeyError(f"node not found for removal: {name}")
    log.debug("remove_node: '%s' not present (tolerated)", name)
    return False


def remove_nodes(model: onnx.ModelProto, names: Iterable[str]) -> list[str]:
    return [n for n in names if remove_node(model, n, tolerant=True)]


# --------------------------------------------------------------------------- #
# initializers
# --------------------------------------------------------------------------- #
def is_initializer(model: onnx.ModelProto, name: str) -> bool:
    return any(i.name == name for i in model.graph.initializer)


def find_initializer_value(model: onnx.ModelProto, name: str) -> np.ndarray:
    for i in model.graph.initializer:
        if i.name == name:
            return numpy_helper.to_array(i)
    raise KeyError(f"initializer not found: {name}")


def add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    value = np.asarray(value, dtype=np.float32)
    model.graph.initializer.append(
        helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=value.shape,
            vals=value.flatten().tolist(),
        )
    )


# --------------------------------------------------------------------------- #
# graph inputs / outputs
# --------------------------------------------------------------------------- #
def clear_outputs(model: onnx.ModelProto) -> None:
    del model.graph.output[:]


def add_output(model: onnx.ModelProto, name: str, shape) -> None:
    model.graph.output.append(
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(shape))
    )


def input_tensors(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    init = {i.name for i in model.graph.initializer}
    return [i for i in model.graph.input if i.name not in init]


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #
def topo_sort(model: onnx.ModelProto) -> None:
    """Kahn topological sort of graph nodes (in-place, stable)."""
    available = {""}
    available.update(i.name for i in model.graph.input)
    available.update(i.name for i in model.graph.initializer)

    pending = list(model.graph.node)
    ordered: list[onnx.NodeProto] = []
    progressed = True
    while pending and progressed:
        progressed = False
        still: list[onnx.NodeProto] = []
        for node in pending:
            if all((inp in available) for inp in node.input):
                ordered.append(node)
                available.update(node.output)
                progressed = True
            else:
                still.append(node)
        pending = still

    if pending:
        # Cycle or dangling input — leave leftovers at the end rather than drop them.
        log.warning("topo_sort: %d node(s) with unresolved inputs left in place", len(pending))
        ordered.extend(pending)

    del model.graph.node[:]
    model.graph.node.extend(ordered)
