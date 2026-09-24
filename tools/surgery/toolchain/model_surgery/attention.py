"""
Generalized YOLO C2PSA attention rewrite: MatMul -> Einsum for the SiMa MLA.

Ultralytics anchor-free detectors (v10 / v11 / yolo26, any size) express their
C2PSA self-attention through two `MatMul` nodes (Q·K then softmax·V). The Modalix
MLA compiler *hangs* on that pattern, but supports the exact-equivalent
`nhwc,nhqc->nhwq` `Einsum` form. This module rewrites every attention block into
that Einsum form, in place, deriving all block-local names from the graph so it
works for **any number of blocks at any `/model.N` location** — no hard-coded
base names or Einsum counts (contrast the single-model reference
`graph_surgery_yolo26m.py`, which pins two specific bases and expects four
Einsums).

An attention block is discovered by its qkv projection Conv, whose name ends
`/attn/qkv/conv/Conv`; the block `base` is that name minus `/qkv/conv/Conv`. Each
block then exposes the fixed Ultralytics `Attention` sub-graph under `base`:

    {base}/qkv/conv/Conv   {base}/Split   {base}/Mul (scale)
    {base}/pe/conv/Conv     {base}/proj/conv/Conv

The replacement is the reference's exact graph — Reshape -> Split -> 2x Transpose
-> Einsum(Q·K) -> Mul(scale) -> Softmax -> Reshape(value) -> Conv(pe)
-> Einsum(attn·value) -> Transpose -> Reshape -> Add -> Conv(proj) — after which
the residual `Add` that consumed the old `proj` output is re-pointed at the new
projection, orphaning the MatMul sub-graph. `prune_dead_nodes` then sweeps it.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
from onnx import helper, numpy_helper

log = logging.getLogger("model_surgery.attention")

# The single MLA-supported attention Einsum equation (batched Q·Kᵀ / attn·V).
SUPPORTED_EINSUM_EQUATION = "nhwc,nhqc->nhwq"

# suffix identifying an attention block's qkv projection Conv
_QKV_SUFFIX = "/qkv/conv/Conv"
_ATTN_QKV_SUFFIX = "/attn" + _QKV_SUFFIX


# --------------------------------------------------------------------------- #
# small graph helpers (self-contained, mirror the reference tool)
# --------------------------------------------------------------------------- #
def find_node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    """Return the uniquely named node, or raise a clear error."""
    for node in model.graph.node:
        if node.name == name:
            return node
    raise ValueError(f"attention rewrite: node not found: {name}")


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    """Static shape for every tensor, from a shape-inferred copy of the graph.

    Built off `infer_shapes` (not the live graph) so the rewrite is robust even
    when the caller skipped onnxsim (`--no-simplify`) and value_info is sparse.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for value in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]:
        dims: list[int] = []
        ok = True
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                ok = False
                break
            dims.append(dim.dim_value)
        if ok:
            shapes[value.name] = dims
    return shapes


def tensor_shape(shapes: dict[str, list[int]], name: str) -> list[int]:
    """Return one statically inferred tensor shape (raises if dynamic/missing)."""
    if name not in shapes:
        raise ValueError(
            f"attention rewrite: no static shape for tensor '{name}'; "
            "export the model with a static input shape"
        )
    return shapes[name]


def node_attributes(node: onnx.NodeProto) -> dict[str, object]:
    """Copy ONNX attributes off an existing node (e.g. Conv strides/pads)."""
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def add_int64_initializer(model: onnx.ModelProto, name: str, values: list[int]) -> None:
    """Add a constant int64 shape tensor used by a replacement Reshape node."""
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)
    )


def split_sizes_attribute(node: onnx.NodeProto) -> list[int] | None:
    """Return Split sizes when an older export stores them as an attribute."""
    for attr in node.attribute:
        if attr.name == "split":
            return list(helper.get_attribute_value(attr))
    return None


def _tag(base: str) -> str:
    """Unique, name-safe prefix fragment derived from a block base path.

    `/model.10/m/m.0/attn` -> `model_10_m_m_0_attn`, guaranteeing per-block
    uniqueness of every replacement tensor/node name.
    """
    return base.strip("/").replace("/", "_").replace(".", "_") or "attn"


# --------------------------------------------------------------------------- #
# discovery / inspection
# --------------------------------------------------------------------------- #
def find_attention_blocks(model: onnx.ModelProto) -> list[str]:
    """Return the `base` path of every C2PSA attention block in the graph.

    Scans for Conv nodes named `.../attn/qkv/conv/Conv`; the block base is that
    name minus the trailing `/qkv/conv/Conv`. Order-preserving, de-duplicated.
    Returns 0..N bases — a model may carry any number of attention blocks.
    """
    bases: list[str] = []
    seen: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "Conv" and node.name.endswith(_ATTN_QKV_SUFFIX):
            base = node.name[: -len(_QKV_SUFFIX)]
            if base not in seen:
                seen.add(base)
                bases.append(base)
    return bases


def has_attention(model: onnx.ModelProto) -> bool:
    """True if the graph carries at least one `/attn/qkv/conv/Conv` block."""
    return bool(find_attention_blocks(model))


def count_matmul(model: onnx.ModelProto) -> int:
    """Number of MatMul nodes remaining in the graph."""
    return sum(1 for node in model.graph.node if node.op_type == "MatMul")


# --------------------------------------------------------------------------- #
# the rewrite
# --------------------------------------------------------------------------- #
def _rewrite_one_block(model: onnx.ModelProto, base: str, shapes: dict[str, list[int]]) -> None:
    """Replace one MatMul attention block (rooted at `base`) with Einsum form."""
    tag = _tag(base)
    qkv = find_node(model, f"{base}/qkv/conv/Conv")
    split = find_node(model, f"{base}/Split")
    pe = find_node(model, f"{base}/pe/conv/Conv")
    proj = find_node(model, f"{base}/proj/conv/Conv")
    scale = find_node(model, f"{base}/Mul")

    # the residual Add is whatever consumes the old projection output
    attention_output = proj.output[0]
    residual_consumers = [n for n in model.graph.node if attention_output in n.input]
    if not residual_consumers:
        raise ValueError(
            f"attention rewrite: no node consumes projection output "
            f"'{attention_output}' for block {base}"
        )

    batch, channels, height, width = tensor_shape(shapes, qkv.output[0])
    q_shape = tensor_shape(shapes, split.output[0])
    v_shape = tensor_shape(shapes, split.output[2])
    heads = q_shape[1]
    if channels % heads != 0:
        raise ValueError(
            f"attention rewrite: {qkv.name} channel depth {channels} "
            f"is not divisible by {heads} heads"
        )

    prefix = f"/sima_exact_attention/{tag}"
    qkv_shape_name = f"{prefix}/qkv_shape"
    feature_shape_name = f"{prefix}/feature_shape"
    add_int64_initializer(model, qkv_shape_name, [batch, heads, channels // heads, height * width])
    add_int64_initializer(model, feature_shape_name, [batch, heads * v_shape[2], height, width])

    qkv_reshaped = f"{prefix}/qkv_reshape_output_0"
    query = f"{prefix}/query_output_0"
    key = f"{prefix}/key_output_0"
    value = f"{prefix}/value_output_0"
    query_transposed = f"{prefix}/query_nhwc_output_0"
    key_transposed = f"{prefix}/key_nhqc_output_0"
    query_key = f"{prefix}/query_key_output_0"
    scaled = f"{prefix}/scaled_output_0"
    probabilities = f"{prefix}/probabilities_output_0"
    value_feature = f"{prefix}/value_feature_output_0"
    positional = f"{prefix}/positional_output_0"
    weighted_nhwq = f"{prefix}/weighted_nhwq_output_0"
    weighted = f"{prefix}/weighted_output_0"
    weighted_feature = f"{prefix}/weighted_feature_output_0"
    attention_sum = f"{prefix}/attention_sum_output_0"
    projected = f"{prefix}/projected_output_0"

    split_sizes = split_sizes_attribute(split)
    replacement_nodes = [
        helper.make_node(
            "Reshape",
            [qkv.output[0], qkv_shape_name],
            [qkv_reshaped],
            name=f"{prefix}/Reshape",
        ),
        helper.make_node(
            "Split",
            [qkv_reshaped] if split_sizes is not None else [qkv_reshaped, split.input[1]],
            [query, key, value],
            name=f"{prefix}/Split",
            axis=2,
            **({"split": split_sizes} if split_sizes is not None else {}),
        ),
        helper.make_node(
            "Transpose", [query], [query_transposed],
            name=f"{prefix}/Query/Transpose", perm=[0, 1, 3, 2],
        ),
        helper.make_node(
            "Transpose", [key], [key_transposed],
            name=f"{prefix}/Key/Transpose", perm=[0, 1, 3, 2],
        ),
        helper.make_node(
            "Einsum", [query_transposed, key_transposed], [query_key],
            name=f"{prefix}/QueryKey/Einsum", equation=SUPPORTED_EINSUM_EQUATION,
        ),
        helper.make_node(
            "Mul", [query_key, scale.input[1]], [scaled], name=f"{prefix}/Scale/Mul",
        ),
        helper.make_node(
            "Softmax", [scaled], [probabilities], name=f"{prefix}/Softmax", axis=-1,
        ),
        helper.make_node(
            "Reshape", [value, feature_shape_name], [value_feature],
            name=f"{prefix}/Value/Reshape",
        ),
        helper.make_node(
            "Conv", [value_feature, pe.input[1], pe.input[2]], [positional],
            name=f"{prefix}/Positional/Conv", **node_attributes(pe),
        ),
        helper.make_node(
            "Einsum", [probabilities, value], [weighted_nhwq],
            name=f"{prefix}/AttentionValue/Einsum", equation=SUPPORTED_EINSUM_EQUATION,
        ),
        helper.make_node(
            "Transpose", [weighted_nhwq], [weighted],
            name=f"{prefix}/Weighted/Transpose", perm=[0, 1, 3, 2],
        ),
        helper.make_node(
            "Reshape", [weighted, feature_shape_name], [weighted_feature],
            name=f"{prefix}/Weighted/Reshape",
        ),
        helper.make_node(
            "Add", [weighted_feature, positional], [attention_sum], name=f"{prefix}/Add",
        ),
        helper.make_node(
            "Conv", [attention_sum, proj.input[1], proj.input[2]], [projected],
            name=f"{prefix}/Projection/Conv", **node_attributes(proj),
        ),
    ]

    insert_index = list(model.graph.node).index(residual_consumers[0])
    for offset, node in enumerate(replacement_nodes):
        model.graph.node.insert(insert_index + offset, node)

    # re-point every consumer of the old projection output at the new one
    for consumer in residual_consumers:
        for index, input_name in enumerate(consumer.input):
            if input_name == attention_output:
                consumer.input[index] = projected
    log.info("rewrote attention block %s -> Einsum (heads=%d, %d consumer(s))",
             base, heads, len(residual_consumers))


def rewrite_attention(model: onnx.ModelProto) -> int:
    """Rewrite every MatMul attention block into Einsum form. Returns block count.

    Edits `model` in place. Does *not* prune the orphaned MatMul sub-graphs —
    call `prune_dead_nodes(model)` afterwards. Safe (no-op) when the graph has no
    attention blocks.
    """
    bases = find_attention_blocks(model)
    if not bases:
        return 0
    shapes = _shape_map(model)
    for base in bases:
        _rewrite_one_block(model, base, shapes)
    return len(bases)


# --------------------------------------------------------------------------- #
# pruning
# --------------------------------------------------------------------------- #
def prune_dead_nodes(model: onnx.ModelProto) -> int:
    """Remove nodes not backward-reachable from the graph outputs.

    After `rewrite_attention` re-points the residual Add, the original
    Split/MatMul/Softmax/Transpose/pe/proj sub-graph is orphaned; this reachability
    sweep deletes it. Initializers no surviving node consumes are dropped too.
    Returns the number of nodes removed.
    """
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    keep: set[int] = set()
    seen: set[str] = set()
    stack = [o.name for o in model.graph.output]
    while stack:
        tensor = stack.pop()
        if tensor in seen:
            continue
        seen.add(tensor)
        node = producer.get(tensor)
        if node is None or id(node) in keep:
            continue
        keep.add(id(node))
        stack.extend(inp for inp in node.input if inp)

    removed = 0
    for node in list(model.graph.node):
        if id(node) not in keep:
            model.graph.node.remove(node)
            removed += 1

    used = {inp for node in model.graph.node for inp in node.input}
    used.update(o.name for o in model.graph.output)
    for init in list(model.graph.initializer):
        if init.name not in used:
            model.graph.initializer.remove(init)

    if removed:
        log.info("attention prune: removed %d orphaned node(s)", removed)
    return removed


# --------------------------------------------------------------------------- #
# one-call surgeon entry point
# --------------------------------------------------------------------------- #
def apply(model: onnx.ModelProto) -> int:
    """Rewrite + prune + hard-verify, for use by the family surgeons.

    Rewrites every attention block to Einsum, prunes the orphaned MatMul
    sub-graphs, then hard-fails if any attention MatMul survives (a
    silently-broken pack must never be emitted). Returns the block count (0 when
    the model has no attention — a safe no-op for the non-attention families).
    """
    if not has_attention(model):
        return 0
    n = rewrite_attention(model)
    prune_dead_nodes(model)
    remaining = count_matmul(model)
    if remaining:
        raise RuntimeError(
            f"attention rewrite left {remaining} MatMul node(s) in the graph — "
            "the C2PSA attention was not fully lowered to Einsum; refusing to emit "
            "a model the MLA compiler will hang on"
        )
    log.info("attention: rewrote %d block(s) to Einsum, 0 MatMul remaining", n)
    return n
