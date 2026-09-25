#!/usr/bin/env python3
"""Recover and de-tessellate the raw MLA output of a pack converted with convert_mpk_b1157.py.

After conversion the runtime returns the raw (tessellated, 16-channel-aligned) MLA output as a
float32 tensor whose 32-bit patterns are the original 16-bit words shifted left by 16. Two helpers:

  recover_words(out)                    -> np.uint16 array of the original OFM words, bit-exact
  detessellate_from_geometry(words, geometry, frame_type="bfloat16")
                                        -> logical tensor [N, D, H, W, C] (float32)
  detessellate_single_channel(words, positions) -> fast path for C=1 outputs (e.g. a pooled embedding)

'geometry' is the dict from the pack's original 0_detessellate.json (kept in the converted archive
as detess_geometry.json). The general path uses ev_transforms.detessellation from the SiMa
ModelSDK / delivery bundle (import path must contain the ev_transforms package). The MLA stores
2-byte elements as byte planes inside each 16-channel block ([16 low bytes][16 high bytes]); the
reference implementation handles that, and the fast C=1 path does it explicitly.

Never convert the float32 output back with astype(bfloat16): NaN-looking byte pairs would be
canonicalized and the data corrupted. Always go through recover_words().
"""
import json, tarfile
import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = None


def recover_words(out):
    """Runtime output (float32 [1,N] from unpack->cast->pass_through) -> uint16 words of the raw OFM."""
    a = np.asarray(out).reshape(-1)
    if a.dtype == np.float32:
        return (np.ascontiguousarray(a).view(np.uint32) >> 16).astype(np.uint16)
    if a.dtype in (np.uint16, np.int16):
        return a.view(np.uint16)
    if a.dtype in (np.int8, np.uint8):
        return np.ascontiguousarray(a).view(np.uint16)
    raise TypeError(f"unexpected output dtype {a.dtype}")


def words_to_float(words, frame_type="bfloat16"):
    if frame_type == "bfloat16":
        if BF16 is None:
            # bf16 -> f32 is a left shift by 16
            return (words.astype(np.uint32) << 16).view(np.float32)
        return words.view(BF16).astype(np.float32)
    if frame_type == "int16":
        return words.view(np.int16).astype(np.float32)
    if frame_type == "float16":
        return words.view(np.float16).astype(np.float32)
    raise ValueError(frame_type)


def geometry_from_pack(pack_path):
    """Read detess_geometry.json (converted pack) or 0_detessellate.json (original pack)."""
    with tarfile.open(pack_path) as t:
        for m in t.getmembers():
            base = m.name.rsplit("/", 1)[-1]
            if base in ("detess_geometry.json",) or base.endswith("_detessellate.json"):
                return json.load(t.extractfile(m))
    raise FileNotFoundError(f"{pack_path}: no detessellate geometry json")


def _first(v):
    return int(v[0]) if isinstance(v, list) else int(v)


def detessellate_from_geometry(words, geometry, frame_type="bfloat16"):
    """General path using ev_transforms.detessellation with the pack geometry.
    Returns float32 array shaped [N, D, H, W, C] (unit dims kept)."""
    from ev_transforms.transforms import detessellation  # from the ModelSDK / delivery bundle
    g = geometry
    W, H, D, C = (_first(g[k]) for k in ("input_width", "input_height", "input_depth", "input_channels"))
    tW, tH, tD, tC = (_first(g[k]) for k in ("tile_width", "tile_height", "tile_depth", "tile_channels"))
    frame_shape = [1, D, H, W, C]
    slice_shape = [tD, tH, tW, tC]
    raw = np.ascontiguousarray(words).view(np.int8).reshape(1, -1)
    out = detessellation(raw, slice_shape=slice_shape, frame_type=frame_type, frame_shape=frame_shape,
                         align_c16=True, cblock=True)
    return np.asarray(out, dtype=np.float32)


def detessellate_single_channel(words, positions, frame_type="bfloat16"):
    """Fast path for outputs with one channel per spatial position (e.g. a pooled [768] embedding):
    each position occupies one 16-channel block = 32 bytes, stored as [16 low bytes][16 high bytes]."""
    raw = np.ascontiguousarray(words).view(np.uint8).reshape(positions, 32)
    vals = (raw[:, 16].astype(np.uint16) << 8) | raw[:, 0].astype(np.uint16)
    return words_to_float(vals, frame_type)


def detessellate_rows(words, rows, channels, tile_rows, tile_channels, frame_type="bfloat16"):
    """Row-major [rows, channels] outputs (transformer hidden states): tiles (tile_rows, tile_channels),
    frame [1, rows, channels]. Uses the reference implementation."""
    from ev_transforms.transforms import detessellation
    raw = np.ascontiguousarray(words).view(np.int8).reshape(1, -1)
    out = detessellation(raw, slice_shape=[tile_rows, tile_channels], frame_type=frame_type,
                         frame_shape=[1, rows, channels], align_c16=True, cblock=True)
    return np.asarray(out, dtype=np.float32).reshape(1, rows, channels)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="self-check: de-tessellate a saved runtime output (.npy) with a pack's geometry")
    ap.add_argument("--npy", required=True, help="float32 output saved from Model.run on the converted pack")
    ap.add_argument("--pack", required=True, help="converted or original pack (for the geometry)")
    ap.add_argument("--frame_type", default="bfloat16")
    a = ap.parse_args()
    w = recover_words(np.load(a.npy))
    g = geometry_from_pack(a.pack)
    ref = detessellate_from_geometry(w, g, a.frame_type)
    print("geometry:", {k: g[k] for k in g if k.startswith(("input_", "tile_"))})
    print("logical shape", ref.shape, "nan", int(np.isnan(ref).sum()), "min", float(np.nanmin(ref)), "max", float(np.nanmax(ref)))
    C = _first(g["input_channels"])
    if C == 1:
        fast = detessellate_single_channel(w, ref.size, a.frame_type)
        print("fast C=1 path equals reference:", bool(np.array_equal(fast.reshape(-1), ref.reshape(-1))))
