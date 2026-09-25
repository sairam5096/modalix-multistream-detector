"""SigLIP2 on Modalix, B1157-runtime variant.

Same as siglip_modalix.pipeline.SiglipModalix but for the *_b1157_mpk.tar.gz packs produced by
patch_packs.py: both towers end at an EV74 unpack -> float32 cast -> pass-through, so the runtime
returns the RAW (tessellated, 16-channel-aligned) MLA output as float32 and this class de-tessellates
in numpy. Tile geometry is passed explicitly (the patched packs no longer carry 0_detessellate.json).
"""
from __future__ import annotations
import numpy as np
import ml_dtypes
from PIL import Image
import pyneat

from siglip_modalix import pipeline as base
from siglip_modalix.transforms import FastTess, FastDetess, tess_ref, detess_ref


def _raw_bytes(arr):
    """Runtime output -> the original MLA OFM byte stream as int8, bit-exact.
    The patched pack's EV74 chain is unpack(bf16) -> cast(float32): each 16-bit OFM word w becomes the
    float32 with bit pattern w << 16, so the inverse is a shift, never a float conversion (which would
    canonicalize NaN-looking byte pairs and corrupt data)."""
    a = np.asarray(arr).reshape(-1)
    if a.dtype == np.int8:
        return a
    if a.dtype == np.uint8:
        return a.view(np.int8)
    if a.dtype in (np.uint16, np.int16) or a.dtype == ml_dtypes.bfloat16:
        return np.ascontiguousarray(a).view(np.int8).reshape(-1)
    if a.dtype == np.float32:
        words = (np.ascontiguousarray(a).view(np.uint32) >> 16).astype(np.uint16)
        return words.view(np.int8).reshape(-1)
    raise TypeError(f"unexpected runtime output dtype {a.dtype}")


class SiglipModalixB1157(base.SiglipModalix):
    def __init__(self, base_dir, text_mpk=None, vision_mpk=None, tess=(16, 48), detess=(3, 192),
                 vision_detess="c16", **kw):
        text_mpk = text_mpk or f"{base_dir}/models/text_b1157_mpk.tar.gz"
        vision_mpk = vision_mpk or f"{base_dir}/models/vision_b1157_mpk.tar.gz"
        self.vision_detess_mode = vision_detess
        # the patched packs carry no detess json: hand the base class the known geometry
        base.mpk_tiles = lambda _p, _t=tuple(tess), _d=tuple(detess): (_t, _d)
        super().__init__(base_dir, text_mpk=text_mpk, vision_mpk=vision_mpk, tess=tess, detess=detess, **kw)

    # B1157 plan: the pack's own cast+tessellate run on the EV74, so the text body takes the fp32
    # embedding [SEQ,1,HID] directly (no numpy tessellation) and returns the raw tessellated OFM.
    def _load_text(self, mpk):
        to = pyneat.ModelOptions(); to.preprocess.kind = pyneat.InputKind.Tensor
        to.preprocess.input_max_width = 1; to.preprocess.input_max_height = self.seq; to.preprocess.input_max_depth = self.hid
        return pyneat.Model(mpk, to), None

    def _text_run(self, ie):
        x = np.ascontiguousarray(ie.reshape(self.seq, 1, self.hid).astype(np.float32))
        t = pyneat.Tensor.from_numpy(x, copy=True, memory=pyneat.TensorMemory.EV74)
        return self.text.run([t], timeout_ms=30000)[0].to_numpy()

    def _mla(self, ie, tessfn, detessfn):
        return detessfn(_raw_bytes(self._text_run(ie)))

    def _build_fast(self, verify):
        (ie0,) = self.embed.run([self.eout], {self.ein: np.zeros((1, self.seq), np.int64)})
        out0 = self._text_run(ie0)
        raw0 = _raw_bytes(out0)
        print(f"[b1157] text raw OFM from runtime: dtype={out0.dtype} shape={out0.shape} -> {raw0.nbytes} bytes", flush=True)
        self.ftess = None
        self.fdetess = FastDetess(self.seq, self.hid, self.dsl, raw0.nbytes)
        if verify:
            assert np.array_equal(self.fdetess(raw0), detess_ref(raw0, self.seq, self.hid, self.dsl)), "fast detessellate mismatch"

    def encode_image(self, image):
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
        t = pyneat.Tensor.from_numpy(np.ascontiguousarray(image), copy=True,
                                     image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74)
        out = self.vision.run([t], timeout_ms=30000)[0].to_numpy()
        return self._vision_detess(out)

    def _vision_detess(self, out):
        """raw vision OFM: 24576 bytes = 768 width positions x one 16-channel block of bf16 (C=1, c16 aligned).
        The MLA stores 2-byte elements as byte planes inside each block: [16 low bytes][16 high bytes],
        so the single value of a block is (block[16] << 8) | block[0]."""
        raw = _raw_bytes(out).view(np.uint8).reshape(768, 32)   # one 32-byte block per position
        mode = self.vision_detess_mode
        if mode in ("c16", "planes"):
            words = (raw[:, 16].astype(np.uint16) << 8) | raw[:, 0].astype(np.uint16)
            return words.view(ml_dtypes.bfloat16).astype(np.float32)
        if mode == "first":                          # naive: first bf16 word of each block
            return raw[:, :2].reshape(-1).view(ml_dtypes.bfloat16).astype(np.float32)
        if mode == "ref":                            # ev_transforms reference: frame [1, W=768, C=1], tile (768, 1)
            from ev_transforms.transforms import detessellation
            return detessellation(raw.reshape(1, -1).view(np.int8), slice_shape=[768, 1], frame_type="bfloat16",
                                  frame_shape=[1, 768, 1], align_c16=True, cblock=True).astype(np.float32).reshape(-1)
        raise ValueError(mode)
