"""
model_surgery — auto-identify a YOLO ONNX and rewrite its detection head so the
graph exposes the SiMa generic box-decoder input contract.

This is the surgery front-end of the Ultralytics -> SiMa conversion toolchain.
It is deliberately self-contained: it depends only on
`onnx`/`numpy` (and, if present, `onnxsim`/`onnxruntime`) so it can run and be
tested off-board on an x86 host, and later be imported by the Docker toolchain's
convert step.

Pipeline:  identify -> audit -> surgery -> re-audit + numeric sanity -> emit
           (surgered .onnx + boxdecoder.json)
"""

__version__ = "0.1.0"
