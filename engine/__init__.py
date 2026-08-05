"""Inference engine: model runners, KV cache and sampling.

Attention runs on the hand-written `flashattn_cuda` kernels; every other
operation stays plain PyTorch.
"""

__version__ = "0.1.0"
