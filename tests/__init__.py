"""Test package.

This file is load-bearing. Without it, pytest's prepend import mode puts
`tests/` itself on `sys.path` rather than the repository root, so
`from tests.parity_utils import ...` resolves only when the working directory
happens to be on `sys.path` too — which is true under `python -m pytest` and
false under a bare `pytest`. Making `tests` a package means pytest inserts the
repository root instead, and both invocations behave the same.
"""
