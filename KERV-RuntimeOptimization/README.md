# KERV-RuntimeOptimization

This directory contains the on-device runtime extension for KERV. It packages
the accuracy-safe operator fusions, persistent buffers, static-tree execution,
CUDA Graph paths, and the FlagScale launch bridge without changing KERV's
acceptance rule, Kalman logic, or action semantics.

```text
KERV-RuntimeOptimization/
|-- KERVRuntimeOptimization/  # importable Python package and operators
|-- FlagScale/
|   |-- configs/              # reproducible FlagScale launch profile
|   |-- inference_kerv.py     # bridge into the KERV inference entry point
|   `-- LICENSE               # FlagScale integration license notice
`-- LICENSE                   # runtime-extension license
```

The Python import package follows CamelCase naming:

```python
from KERVRuntimeOptimization.embodied_ops import configure_kerv_ops
```

Operators remain registered under `torch.ops.flagos_embodied` so the runtime
can be consumed through the existing FlagOS operator interface. The top-level
[`run_kerv.py`](../run_kerv.py) launcher adds this directory to `PYTHONPATH`
and passes the bundled configuration to FlagScale.

See [`docs/OPERATORS.md`](../docs/OPERATORS.md) for the released operator
surface and fallback policy.
