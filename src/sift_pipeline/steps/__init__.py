"""
Importing this package registers every available Step 1 backend into
STEP1_REGISTRY (see registry.py). Each backend lives in its own module and
self-registers via the @register_step1("name") decorator on import.

Currently implemented:
    - lib_step1.py  -> "lib"     (OpenCV / library reference)

Planned (not yet implemented -- add a module + uncomment the import when
ready, nothing else needs to change):
    - cpu_step1.py    -> "cpu"      (pure NumPy Gaussian/DoG pyramid)
    - gpu_v1_step1.py -> "gpu_v1"   (naive Numba CUDA kernel)
    - gpu_v2_step1.py -> "gpu_v2"   (shared-memory tiled kernel)
    - gpu_v3_step1.py -> "gpu_v3"   (fused blur + DoG kernel)
"""

from . import lib_step1  # noqa: F401  (import triggers registration)

# from . import cpu_step1       # noqa: F401
# from . import gpu_v1_step1    # noqa: F401
# from . import gpu_v2_step1    # noqa: F401
# from . import gpu_v3_step1    # noqa: F401
