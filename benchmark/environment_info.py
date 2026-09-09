"""Compatibility import for the ROCm/environment inspection adapter.

New application code belongs under ``bench_app``. This module keeps
existing CLI scripts and external imports (``import environment_info``,
``./environment_info.py`` as a standalone entry point) working while the
layout is migrated - see docs/architecture.md's migration rule.
"""

from bench_app.adapters.outbound.rocm_environment import (
    gather,
    gpu_info,
    host_info,
    llama_cpp_build,
    main,
    rocm_version,
    run_id,
    short_rocm_version,
    slugify_version,
)

__all__ = [
    "gather",
    "gpu_info",
    "host_info",
    "llama_cpp_build",
    "main",
    "rocm_version",
    "run_id",
    "short_rocm_version",
    "slugify_version",
]

if __name__ == "__main__":
    main()
