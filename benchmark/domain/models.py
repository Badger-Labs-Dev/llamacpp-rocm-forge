"""Core benchmark domain model: the configuration knobs one llama-bench
probe is run with, and the outcome of running a full depth-curve for one
(model, series, config).

Pure dataclasses - no I/O, no CLI, no Docker/filesystem coupling. Moved
out of run_bench.py so application-layer modules (run_curve.py,
auto_tune.py, moe_sweep.py, dense_sweep.py) can depend on this without
depending on run_bench.py itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# KV cache dtypes that require flash attention (llama.cpp hard requirement -
# the non-fused attention path can't read a quantized KV cache at all, this
# isn't just a perf preference).
KV_TYPES_REQUIRING_FA = frozenset({"q8_0", "q4_0", "q5_0", "q5_1", "q4_1", "iq4_nl"})


@dataclass(frozen=True)
class BenchConfig:
    ubatch: int = 2048
    batch: int = 2048
    ctk: str = "f16"
    ctv: str = "f16"
    flash_attn: str = "auto"  # llama-bench's -fa: "auto" | "on" | "off".
        # "auto" lets llama.cpp decide per model/backend at load time
        # whether the fused kernel applies (some architectures - KQ-bias
        # models, certain hybrid/SSM models, unsupported head dims - fall
        # back to the ordinary path regardless of this flag, silently).
        # We stopped sweeping flash-attn on/off: forcing "on" doesn't help
        # on architectures where FA can't apply anyway, and "auto" is
        # already llama.cpp's own considered default as of this build.
    block_count: int | None = None
    n_cpu_layers: int = 0  # Dense-model search coordinate. When block_count
        # is known, run_curve translates this to --ngl = block_count + 1 -
        # n_cpu_layers; llama.cpp counts the output layer in addition to the
        # transformer blocks. MoE/legacy probes leave block_count unset and retain
        # the historical --ngl 99 full-offload request.
    n_cpu_moe: int = 0  # llama-bench's -ncmoe/--n-cpu-moe: moves the MoE
        # feed-forward (expert) weights of the first N layers to CPU RAM,
        # keeping attention/shared weights on GPU. 0 (default) is a no-op
        # even for dense models - only meaningful for MoE models, where
        # every expert must otherwise be resident in VRAM regardless of
        # how few are active per token (see application/moe_sweep.py).

    def tag(self) -> str:
        base = f"ub{self.ubatch}_b{self.batch}_kv{self.ctk}-{self.ctv}_fa{self.flash_attn}"
        if self.block_count is not None:
            base = f"{base}_ngl{self.gpu_layers}"
        return f"{base}_ncmoe{self.n_cpu_moe}" if self.n_cpu_moe else base

    @property
    def gpu_layers(self) -> int:
        if self.block_count is None:
            return 99
        max_gpu_layers = self.block_count + 1
        return max(0, min(max_gpu_layers, max_gpu_layers - self.n_cpu_layers))

    def validate(self) -> "BenchConfig":
        """Force flash-attn on if the KV cache dtype requires it - "auto"
        is not guaranteed to enable FA, but a quantized KV cache can only
        be read via the fused kernel, so "auto" alone isn't safe here."""
        if (self.ctk in KV_TYPES_REQUIRING_FA or self.ctv in KV_TYPES_REQUIRING_FA) and self.flash_attn == "off":
            return replace(self, flash_attn="on")
        return self


@dataclass
class RunResult:
    model: str
    series: str  # "prefill" or "generation"
    config: BenchConfig
    status: str  # "ok" (all depths ran), "partial" (stopped early after a
                 # depth failed/timed out, but at least one depth succeeded),
                 # "failed" (no depth produced results)
    return_code: int
    jsonl_path: Path
    stderr_path: Path
    command: list[str]
    depths_run: tuple[int, ...] = ()
    depths_skipped: tuple[int, ...] = ()
    stop_reason: str | None = None  # e.g. "depth 65536 timed out after 300s"
