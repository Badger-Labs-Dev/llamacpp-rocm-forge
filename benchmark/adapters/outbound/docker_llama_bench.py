"""Docker adapter for one llama-bench probe.

The benchmark domain describes configuration values; this adapter owns the
Docker/llama.cpp command-line representation of those values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaBenchProbe:
    model_container_path: str
    series: str
    batch: int
    ubatch: int
    flash_attn: str
    n_cpu_moe: int
    ctk: str
    ctv: str
    device: str
    depth: int
    repetitions: int
    gpu_layers: int
    prefill_tokens: int
    generation_tokens: int


def docker_gpu_args(gpu_gids: list[str]) -> list[str]:
    args = ["--device", "/dev/dri", "--device", "/dev/kfd"]
    for gid in gpu_gids:
        args += ["--group-add", gid]
    return [*args, "--security-opt", "seccomp=unconfined", "--ipc=host"]


def llama_bench_command(probe: LlamaBenchProbe) -> list[str]:
    command = [
        "llama-bench", "-m", probe.model_container_path,
        "-o", "jsonl", "-oe", "jsonl", "-r", str(probe.repetitions),
        "-b", str(probe.batch), "-ub", str(probe.ubatch),
        "-ngl", str(probe.gpu_layers), "-fa", probe.flash_attn,
        "-ncmoe", str(probe.n_cpu_moe), "-mmp", "0",
        "-ctk", probe.ctk, "-ctv", probe.ctv, "-dev", probe.device,
        "-d", str(probe.depth), "--progress",
    ]
    return command + (
        ["-p", str(probe.prefill_tokens), "-n", "0"]
        if probe.series == "prefill"
        else ["-p", "0", "-n", str(probe.generation_tokens)]
    )


def docker_command(
    *,
    image: str,
    container_name: str,
    gpu_gids: list[str],
    model_directory: str,
    probe: LlamaBenchProbe,
) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", container_name,
        *docker_gpu_args(gpu_gids), "-v", f"{model_directory}:/models:ro",
        image, *llama_bench_command(probe),
    ]
