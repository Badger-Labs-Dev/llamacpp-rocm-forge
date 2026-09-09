"""Docker container execution adapter: runs one llama-bench probe via
`docker run`, with VRAM-leak protection (timeout -> `docker kill`) and
process-lifetime container tracking so an interrupted script still
cleans up after itself.

Extracted from run_bench.py's run_one()/`_ACTIVE_CONTAINERS` per step 7
of the refactor roadmap. This is the most behavior-critical code in the
file - a bug here leaks VRAM on real hardware - so this adapter changes
nothing about *when* to kill a container or *what* counts as a timeout,
only *where* that logic lives. Validated with a supervised real-GPU
smoke test (see docs/architecture.md), not just mocked unit tests.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import IO

# Names of docker containers currently running a probe, so they can be
# force-killed if the parent script is interrupted (Ctrl-C, SIGTERM, an
# unhandled exception). `docker run --rm` alone does not protect against
# this: it only removes the container after IT exits, which does not
# happen just because the client/parent process died.
_ACTIVE_CONTAINERS: set[str] = set()


def new_container_name() -> str:
    return f"r9700-llm-bench-{uuid.uuid4().hex[:12]}"


def kill_active_containers() -> None:
    """`docker kill` every container this process is currently tracking.
    Safe to call from a signal handler or atexit hook - iterates a copy
    since kill_active_containers() itself mutates _ACTIVE_CONTAINERS."""
    for name in list(_ACTIVE_CONTAINERS):
        subprocess.run(["docker", "kill", name], capture_output=True)
        _ACTIVE_CONTAINERS.discard(name)


@dataclass
class ProbeOutcome:
    returncode: int
    timed_out: bool
    elapsed_seconds: float


def run_probe(
    docker_cmd: list[str],
    *,
    container_name: str,
    jsonl_file: IO[str],
    stderr_file: IO[str],
    timeout_seconds: int,
) -> ProbeOutcome:
    """Run one `docker run ...` invocation, tracking its container name
    for cleanup and `docker kill`-ing it on timeout.

    jsonl_file/stderr_file are already-open file objects the caller owns
    (this function writes to but does not close them). A timeout writes
    a marker line to stderr_file before returning, matching run_one()'s
    original inline behavior.
    """
    _ACTIVE_CONTAINERS.add(container_name)
    started_at = time.monotonic()
    try:
        proc = subprocess.run(
            docker_cmd, stdout=jsonl_file, stderr=stderr_file,
            timeout=timeout_seconds,
        )
        return ProbeOutcome(
            returncode=proc.returncode, timed_out=False,
            elapsed_seconds=time.monotonic() - started_at,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run(timeout=...) only kills the direct child (the
        # docker CLI), not the container it started - without this
        # explicit `docker kill`, a timed-out probe leaks VRAM exactly
        # like an interrupted script would.
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        stderr_file.write(f"\n[TIMEOUT after {timeout_seconds}s - container killed]\n")
        stderr_file.flush()
        return ProbeOutcome(
            returncode=-1, timed_out=True,
            elapsed_seconds=time.monotonic() - started_at,
        )
    finally:
        _ACTIVE_CONTAINERS.discard(container_name)
