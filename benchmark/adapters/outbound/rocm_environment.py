#!/usr/bin/env python3
"""Gather environment/version metadata for a benchmark run: ROCm version,
llama.cpp build, GPU model/VRAM, host kernel. Used to build run-ids and
metadata.json so results stay comparable (and distinguishable) across
llama.cpp/ROCm upgrades over time.

Usage:
    ./environment_info.py --image llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench --device ROCm0
    ./environment_info.py --image llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (result.stdout or "") + (result.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def rocm_version(image: str) -> str | None:
    """Query the installed ROCm core meta-package version inside the
    image. ROCm 10's apt packaging split what used to be a single
    'rocm-core' package into gfx-target-specific meta-packages
    (amdrocm-core{version}-{gfx_target}, e.g. amdrocm-core10.0-gfx1201);
    there is no bare 'rocm-core' package to query anymore, so this
    pattern-matches whichever amdrocm-core* package is actually
    installed instead of hardcoding one name."""
    output = _run([
        "docker", "run", "--rm", "--entrypoint", "bash", image, "-c",
        "dpkg-query -W -f='${Package} ${Version}\\n' 'amdrocm-core*' 2>/dev/null | head -1",
    ])
    output = output.strip()
    if not output:
        return None
    parts = output.split(" ", 1)
    return parts[1] if len(parts) == 2 else None


def llama_cpp_build(image: str) -> dict:
    """Read the llama.cpp identity/commit files baked into the image by
    Dockerfile.rocm-10.0.0.ubuntu26 (see its STAGE 2 comments), instead
    of parsing `llama-cli --version` stdout. Two reasons this replaced
    the old regex-on-stdout approach:
      - llama-bench (what run_bench.py actually benchmarks) has no
        --version flag at all - only llama-cli/llama-server do.
      - Even where --version exists, a locally-built image reports
        'version: 0.4.0-dev (build 1, commit <sha>)' - the '(build N)'
        counter is meaningless for a local build (it's always "1"
        regardless of which commit was built), so it can't distinguish
        two different `master` builds from each other the way the old
        upstream-CI build-number scheme could.
    identity is a release tag (v0.4.0) when LLAMA_CPP_REF was a tag, or
    the short commit SHA otherwise (branch or raw commit ref) - see
    run_id() below for why that's the right thing to key run directories
    on."""
    identity = _run(["docker", "run", "--rm", "--entrypoint", "cat", image, "/app/.llama-cpp-identity"]).strip()
    commit = _run(["docker", "run", "--rm", "--entrypoint", "cat", image, "/app/.llama-cpp-commit"]).strip()
    return {
        "llama_cpp_identity": identity or None,
        "llama_cpp_commit": commit or None,
    }


def gpu_info(rocm_smi_device_index: int = 0) -> dict:
    """Query GPU model + VRAM from the HOST's rocm-smi (not the container,
    which typically doesn't ship rocm-smi in a lean runtime image)."""
    name_output = _run(["rocm-smi", "--showproductname"])
    vram_output = _run(["rocm-smi", "--showmeminfo", "vram"])

    name = None
    for line in name_output.splitlines():
        if f"GPU[{rocm_smi_device_index}]" in line and "Card Series" in line:
            # Line shape: "GPU[0]\t\t: Card Series: \t\tAMD Radeon AI PRO R9700"
            name = line.split(":", 2)[-1].strip()
            break

    vram_bytes = None
    for line in vram_output.splitlines():
        if f"GPU[{rocm_smi_device_index}]" in line and "VRAM Total Memory" in line:
            match = re.search(r"(\d+)\s*$", line.strip())
            if match:
                vram_bytes = int(match.group(1))
            break

    return {
        "gpu_name": name,
        "gpu_vram_bytes": vram_bytes,
        "gpu_vram_gib": round(vram_bytes / (1024 ** 3), 1) if vram_bytes else None,
    }


def host_info() -> dict:
    kernel = _run(["uname", "-r"]).strip() or None
    return {"host_kernel": kernel}


def gather(image: str, rocm_smi_device_index: int = 0) -> dict:
    info = {"image": image}
    info["rocm_version"] = rocm_version(image)
    info.update(llama_cpp_build(image))
    info.update(gpu_info(rocm_smi_device_index))
    info.update(host_info())
    return info


def slugify_version(version: str | None, prefix: str) -> str:
    """Turn a version string into a filesystem-safe run-id fragment."""
    if not version:
        return f"{prefix}unknown"
    safe = re.sub(r"[^A-Za-z0-9.]+", "-", version).strip("-")
    return f"{prefix}{safe}"


def short_rocm_version(version: str | None) -> str:
    """Extract the semver-like prefix from a full package version string,
    e.g. '7.2.4.70204-93~24.04' -> '7.2.4'. Falls back to the raw string
    (slugified) if it doesn't look like a dotted version."""
    if not version:
        return "unknown"
    match = re.match(r"(\d+\.\d+\.\d+)", version)
    if match:
        return match.group(1)
    return slugify_version(version, "")


def run_id(env: dict) -> str:
    """Build a run-id from ROCm version + llama.cpp identity, e.g.
    rocm10.0.0_llamacppv0.4.0 for a release build or
    rocm10.0.0_llamacpp5266f24 for a `master`/commit build. No
    date/timestamp - reruns with the same versions collide on purpose
    (run_bench.py refuses to overwrite unless --force), since the
    versions ARE the identity we care about tracking. Uses
    llama_cpp_identity (a slugified release tag, or a short commit SHA)
    rather than a raw build counter, since a locally-built image's build
    counter is always "1" regardless of which commit was actually
    built - it can't distinguish two different `master` builds the way
    upstream's CI build-number scheme could."""
    rocm_part = f"rocm{short_rocm_version(env.get('rocm_version'))}"
    identity = env.get("llama_cpp_identity")
    llama_part = slugify_version(identity, "llamacpp") if identity else "llamacppunknown"
    return f"{rocm_part}_{llama_part}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Docker image to inspect")
    parser.add_argument("--device-index", type=int, default=0, help="rocm-smi GPU index to report (default: 0)")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gather(args.image, args.device_index)
    if args.json:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for key, value in env.items():
            print(f"{key}: {value}")
        print(f"run_id: {run_id(env)}")


if __name__ == "__main__":
    main()
