#!/usr/bin/env python3
"""Gather environment/version metadata for a benchmark run: ROCm version,
llama.cpp build, GPU model/VRAM, host kernel. Used to build run-ids and
metadata.json so results stay comparable (and distinguishable) across
llama.cpp/ROCm upgrades over time.

Usage:
    ./environment_info.py --image r9700-llm-bench:rocm-7.2.4 --device ROCm0
    ./environment_info.py --image r9700-llm-bench:rocm-7.2.4 --json
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
    """Query the installed rocm-core package version inside the image."""
    output = _run(["docker", "run", "--rm", image, "dpkg-query", "-W", "-f=${Version}", "rocm-core"])
    output = output.strip()
    return output or None


def llama_cpp_build(image: str) -> dict:
    """Parse llama-cli --version output: 'version: 9974 (94761d304)'."""
    output = _run(["docker", "run", "--rm", image, "llama-cli", "--version"])
    match = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", output)
    if not match:
        return {"build_number": None, "build_commit": None}
    return {"build_number": int(match.group(1)), "build_commit": match.group(2)}


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
    """Build a run-id from ROCm version + llama.cpp build number, e.g.
    rocm7.2.4_llamacpp9974. No date/timestamp - reruns with the same
    versions collide on purpose (run_bench.py refuses to overwrite unless
    --force), since the versions ARE the identity we care about tracking."""
    rocm_part = f"rocm{short_rocm_version(env.get('rocm_version'))}"
    build_number = env.get("build_number")
    llama_part = f"llamacpp{build_number}" if build_number is not None else "llamacppunknown"
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
