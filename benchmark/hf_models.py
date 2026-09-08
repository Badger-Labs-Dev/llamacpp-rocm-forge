#!/usr/bin/env python3
"""Resolve Hugging Face model references (URIs, repo IDs, ollama-style tags)
into local file paths, downloading via the Hugging Face cache if needed.

Supports the forms a user is likely to copy off the model card's "Download
with hf CLI" / "Use this model" buttons:

    hf://org/repo/path/to/file.gguf     -> download that exact file
    org/repo                            -> no filename given: list available
                                            .gguf files in the repo and let
                                            the user pick one interactively
                                            (or --file / non-interactive)
    org/repo:QUANT                      -> ollama-style tag; matches a
                                            *_QUANT.gguf file case-insensitively
    org/repo/file.gguf                  -> same as hf:// form, prefix optional

Downloads use huggingface_hub.hf_hub_download, which is itself already
cache-aware: an already-cached file returns instantly with no network
call, a missing one downloads and populates the cache. Nothing here
bypasses or duplicates that cache.

Usage (as a library, from run_bench.py):
    from hf_models import resolve_hf_reference
    local_path = resolve_hf_reference("hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")

Usage (standalone, to just list or download):
    ./hf_models.py list unsloth/Qwen3.6-35B-A3B-GGUF
    ./hf_models.py resolve hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    ./hf_models.py resolve unsloth/Qwen3.6-35B-A3B-GGUF          # interactive picker
    ./hf_models.py resolve unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL
"""

from __future__ import annotations

import argparse
import re
import sys

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # older huggingface_hub versions
        from huggingface_hub.utils import HfHubHTTPError
    # hf_hub_download's tqdm progress bar assumes a real interactive
    # terminal; when stdout isn't one (piped, redirected, run under a
    # supervising process/log capture) its carriage-return redraws can
    # render as a flood of near-blank lines instead of updating in place.
    # Downloads are a one-time, background part of this tool - not worth
    # that risk - so bars are disabled unconditionally.
    disable_progress_bars()
except ImportError:
    HfApi = None
    hf_hub_download = None
    HfHubHTTPError = Exception


class HfReferenceError(Exception):
    pass


def _require_huggingface_hub() -> None:
    if HfApi is None:
        raise HfReferenceError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )


def looks_like_hf_reference(value: str) -> bool:
    """True if `value` looks like something this module should handle,
    rather than a plain filesystem path. Deliberately conservative: plain
    paths (./foo.gguf, /home/.../foo.gguf, C:\\...) must never be
    misdetected as HF references."""
    if value.startswith("hf://"):
        return True
    if value.startswith(("/", "./", "../", "~")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", value):  # Windows drive letter
        return False
    # org/repo, org/repo/file.gguf, or org/repo:quant - exactly one slash
    # before an optional further path, no leading slash.
    return bool(re.match(r"^[\w.-]+/[\w.-]+(/.*)?(:[\w.-]+)?$", value))


def parse_hf_reference(value: str) -> tuple[str, str | None]:
    """Parse a reference into (repo_id, filename_or_none).

    filename is None when the caller only gave org/repo (or org/repo:quant
    with no matching file resolved yet) and needs list_gguf_files() /
    match_quant() to find the actual file.
    """
    value = value.removeprefix("hf://")

    if ":" in value and "/" in value.split(":")[0]:
        # ollama-style org/repo:QUANT
        repo_part, quant = value.rsplit(":", 1)
        return repo_part, f"__quant__:{quant}"

    parts = value.split("/")
    if len(parts) < 2:
        raise HfReferenceError(f"Not a valid Hugging Face reference: {value!r}")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, filename


def list_gguf_files(repo_id: str) -> list[dict]:
    """List every .gguf file in a repo with its size, largest-context-first
    is not guaranteed - sorted by path for stable, predictable output."""
    _require_huggingface_hub()
    api = HfApi()
    try:
        entries = api.list_repo_tree(repo_id, recursive=True)
    except HfHubHTTPError as e:
        raise HfReferenceError(f"Could not list files for {repo_id}: {e}") from e

    files = []
    for entry in entries:
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path and path.endswith(".gguf"):
            files.append({"path": path, "size_bytes": size})
    files.sort(key=lambda f: f["path"])
    return files


def format_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "unknown size"
    gib = size_bytes / (1024 ** 3)
    return f"{gib:.1f} GiB"


def match_quant(files: list[dict], quant: str) -> dict | None:
    """Match an ollama-style quant tag (e.g. 'UD-Q4_K_XL') against a
    repo's .gguf filenames, case-insensitively, matching the stem before
    the .gguf extension."""
    quant_lower = quant.lower()
    for f in files:
        stem = f["path"].rsplit("/", 1)[-1].removesuffix(".gguf").lower()
        if stem == quant_lower or stem.endswith(f"-{quant_lower}") or stem.endswith(f"_{quant_lower}"):
            return f
    return None


def prompt_pick_file(files: list[dict], repo_id: str) -> dict:
    """Interactive picker: numbered list, read a choice from stdin.
    Raises HfReferenceError if stdin isn't interactive (e.g. running in a
    script/cron) so callers get a clear error instead of hanging."""
    if not sys.stdin.isatty():
        names = ", ".join(f["path"] for f in files)
        raise HfReferenceError(
            f"{repo_id} has multiple .gguf files and no filename/quant was "
            f"specified, and stdin is not interactive to prompt. Available "
            f"files: {names}"
        )

    print(f"\nMultiple .gguf files found in {repo_id}:\n", file=sys.stderr)
    for i, f in enumerate(files, start=1):
        print(f"  [{i}] {f['path']}  ({format_size(f['size_bytes'])})", file=sys.stderr)
    print(file=sys.stderr)

    while True:
        choice = input(f"Pick a file [1-{len(files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("Invalid choice, try again.", file=sys.stderr)


def resolve_hf_reference(reference: str, quiet: bool = False) -> str:
    """Resolve a reference (hf:// URI, org/repo, org/repo:quant, or
    org/repo/file.gguf) to a local file path, downloading via the HF cache
    if not already present. Prints progress to stderr unless quiet=True.
    """
    _require_huggingface_hub()
    repo_id, filename = parse_hf_reference(reference)

    if filename is not None and filename.startswith("__quant__:"):
        quant = filename.removeprefix("__quant__:")
        files = list_gguf_files(repo_id)
        if not files:
            raise HfReferenceError(f"No .gguf files found in {repo_id}")
        match = match_quant(files, quant)
        if match is None:
            available = ", ".join(f["path"] for f in files)
            raise HfReferenceError(
                f"No file matching quant {quant!r} in {repo_id}. Available: {available}"
            )
        filename = match["path"]
    elif filename is None:
        files = list_gguf_files(repo_id)
        if not files:
            raise HfReferenceError(f"No .gguf files found in {repo_id}")
        if len(files) == 1:
            filename = files[0]["path"]
        else:
            filename = prompt_pick_file(files, repo_id)["path"]

    if not quiet:
        print(f"Resolving {repo_id}/{filename} via Hugging Face cache...", file=sys.stderr)

    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except HfHubHTTPError as e:
        raise HfReferenceError(f"Download failed for {repo_id}/{filename}: {e}") from e

    if not quiet:
        print(f"  -> {local_path}", file=sys.stderr)
    return local_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List .gguf files in a repo")
    list_p.add_argument("repo_id", help="e.g. unsloth/Qwen3.6-35B-A3B-GGUF")

    resolve_p = sub.add_parser("resolve", help="Resolve a reference to a local path, downloading if needed")
    resolve_p.add_argument("reference", help="hf:// URI, org/repo, org/repo:quant, or org/repo/file.gguf")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "list":
            files = list_gguf_files(args.repo_id)
            if not files:
                print(f"No .gguf files found in {args.repo_id}")
                return
            for f in files:
                print(f"{f['path']:60s} {format_size(f['size_bytes'])}")
        elif args.command == "resolve":
            path = resolve_hf_reference(args.reference)
            print(path)
    except HfReferenceError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
