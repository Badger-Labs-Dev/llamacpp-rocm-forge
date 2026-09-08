# r9700-llm-bench

llama.cpp + ROCm 7.2.4 benchmarking on an AMD Radeon AI PRO R9700 (gfx1201),
running on plain Docker (Ubuntu 24.04 base) instead of Toolbx/Podman.

Forked out of [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes),
which targets a different GPU (Strix Halo APU, gfx1151) and a different host
setup (Fedora Toolbx + their own "AI Toolbox Cockpit" orchestration tooling
across their own SSH hosts). This repo strips that down to what actually
applies to our hardware and workflow.

## Contents

- `docker/` - the Dockerfile (Ubuntu 24.04 base, ROCm 7.2.4, gfx1201 target)
  plus its direct build dependencies (grammar patch, VRAM estimator script).
- `benchmark/` - upstream's benchmark orchestration scripts, kept for
  reference/adaptation. `UPSTREAM_RUNBOOK_REFERENCE.md` is their agent runbook;
  it assumes their SSH hosts, `llama-cockpit` CLI, and Toolbx - none of which
  we have. Treat it as a spec for the benchmark *protocol* (depths, batch
  sizes, repetitions, etc.), not a runnable procedure here.
- `docs/` - reference docs (VRAM estimation, local build notes) carried over
  as-is.

## Status

Not yet adapted to run standalone. Next step: write a script that drives
`llama-bench` through the same depth/ubatch sweep the upstream runbook
defines, targeting a plain `docker run` container instead of their Toolbx +
Cockpit stack.

## Building the image

```bash
cd docker
docker build -f Dockerfile.rocm-7.2.4-rdma-fix.ubuntu -t r9700-llm-bench:rocm-7.2.4 .
```

## Running the container

```bash
docker run -it --rm \
  --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v ~/models:/models \
  r9700-llm-bench:rocm-7.2.4
```
