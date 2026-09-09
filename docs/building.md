# Building the Docker image

The benchmark uses a local Docker image rather than Toolbx or Podman. It is built for ROCm 10.0.0 on Ubuntu 26.04, and defaults to a single GPU target — the R9700 (`gfx1201`) — but can build for any gfx target(s) AMD's ROCm 10.0.0 apt repo publishes; see "Building for multiple GPUs" below.

The Dockerfile (`docker/Dockerfile.rocm-10.0.0.ubuntu26`) doesn't pin a llama.cpp version itself — that's a build-time `LLAMA_CPP_REF` argument, so the same file builds a tagged release, `master`, or a specific commit.

This repo is scoped to building/tagging images and running the benchmark itself — not to running `llama-server` as a persistent service. That belongs in `homelab-llm-router`; see "Running the server" below.

## Where this build comes from

`docker/Dockerfile.rocm-10.0.0.ubuntu26` is not a from-scratch build; it adapts two upstream sources, which are also credited in the file's own header comment:

- **ROCm packages**: pulled directly from AMD's official apt repo, per [AMD's ROCm 10.0.0 install docs](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=radeon&w=compute&gpu=amd-radeon-ai-pro-r9700&gfx=gfx1201&os=ubuntu&ubuntu-ver=26.04&i=pkgman) for this exact GPU/OS/ROCm combination ("Package manager (apt)" install method). AMD hasn't published a ROCm 10.0.0 `rocm/dev-ubuntu-26.04:...-complete` container image yet, so instead of `FROM`-ing one (what upstream llama.cpp's own Dockerfile does), STAGE 1 starts from plain `ubuntu:26.04` and installs the ROCm apt repo/keyring/packages by hand, following those docs.
- **llama.cpp build structure**: the build/runtime-split staging (a `builder` stage that compiles, then several slim final stages that each `COPY --from=builder` just one binary) mirrors [llama.cpp's own official ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (`.devops/rocm.Dockerfile` in that repo) — same `GGML_HIP=ON`/`AMDGPU_TARGETS`/`GGML_BACKEND_DL=ON` cmake flags, same `light`/`server`/(their `full`, our `bench`) split pattern. Adapted rather than copied wholesale: their base image is `rocm/dev-ubuntu-*-complete` (not available for ROCm 10.0.0/Ubuntu 26.04 yet, per above), and their `ROCM_DOCKER_ARCH` builds a fat multi-GPU binary for every gfx target at once, where this Dockerfile currently builds one target (`gfx1201`) at a time via `AMDGPU_TARGETS=${GFX_TARGET}`.

## Build targets

The Dockerfile has three final targets, each producing a single-binary image:

- `bench` — `llama-bench` only. What `run_bench.py` drives.
- `server` — `llama-server` only, with `LLAMA_ARG_*` env var defaults set (host/port/`-ngl`). Not run from this repo (see "Running the server" below) — built here so a benchmarked image can be handed off elsewhere.
- `light` — `llama-cli` only, for quick interactive checks.

Tag convention: `<image>:rocm_<rocm-version>-llama_<llama-cpp-ref>-<gfx-target>-<target>`, e.g. `llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench`. Underscore binds a label to its value (`rocm_10.0.0`, `llama_v0.4.0`); hyphen separates distinct fields — keeps the version numbers, the gfx target, and the build target from running together.

The gfx-target segment (`gfx1201`) is required, not cosmetic, same reason as the `-bench`/`-server`/`-light` suffix: building two different `GFX_TARGET`s with the same tag makes the second build silently overwrite the first in `docker images`. See "Building for multiple GPUs" below for building more than one target in one invocation.

`bench` and `light` have no fixed `ENTRYPOINT` (their binary is invoked with different args every run, so `/app` is just added to `PATH` instead — `docker run image llama-bench -m ...` works the way running it on a normal host would). `server` keeps a fixed entrypoint since it's a persistent daemon.

## Building (recommended: `docker buildx bake`)

Tags aren't hand-typed — `docker/docker-bake.hcl` computes them from `ROCM_VERSION`/`LLAMA_CPP_REF` and knows about all three targets, so a plain `docker buildx bake` (no `-t`, no `--target`) builds `bench`+`server`+`light` together with correct tags on all three:

```bash
cd docker
docker buildx bake
```

Build just one target by naming it:

```bash
docker buildx bake bench
```

Override the llama.cpp ref (a release tag, `master`, or a full commit SHA — see below) for one build without editing the file:

```bash
LLAMA_CPP_REF=master docker buildx bake bench
```

`docker buildx bake --print [target]` shows the resolved tags/args without building anything — useful for checking what a tag will come out as before committing to a real build.

## Building for multiple GPUs

By default, both `bake` and `make` build a single-arch image for `gfx1201` (the R9700) — one `AMDGPU_TARGETS` value baked into the compiled binary, one image per build target. This is deliberately *not* a fat multi-arch image the way [upstream llama.cpp's Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) builds one (`AMDGPU_TARGETS='gfx908;gfx90a;...'`, every arch in one binary) — see [docs/building.md#where-this-build-comes-from](#where-this-build-comes-from) for why. Single-arch keeps each image smaller and each build faster; the tradeoff is one image per GPU instead of one image that runs anywhere.

To build for more than one GPU, pass a list — `docker-bake.hcl`'s `GFX_TARGETS` (comma-separated) or the Makefile's `GFX_TARGETS` (space-separated) — and each entry fans out into its own fully-tagged image, naming which gfx target it's for:

```bash
# bake: comma-separated
GFX_TARGETS=gfx1201,gfx1151 docker buildx bake bench
# -> llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench
# -> llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1151-bench

# make: space-separated
make bench GFX_TARGETS="gfx1201 gfx1151"
```

Each entry must be a gfx target AMD actually publishes ROCm 10.0.0 apt meta-packages for (`amdrocm10.0-<gfx>`, `amdrocm-core-dev10.0-<gfx>`) — check [AMD's meta-packages table](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman#rocm-install-meta-packages) for the current list. There's no built-in "build for every supported arch" default — `GFX_TARGETS` is meant to stay an explicit, maintained set matching the GPUs actually in use, not silently balloon build time by building archs nobody runs.

### Why bake over hand-typed `docker build -t ...`?

Nothing about Docker auto-generates a tag from `--build-arg` values — that's always the caller's job. Two ways to automate it were considered:

| Approach | Auto tag from vars | Build-all default | Standard/portable | New tooling |
|---|---|---|---|---|
| **`docker buildx bake`** | Yes, native | Yes (`default` group) | Ships with Docker itself | HCL syntax, small for this use case |
| `Makefile` (see `docker/Makefile`) | Yes, via `make` vars | Yes (`make all`) | Needs `make` | Familiar if you already know Make |
| Hand-typed `docker build -t ...` | No | No | Nothing extra | None, but every tag typed by hand |

`bake` won: it's Docker's own tool (not a repo-specific wrapper script to maintain), and it's the only option here with a real `default` group — `docker buildx bake` alone builds all three targets, `docker buildx bake bench` narrows to one. `docker/Makefile` is kept in the repo only as a working comparison, not the documented path.

## Building by hand (what `bake`/`make` are doing under the hood)

Useful for understanding, or if you don't have `buildx`/`make` available. From the repository root:

```bash
cd docker
docker build \
  --build-arg LLAMA_CPP_REF=v0.4.0 \
  --build-arg GFX_TARGET=gfx1201 \
  --target bench \
  -t llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench \
  -f Dockerfile.rocm-10.0.0.ubuntu26 \
  .
```

The benchmark's default `--image` value is `llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench`, matching this build, so no extra flag is needed after it completes. Check [llama.cpp's releases page](https://github.com/ggml-org/llama.cpp/releases) for the current tag and adjust `LLAMA_CPP_REF`/the image tag together if you're building a newer one.

The `server` image builds the same way with `--target server` and a `-server` tag suffix instead:

```bash
docker build \
  --build-arg LLAMA_CPP_REF=v0.4.0 \
  --build-arg GFX_TARGET=gfx1201 \
  --target server \
  -t llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-server \
  -f Dockerfile.rocm-10.0.0.ubuntu26 \
  .
```

## Build `master` or a specific commit

`LLAMA_CPP_REF` accepts a release tag, a branch name, or a full 40-character commit SHA (short/abbreviated hashes don't work; GitHub's anonymous fetch-by-hash requires the full SHA):

```bash
LLAMA_CPP_REF=master docker buildx bake bench
# or by hand:
docker build \
  --build-arg LLAMA_CPP_REF=master \
  --build-arg GFX_TARGET=gfx1201 \
  --target bench \
  -t llamacpp-rocm-forge:rocm_10.0.0-llama_master-gfx1201-bench \
  -f Dockerfile.rocm-10.0.0.ubuntu26 \
  .
```

Tag the image with whatever makes the build identifiable to you (beyond the required target/gfx suffixes) — the run-id the benchmark records is derived from what's actually baked into the image (see below), not from the docker tag, so the rest of the tag is just for your own bookkeeping.

## Check the image

```bash
docker run --rm llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench llama-bench --help
docker run --rm --entrypoint cat llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench /app/.llama-cpp-identity
```

The last command shows what `run_id()` will call this build in results directory names — a release tag (`v0.4.0`) if `LLAMA_CPP_REF` was a tag, otherwise a short commit SHA.

GPU access is supplied by `run_bench.py` when it starts each benchmark container. The script passes `/dev/dri`, `/dev/kfd`, and the host's numeric `video` and `render` GIDs. It targets `ROCm0`, the discrete R9700; this host also exposes the Ryzen iGPU as `ROCm1`.

## Running the server

`docker/legacy/docker-compose.yml` is a retired reference copy — see its header comment. The real, actively-deployed `llama-server` stack lives in `homelab-llm-router`, alongside that repo's other always-on compose stacks (`vllm/`, `litellm-headroom/`), not here. This repo's job stops at building and tagging the `server` image.

## Older ROCm 7.2.4 image

`docker/legacy/Dockerfile.rocm-7.2.4-rdma-fix.ubuntu` is retired — kept only as a short-term reference, not actively built or maintained. See the header comment in that file for why.

For the surrounding benchmark workflow, return to the [README](../README.md).
