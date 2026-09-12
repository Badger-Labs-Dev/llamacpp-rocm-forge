# Building the Docker image

The benchmark uses local Docker images rather than Toolbx or Podman. They are built for ROCm 10.0.0 on Ubuntu 26.04 for two local hardware profiles: the R9700 (`gfx1201`, HIP UMA **off**) and Ryzen 5 9600X iGPU (`gfx1036`, HIP UMA **on**).

The Dockerfile (`docker/Dockerfile.rocm-10.0.0.ubuntu26`) doesn't pin a llama.cpp version itself: that's a build-time `LLAMA_CPP_REF` argument, so the same file builds a tagged release, `master`, or a specific commit.

This repo is scoped to building/tagging images and running the benchmark itself, not to running `llama-server` as a persistent service. That belongs in `homelab-llm-router`; see "Running the server" below.

## Where this build comes from

`docker/Dockerfile.rocm-10.0.0.ubuntu26` is not a from-scratch build; it adapts two upstream sources, which are also credited in the file's own header comment:

- **ROCm packages**: pulled directly from AMD's official apt repo, per [AMD's ROCm 10.0.0 install docs](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=radeon&w=compute&gpu=amd-radeon-ai-pro-r9700&gfx=gfx1201&os=ubuntu&ubuntu-ver=26.04&i=pkgman) for this exact GPU/OS/ROCm combination ("Package manager (apt)" install method). AMD hasn't published a ROCm 10.0.0 `rocm/dev-ubuntu-26.04:...-complete` container image yet, so instead of `FROM`-ing one (what upstream llama.cpp's own Dockerfile does), STAGE 1 starts from plain `ubuntu:26.04` and installs the ROCm apt repo/keyring/packages by hand, following those docs.
- **llama.cpp build structure**: the build/runtime-split staging (a `builder` stage that compiles, then several slim final stages that each `COPY --from=builder` just one binary) mirrors [llama.cpp's own official ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (`.devops/rocm.Dockerfile` in that repo), using the same `GGML_HIP=ON`/`AMDGPU_TARGETS`/`GGML_BACKEND_DL=ON` cmake flags and the same `light`/`server`/(their `full`, our `bench`) split pattern. Adapted rather than copied wholesale: their base image is `rocm/dev-ubuntu-*-complete` (not available for ROCm 10.0.0/Ubuntu 26.04 yet, per above). Each named Bake profile builds one architecture-specific binary; `gfx1036` additionally sets llama.cpp's `LLAMA_HIP_UMA=ON` so the iGPU can allocate from system memory, while the R9700 profile keeps it off because it slows discrete GPUs.

## Build targets

The Dockerfile has three final targets, each producing a single-binary image:

- `bench`: `llama-bench` only. What `run_bench.py` drives.
- `server`: `llama-server` only, with `LLAMA_ARG_*` env var defaults set (host/port/`-ngl`). Not run from this repo (see "Running the server" below); built here so a benchmarked image can be handed off elsewhere.
- `light`: `llama-cli` only, for quick interactive checks.

Tag convention: `<image>:rocm_<rocm-version>-llama_<llama-cpp-ref>-<gfx-target>-<target>`, e.g. `llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench`. Underscore binds a label to its value (`rocm_10.0.0`, `llama_v0.4.0`); hyphen separates distinct fields, keeping the version numbers, the gfx target, and the build target from running together.

The gfx-target segment (`gfx1201`) is required, not cosmetic, same reason as the `-bench`/`-server`/`-light` suffix: building two different `GFX_TARGET`s with the same tag makes the second build silently overwrite the first in `docker images`. See "Building for the local GPUs" below for the named hardware profiles.

`bench` and `light` have no fixed `ENTRYPOINT` (their binary is invoked with different args every run, so `/app` is just added to `PATH` instead: `docker run image llama-bench -m ...` works the way running it on a normal host would). `server` keeps a fixed entrypoint since it's a persistent daemon.

## Building (recommended: `docker buildx bake`)

Tags aren't hand-typed: `docker/docker-bake.hcl` computes them from `ROCM_VERSION`/`LLAMA_CPP_REF` and knows about all three targets, so a plain `docker buildx bake` (no `-t`, no `--target`) builds `bench`+`server`+`light` together with correct tags on all three:

```bash
cd docker
docker buildx bake
```

Build just one target by naming it:

```bash
docker buildx bake bench
```

Override the llama.cpp ref (a release tag, `master`, or a full commit SHA, see below) for one build without editing the file:

```bash
LLAMA_CPP_REF=master docker buildx bake bench
```

`docker buildx bake --print [target]` shows the resolved tags/args without building anything, useful for checking what a tag will come out as before committing to a real build.

## Building for the local GPUs

The recommended path is a named hardware profile in `docker-bake.hcl`, not a generic target matrix. Each profile makes its architecture and memory-allocation policy reviewable in one place:

| Profile | GPU | `GFX_TARGET` | `LLAMA_HIP_UMA` |
|---|---|---|---|
| `*-gfx1201` | Radeon AI PRO R9700 | `gfx1201` | `OFF` |
| `*-gfx1036` | Ryzen 5 9600X integrated Radeon Graphics | `gfx1036` | `ON` |

`LLAMA_HIP_UMA=ON` enables HIP managed allocations, allowing the iGPU to use system memory beyond its BIOS-reserved frame buffer. It is deliberately **off** for the R9700: llama.cpp documents a performance penalty on discrete GPUs.

A plain Bake command builds both profiles with distinct tags:

```bash
# both bench images
docker buildx bake bench

# all three final images for both GPUs
docker buildx bake

# one iGPU image while iterating
docker buildx bake bench-gfx1036
```

The resulting bench tags are:

```text
llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench
llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1036-bench
```

The tags must retain their gfx segment: otherwise one architecture-specific build would silently replace the other in the local image store.

### ROCm runtime package selection and image size

The `builder` stage continues to install AMD's broad target-specific `amdrocm-core-dev10.0-gfx*` meta-package. Building llama.cpp needs its HIP compiler, headers, static libraries, and CMake configuration, so narrowing that package is out of scope.

Named final images use a narrower runtime set instead: `amdrocm-runtime10.0` plus `amdrocm-blas10.0-gfx*`. The former `amdrocm10.0-gfx*` meta-package also installs CK, ROCSHMEM, and DNN workloads that llama.cpp does not appear to require at runtime. The selected packages cover the HIP runtime and target-specific BLAS libraries required by llama.cpp's dynamically loaded HIP backend.

These selected packages install shared objects beneath `/opt/rocm/core-10.0/lib`, while the broad core package exposes conventional `/opt/rocm/lib` and `/opt/rocm/lib64` paths. The runtime stage creates compatibility symlinks for those paths, writes them to `ld.so.conf.d`, and runs `ldconfig`. This deliberately uses the dynamic linker cache rather than `LD_LIBRARY_PATH`, so `dlopen()` of the HIP backend resolves consistently regardless of environment overrides.

Package sizes vary by AMD release, but the validated ROCm 10.0 package experiment reduced the shared ROCm image layer approximately as follows:

| Profile | Previous shared layer | Expected selected-package layer | Approx. reduction |
|---|---:|---:|---:|
| `gfx1036` | 6.49 GB | ~2.68 GB | ~3.81 GB |
| `gfx1201` | 7.93 GB | ~3.02 GB | ~4.91 GB |

These are expected image-layer reductions, not a model-execution guarantee. Run device enumeration and real GPU inference or benchmarking on the intended hardware before adopting a rebuilt image.

### Fat multi-arch image (`GFX_TARGET=all`)

The Dockerfile still supports an explicit hand-built fat image. Its `GFX_TARGETS_ALL` list now includes `gfx1036`, and it deliberately continues to install AMD's broad unsuffixed all-architecture `amdrocm10.0` meta-package: a minimal package selection for this multi-architecture path has not yet been validated. It keeps HIP UMA off by default, so it is appropriate for portable discrete-GPU images, not as the recommended iGPU image. Use the named `*-gfx1036` profiles for the Ryzen iGPU.

### Makefile alternative

`docker buildx bake` is the documented path. The retained Makefile mirrors the two local profiles by default:

```bash
make bench
# override to build only the R9700 image
make bench GFX_TARGETS="gfx1201"
```

`HIP_UMA_TARGETS` defaults to `gfx1036`; it adds `--build-arg ENABLE_HIP_UMA=ON` only for matching targets.

### Why bake over hand-typed `docker build -t ...`?

Nothing about Docker auto-generates a tag from `--build-arg` values; that's always the caller's job. Two ways to automate it were considered:

| Approach | Auto tag from vars | Build-all default | Standard/portable | New tooling |
|---|---|---|---|---|
| **`docker buildx bake`** | Yes, native | Yes (`default` group) | Ships with Docker itself | HCL syntax, small for this use case |
| `Makefile` (see `docker/Makefile`) | Yes, via `make` vars | Yes (`make all`) | Needs `make` | Familiar if you already know Make |
| Hand-typed `docker build -t ...` | No | No | Nothing extra | None, but every tag typed by hand |

`bake` won: it's Docker's own tool (not a repo-specific wrapper script to maintain), and it's the only option here with a real `default` group. A plain `docker buildx bake` builds all three targets; `docker buildx bake bench` narrows to one. `docker/Makefile` is kept in the repo only as a working comparison, not the documented path.

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

Tag the image with whatever makes the build identifiable to you (beyond the required target/gfx suffixes). The run-id the benchmark records is derived from what's actually baked into the image (see below), not from the docker tag, so the rest of the tag is just for your own bookkeeping.

## Check the image

```bash
docker run --rm llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench llama-bench --help
docker run --rm --entrypoint cat llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench /app/.llama-cpp-identity
```

The last command shows what `run_id()` will call this build in results directory names: a release tag (`v0.4.0`) if `LLAMA_CPP_REF` was a tag, otherwise a short commit SHA.

GPU access is supplied by `run_bench.py` when it starts each benchmark container. The script passes `/dev/dri`, `/dev/kfd`, and the host's numeric `video` and `render` GIDs. It defaults to `ROCm0`, the discrete R9700; the iGPU is currently `ROCm1`, but enumerate devices inside the container before selecting it because the ordering is not a stable contract.

## Running the server

`docker/legacy/docker-compose.yml` is a retired reference copy; see its header comment. The real, actively-deployed `llama-server` stack lives in `homelab-llm-router`, alongside that repo's other always-on compose stacks (`vllm/`, `litellm-headroom/`), not here. This repo's job stops at building and tagging the `server` image.

## Older ROCm 7.2.4 image

`docker/legacy/Dockerfile.rocm-7.2.4-rdma-fix.ubuntu` is retired, kept only as a short-term reference, not actively built or maintained. See the header comment in that file for why.

For the surrounding benchmark workflow, return to the [README](../README.md).
