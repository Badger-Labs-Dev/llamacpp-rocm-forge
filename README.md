# llamacpp-rocm-forge

Build llama.cpp Docker images for AMD ROCm — one image per binary, one per GPU profile.

## What this repo does

Compiles [llama.cpp](https://github.com/ggerganov/llama.cpp) with ROCm support and ships the result as lean Docker images:

| Image | Contents |
|-------|----------|
| `server` | `llama-server` + shared libs |
| `light` | `llama-cli` + shared libs |

Each image is built for two GPU profiles:

- **R9700** (`gfx1201`) — discrete GPU, HIP UMA disabled
- **iGPU** (`gfx1036`) — integrated GPU, HIP UMA enabled (required for system-memory allocations on AMD iGPUs)

## Quick start

```bash
# Build all images for both GPUs
docker buildx bake

# Build only the server image for the R9700
docker buildx bake server-gfx1201

# Override the llama.cpp commit ref
LLAMA_CPP_REF=master docker buildx bake
```

Images are tagged as:

```
llamacpp-rocm-forge:rocm_10.0.0-llama_<ref>-<gfx>-<target>
```

e.g. `llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-server`

### Using the images

```bash
# Serve a model
docker run --rm -p 8080:8080 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-server \
  llama-server --model /models/model.gguf --host 0.0.0.0 --port 8080

# Run a single inference
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-light \
  llama-cli -m /models/model.gguf -n 256 -p "Hello"
```

## Image identity

Every image carries two identity files:

```bash
docker run --rm <image> cat /app/.llama-cpp-identity
docker run --rm <image> cat /app/.rocm-meta-version
```

These are the recommended way to identify a build at runtime — more stable than parsing `--version` stdout, whose format has changed upstream before.

## Building from scratch

See [docs/building.md](docs/building.md) for the full build system reference, including:

- The `docker-bake.hcl` declarative build configuration
- The alternative `Makefile` (kept for comparison)
- HIP UMA configuration details
- Multi-arch build notes

## Architecture

See [docs/architecture.md](docs/architecture.md) for a deep dive into the Dockerfile stages, the build pipeline, and design decisions.

## Why Docker images instead of bare binaries?

- **Reproducibility**: the same Dockerfile + commit ref produces the same image every time
- **Portability**: works on any ROCm-capable host without installing build dependencies
- **Isolation**: each image carries only what it needs — no system-level library conflicts
- **Identity**: the `.llama-cpp-identity` file makes it trivial to verify which build you're running

## License

llama.cpp is licensed under MIT. This repo's Dockerfiles and build configuration inherit that license.
