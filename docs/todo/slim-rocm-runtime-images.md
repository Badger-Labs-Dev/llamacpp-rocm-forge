# Slim ROCm 10 runtime images

## Goal

Reduce the final llama.cpp ROCm runtime images without changing the supported local hardware profiles or their behavior:

- R9700: `gfx1201`, `LLAMA_HIP_UMA=OFF`
- Ryzen iGPU: `gfx1036`, `LLAMA_HIP_UMA=ON`
- Preserve the existing `bench`, `server`, and `light` final targets and tags.

The validated package experiment indicates these approximate shared-layer reductions:

| Profile | Current shared image layer | Target shared image layer |
|---|---:|---:|
| `gfx1036` | 6.49 GB | ~2.68 GB |
| `gfx1201` | 7.93 GB | ~3.02 GB |

## Evidence

The current runtime stage installs the broad `amdrocm10.0-gfx*` meta-package. That package depends on `amdrocm-core10.0-gfx*`, which brings workloads that llama.cpp does not appear to need at runtime:

- `amdrocm-ck10.0`: ~2.39 GB
- `amdrocm-rocshmem10.0`: ~514 MB
- `amdrocm-dnn-host10.0` plus `amdrocm-dnn10.0-gfx1201`: ~908 MB on `gfx1201`

A research-only runtime image built with the selected package set below:

- built successfully for both `gfx1036` and `gfx1201`;
- had no missing dependencies in `ldd /app/libggml-hip.so`;
- enumerated `ROCm0` R9700 and `ROCm1` integrated Radeon with real GPU device nodes and numeric host GIDs.

This proves HIP backend loading/device discovery, but not every model/kernel execution path. Real inference or benchmarking remains required before merging.

## Implementation plan

### 1. Replace the broad runtime meta-package

**File:** `docker/Dockerfile.rocm-10.0.0.ubuntu26`, `runtime-base` stage.

Keep the builder package unchanged:

```dockerfile
amdrocm-core-dev${ROCM_META_VERSION}-${GFX_TARGET}
```

It legitimately needs compilers, headers, and the HIP build environment.

For named single-target profiles, replace:

```dockerfile
amdrocm${ROCM_META_VERSION}-${GFX_TARGET}
```

with:

```dockerfile
amdrocm-runtime${ROCM_META_VERSION}
amdrocm-blas${ROCM_META_VERSION}-${GFX_TARGET}
```

Retain the broad unsuffixed `amdrocm${ROCM_META_VERSION}` path for `GFX_TARGET=all` until a separate fat-image package selection has been tested.

### 2. Restore the ROCm compatibility library paths

The broad core meta-package supplies conventional paths such as `/opt/rocm/lib`; the selected packages place payloads under `/opt/rocm/core-10.0/lib`.

Before the current linker-cache setup, create the compatibility symlinks and register them:

```dockerfile
RUN ln -s core-${ROCM_META_VERSION}/lib /opt/rocm/lib \
    && ln -s core-${ROCM_META_VERSION}/lib /opt/rocm/lib64 \
    && printf '%s\n' /opt/rocm/lib /opt/rocm/lib64 \
       > /etc/ld.so.conf.d/rocm.conf \
    && ldconfig
```

Do not replace this with `LD_LIBRARY_PATH`. The image must use the dynamic linker cache so the dynamically loaded HIP backend resolves consistently.

### 3. Update comments and documentation

Update the Dockerfile runtime-stage rationale and `docs/building.md` to explain:

- why the former broad AMD meta-package is no longer used for named single-target runtime images;
- why `amdrocm-runtime` plus target-specific `amdrocm-blas` is the selected llama.cpp runtime set;
- why ROCm compatibility symlinks are necessary;
- why the builder remains broad;
- why `GFX_TARGET=all` remains broad until separately validated.

Add approximate expected image-size reductions, labelled as version-dependent because AMD package sizes may change with ROCm releases.

### 4. Validate build configuration and build every final image

```bash
cd docker
docker buildx bake --print
docker buildx bake
```

Confirm all six leaf targets keep distinct tags and correct arguments:

- `bench-gfx1201`, `server-gfx1201`, `light-gfx1201`
- `bench-gfx1036`, `server-gfx1036`, `light-gfx1036`

### 5. Smoke-test static and runtime behavior

For each hardware profile:

1. Confirm the HIP backend has no unresolved libraries:

   ```bash
   docker run --rm --entrypoint sh <image> \
     -c 'ldd /app/libggml-hip.so | grep "not found" && exit 1 || true'
   ```

2. Confirm the final-stage executable starts:

   ```bash
   docker run --rm <bench-image> llama-bench --help
   docker run --rm <light-image> llama-cli --help
   docker run --rm <server-image> --help
   ```

3. Run `llama-cli --list-devices` with `/dev/kfd`, `/dev/dri`, and the host's numeric `video`/`render` GIDs.

4. Confirm the Bake-resolved tags and UMA values remain correct.

### 6. Validate real GPU execution

Device enumeration is necessary but insufficient. For each GPU:

- select the intended ROCm device explicitly (`ROCm0` for the R9700; `ROCm1` for the iGPU);
- load a real GGUF;
- run a short `llama-cli` generation or `llama-bench` workload;
- compare successful completion and basic throughput to the existing image.

For the iGPU, retain `LLAMA_HIP_UMA=ON` and use a model/configuration that exercises system-memory allocation beyond the BIOS-reserved framebuffer.

### 7. Measure, document, and commit

After real workload validation:

```bash
docker system df -v
docker image inspect <each-image>
```

Record before/after shared-layer sizes in the docs. Commit Dockerfile and documentation changes together using a focused Conventional Commit, e.g.:

```text
feat: slim ROCm runtime images
```

## Out of scope

- Slimming the builder image or its BuildKit cache.
- Pruning the existing BuildKit cache (currently separate from final-image size).
- Changing Bake profile structure, image tags, or UMA policy.
- Slimming the `GFX_TARGET=all` fat-image path before dedicated testing.
