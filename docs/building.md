# Building the Docker image

The benchmark uses a local Docker image rather than Toolbx or Podman. It is built for ROCm 7.2.4 on the R9700 (`gfx1201`) and uses Ubuntu 24.04.

From the repository root:

```bash
cd docker
docker build \
  -f Dockerfile.rocm-7.2.4-rdma-fix.ubuntu \
  -t r9700-llm-bench:rocm-7.2.4 \
  .
```

The benchmark's default `--image` value is `r9700-llm-bench:rocm-7.2.4`, so no extra flag is needed after this build completes.

## Check the image

```bash
docker run --rm r9700-llm-bench:rocm-7.2.4 llama-bench --version
docker run --rm r9700-llm-bench:rocm-7.2.4 llama-bench --help
```

GPU access is supplied by `run_bench.py` when it starts each benchmark container. The script passes `/dev/dri`, `/dev/kfd`, and the host's numeric `video` and `render` GIDs. It targets `ROCm0`, the discrete R9700; this host also exposes the Ryzen iGPU as `ROCm1`.

For the surrounding benchmark workflow, return to the [README](../README.md).