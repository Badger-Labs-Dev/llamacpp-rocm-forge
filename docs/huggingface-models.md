# Hugging Face model references

`--model` accepts a local GGUF path and the common references Hugging Face shows in its UI. The driver resolves Hugging Face references through `huggingface_hub.hf_hub_download`, so it reuses the standard local cache when a file is already present and downloads only when needed.

## Accepted forms

```bash
# The exact URI from “Download with hf CLI”
uv run benchmark/run_bench.py \
  --model "hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
  --full-sweep

# The quant tag from “Use this model → Ollama”; omit hf.co/
uv run benchmark/run_bench.py \
  --model "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL" \
  --full-sweep

# A repository with no selected GGUF; interactive terminal picker
uv run benchmark/run_bench.py \
  --model "unsloth/Qwen3.6-35B-A3B-GGUF" \
  --full-sweep

# A local path
uv run benchmark/run_bench.py \
  --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --full-sweep
```

A bare repository reference opens an interactive picker only when stdin is a terminal. In a script or cron job, supply an exact URI, tag, or local file path.

## Cache and Docker mounts

Hugging Face stores a downloaded file at `snapshots/<hash>/model.gguf`, where that apparent GGUF file is normally a symlink to `blobs/<content-hash>`. The blob has no `.gguf` extension and sits outside the snapshot directory.

Before creating the Docker bind mount, the benchmark driver resolves the symlink and mounts the real blob's parent directory. llama.cpp recognizes GGUF magic bytes, so an extensionless blob loads correctly. The results slug still uses the original readable filename rather than the content hash.

You can still use a dedicated model directory or an `hf download --local-dir ...` workflow. That can be useful for a NAS, a shared model store, or isolation from other tools that use the cache. It is not required for correctness or normal local performance.
