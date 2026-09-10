#!/usr/bin/env bash
# Reproducible package-01 live-GPU probes. Do not use this to change campaign behavior.
set -uo pipefail

OUT_DIR="$(cd "$(dirname "$0")" && pwd)/2026-09-10"
IMAGE="llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench"
TINY_BLOB="/home/wilsonrm/.cache/huggingface/hub/models--tensorblock--tinyllama-15M-GGUF/blobs/aa933426e277777f8e225080b40ec9632111993098a75fe693109983babfd5bb"
QWEN_BLOB="/home/wilsonrm/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/blobs/493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb"
VIDEO_GID=983
RENDER_GID=987
mkdir -p "$OUT_DIR"

record_gpu() {
  local label="$1"
  {
    date --iso-8601=seconds
    rocm-smi --showproductname --showmeminfo vram --csv
  } > "$OUT_DIR/${label}.gpu.csv" 2>&1
}

run_probe() {
  local label="$1"
  local blob="$2"
  local depth="$3"
  local ctk="$4"
  local nkvo="$5"
  local timeout_seconds="$6"
  local name="kvcal-${label}-$RANDOM"
  local model_dir
  model_dir="$(dirname "$blob")"

  {
    printf '# timestamp: '; date --iso-8601=seconds
    printf '# image: %s\n' "$IMAGE"
    printf '# model blob: %s\n' "$blob"
    printf '# command: timeout --signal=TERM 300s docker run --rm --name %s --device /dev/dri --device /dev/kfd --group-add %s --group-add %s --security-opt seccomp=unconfined --ipc=host -v %s:/models:ro %s llama-bench -v -m /models/%s -o jsonl -oe jsonl -r 1 -b 2048 -ub 2048 -ngl 0 -nkvo %s -fa auto -ctk %s -ctv %s -dev ROCm0 -d %s -p 2048 -n 0 --progress\n' "$name" "$VIDEO_GID" "$RENDER_GID" "$model_dir" "$IMAGE" "$(basename "$blob")" "$nkvo" "$ctk" "$ctk" "$depth"
  } > "$OUT_DIR/${label}.command.txt"

  record_gpu "${label}.before"
  set +e
  timeout --signal=TERM "$timeout_seconds" docker run --rm --name "$name" \
    --device /dev/dri --device /dev/kfd --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined --ipc=host -v "$model_dir:/models:ro" "$IMAGE" \
    llama-bench -v -m "/models/$(basename "$blob")" -o jsonl -oe jsonl -r 1 \
    -b 2048 -ub 2048 -ngl 0 -nkvo "$nkvo" -fa auto -ctk "$ctk" -ctv "$ctk" \
    -dev ROCm0 -d "$depth" -p 2048 -n 0 --progress \
    > "$OUT_DIR/${label}.jsonl" 2> "$OUT_DIR/${label}.stderr.log"
  local status=$?
  set -e
  printf '%s\n' "$status" > "$OUT_DIR/${label}.exit-code"
  docker rm -f "$name" >/dev/null 2>&1 || true
  record_gpu "${label}.after"
}

set -e
record_gpu baseline
{
  printf 'timestamp='; date --iso-8601=seconds
  printf 'image_identity='; docker run --rm "$IMAGE" cat /app/.llama-cpp-identity
  printf 'image_commit='; docker run --rm "$IMAGE" cat /app/.llama-cpp-commit
} > "$OUT_DIR/image-identity.txt"

# Required small-model controls: default KV-offload semantics, explicit host KV,
# then both quantized KV types. Actual placement is established by the logs.
run_probe A-default-f16 "$TINY_BLOB" 4096 f16 0 300
run_probe B-host-f16 "$TINY_BLOB" 4096 f16 1 300
run_probe C-default-q8_0 "$TINY_BLOB" 4096 q8_0 0 300
run_probe D-default-q4_0 "$TINY_BLOB" 4096 q4_0 0 300

# Target Qwen3.8: one comfortably feasible depth and one prior observed stress depth.
run_probe E-qwen3_8-f16-depth-2048 "$QWEN_BLOB" 2048 f16 0 300
run_probe H-qwen3_8-f16-depth-63488 "$QWEN_BLOB" 63488 f16 0 300
run_probe F-qwen3_8-f16-depth-96256 "$QWEN_BLOB" 96256 f16 0 300
record_gpu final

docker ps -a --filter 'name=kvcal-' --format '{{.Names}} {{.Status}}' > "$OUT_DIR/remaining-containers.txt"
