#!/usr/bin/env bash
# Follow-up to run-calibration-2026-09-10.sh: TinyLlama's head dimension (48)
# cannot represent q8_0/q4_0 cache blocks (32), so use Qwen3.8 (256-wide KV heads).
set -euo pipefail
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/2026-09-10"
IMAGE="llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench"
BLOB="/home/wilsonrm/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/blobs/493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb"
for type in q8_0 q4_0; do
  label="G-qwen3_8-${type}-depth-2048"
  name="kvcal-${label}-$RANDOM"
  {
    printf '# timestamp: '; date --iso-8601=seconds
    printf '# command: timeout --signal=TERM 300s docker run --rm --name %s --device /dev/dri --device /dev/kfd --group-add 983 --group-add 987 --security-opt seccomp=unconfined --ipc=host -v %s:/models:ro %s llama-bench -v -m /models/%s -o jsonl -oe jsonl -r 1 -b 2048 -ub 2048 -ngl 0 -nkvo 0 -fa auto -ctk %s -ctv %s -dev ROCm0 -d 2048 -p 2048 -n 0 --progress\n' "$name" "$(dirname "$BLOB")" "$IMAGE" "$(basename "$BLOB")" "$type" "$type"
  } > "$OUT_DIR/${label}.command.txt"
  { date --iso-8601=seconds; rocm-smi --showproductname --showmeminfo vram --csv; } > "$OUT_DIR/${label}.before.gpu.csv" 2>&1
  set +e
  timeout --signal=TERM 300s docker run --rm --name "$name" --device /dev/dri --device /dev/kfd --group-add 983 --group-add 987 --security-opt seccomp=unconfined --ipc=host -v "$(dirname "$BLOB"):/models:ro" "$IMAGE" llama-bench -v -m "/models/$(basename "$BLOB")" -o jsonl -oe jsonl -r 1 -b 2048 -ub 2048 -ngl 0 -nkvo 0 -fa auto -ctk "$type" -ctv "$type" -dev ROCm0 -d 2048 -p 2048 -n 0 --progress > "$OUT_DIR/${label}.jsonl" 2> "$OUT_DIR/${label}.stderr.log"
  code=$?
  set -e
  printf '%s\n' "$code" > "$OUT_DIR/${label}.exit-code"
  docker rm -f "$name" >/dev/null 2>&1 || true
  { date --iso-8601=seconds; rocm-smi --showproductname --showmeminfo vram --csv; } > "$OUT_DIR/${label}.after.gpu.csv" 2>&1
done
docker ps -a --filter 'name=kvcal-' --format '{{.Names}} {{.Status}}' > "$OUT_DIR/remaining-containers.txt"
