# Prompt: investigate exact KV/state accounting for hybrid architectures (Qwen3.5/Qwen3.6/Qwen3.8)

## Purpose of this file

Pick this up in a **new session** once the current real-hardware benchmark run
against `qwen3-6-35b-a3b-ud-q4-k-xl` (and ideally a Qwen3.8-family run too) has
completed and produced results under `results/`. The goal of this task is
**evaluation and planning only** — read real campaign logs/results, and
produce a written plan for whether/how to add exact static feasibility
accounting for hybrid architectures. **Do not implement anything.** A
follow-up task will turn the plan into an actual work package (tentatively
"WP08") after this plan is reviewed.

## Background (from the KV-CACHE-REFACTOR series)

- `benchmark/domain/kv_feasibility.py` has an exact static byte-accounting
  formula for **dense, conventional-attention** architectures only
  (`assess_gpu_kv_feasibility` / `_conventional_dense_kv_bytes`), built and
  calibrated in WP01 (`KV-CACHE-REFACTOR-01-CALIBRATION.md`) and
  `docs/kv-feasibility-calibration.md`.
- Qwen3.5/Qwen3.6/Qwen3.8 (`general.architecture=qwen35` / `qwen35moe`,
  `LLM_ARCH_QWEN35`/`LLM_ARCH_QWEN35MOE` in llama.cpp) are **hybrid**:
  some layers are ordinary attention with a real KV cache, others are
  recurrent/SSM (Mamba-style) state layers with a completely different
  memory shape (`ssm.conv_kernel`, `ssm.state_size`, `ssm.inner_size`,
  `ssm.group_count` GGUF metadata, not `attention.key_length`).
- The calibration decision in WP01 was deliberate and is currently correct:
  **`unknown`/no-static-prune for all hybrid architectures.** Every
  depth/dtype combo for these models always gets a real Docker/llama-bench
  probe; nothing is statically excluded or statically cleared. This trades
  wasted probe/watchdog time on doomed-to-fail combos for correctness (never
  wrongly pruning something that would have fit).
- Confirmed live on real hardware (2026-09-10 session): a static plan run
  against `unsloth/Qwen3.8-27B-GGUF` (Q6_K_M) correctly returned `unknown`
  at every depth (0 through 260,096) for every dtype (q4_0/q8_0/f16) — zero
  exclusions, as designed.
- Also confirmed by direct probe: a real Docker/llama-bench run against
  Qwen3.8-27B at `-ngl 99 -d 30720 -ctk q4_0` succeeded with GPU-resident KV
  (`ROCm0 KV buffer size = 549.00 MiB`), so the model does run and the
  pipeline handles it correctly today — just always via runtime probing,
  never via static exclusion.

## The open question

Is it worth building exact static feasibility accounting for hybrid
architectures, the same way WP01 built it for dense architectures? The
payoff is purely **wall-clock efficiency** on benchmark sweeps (skip
combos that are provably going to OOM, instead of paying a full container
launch + model load + watchdog timeout to discover that at runtime) — it
never changes measured throughput numbers for combos that do run, and it
never unlocks anything currently blocked.

The size of that payoff is unknown and depends entirely on how many
depth/dtype combos in a real hybrid-architecture campaign actually fail at
runtime vs. how many just succeed (only the failing ones represent wasted
time worth pruning). That's exactly what this investigation should
establish from real logs before any implementation work is proposed.

## What to do in the new session

1. **Locate the real campaign results** to analyze. As of when this prompt
   was written, a run was in progress:
   ```
   uv run benchmark/run_bench.py --model hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
   ```
   writing to `results/qwen3-6-35b-a3b-ud-q4-k-xl/rocm10.0.0_llamacppv0.4.0/`
   (`tuning/`, `moe-tuning/`, and eventually final curve subdirectories).
   Check whether a Qwen3.8-family (dense-attention-heavier hybrid split,
   different `full_attention_interval`/block ratio) campaign was also run;
   if not, note that as a gap in the evidence rather than assuming the
   Qwen3.6 MoE-hybrid results generalize to Qwen3.8.

2. **Read every stderr.log and jsonl artifact from the completed run(s)**
   under `results/<model-slug>/<run_id>/`. For each probe, extract:
   - depth, dtype (`ctk`/`ctv`), `n_cpu_moe` (or `n_gpu_layers` for dense),
     status (`ok` / failed / timeout), and if failed, the exact error
     class (OOM/`cudaMalloc failed`, config rejection, timeout-killed).
   - Wall-clock cost per probe (container start to container exit), broken
     out by phase (KV-dtype tuning at max depth vs. MoE offload
     bisection vs. final curve) — this run already showed those phases
     have very different per-probe costs (~215s for a full tuning-phase
     reload+deep-prefill vs. ~16s for a depth=0 MoE-bisection probe), which
     matters for estimating time saved by pruning.

3. **Quantify how much runtime pruning would actually save**, in probe-count
   and wall-clock terms, if exact hybrid accounting existed and correctly
   pre-excluded every combo that failed for VRAM/allocation reasons (not
   combos that failed for other reasons — those wouldn't be pruned by a KV
   accounting fix). Be honest if the number is small: since only a minority
   of layers in these models are attention-KV layers (recurrent-state layers
   are roughly constant-size regardless of context depth), it's plausible
   that most depths simply fit and there's little to prune — say so
   explicitly if the evidence points that way, don't manufacture a case for
   building this if the numbers don't support it.

4. **If the evidence justifies further investigation**, sketch (don't
   implement) what exact hybrid accounting would require:
   - Source citations in the exact llama.cpp revision baked into the
     benchmark image (`src/llama-memory-hybrid.cpp`,
     `src/llama-memory-recurrent.cpp`, `src/llama-arch.cpp`'s
     `LLM_ARCH_QWEN35`/`LLM_ARCH_QWEN35MOE` classification — already partly
     cited in `docs/kv-feasibility-calibration.md`) for exactly which
     layers get attention-KV memory vs. recurrent-state memory, and the
     byte formula for each.
   - Whether a calibration probe pass (like WP01's) is needed to validate
     the derived recurrent-state formula against a real measured buffer
     size, the same way `_conventional_dense_kv_bytes` was validated.
   - Whether a hybrid-specific runtime reserve is calibratable, or whether
     `unknown` should remain the answer for the non-KV reserve even if KV
     itself becomes exactly accountable.
   - Rough shape of the new `domain/kv_feasibility.py` function this would
     require, parallel to the existing dense one, and how
     `domain/kv_depth_planner.py` would route to it.

5. **Write up findings and a recommendation** (build it / don't build it
   yet / build a narrower version) with the supporting numbers from step 3.
   This is a planning document, not code — no production files should
   change as a result of this investigation. If it turns out worth doing,
   the actual implementation becomes a separate, explicitly-approved task
   (WP08), scoped and TDD'd like every other package in this series.

## Constraints

- Investigation and planning only. **Do not modify
  `domain/kv_feasibility.py`, `domain/kv_depth_planner.py`, or any other
  production code in this task.**
- Do not touch or delete anything under `results/` — read-only.
- If the evidence is incomplete (e.g. no Qwen3.8-family campaign completed
  yet, or the running campaign got interrupted), say so plainly rather than
  extrapolating from a partial or single-model dataset.

## Two related, smaller follow-ups noticed during the 2026-09-10 live run

Neither of these is part of the hybrid-accounting investigation above, but
they came up while watching that run live and are worth picking up
separately (small, independently scoped, real code changes — each should
get its own explicitly-approved task, not be folded silently into WP08):

1. **`run_bench.py` doesn't write a log file.** It only `print()`s to
   stdout with `flush=True`. For a campaign that runs for hours, there's no
   way to `tail -f` progress from a second terminal unless the user
   remembers to pipe through `tee` themselves
   (`uv run benchmark/run_bench.py --model ... 2>&1 | tee -a run.log`).
   Worth considering: always write a persistent log (e.g.
   `results/<model-slug>/<run_id>/campaign.log`) alongside the JSONL
   artifacts, so a run started without `tee` can still be inspected live
   or after the fact, and so this coding agent can `tail`/`read_file` a
   running campaign's log directly in a later session without relying on
   the user's terminal scrollback.

2. **The campaign progress ETA (`domain/progress.py`'s `ProbeProgress.snapshot`)
   is misleading during phase transitions.** It computes
   `remaining * (statistics.median(all_durations_so_far) + expected_gap_seconds)`
   as one global median across the entire run. But probe cost varies
   hugely by phase — observed on the 2026-09-10 `qwen3-6-35b-a3b-ud-q4-k-xl`
   run: KV-dtype tuning probes at the deepest requested depth (full model
   reload + 129,024-token prefill) cost **~214-216s each**, while MoE
   `n_cpu_moe` bisection probes at depth=0 (model reload, near-zero
   prefill) cost **~16-17s each**. Early in a run, the median is dominated
   by whichever phase has run so far, producing a wildly wrong ETA (e.g.
   920+ minutes based on 3 tuning-phase samples, dropping to 642 minutes
   within seconds as fast depth=0 probes start landing) that has nothing to
   do with GPU offload state and everything to do with median-mixing
   across phases. A phase-aware median (bucket by phase, or at least by
   depth-bucket) or a recency-weighted estimate (e.g. EWMA, or median of
   just the last N samples) would track reality far better. This is a
   narrow, well-contained domain-logic fix in `domain/progress.py` with an
   existing pure-function test file to extend — low risk, but still real
   production code needing its own TDD pass and explicit go-ahead before
   implementation, same as the hybrid-accounting work above.
