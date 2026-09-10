# KV-cache feasibility refactor — implementation map

## Goal

Avoid wasting five-minute watchdog probes on context depths that are **provably impossible** because GPU-resident KV cache plus an explicit runtime reserve cannot fit. Preserve real runtime probing for every case that is uncertain or may fail for allocator/workspace/performance reasons.

The benchmark’s standard runtime configuration is fixed:

```text
batch=2048
ubatch=2048
flash_attn=auto
```

This refactor deliberately removes batch/ubatch optimization from normal campaigns. It does **not** add a batch/ubatch fallback ladder.

## Locked behavioral decisions

1. KV dtype processing order is `q4_0`, then `q8_0`, then `f16`. This is reliability ordering, not an automatic preference for q4.
2. KV dtype selection is coverage-first, then throughput: a dtype must successfully complete a target depth before its throughput is compared with other dtypes at that same depth.
3. Static exclusion is per `(KV dtype, depth)`, never a global truncation of the campaign’s depth list.
4. A static exclusion is allowed only from exact, documented, GPU-resident KV accounting plus a validated reserve. Upper-bound or incomplete estimators return `unknown` and must not prune.
5. The existing full-model-offload estimate stays advisory. It cannot prove `-ngl 0` is impossible because model weights can move to host RAM while KV remains on GPU.
6. Default llama.cpp semantics remain unchanged: `no_kv_offload=false`. Do not add `-nkvo 1` merely to make a context fit.
7. A runtime timeout is not proof of static infeasibility. Distinguish clean allocation failure, timeout, unsupported/configuration error, and successful execution.
8. A campaign that completes all planned eligible depths is `finished`, even when larger original depths were statically excluded.

## Relevant established facts

- Upstream llama.cpp v0.4.0 `llama-bench` supports `--no-kv-offload`; the default is false.
- With the default, `-ngl 0` does not imply host KV: weights may be on CPU while KV buffers remain on the GPU backend.
- Quantized KV requires flash attention in this benchmark’s model validation rules.
- `Qwen3.5` is hybrid in llama.cpp v0.4.0. The current `estimate_full_offload_vram()` treats every transformer block as a normal f16 attention KV allocation and is intentionally conservative; it must not be reused as a hard bound.
- Existing campaign failure policy conflates allocation errors and watchdog timeouts during dense `--ngl` probing. This work must not perpetuate that conflation.

## Work-package dependency graph

```text
01 calibration / source evidence ──┐
                                  ├──> 02 exact domain feasibility model ──> 04 dtype planner
03 remove batch/ubatch grid ───────┘                                         │
                                                                              ├──> 05 application orchestration
                                                                              └──> 06 artifacts, status, docs
                                                                                       │
                                                                                       └──> 07 verification
```

- **01** is a prerequisite for enabling hard exclusions for Qwen3.5/hybrid models. It may produce `unknown` support instead if the exact allocation cannot be established.
- **03** is independent and can be implemented immediately.
- **02** must expose pure, tested APIs before **04**.
- **04**, **05**, and **06** are separable only after their data contracts are agreed; preserve backward-compatible manifest construction until the coordinated merge.
- **07** is the final integration gate.

## Work packages

| File | Deliverable | Depends on |
|---|---|---|
| `KV-CACHE-REFACTOR-01-CALIBRATION.md` | Evidence and policy inputs for exact accounting/reserve | none |
| `KV-CACHE-REFACTOR-02-DOMAIN-FEASIBILITY.md` | Pure exact/unknown KV feasibility API and tests | 01 for Qwen hard support |
| `KV-CACHE-REFACTOR-03-FIXED-BATCH-CONFIG.md` | Removal of the batch/ubatch grid | none |
| `KV-CACHE-REFACTOR-04-DTYPE-DEPTH-PLANNER.md` | Pure per-dtype eligibility planner | 02 |
| `KV-CACHE-REFACTOR-05-CAMPAIGN-ORCHESTRATION.md` | Tuning/preflight/final curve flow | 03, 04 |
| `KV-CACHE-REFACTOR-06-ARTIFACTS-STATUS-DOCS.md` | Manifest/status/docs semantics | 04, 05 |
| `KV-CACHE-REFACTOR-07-VERIFICATION.md` | Unit/integration/GPU release gate | all applicable packages |

## Non-goals

- Do not alter model quantization, model loading, or default KV placement semantics.
- Do not claim that q4/q8/f16 measurements are interchangeable; their cache dtype is part of the result configuration.
- Do not change pre-existing `results/` artifacts, especially deleted/untracked result files already in the worktree.
- Do not replace real runtime probes with static estimates.
- Do not add a new automatic batch/ubatch performance optimizer as part of this refactor.

## Common implementation rules

- Use strict TDD: create one focused failing test, run it and observe the expected failure, add minimal production code, then run focused and full tests.
- Keep arithmetic and selection logic in `benchmark/domain/`; application modules should orchestrate I/O only.
- Treat unknown data conservatively in favor of execution, not pruning.
- Preserve requested/trained context data in all artifacts; planned depth eligibility is an additional record, not a replacement.
- Before claiming completion, run the relevant test slice and record actual command output in the PR/session handoff.
