"""Application contract published by the benchmark app for the static viewer."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, NoReturn

SCHEMA_VERSION = 1

_REQUIRED_RUN_FIELDS = {
    "run_id", "status", "mode", "environment", "final_config",
    "run_completed_at", "depths_tested", "prefill_depth0_ts",
    "generation_depth0_ts", "curve", "tuning_log", "moe_offload_curve",
}

_FINAL_CONFIG_FIELDS = {
    "ubatch", "batch", "ctk", "ctv", "flash_attn", "gpu_layers",
    "block_count", "n_cpu_layers", "n_cpu_moe",
}


def _fail(message: str) -> NoReturn:
    raise ValueError(f"viewer dataset: {message}")


def _is_integer(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _is_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _check_curve_row(row: Any, where: str) -> None:
    if not isinstance(row, dict):
        _fail(f"{where} curve row must be an object, got {type(row).__name__}")
    if row.get("series") not in ("prefill", "generation"):
        _fail(f"{where} curve row missing/invalid 'series'")
    if not _is_integer(row.get("n_depth")):
        _fail(f"{where} curve row missing/invalid 'n_depth'")
    avg_ts = row.get("avg_ts")
    if "avg_ts" not in row or (avg_ts is not None and not _is_number(avg_ts)):
        _fail(f"{where} curve row 'avg_ts' must be number or null")
    if row.get("status") not in ("ok", "partial", "failed"):
        _fail(f"{where} curve row missing/invalid 'status'")


def _check_tuning_stage(stage: Any, where: str) -> None:
    if not isinstance(stage, dict):
        _fail(f"{where} tuning_log entry must be an object, got {type(stage).__name__}")
    if not isinstance(stage.get("stage"), str):
        _fail(f"{where} tuning_log entry missing/invalid 'stage'")
    scores = stage.get("scores")
    if not isinstance(scores, dict) or not scores:
        _fail(f"{where} tuning_log entry 'scores' must be a non-empty object")
    if not all(_is_number(v) for v in scores.values()):
        _fail(f"{where} tuning_log entry 'scores' values must be finite numbers")
    winner = stage.get("winner")
    if not isinstance(winner, str) and not _is_number(winner):
        _fail(f"{where} tuning_log entry 'winner' must be a string or number")


def _check_moe_offload_curve(curve: Any, where: str) -> None:
    if curve is None:
        return
    if not isinstance(curve, dict):
        _fail(f"{where} moe_offload_curve must be an object or null")
    if curve.get("mode") not in ("quick", "thorough"):
        _fail(f"{where} moe_offload_curve 'mode' must be 'quick' or 'thorough'")
    for key in ("expert_count", "block_count"):
        if not _is_integer(curve.get(key)):
            _fail(f"{where} moe_offload_curve missing/invalid {key!r}")
    if "expert_used_count" not in curve or (
        curve["expert_used_count"] is not None and not _is_integer(curve["expert_used_count"])
    ):
        _fail(f"{where} moe_offload_curve 'expert_used_count' must be integer or null")
    if "candidates_tested" in curve and not (
        isinstance(curve["candidates_tested"], list)
        and all(_is_integer(value) for value in curve["candidates_tested"])
    ):
        _fail(f"{where} moe_offload_curve 'candidates_tested' must be a list of integers")
    by_depth = curve.get("by_depth")
    if not isinstance(by_depth, list):
        _fail(f"{where} moe_offload_curve 'by_depth' must be a list")
    for point in by_depth:
        if not isinstance(point, dict) or not _is_integer(point.get("depth")):
            _fail(f"{where} moe_offload_curve by_depth entry missing/invalid 'depth'")
        results = point.get("results")
        boundary = point.get("min_ncmoe_that_fits")
        if "min_ncmoe_that_fits" not in point or (boundary is not None and not _is_integer(boundary)):
            _fail(f"{where} moe_offload_curve 'min_ncmoe_that_fits' must be integer or null")
        if not isinstance(results, list):
            _fail(f"{where} moe_offload_curve by_depth entry 'results' must be a list")
        for result in results:
            if not isinstance(result, dict) or not _is_integer(result.get("n_cpu_moe")):
                _fail(f"{where} moe_offload_curve result missing/invalid 'n_cpu_moe'")
            if result.get("status") not in ("ok", "failed"):
                _fail(f"{where} moe_offload_curve result has invalid 'status'")
            avg_ts = result.get("avg_ts")
            if "avg_ts" not in result or (avg_ts is not None and not _is_number(avg_ts)):
                _fail(f"{where} moe_offload_curve result 'avg_ts' must be number or null")


def _check_dense_offload_curve(curve: Any, where: str) -> None:
    if curve is None:
        return
    if not isinstance(curve, dict):
        _fail(f"{where} dense_offload_curve must be an object or null")
    if curve.get("mode") not in ("quick", "thorough"):
        _fail(f"{where} dense_offload_curve 'mode' must be 'quick' or 'thorough'")
    for key in ("block_count", "max_gpu_layers"):
        if not _is_integer(curve.get(key)):
            _fail(f"{where} dense_offload_curve missing/invalid {key!r}")
    if "final_ngl" not in curve or (curve["final_ngl"] is not None and not _is_integer(curve["final_ngl"])):
        _fail(f"{where} dense_offload_curve 'final_ngl' must be integer or null")
    by_depth = curve.get("by_depth")
    if not isinstance(by_depth, list):
        _fail(f"{where} dense_offload_curve 'by_depth' must be a list")
    for depth in by_depth:
        if not isinstance(depth, dict) or not _is_integer(depth.get("depth")):
            _fail(f"{where} dense_offload_curve by_depth entry missing/invalid 'depth'")
        boundary = depth.get("max_ngl_that_fits")
        if "max_ngl_that_fits" not in depth or (boundary is not None and not _is_integer(boundary)):
            _fail(f"{where} dense_offload_curve 'max_ngl_that_fits' must be integer or null")
        if "estimate" in depth and depth["estimate"] is not None and not isinstance(depth["estimate"], dict):
            _fail(f"{where} dense_offload_curve 'estimate' must be an object or null")
        results = depth.get("results")
        if not isinstance(results, list):
            _fail(f"{where} dense_offload_curve by_depth entry 'results' must be a list")
        for point in results:
            if not isinstance(point, dict):
                _fail(f"{where} dense_offload_curve result must be an object")
            if not _is_integer(point.get("n_gpu_layers")):
                _fail(f"{where} dense_offload_curve result missing/invalid 'n_gpu_layers'")
            if not _is_integer(point.get("n_cpu_layers")):
                _fail(f"{where} dense_offload_curve result missing/invalid 'n_cpu_layers'")
            if point.get("status") not in ("ok", "failed"):
                _fail(f"{where} dense_offload_curve result has invalid 'status'")
            avg_ts = point.get("avg_ts")
            if "avg_ts" not in point or (avg_ts is not None and not _is_number(avg_ts)):
                _fail(f"{where} dense_offload_curve result 'avg_ts' must be number or null")


def validate_viewer_dataset(dataset: dict) -> None:
    """Validate the published viewer dataset against the stable contract.

    Checks field presence *and* the value types/shapes that
    contracts/viewer-dataset.schema.json and the viewer's TypeScript types
    both assume - a payload that only satisfies key-presence can still be
    schema-valid-but-unrenderable (e.g. an empty {} moe_offload_curve, or a
    tuning_log entry with no 'scores'), which previously passed here.
    """
    if not isinstance(dataset, dict):
        _fail(f"root value must be an object, got {type(dataset).__name__}")
    required_root = {"schema_version", "generated_at", "models"}
    missing = required_root - dataset.keys()
    if missing:
        _fail(f"missing root fields: {sorted(missing)}")
    if dataset["schema_version"] != SCHEMA_VERSION:
        _fail(f"unsupported schema version: {dataset['schema_version']!r}")
    if not isinstance(dataset["generated_at"], str) or not _is_datetime(dataset["generated_at"]):
        _fail("'generated_at' must be a date-time string")
    if not isinstance(dataset["models"], list):
        _fail("'models' must be a list")

    for model in dataset["models"]:
        if not isinstance(model, dict):
            _fail(f"model must be an object, got {type(model).__name__}")
        if not {"model_slug", "runs"} <= model.keys():
            _fail("model is missing 'model_slug' or 'runs'")
        if not isinstance(model["model_slug"], str):
            _fail(f"model {model.get('model_slug')!r} has non-string model_slug")
        for key in ("model_filename", "model_architecture", "model_name"):
            if key in model and model[key] is not None and not isinstance(model[key], str):
                _fail(f"model {model['model_slug']!r} {key!r} must be a string or null")
        if "model_context_length" in model and model["model_context_length"] is not None and not _is_integer(model["model_context_length"]):
            _fail(f"model {model['model_slug']!r} 'model_context_length' must be an integer or null")
        runs = model["runs"]
        if not isinstance(runs, list):
            _fail(f"model {model['model_slug']!r} 'runs' must be a list")

        for run in runs:
            where = f"model {model['model_slug']!r}"
            if not isinstance(run, dict):
                _fail(f"{where} run must be an object, got {type(run).__name__}")
            missing_run = _REQUIRED_RUN_FIELDS - run.keys()
            if missing_run:
                _fail(f"{where} run {run.get('run_id')!r} is incomplete: missing {sorted(missing_run)}")
            if not isinstance(run["run_id"], str):
                _fail(f"{where} run has non-string run_id")
            where = f"{where} run {run['run_id']!r}"
            if run["status"] not in ("finished", "partial", "failed"):
                _fail(f"{where} 'status' must be finished, partial, or failed")
            if run["run_completed_at"] is not None and not isinstance(run["run_completed_at"], str):
                _fail(f"{where} 'run_completed_at' must be a string or null")
            if not isinstance(run["depths_tested"], list) or not all(_is_integer(depth) for depth in run["depths_tested"]):
                _fail(f"{where} 'depths_tested' must be a list of integers")
            for key in ("prefill_depth0_ts", "generation_depth0_ts"):
                value = run[key]
                if value is not None and not _is_number(value):
                    _fail(f"{where} {key!r} must be a finite number or null")
            if not isinstance(run["mode"], str):
                _fail(f"{where} 'mode' must be a string")
            if not isinstance(run["environment"], dict):
                _fail(f"{where} 'environment' must be an object")
            if not isinstance(run["final_config"], dict):
                _fail(f"{where} 'final_config' must be an object")
            config = run["final_config"]
            missing_config = _FINAL_CONFIG_FIELDS - config.keys()
            if missing_config:
                _fail(f"{where} 'final_config' is missing {sorted(missing_config)}")
            for key in ("ubatch", "batch", "gpu_layers", "n_cpu_layers", "n_cpu_moe"):
                if not _is_integer(config[key]):
                    _fail(f"{where} 'final_config.{key}' must be an integer")
            if config["block_count"] is not None and not _is_integer(config["block_count"]):
                _fail(f"{where} 'final_config.block_count' must be integer or null")
            for key in ("ctk", "ctv", "flash_attn"):
                if not isinstance(config[key], str):
                    _fail(f"{where} 'final_config.{key}' must be a string")
            curve = run["curve"]
            if not isinstance(curve, list):
                _fail(f"{where} 'curve' must be a list")
            for row in curve:
                _check_curve_row(row, where)
            tuning_log = run["tuning_log"]
            if not isinstance(tuning_log, list):
                _fail(f"{where} 'tuning_log' must be a list")
            for stage in tuning_log:
                _check_tuning_stage(stage, where)
            _check_moe_offload_curve(run["moe_offload_curve"], where)
            _check_dense_offload_curve(run.get("dense_offload_curve"), where)
