"""Application contract published by the benchmark app for the static viewer."""

from __future__ import annotations

from typing import Any, NoReturn

SCHEMA_VERSION = 1

_REQUIRED_RUN_FIELDS = {
    "run_id", "status", "mode", "environment", "final_config",
    "curve", "tuning_log", "moe_offload_curve",
}


def _fail(message: str) -> NoReturn:
    raise ValueError(f"viewer dataset: {message}")


def _check_curve_row(row: Any, where: str) -> None:
    if not isinstance(row, dict):
        _fail(f"{where} curve row must be an object, got {type(row).__name__}")
    if not isinstance(row.get("series"), str):
        _fail(f"{where} curve row missing/invalid 'series'")
    if not isinstance(row.get("n_depth"), int):
        _fail(f"{where} curve row missing/invalid 'n_depth'")
    avg_ts = row.get("avg_ts")
    if avg_ts is not None and not isinstance(avg_ts, (int, float)):
        _fail(f"{where} curve row 'avg_ts' must be number or null")
    if not isinstance(row.get("status"), str):
        _fail(f"{where} curve row missing/invalid 'status'")


def _check_tuning_stage(stage: Any, where: str) -> None:
    if not isinstance(stage, dict):
        _fail(f"{where} tuning_log entry must be an object, got {type(stage).__name__}")
    if not isinstance(stage.get("stage"), str):
        _fail(f"{where} tuning_log entry missing/invalid 'stage'")
    scores = stage.get("scores")
    if not isinstance(scores, dict) or not scores:
        _fail(f"{where} tuning_log entry 'scores' must be a non-empty object")
    if not all(isinstance(v, (int, float)) for v in scores.values()):
        _fail(f"{where} tuning_log entry 'scores' values must be numbers")
    if "winner" not in stage:
        _fail(f"{where} tuning_log entry missing 'winner'")


def _check_moe_offload_curve(curve: Any, where: str) -> None:
    if curve is None:
        return
    if not isinstance(curve, dict):
        _fail(f"{where} moe_offload_curve must be an object or null")
    if curve.get("mode") not in ("quick", "thorough"):
        _fail(f"{where} moe_offload_curve 'mode' must be 'quick' or 'thorough'")
    if not isinstance(curve.get("block_count"), int):
        _fail(f"{where} moe_offload_curve missing/invalid 'block_count'")
    by_depth = curve.get("by_depth")
    if not isinstance(by_depth, list):
        _fail(f"{where} moe_offload_curve 'by_depth' must be a list")
    for point in by_depth:
        if not isinstance(point, dict) or not isinstance(point.get("depth"), int):
            _fail(f"{where} moe_offload_curve by_depth entry missing/invalid 'depth'")
        results = point.get("results")
        if not isinstance(results, list):
            _fail(f"{where} moe_offload_curve by_depth entry 'results' must be a list")


def _check_dense_offload_curve(curve: Any, where: str) -> None:
    if curve is None:
        return
    if not isinstance(curve, dict):
        _fail(f"{where} dense_offload_curve must be an object or null")
    if curve.get("mode") not in ("quick", "thorough"):
        _fail(f"{where} dense_offload_curve 'mode' must be 'quick' or 'thorough'")
    for key in ("block_count", "max_gpu_layers"):
        if not isinstance(curve.get(key), int):
            _fail(f"{where} dense_offload_curve missing/invalid {key!r}")
    if curve.get("final_ngl") is not None and not isinstance(curve.get("final_ngl"), int):
        _fail(f"{where} dense_offload_curve 'final_ngl' must be integer or null")
    by_depth = curve.get("by_depth")
    if not isinstance(by_depth, list):
        _fail(f"{where} dense_offload_curve 'by_depth' must be a list")
    for depth in by_depth:
        if not isinstance(depth, dict) or not isinstance(depth.get("depth"), int):
            _fail(f"{where} dense_offload_curve by_depth entry missing/invalid 'depth'")
        boundary = depth.get("max_ngl_that_fits")
        if boundary is not None and not isinstance(boundary, int):
            _fail(f"{where} dense_offload_curve 'max_ngl_that_fits' must be integer or null")
        results = depth.get("results")
        if not isinstance(results, list):
            _fail(f"{where} dense_offload_curve by_depth entry 'results' must be a list")
        for point in results:
            if not isinstance(point, dict):
                _fail(f"{where} dense_offload_curve result must be an object")
            if not isinstance(point.get("n_gpu_layers"), int):
                _fail(f"{where} dense_offload_curve result missing/invalid 'n_gpu_layers'")
            if not isinstance(point.get("n_cpu_layers"), int):
                _fail(f"{where} dense_offload_curve result missing/invalid 'n_cpu_layers'")
            if point.get("status") not in ("ok", "failed"):
                _fail(f"{where} dense_offload_curve result has invalid 'status'")
            avg_ts = point.get("avg_ts")
            if avg_ts is not None and not isinstance(avg_ts, (int, float)):
                _fail(f"{where} dense_offload_curve result 'avg_ts' must be number or null")


def validate_viewer_dataset(dataset: dict) -> None:
    """Validate the published viewer dataset against the stable contract.

    Checks field presence *and* the value types/shapes that
    contracts/viewer-dataset.schema.json and the viewer's TypeScript types
    both assume - a payload that only satisfies key-presence can still be
    schema-valid-but-unrenderable (e.g. an empty {} moe_offload_curve, or a
    tuning_log entry with no 'scores'), which previously passed here.
    """
    required_root = {"schema_version", "generated_at", "models"}
    missing = required_root - dataset.keys()
    if missing:
        _fail(f"missing root fields: {sorted(missing)}")
    if dataset["schema_version"] != SCHEMA_VERSION:
        _fail(f"unsupported schema version: {dataset['schema_version']!r}")
    if not isinstance(dataset["generated_at"], str):
        _fail("'generated_at' must be a string")
    if not isinstance(dataset["models"], list):
        _fail("'models' must be a list")

    for model in dataset["models"]:
        if not isinstance(model, dict):
            _fail(f"model must be an object, got {type(model).__name__}")
        if not {"model_slug", "runs"} <= model.keys():
            _fail("model is missing 'model_slug' or 'runs'")
        if not isinstance(model["model_slug"], str):
            _fail(f"model {model.get('model_slug')!r} has non-string model_slug")
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
            if not isinstance(run["status"], str):
                _fail(f"{where} 'status' must be a string")
            if not isinstance(run["mode"], str):
                _fail(f"{where} 'mode' must be a string")
            if not isinstance(run["environment"], dict):
                _fail(f"{where} 'environment' must be an object")
            if not isinstance(run["final_config"], dict):
                _fail(f"{where} 'final_config' must be an object")
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
