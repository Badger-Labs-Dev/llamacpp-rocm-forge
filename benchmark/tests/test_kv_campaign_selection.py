"""Application-level selection tests for planned KV dtype/depth probes."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from application.auto_tune import select_kv_config
from domain.kv_depth_planner import DtypeDepthPlan, KvDepthPlan
from domain.models import BenchConfig


def _plan(*, q4=(0, 96_256), q8=(0, 96_256), f16=(0,)):
    requested = (0, 96_256)
    return KvDepthPlan(
        requested_depths=requested,
        prefill_tokens=2048,
        dtype_plans=tuple(
            DtypeDepthPlan(
                ctk=ctk,
                ctv=ctk,
                requested_depths=requested,
                runnable_depths=depths,
                eligible_depths=depths,
                unknown_depths=(),
                exclusions=(),
                assessments=(),
            )
            for ctk, depths in (("q4_0", q4), ("q8_0", q8), ("f16", f16))
        ),
    )


class KvCampaignSelectionTests(unittest.TestCase):
    def test_excluded_f16_is_never_probed_while_q4_at_the_depth_is_probed(self):
        calls = []

        def probe(config, depth):
            calls.append((config.ctk, depth, config.batch, config.ubatch))
            return SimpleNamespace(status="ok", jsonl_path=Path(f"/{config.ctk}-{depth}.jsonl"))

        selection = select_kv_config(
            plan=_plan(q4=(0, 96_256), q8=(0,), f16=(0,)),
            probe=probe,
            mean_throughput=lambda path: 10.0,
        )

        self.assertEqual(calls, [("q4_0", 96_256, 2048, 2048)])
        self.assertEqual(selection.config.ctk, "q4_0")
        self.assertEqual(selection.target_depth, 96_256)
        self.assertEqual(selection.runnable_depths, (0, 96_256))

    def test_timeout_does_not_exclude_other_dtypes_and_descends_after_no_success(self):
        calls = []

        def probe(config, depth):
            calls.append((config.ctk, depth))
            status = "partial" if depth == 96_256 else "ok"
            return SimpleNamespace(status=status, jsonl_path=Path(f"/{config.ctk}-{depth}.jsonl"))

        selection = select_kv_config(
            plan=_plan(), probe=probe, mean_throughput=lambda path: 10.0,
        )

        self.assertEqual(calls, [
            ("q4_0", 96_256), ("q8_0", 96_256),
            ("q4_0", 0), ("q8_0", 0), ("f16", 0),
        ])
        self.assertEqual(selection.config.ctk, "q4_0")
        self.assertEqual(selection.target_depth, 0)

    def test_higher_throughput_wins_only_among_same_depth_successes(self):
        scores = {"q4_0": 30.0, "q8_0": 50.0, "f16": 40.0}

        selection = select_kv_config(
            plan=_plan(q4=(0,), q8=(0,), f16=(0,)),
            probe=lambda config, depth: SimpleNamespace(status="ok", jsonl_path=Path(config.ctk)),
            mean_throughput=lambda path: scores[path.name],
        )

        self.assertEqual(selection.config, BenchConfig(batch=2048, ubatch=2048, ctk="q8_0", ctv="q8_0").validate())
        self.assertEqual(selection.target_depth, 0)


if __name__ == "__main__":
    unittest.main()
