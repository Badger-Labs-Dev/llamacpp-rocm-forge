"""Tests for the Docker probe execution adapter."""

import io
import subprocess
import unittest
from unittest import mock

from adapters.outbound import docker_runner
from adapters.outbound.docker_runner import (
    ProbeOutcome,
    kill_active_containers,
    new_container_name,
    run_probe,
)


class NewContainerNameTests(unittest.TestCase):
    def test_produces_unique_prefixed_names(self):
        a, b = new_container_name(), new_container_name()
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("llamacpp-rocm-forge-"))


class RunProbeTests(unittest.TestCase):
    def setUp(self):
        docker_runner._ACTIVE_CONTAINERS.clear()

    def test_success_returns_returncode_and_clears_tracking(self):
        jsonl_f = io.StringIO()
        stderr_f = io.StringIO()
        fake_proc = mock.Mock(returncode=0)
        with mock.patch.object(docker_runner.subprocess, "run", return_value=fake_proc) as run:
            outcome = run_probe(
                ["docker", "run", "..."], container_name="fake-1",
                jsonl_file=jsonl_f, stderr_file=stderr_f, timeout_seconds=300,
            )
        self.assertEqual(outcome.returncode, 0)
        self.assertFalse(outcome.timed_out)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0)
        self.assertEqual(docker_runner._ACTIVE_CONTAINERS, set())
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["timeout"], 300)

    def test_nonzero_returncode_is_not_treated_as_a_timeout(self):
        fake_proc = mock.Mock(returncode=1)
        with mock.patch.object(docker_runner.subprocess, "run", return_value=fake_proc):
            outcome = run_probe(
                ["docker", "run", "..."], container_name="fake-2",
                jsonl_file=io.StringIO(), stderr_file=io.StringIO(), timeout_seconds=300,
            )
        self.assertEqual(outcome.returncode, 1)
        self.assertFalse(outcome.timed_out)

    def test_timeout_kills_the_container_and_writes_stderr_marker(self):
        stderr_f = io.StringIO()
        killed = []

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["docker", "kill"]:
                killed.append(cmd[2])
                return mock.Mock(returncode=0)
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        with mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run):
            outcome = run_probe(
                ["docker", "run", "..."], container_name="fake-3",
                jsonl_file=io.StringIO(), stderr_file=stderr_f, timeout_seconds=5,
            )

        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.returncode, -1)
        self.assertEqual(killed, ["fake-3"])
        self.assertIn("TIMEOUT after 5s", stderr_f.getvalue())
        self.assertEqual(docker_runner._ACTIVE_CONTAINERS, set())

    def test_container_is_tracked_as_active_during_the_call(self):
        # Verify _ACTIVE_CONTAINERS actually contains the name mid-call,
        # not just before/after - this is what a signal handler relies on.
        seen_during_call = {}

        def fake_run(cmd, **kwargs):
            seen_during_call["active"] = set(docker_runner._ACTIVE_CONTAINERS)
            return mock.Mock(returncode=0)

        with mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run):
            run_probe(
                ["docker", "run", "..."], container_name="fake-4",
                jsonl_file=io.StringIO(), stderr_file=io.StringIO(), timeout_seconds=300,
            )
        self.assertEqual(seen_during_call["active"], {"fake-4"})


class KillActiveContainersTests(unittest.TestCase):
    def setUp(self):
        docker_runner._ACTIVE_CONTAINERS.clear()

    def tearDown(self):
        docker_runner._ACTIVE_CONTAINERS.clear()

    def test_kills_and_clears_every_tracked_container(self):
        docker_runner._ACTIVE_CONTAINERS.update({"c1", "c2"})
        killed = []
        with mock.patch.object(
            docker_runner.subprocess, "run",
            side_effect=lambda cmd, **kw: killed.append(cmd[2]) or mock.Mock(),
        ):
            kill_active_containers()
        self.assertEqual(set(killed), {"c1", "c2"})
        self.assertEqual(docker_runner._ACTIVE_CONTAINERS, set())

    def test_noop_when_nothing_is_tracked(self):
        with mock.patch.object(docker_runner.subprocess, "run") as run:
            kill_active_containers()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
