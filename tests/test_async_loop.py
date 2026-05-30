"""
tests/test_async_loop.py

Verifies that:
1. _run_parallel_branches uses get_running_loop() not get_event_loop()
2. The parallel triad executes all four branches concurrently without
   raising RuntimeError or DeprecationWarning from event loop access
3. Branch results are correctly stored on VignetteRun

Uses a mocked Vertex AI client — no live GCP credentials required.
"""

import asyncio
import inspect
import unittest
from unittest.mock import MagicMock

# Stubs for external dependencies are installed by conftest.py
# ── Now import the runner ─────────────────────────────────────────────────────
from yentlguard.agent.runner import VignetteRun, YentlGuardRunner


def _make_mock_response(esi_text: str = "ESI: 3\nRationale: Normal vitals.") -> MagicMock:
    """Build a minimal mock GenerateContentResponse."""
    resp = MagicMock()
    resp.text = esi_text
    resp.candidates = [MagicMock()]
    resp.candidates[0].logprobs_result = None  # delta_m will return None (mocked)
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.thoughts_token_count = None
    resp.usage_metadata.candidates_token_count = 42
    return resp


class TestGetRunningLoop(unittest.IsolatedAsyncioTestCase):
    """
    Verify get_running_loop() resolves correctly inside _run_parallel_branches.

    IsolatedAsyncioTestCase runs each async test in its own event loop,
    matching the environment where _run_parallel_branches executes.
    """

    async def test_running_loop_resolves_inside_async_context(self):
        """
        get_running_loop() must return the active loop without raising.
        In Python 3.10+ get_event_loop() raises DeprecationWarning or
        RuntimeError when no current loop is set; get_running_loop() is safe.
        """
        loop = asyncio.get_running_loop()
        self.assertIsNotNone(loop)
        self.assertIsInstance(loop, asyncio.AbstractEventLoop)

    async def test_parallel_branches_use_running_loop(self):
        """
        _run_parallel_branches must not raise RuntimeError or
        DeprecationWarning from event loop access.
        """
        runner = YentlGuardRunner(
            model_version="gemini-2.5-pro",
            thinking_budget="medium",
        )

        # Mock the Vertex AI client to return a canned response
        mock_response = _make_mock_response()
        runner._client = MagicMock()
        runner._client.models.generate_content = MagicMock(return_value=mock_response)

        config = MagicMock()
        run = VignetteRun(
            vignette_id="ED_TEST_001",
            demographic_variant="female",
            model_version="gemini-2.5-pro",
            thinking_budget="medium",
        )
        # Set up minimal pass1 state so CRR can be attempted
        run.pass1_esi = "3"
        run.baseline_delta_m = 2.5
        run.pass1_delta_m = MagicMock()
        run.pass1_delta_m.delta_m = 0.4

        branches = {
            "corrective": "Corrective prompt text",
            "3a": "Distractor A prompt",
            "3b": "Distractor B prompt",
            "3c": "Distractor C prompt",
        }

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            results = await runner._run_parallel_branches(
                branches=branches,
                config=config,
                vignette_id="ED_TEST_001",
                demographic_variant="female",
                run=run,
            )

        # All four branches must return results
        self.assertEqual(set(results.keys()), {"corrective", "3a", "3b", "3c"})

    async def test_all_four_branches_execute(self):
        """
        asyncio.gather must execute all four branches, not short-circuit
        on individual failures.
        """
        runner = YentlGuardRunner(
            model_version="gemini-2.5-pro",
            thinking_budget="medium",
        )

        call_count = {"n": 0}

        def mock_generate(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_response()

        runner._client = MagicMock()
        runner._client.models.generate_content = MagicMock(side_effect=mock_generate)

        run = VignetteRun(
            vignette_id="ED_TEST_002",
            demographic_variant="female",
            model_version="gemini-2.5-pro",
            thinking_budget="medium",
        )
        run.pass1_esi = "3"
        run.baseline_delta_m = None  # no baseline — CRR will be None, branches still run

        branches = {
            "corrective": "p1",
            "3a": "p2",
            "3b": "p3",
            "3c": "p4",
        }

        results = await runner._run_parallel_branches(
            branches=branches,
            config=MagicMock(),
            vignette_id="ED_TEST_002",
            demographic_variant="female",
            run=run,
        )

        self.assertEqual(call_count["n"], 4, "All four branches must call generate_content")
        self.assertEqual(len(results), 4, "All four results must be returned")

    async def test_branch_failure_does_not_cancel_others(self):
        """
        If one branch raises, asyncio.gather must complete the remaining
        three branches and store the error on the failed branch result only.
        """
        runner = YentlGuardRunner(
            model_version="gemini-2.5-pro",
            thinking_budget="medium",
        )

        def mock_generate_with_failure(*args, **kwargs):
            # Inspect the prompt to simulate failure on branch 3b only
            prompt = args[0] if args else kwargs.get("contents", "")
            if "3b_fail" in str(prompt):
                raise RuntimeError("Simulated Vertex AI timeout")
            return _make_mock_response()

        runner._client = MagicMock()
        runner._client.models.generate_content = MagicMock(side_effect=mock_generate_with_failure)

        run = VignetteRun(
            vignette_id="ED_TEST_003",
            demographic_variant="female",
            model_version="gemini-2.5-pro",
            thinking_budget="medium",
        )
        run.pass1_esi = "3"
        run.baseline_delta_m = None

        branches = {
            "corrective": "corrective_prompt",
            "3a": "distractor_a_prompt",
            "3b": "3b_fail_prompt",  # triggers RuntimeError
            "3c": "distractor_c_prompt",
        }

        results = await runner._run_parallel_branches(
            branches=branches,
            config=MagicMock(),
            vignette_id="ED_TEST_003",
            demographic_variant="female",
            run=run,
        )

        # All four keys must be present regardless of individual failure
        self.assertEqual(set(results.keys()), {"corrective", "3a", "3b", "3c"})

        # Failed branch must have error set, not raise
        self.assertIsNotNone(results["3b"]["error"])
        self.assertIn("Simulated Vertex AI timeout", results["3b"]["error"])

        # Successful branches must have no error
        self.assertIsNone(results["corrective"]["error"])
        self.assertIsNone(results["3a"]["error"])
        self.assertIsNone(results["3c"]["error"])


class TestGetRunningLoopSourceInspection(unittest.TestCase):
    """
    Static verification that the source code uses get_running_loop,
    not the deprecated get_event_loop.
    """

    def test_source_uses_get_running_loop(self):
        import yentlguard.agent.runner as runner_module

        source = inspect.getsource(runner_module)
        self.assertNotIn(
            "get_event_loop",
            source,
            "get_event_loop() is deprecated in Python 3.10+. "
            "Use get_running_loop() inside async functions.",
        )
        self.assertIn(
            "get_running_loop",
            source,
            "get_running_loop() must be used in _run_parallel_branches.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
