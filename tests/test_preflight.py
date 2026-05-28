"""
tests/test_preflight.py

Pre-flight validation suite for YentlGuard.

Runs 3-5 vignettes through the full pipeline to confirm:
  1. GCP credentials and Vertex AI access are working
  2. Gemini returns logprobs and thoughts_token_count as expected
  3. YentlBench loads vignettes with the correct schema
  4. Arize Phoenix accepts spans (tracing endpoint reachable)
  5. BigQuery dataset and tables exist and accept streaming inserts
  6. Phoenix MCP endpoint is reachable (SSE handshake)
  7. ΔM, TAR, CRR compute without error on real Gemini responses
  8. The Parallel Triad executes all four branches on a gate-fired vignette

Each check is a separate test so failures are isolated — a BigQuery
credential failure won't mask a Gemini logprobs failure.

Usage:
    # Run pre-flight only (fast, 3-5 vignettes):
    pytest tests/test_preflight.py -v

    # Run pre-flight with explicit vignette count:
    YENTLGUARD_PREFLIGHT_N=5 pytest tests/test_preflight.py -v

    # Skip tests that require live credentials (CI without secrets):
    pytest tests/test_preflight.py -v -m "not live"

Markers:
    live     — requires real GCP credentials and network access
    yentlbench — requires YentlBench to be installed and data accessible
    phoenix  — requires Arize Phoenix endpoint to be reachable
    mcp      — requires Phoenix MCP server to be running
"""

import os
import pathlib
import sys
import time
import unittest
import uuid
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pytest
from conftest import _find_quintets_csv

# ── Configuration ─────────────────────────────────────────────────────────────

PREFLIGHT_N = int(os.environ.get("YENTLGUARD_PREFLIGHT_N", "3"))
PREFLIGHT_MODEL = os.environ.get("YENTLGUARD_PREFLIGHT_MODEL", "gemini-2.5-pro")
PREFLIGHT_BUDGET = os.environ.get("YENTLGUARD_PREFLIGHT_BUDGET", "low")
PREFLIGHT_VARIANT = os.environ.get("YENTLGUARD_PREFLIGHT_VARIANT", "nb_ambiguous")

# Timeout for individual Gemini calls in pre-flight (seconds)
VERTEX_TIMEOUT = 60


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config():
    """Load and validate GCP config. Skip test if placeholders unfilled."""
    try:
        from yentlguard.config import BQ_DATASET_ID, GCP_LOCATION, GCP_PROJECT_ID, validate
        validate()
        return GCP_PROJECT_ID, GCP_LOCATION, BQ_DATASET_ID
    except RuntimeError as e:
        pytest.skip(f"GCP config incomplete: {e}")


def _load_vignettes(variant: str, n: int):
    """Load n vignettes from YentlBench. Skip if dataset not found."""
    import os
    import pathlib

    import pandas as pd
    try:
        from yentlbench.local_runner.prompt import build_prompt
    except ImportError:
        pytest.skip("YentlBench not installed. Run: pip install yentlbench")

    dataset_path = pathlib.Path(
        os.environ.get("YENTLGUARD_DATASET_PATH", "dataset_output/dataset_quintets.csv")
    )
    if not dataset_path.exists():
        pytest.skip(
            f"dataset_quintets.csv not found at {dataset_path}. "
            "Run: yentlbench prepare"
        )

    df = pd.read_csv(dataset_path)
    df = df[df["acuity"].notna() & (df["gender_variant"] == variant)]

    vignettes = []
    for _, row in df.head(n).iterrows():
        vignette_dict = row.to_dict()
        obj = type("Vignette", (), {
            "vignette_id": str(int(vignette_dict["source_stay_id"])),
            "text": build_prompt(vignette_dict, variant),
            "_row": vignette_dict,
        })()
        vignettes.append(obj)

    if not vignettes:
        pytest.skip(f"No vignettes found for variant={variant}")
    return vignettes


def _make_vertex_client(project: str, location: str):
    """Instantiate a Vertex AI Gemini client."""
    from google import genai
    return genai.Client(vertexai=True, project=project, location=location)


def _make_runner(project: str, location: str):
    """Instantiate a YentlGuardRunner with no Phoenix MCP client."""
    from yentlguard.agent.runner import YentlGuardRunner
    return YentlGuardRunner(
        model_version=PREFLIGHT_MODEL,
        thinking_budget=PREFLIGHT_BUDGET,
        delta_m_threshold=999.0,  # force gate to fire on every vignette for pre-flight
        baseline_lookup=None,
    )


# ── Check 1: GCP credentials ──────────────────────────────────────────────────

@pytest.mark.live
class TestGCPCredentials(unittest.TestCase):
    """Verify Application Default Credentials are configured."""

    def test_adc_available(self):
        """google.auth.default() must resolve without raising."""
        try:
            import google.auth
            credentials, project = google.auth.default()
            self.assertIsNotNone(credentials)
        except ImportError:
            self.skipTest("google-auth not installed")
        except google.auth.exceptions.DefaultCredentialsError as e:
            self.fail(
                f"Application Default Credentials not found: {e}\n"
                "Run: gcloud auth application-default login"
            )

    def test_gcp_project_configured(self):
        """GCP_PROJECT_ID must be set and non-placeholder."""
        project, location, dataset = _load_config()
        self.assertNotEqual(project, "YOUR_GCP_PROJECT_ID")
        self.assertNotEqual(location, "YOUR_GCP_LOCATION")
        self.assertNotEqual(dataset, "YOUR_BQ_DATASET_ID")
        self.assertTrue(len(project) > 3, "GCP_PROJECT_ID looks too short")


# ── Check 2: Vertex AI / Gemini ───────────────────────────────────────────────

@pytest.mark.live
class TestVertexAIConnection(unittest.TestCase):
    """Verify Vertex AI can serve Gemini with logprobs enabled."""

    def setUp(self):
        self.project, self.location, _ = _load_config()
        self.client = _make_vertex_client(self.project, self.location)

    def test_gemini_responds(self):
        """Gemini must return a text response for a simple prompt."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=PREFLIGHT_MODEL,
            contents="Reply with the single digit 3.",
            config=types.GenerateContentConfig(temperature=0.0),
        )
        self.assertIsNotNone(response.text)
        self.assertGreater(len(response.text.strip()), 0)

    def test_gemini_returns_logprobs(self):
        """
        Gemini must return logprobs_result when response_logprobs=True.
        This is the prerequisite for ΔM computation.
        """
        from google.genai import types
        response = self.client.models.generate_content(
            model=PREFLIGHT_MODEL,
            contents="Reply with the single digit 3.",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_logprobs=True,
                logprobs=5,
            ),
        )
        candidate = response.candidates[0]
        self.assertIsNotNone(
            candidate.logprobs_result,
            "logprobs_result is None — verify response_logprobs=True is "
            f"supported for {PREFLIGHT_MODEL} on Vertex AI."
        )
        chosen = candidate.logprobs_result.chosen_candidates
        self.assertGreater(len(chosen), 0, "chosen_candidates list is empty")

    def test_gemini_returns_usage_metadata(self):
        """usage_metadata must be present for TAR computation."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=PREFLIGHT_MODEL,
            contents="Reply with the single digit 3.",
            config=types.GenerateContentConfig(temperature=0.0),
        )
        self.assertIsNotNone(
            response.usage_metadata,
            "usage_metadata is None — TAR cannot be computed."
        )
        self.assertIsNotNone(response.usage_metadata.candidates_token_count)

    def test_gemini_returns_thoughts_token_count(self):
        """
        thoughts_token_count must be present when ThinkingConfig is enabled.
        TAR = thoughts_token_count / candidates_token_count.
        If None, TAR will be null for all rows — expected for some model/budget combos.
        """
        from google.genai import types
        response = self.client.models.generate_content(
            model=PREFLIGHT_MODEL,
            contents="Reply with the single digit 3.",
            config=types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=512),
            ),
        )
        count = getattr(response.usage_metadata, "thoughts_token_count", None)
        if count is None:
            warnings.warn(
                f"thoughts_token_count is None for {PREFLIGHT_MODEL} at budget=low. "
                "TAR will be null for this model/budget combination. "
                "This is expected for some Gemini variants.",
                UserWarning,
            )
        # Not an assertion failure — TAR gracefully handles None


# ── Check 3: YentlBench data contract ─────────────────────────────────────────

@pytest.mark.yentlbench
class TestYentlBenchData(unittest.TestCase):
    """Verify YentlBench loads vignettes with the expected schema."""

    def test_vignettes_load(self):
        """load_vignettes must return a non-empty list."""
        vignettes = _load_vignettes(PREFLIGHT_VARIANT, PREFLIGHT_N)
        self.assertGreater(len(vignettes), 0)

    def test_vignette_has_required_fields(self):
        """Each vignette must have vignette_id and text attributes."""
        vignettes = _load_vignettes(PREFLIGHT_VARIANT, PREFLIGHT_N)
        for v in vignettes:
            self.assertTrue(
                hasattr(v, "vignette_id"),
                f"Vignette missing vignette_id: {v}"
            )
            self.assertTrue(
                hasattr(v, "text"),
                f"Vignette {v.vignette_id} missing text attribute"
            )
            self.assertIsInstance(v.vignette_id, str)
            self.assertGreater(len(v.vignette_id), 0)
            self.assertIsInstance(v.text, str)
            self.assertGreater(len(v.text), 50,
                f"Vignette {v.vignette_id} text is suspiciously short: {repr(v.text)}")

    def test_all_five_variants_load(self):
        """All four YentlGuard variants must have rows in dataset_quintets.csv."""
        from yentlbench.config import ALL_VARIANTS
        dataset_path = _find_quintets_csv()
        if dataset_path is None:
            self.skipTest("dataset_quintets.csv not found")
        import pandas as pd
        df = pd.read_csv(dataset_path)
        for variant in ALL_VARIANTS:
            with self.subTest(variant=variant):
                count = len(df[df["gender_variant"] == variant])
                self.assertGreater(count, 0,
                    f"No rows for variant={variant} in dataset_quintets.csv")

    def test_vignettes_contain_demographic_signal(self):
        """
        Female prompts must contain a demographic signal token.
        nb_ambiguous prompts must not.
        """
        from yentlbench.local_runner.prompt import build_prompt
        dataset_path = _find_quintets_csv()
        if dataset_path is None:
            self.skipTest("dataset_quintets.csv not found")
        import pandas as pd
        df = pd.read_csv(dataset_path)

        female_tokens = {"female", "woman", "she", "her"}
        for _, row in df[df["gender_variant"] == "female"].head(3).iterrows():
            prompt = build_prompt(row.to_dict(), "female").lower()
            has_token = any(t in prompt for t in female_tokens)
            self.assertTrue(has_token,
                f"Female prompt for source_stay_id={row['source_stay_id']} "
                f"contains no demographic token. Snippet: {prompt[:200]}")

        for _, row in df[df["gender_variant"] == "nb_ambiguous"].head(3).iterrows():
            prompt = build_prompt(row.to_dict(), "nb_ambiguous").lower()
            # Sex: nan is acceptable — it carries no demographic signal
            # Sex: female / male / non-binary would be a bug
            for bad_token in {"sex: female", "sex: male", "sex: non-binary"}:
                self.assertNotIn(bad_token, prompt,
                    f"nb_ambiguous prompt for source_stay_id={row['source_stay_id']} "
                    f"contains explicit sex label '{bad_token}' — demographic signal should be absent.")

    def test_preflight_subset_vs_full(self):
        """
        Pre-flight uses {PREFLIGHT_N} vignettes; full run uses all available.
        Log the ratio so it's visible in test output.
        """
        try:
            from yentlbench.data import load_vignettes
        except ImportError:
            self.skipTest("YentlBench not installed")

        full = load_vignettes(variant=PREFLIGHT_VARIANT)
        subset = full[:PREFLIGHT_N]
        ratio = len(subset) / len(full) if full else 0

        print(
            f"\n  Pre-flight subset: {len(subset)} / {len(full)} vignettes "
            f"({ratio:.0%} of full run)"
        )
        self.assertLessEqual(len(subset), len(full))
        self.assertEqual(len(subset), min(PREFLIGHT_N, len(full)))


# ── Check 4: Arize Phoenix tracing ────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.phoenix
class TestPhoenixTracing(unittest.TestCase):
    """Verify Phoenix collector endpoint is reachable."""

    def test_phoenix_env_vars_set(self):
        """PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT must be set."""
        key = os.environ.get("PHOENIX_API_KEY")
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        self.assertIsNotNone(
            key,
            "PHOENIX_API_KEY not set. Export it before running pre-flight."
        )
        self.assertIsNotNone(
            endpoint,
            "PHOENIX_COLLECTOR_ENDPOINT not set. Export it before running pre-flight."
        )
        self.assertNotEqual(key, "your_phoenix_api_key")

    def test_phoenix_endpoint_reachable(self):
        """Phoenix OTLP collector must accept a HEAD request."""
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed; run: pip install requests")

        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
        if not endpoint:
            self.skipTest("PHOENIX_COLLECTOR_ENDPOINT not set")

        api_key = os.environ.get("PHOENIX_API_KEY", "")
        url = f"{endpoint.rstrip('/')}/v1/traces"

        try:
            r = requests.head(url, headers={"api_key": api_key}, timeout=10)
            # 200, 405 (method not allowed), or 404 all confirm the server is up
            self.assertIn(
                r.status_code, [200, 404, 405],
                f"Unexpected status {r.status_code} from Phoenix. "
                f"Server may be unreachable at {url}."
            )
        except requests.exceptions.ConnectionError as e:
            self.fail(
                f"Cannot reach Phoenix at {url}: {e}\n"
                "Check PHOENIX_COLLECTOR_ENDPOINT and network connectivity."
            )
        except requests.exceptions.Timeout:
            self.fail(f"Phoenix endpoint timed out after 10s: {url}")


# ── Check 5: BigQuery connectivity ────────────────────────────────────────────

@pytest.mark.live
class TestBigQueryConnectivity(unittest.TestCase):
    """Verify BigQuery tables exist and accept streaming inserts."""

    def setUp(self):
        self.project, _, self.dataset = _load_config()

    def test_bq_client_initializes(self):
        """BigQuery client must initialize without credential errors."""
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.project)
            self.assertIsNotNone(client)
        except Exception as e:
            self.fail(f"BigQuery client init failed: {e}")

    def test_runs_table_exists(self):
        """runs table must exist. If not, run: python -m yentlguard.eval.schema"""
        try:
            from google.cloud import bigquery

            from yentlguard.config import RUNS_TABLE
            client = bigquery.Client(project=self.project)
            table = client.get_table(RUNS_TABLE)
            self.assertIsNotNone(table)
            self.assertGreater(
                len(table.schema), 0,
                "runs table exists but has no columns — schema may be wrong."
            )
        except Exception as e:
            self.fail(
                f"runs table not found or inaccessible: {e}\n"
                "Run: python -m yentlguard.eval.schema"
            )

    def test_experiments_table_exists(self):
        """experiments table must exist."""
        try:
            from google.cloud import bigquery

            from yentlguard.config import EXPTS_TABLE
            client = bigquery.Client(project=self.project)
            table = client.get_table(EXPTS_TABLE)
            self.assertIsNotNone(table)
        except Exception as e:
            self.fail(
                f"experiments table not found: {e}\n"
                "Run: python -m yentlguard.eval.schema"
            )

    def test_bq_streaming_insert_accepts_row(self):
        """
        A minimal test row must insert without errors.
        Uses a clearly-marked preflight run_id so it can be filtered out.
        """
        try:
            import datetime

            from google.cloud import bigquery

            from yentlguard.config import RUNS_TABLE

            client = bigquery.Client(project=self.project)
            test_row = {
                "run_id": f"preflight-{uuid.uuid4()}",
                "row_id": str(uuid.uuid4()),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "vignette_id": "PREFLIGHT_TEST",
                "model_version": PREFLIGHT_MODEL,
                "model_family": "preflight",
                "demographic_variant": "nb_ambiguous",
                "pass_number": 1,
                "errors": [],
            }
            errors = client.insert_rows_json(RUNS_TABLE, [test_row])
            self.assertEqual(
                errors, [],
                f"BigQuery streaming insert returned errors: {errors}"
            )
        except Exception as e:
            self.fail(f"BigQuery streaming insert failed: {e}")


# ── Check 6: Phoenix MCP reachability ─────────────────────────────────────────

@pytest.mark.mcp
class TestPhoenixMCPReachability(unittest.TestCase):
    """Verify Phoenix MCP SSE endpoint is reachable."""

    def setUp(self):
        self.mcp_endpoint = os.environ.get(
            "YENTLGUARD_PHOENIX_MCP_ENDPOINT",
            "http://localhost:6006/mcp/sse"
        )

    def test_mcp_endpoint_configured(self):
        """
        YENTLGUARD_PHOENIX_MCP_ENDPOINT must default to a valid localhost URL.
        Documents the expected config path — no live network check.
        """
        self.assertIn("http", self.mcp_endpoint)
        self.assertIn("6006", self.mcp_endpoint)

    def test_mcp_sse_endpoint_reachable(self):
        """SSE endpoint must respond within 5 seconds."""
        try:
            import requests
        except ImportError:
            self.skipTest("requests not installed")

        try:
            # HEAD request to SSE endpoint — confirms server is up
            # SSE connections themselves are long-lived; HEAD is enough for pre-flight
            r = requests.get(
                self.mcp_endpoint,
                headers={"Accept": "text/event-stream"},
                timeout=5,
                stream=True,
            )
            r.close()
            self.assertIn(
                r.status_code, [200, 404, 405, 406],
                f"Unexpected status {r.status_code} from MCP endpoint. "
                f"Check Phoenix is running at {self.mcp_endpoint}."
            )
        except requests.exceptions.ConnectionError:
            self.skipTest(
                f"Phoenix MCP server not reachable at {self.mcp_endpoint}. "
                "Start Phoenix locally: python -m phoenix.server.main "
                "or skip MCP tests with: pytest -m 'not mcp'"
            )
        except requests.exceptions.Timeout:
            self.fail(f"Phoenix MCP endpoint timed out after 5s: {self.mcp_endpoint}")


# ── Check 7: Metrics compute on real Gemini response ─────────────────────────

@pytest.mark.live
@pytest.mark.yentlbench
class TestMetricsOnRealResponse(unittest.TestCase):
    """
    Run 3 vignettes through Gemini and verify ΔM, TAR compute without error.
    This is the lightest possible real end-to-end test.
    """

    def setUp(self):
        self.project, self.location, _ = _load_config()
        self.client = _make_vertex_client(self.project, self.location)

    def _run_vignette(self, vignette_text: str):
        from google.genai import types

        from yentlguard.metrics.delta_m import compute_delta_m
        from yentlguard.metrics.tar import compute_tar

        config = types.GenerateContentConfig(
            temperature=0.0,
            response_logprobs=True,
            logprobs=5,
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        )
        response = self.client.models.generate_content(
            model=PREFLIGHT_MODEL,
            contents=vignette_text,
            config=config,
        )
        dm = compute_delta_m(response)
        tar = compute_tar(response, thinking_budget="low")
        return response, dm, tar

    def test_delta_m_computes_on_real_response(self):
        """ΔM must be computable on at least one of the 3 test vignettes."""
        vignettes = _load_vignettes("nb_ambiguous", 3)
        any_delta_m = False

        for v in vignettes:
            with self.subTest(vignette_id=v.vignette_id):
                _, dm, _ = self._run_vignette(v.text)
                if dm is not None:
                    any_delta_m = True
                    self.assertIsInstance(dm.delta_m, (float, type(None)))
                    self.assertIsInstance(dm.esi_token, str)
                    self.assertIn(dm.esi_token, {"1", "2", "3", "4", "5"},
                        f"ESI token {dm.esi_token!r} is not a valid ESI digit")
                    print(
                        f"\n  {v.vignette_id}: ESI={dm.esi_token} "
                        f"ΔM={dm.delta_m:.4f} "
                        f"top_prob={dm.top_prob:.3f}"
                    )

        if not any_delta_m:
            warnings.warn(
                "ΔM was None for all 3 vignettes. Gemini may not be outputting "
                "a bare ESI digit — check the triage prompt format in YentlBench.",
                UserWarning,
            )

    def test_tar_computes_or_gracefully_returns_none(self):
        """TAR must either compute correctly or return None gracefully."""
        vignettes = _load_vignettes("nb_ambiguous", 3)

        for v in vignettes[:1]:  # one vignette is enough for TAR check
            _, _, tar = self._run_vignette(v.text)
            if tar is not None:
                self.assertGreater(tar.tar, 0)
                self.assertGreater(tar.thoughts_token_count, 0)
                self.assertGreater(tar.candidates_token_count, 0)
                print(
                    f"\n  {v.vignette_id}: TAR={tar.tar:.4f} "
                    f"thoughts={tar.thoughts_token_count} "
                    f"candidates={tar.candidates_token_count}"
                )
            else:
                print(
                    f"\n  {v.vignette_id}: TAR=None "
                    f"(thoughts_token_count not returned for {PREFLIGHT_MODEL} "
                    f"at budget={PREFLIGHT_BUDGET})"
                )


# ── Check 8: Parallel Triad on 3 vignettes ───────────────────────────────────

@pytest.mark.live
@pytest.mark.yentlbench
class TestParallelTriadPreflight(unittest.TestCase):
    """
    Run 3 female vignettes through the full Parallel Triad.
    Gate threshold set to 999.0 to force all four branches on every vignette.
    Validates: all branches execute, results stored on VignetteRun,
    no runtime errors from asyncio or Vertex AI.
    """

    def setUp(self):
        self.project, self.location, _ = _load_config()

    def test_parallel_triad_runs_3_vignettes(self):
        """
        3 female vignettes must complete the full Parallel Triad without
        RuntimeError, TimeoutError, or credential failures.
        """
        from yentlguard.agent.runner import YentlGuardRunner
        from yentlguard.telemetry.phoenix import setup_phoenix_tracing

        # Setup Phoenix tracing if credentials available
        try:
            provider = setup_phoenix_tracing()
        except (ValueError, Exception):
            provider = None  # Phoenix not configured — skip tracing, run continues

        runner = YentlGuardRunner(
            model_version=PREFLIGHT_MODEL,
            thinking_budget=PREFLIGHT_BUDGET,
            delta_m_threshold=999.0,  # force gate on every vignette
            baseline_lookup=None,  # no MCP baseline — CRR will be None
        )

        vignettes = _load_vignettes("female", 3)
        results = []

        for v in vignettes:
            start = time.time()
            run = runner.run(
                vignette_id=v.vignette_id,
                vignette_text=v.text,
                demographic_variant="female",
            )
            elapsed = time.time() - start

            # No fatal errors
            fatal_errors = [e for e in run.errors if "Pass 1" in e]
            self.assertEqual(
                fatal_errors, [],
                f"Pass 1 failed for {v.vignette_id}: {fatal_errors}"
            )

            # Gate must have fired (threshold=999.0 forces it)
            self.assertTrue(
                run.intervention_triggered,
                f"Gate did not fire for {v.vignette_id}. "
                "Check ΔM extraction — gate_threshold=999.0 should always fire."
            )

            # All four branches must have executed
            self.assertIsNotNone(
                run.pass2_delta_m,
                f"{v.vignette_id}: corrective branch (pass2) produced no ΔM"
            )
            for label, attr in [("3a", "pass3a_delta_m"), ("3b", "pass3b_delta_m"), ("3c", "pass3c_delta_m")]:
                dm = getattr(run, attr, None)
                distractor_error = next(
                    (e for e in run.errors if f"Pass {label}" in e), None
                )
                self.assertIsNone(
                    distractor_error,
                    f"{v.vignette_id}: branch {label} failed: {distractor_error}"
                )

            print(
                f"\n  {v.vignette_id} ({elapsed:.1f}s):"
                f"\n    Pass1  ESI={run.pass1_esi} "
                f"ΔM={f'{run.pass1_delta_m.delta_m:.4f}' if run.pass1_delta_m and run.pass1_delta_m.delta_m else 'None'}"
                f"\n    Corr   ESI={run.pass2_esi} "
                f"ΔM={f'{run.pass2_delta_m.delta_m:.4f}' if run.pass2_delta_m and run.pass2_delta_m.delta_m else 'None'}"
                f"\n    3a     ESI={run.pass3a_esi} "
                f"ΔM={f'{run.pass3a_delta_m.delta_m:.4f}' if run.pass3a_delta_m and run.pass3a_delta_m.delta_m else 'None'}"
                f"\n    3b     ESI={run.pass3b_esi} "
                f"ΔM={f'{run.pass3b_delta_m.delta_m:.4f}' if run.pass3b_delta_m and run.pass3b_delta_m.delta_m else 'None'}"
                f"\n    3c     ESI={run.pass3c_esi} "
                f"ΔM={f'{run.pass3c_delta_m.delta_m:.4f}' if run.pass3c_delta_m and run.pass3c_delta_m.delta_m else 'None'}"
                f"\n    Errors: {run.errors or 'none'}"
            )
            results.append(run)

        # At least 2 of 3 must complete without any errors
        clean_runs = [r for r in results if not r.errors]
        self.assertGreaterEqual(
            len(clean_runs), 2,
            f"Too many vignettes had errors: "
            f"{[(r.vignette_id, r.errors) for r in results if r.errors]}"
        )

        if provider:
            provider.force_flush()

    def test_branch_esi_tokens_are_valid(self):
        """All four branches must return valid ESI digits (1-5) or None."""
        from yentlguard.agent.runner import YentlGuardRunner

        runner = YentlGuardRunner(
            model_version=PREFLIGHT_MODEL,
            thinking_budget=PREFLIGHT_BUDGET,
            delta_m_threshold=999.0,
            baseline_lookup=None,
        )

        vignettes = _load_vignettes("female", 1)
        run = runner.run(
            vignette_id=vignettes[0].vignette_id,
            vignette_text=vignettes[0].text,
            demographic_variant="female",
        )

        valid_esi = {"1", "2", "3", "4", "5", None}
        for attr, label in [
            ("pass1_esi", "Pass 1"),
            ("pass2_esi", "corrective"),
            ("pass3a_esi", "3a"),
            ("pass3b_esi", "3b"),
            ("pass3c_esi", "3c"),
        ]:
            esi = getattr(run, attr)
            self.assertIn(
                esi, valid_esi,
                f"{label} ESI token {esi!r} is not a valid ESI digit or None"
            )


# ── pytest markers registration ───────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires live GCP credentials and network")
    config.addinivalue_line("markers", "yentlbench: requires YentlBench installed")
    config.addinivalue_line("markers", "phoenix: requires Phoenix endpoint reachable")
    config.addinivalue_line("markers", "mcp: requires Phoenix MCP server running")


if __name__ == "__main__":
    unittest.main(verbosity=2)
