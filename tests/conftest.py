"""
tests/conftest.py

Installs stubs for all external dependencies (openinference, opentelemetry,
google.genai, vertexai, google.cloud.bigquery) before any test module
imports from yentlguard. This prevents ModuleNotFoundError from packages
that require live GCP credentials or are not installed in the test environment.

All stubs are registered in sys.modules at collection time so that
yentlguard/__init__.py → telemetry/phoenix.py → openinference import chain
resolves cleanly without hitting the real packages.
"""

import os
import pathlib
import sys
import types
from unittest.mock import MagicMock


def _stub(name: str) -> types.ModuleType:
    """Create and register a stub module if not already present."""
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


# Skip stubbing if running preflight tests, which require live libraries.
_is_preflight = any("test_preflight.py" in arg for arg in sys.argv)

if not _is_preflight:
    # ── Register all stub modules ─────────────────────────────────────────────────

    _STUB_NAMES = [
        # OpenInference

    "openinference",
    "openinference.instrumentation",
    "openinference.instrumentation.google_genai",
    # OpenTelemetry core
    "opentelemetry",
    "opentelemetry.trace",
    # OpenTelemetry SDK
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.sdk.resources",
    # OpenTelemetry OTLP exporter chain
    "opentelemetry.exporter",
    "opentelemetry.exporter.otlp",
    "opentelemetry.exporter.otlp.proto",
    "opentelemetry.exporter.otlp.proto.http",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    # Google GenAI / Vertex AI
    "google",
    "google.genai",
    "google.genai.types",
    "google.cloud",
    "google.cloud.bigquery",
    # Vertex AI
    "vertexai",
    "vertexai.preview",
    "vertexai.preview.evaluation",
]

    for _name in _STUB_NAMES:
        _stub(_name)

    # ── Attribute stubs required by specific import statements ────────────────────

    # openinference.instrumentation.google_genai.GoogleGenAIInstrumentor
    _oi = sys.modules["openinference.instrumentation.google_genai"]
    _oi.GoogleGenAIInstrumentor = MagicMock(return_value=MagicMock())

    # opentelemetry.sdk.trace.export.BatchSpanProcessor
    _export = sys.modules["opentelemetry.sdk.trace.export"]
    _export.BatchSpanProcessor = MagicMock()

    # opentelemetry.sdk.trace.TracerProvider
    _sdk_trace = sys.modules["opentelemetry.sdk.trace"]
    _sdk_trace.TracerProvider = MagicMock(return_value=MagicMock())

    # opentelemetry.sdk.resources.Resource
    _resources = sys.modules["opentelemetry.sdk.resources"]
    _resources.Resource = MagicMock()
    _resources.Resource.create = MagicMock(return_value=MagicMock())

    # opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter
    _otlp = sys.modules["opentelemetry.exporter.otlp.proto.http.trace_exporter"]
    _otlp.OTLPSpanExporter = MagicMock(return_value=MagicMock())

    # opentelemetry.trace — tracer, span, status stubs
    _trace = sys.modules["opentelemetry.trace"]
    _trace.get_tracer = MagicMock(return_value=MagicMock())
    _trace.get_current_span = MagicMock(return_value=MagicMock())
    _trace.SpanKind = MagicMock()
    _trace.Status = MagicMock()
    _trace.StatusCode = MagicMock()
    _trace.Span = MagicMock          # annotation.py imports Span as a type
    _trace.NonRecordingSpan = MagicMock()

    # yentlguard.config stub — must be registered before yentlguard submodules import it
    import types as _types
    _cfg = _types.ModuleType("yentlguard.config")
    _cfg.GCP_PROJECT_ID = "test-project"
    _cfg.GCP_LOCATION   = "us-central1"
    _cfg.BQ_DATASET_ID  = "test_dataset"
    _cfg.FULL_DATASET   = "test-project.test_dataset"
    _cfg.RUNS_TABLE     = "test-project.test_dataset.runs"
    _cfg.EXPTS_TABLE    = "test-project.test_dataset.experiments"
    _cfg.BQ_LOCATION    = "US"
    _cfg.validate       = MagicMock()
    sys.modules["yentlguard.config"] = _cfg

    # google.genai stubs
    _genai = sys.modules["google.genai"]
    _genai.Client = MagicMock(return_value=MagicMock())

    _genai_types = sys.modules["google.genai.types"]
    _genai_types.ThinkingConfig = MagicMock(return_value=MagicMock())
    _genai_types.GenerateContentConfig = MagicMock(return_value=MagicMock())

    # google.cloud.bigquery stubs
    _bq = sys.modules["google.cloud.bigquery"]
    _bq.Client = MagicMock(return_value=MagicMock())
    _bq.SchemaField = MagicMock()
    _bq.Dataset = MagicMock()
    _bq.Table = MagicMock()
    _bq.TimePartitioning = MagicMock()
    _bq.TimePartitioningType = MagicMock()
    _bq.ArrayQueryParameter = MagicMock()
    _bq.ScalarQueryParameter = MagicMock()
    _bq.QueryJobConfig = MagicMock()


    # vertexai stubs
    _vai = sys.modules["vertexai"]
    _vai.init = MagicMock()

    _vai_eval = sys.modules["vertexai.preview.evaluation"]
    _vai_eval.EvalTask = MagicMock()
    _vai_eval.PointwiseMetric = MagicMock()


def _find_quintets_csv() -> pathlib.Path | None:
    """
    Locate dataset_quintets.csv. Checks:
    1. YENTLGUARD_DATASET_PATH env var
    2. ./dataset_output/dataset_quintets.csv (YentlBench default)
    3. ./dataset_quintets.csv (flat layout)
    """
    env = os.environ.get("YENTLGUARD_DATASET_PATH")
    if env:
        p = pathlib.Path(env)
        if p.exists():
            return p

    candidates = [
        pathlib.Path("dataset_output/dataset_quintets.csv"),
        pathlib.Path("dataset_quintets.csv"),
        pathlib.Path("../dataset_output/dataset_quintets.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
