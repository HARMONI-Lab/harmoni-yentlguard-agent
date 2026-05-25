"""
Arize Phoenix telemetry initialization.

Sets up OpenInference auto-instrumentation for the Google GenAI SDK so that
every Gemini API call — including full response metadata (logprobs,
thoughts_token_count, safety_ratings) — streams into Phoenix as OTel spans.
"""

import logging
import os

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_phoenix_tracing(
    api_key: str | None = None,
    collector_endpoint: str | None = None,
    project_name: str = "yentlguard",
) -> trace_sdk.TracerProvider:
    """
    Initialize OpenInference instrumentation and wire it to Arize Phoenix.

    Reads PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT from environment
    if not passed explicitly. Must be called before any google.genai client
    is instantiated so the instrumentor patches the SDK at import time.

    Parameters
    ----------
    api_key:
        Arize Phoenix API key. Falls back to PHOENIX_API_KEY env var.
    collector_endpoint:
        OTLP collector URL. Falls back to PHOENIX_COLLECTOR_ENDPOINT env var.
    project_name:
        Tag applied to all spans for filtering inside Phoenix UI.

    Returns
    -------
    TracerProvider
        The configured provider. Retain a reference if you need to force-flush
        spans before process exit (call provider.force_flush()).
    """
    api_key = api_key or os.environ.get("PHOENIX_API_KEY")
    collector_endpoint = collector_endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")

    if not api_key:
        raise ValueError(
            "Phoenix API key required. Set PHOENIX_API_KEY or pass api_key= explicitly."
        )
    if not collector_endpoint:
        raise ValueError(
            "Phoenix collector endpoint required. "
            "Set PHOENIX_COLLECTOR_ENDPOINT or pass collector_endpoint= explicitly."
        )

    exporter = OTLPSpanExporter(
        endpoint=f"{collector_endpoint.rstrip('/')}/v1/traces",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    # Attach project tag to all spans via resource attributes
    from opentelemetry.sdk.resources import Resource
    resource = Resource.create({"project.name": project_name})

    provider = trace_sdk.TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Patch the google-genai SDK so every call captures full response metadata.
    # This must happen before client instantiation.
    GoogleGenAIInstrumentor().instrument(tracer_provider=provider)

    logger.info(
        "YentlGuard → Phoenix tracing active. Project: %s. Endpoint: %s",
        project_name,
        collector_endpoint,
    )
    return provider
