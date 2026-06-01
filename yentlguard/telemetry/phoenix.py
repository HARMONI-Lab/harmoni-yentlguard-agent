"""
YentlGuard telemetry — Phoenix tracing via phoenix.otel.register.

Replaces the previous manual OTLPSpanExporter + GoogleGenAIInstrumentor setup.
phoenix.otel.register(auto_instrument=True) patches both the google-genai SDK
and the google-adk SDK in a single call, which is required now that the agent
layer uses ADK.

Environment variables (unchanged from previous version):
    PHOENIX_API_KEY              — Phoenix Cloud API key (format: px_live_...)
    PHOENIX_COLLECTOR_ENDPOINT   — space-scoped OTLP endpoint
                                   (e.g. https://app.phoenix.arize.com/s/your-space)
    PHOENIX_PROJECT_NAME         — project tag on all spans (default: yentlguard)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_provider: Any = None


def setup_phoenix_tracing(
    project_name: str | None = None,
    batch: bool = True,
) -> Any:
    """
    Initialize Phoenix tracing. Idempotent — returns the existing provider on
    subsequent calls.

    Parameters
    ----------
    project_name:
        Phoenix project tag applied to all spans. Defaults to
        PHOENIX_PROJECT_NAME env var, then "yentlguard".
    batch:
        True (default) for CLI runs — spans are batched before export.
        False for interactive adk web sessions — flushes per turn.

    Returns
    -------
    TracerProvider or None if PHOENIX_API_KEY is not set.
    """
    global _provider
    if _provider is not None:
        return _provider

    api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "PHOENIX_API_KEY not set — Phoenix tracing disabled. Spans will not be exported."
        )
        return None

    from phoenix.otel import register

    resolved_project = project_name or os.environ.get("PHOENIX_PROJECT_NAME", "yentlguard")

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if endpoint and not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint.rstrip('/')}/v1/traces"

    _provider = register(
        project_name=resolved_project,
        endpoint=endpoint if endpoint else None,
        batch=batch,
        auto_instrument=True,
        verbose=False,
    )
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
    logger.info("YentlGuard → Phoenix tracing active. Project: %s", resolved_project)
    return _provider
