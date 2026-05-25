"""
YentlGuard span annotation layer.

Two responsibilities:

1. **Span enrichment** — attaches custom attributes to the auto-instrumented
   generation span that OpenInference creates. This puts vignette_id,
   demographic_variant, delta_m, tar, and raw logprob metadata directly
   onto the span Phoenix already captured, so filtering in the UI works
   without a join.

2. **Child span creation** — creates a nested span hierarchy under each
   vignette trace:

   vignette_trace (root)
   ├── pass1 (generation span — auto-created by OpenInference, enriched here)
   │   ├── pass1.metrics (child: ΔM + TAR as attributes)
   │   │   ├── delta_m (grandchild: full logprob breakdown)
   │   │   └── tar (grandchild: token count breakdown)
   ├── correction_gate (child: gate decision + threshold + demographic trigger)
   ├── mcp.baseline_lookup (child: Phoenix MCP query result)
   │
   │   [Parallel Triad — all four branches spawn concurrently]
   ├── pass2 / corrective (enriched: ΔM, pass_number=2)
   │   └── pass2.metrics → delta_m grandchild
   ├── pass3 / 3a Pure Clinical Anchor (enriched: ΔM, pass_number=3)
   │   └── pass3.metrics → delta_m grandchild
   ├── pass4 / 3b Forced Parsing Anchor (enriched: ΔM, pass_number=4)
   │   └── pass4.metrics → delta_m grandchild
   ├── pass5 / 3c Protocol Anchor (enriched: ΔM, pass_number=5)
   │   └── pass5.metrics → delta_m grandchild
   └── crr (child: corrective CRR result — distractor CRRs on BQ row, not span)

This gives maximum Phoenix observability: you can slice at any level,
from aggregate sycophancy gap down to the exact logprob of the runner-up ESI token.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from yentlguard.metrics.crr import CRRResult
from yentlguard.metrics.delta_m import DeltaMResult
from yentlguard.metrics.tar import TARResult

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("yentlguard.annotation", "0.1.0")


# ── Attribute helpers ──────────────────────────────────────────────────────────

def _safe_set(span: Span, key: str, value) -> None:
    """Set a span attribute, silently skipping None values."""
    if value is not None:
        try:
            span.set_attribute(key, value)
        except Exception as e:
            logger.debug("Span attribute set failed for %s: %s", key, e)


def _set_delta_m_attributes(span: Span, result: DeltaMResult, prefix: str = "") -> None:
    """Write all ΔM fields onto a span with an optional key prefix."""
    p = f"{prefix}." if prefix else ""
    _safe_set(span, f"{p}delta_m", result.delta_m)
    _safe_set(span, f"{p}esi_token", result.esi_token)
    _safe_set(span, f"{p}top_logprob", result.top_logprob)
    _safe_set(span, f"{p}top_prob", result.top_prob)
    _safe_set(span, f"{p}runner_up_token", result.runner_up_token)
    _safe_set(span, f"{p}runner_up_logprob", result.runner_up_logprob)
    _safe_set(span, f"{p}runner_up_prob", result.runner_up_prob)
    _safe_set(span, f"{p}token_index", result.token_index)
    _safe_set(span, f"{p}is_low_confidence", result.is_low_confidence)


def _set_tar_attributes(span: Span, result: TARResult, prefix: str = "") -> None:
    """Write all TAR fields onto a span."""
    p = f"{prefix}." if prefix else ""
    _safe_set(span, f"{p}tar", result.tar)
    _safe_set(span, f"{p}thoughts_token_count", result.thoughts_token_count)
    _safe_set(span, f"{p}candidates_token_count", result.candidates_token_count)
    _safe_set(span, f"{p}thinking_budget", result.thinking_budget)
    _safe_set(span, f"{p}is_high_friction", result.is_high_friction)


def _set_crr_attributes(span: Span, result: CRRResult) -> None:
    """Write all CRR fields onto a span."""
    _safe_set(span, "crr", result.crr)
    _safe_set(span, "delta_m_baseline", result.delta_m_baseline)
    _safe_set(span, "delta_m_pass1", result.delta_m_pass1)
    _safe_set(span, "delta_m_pass2", result.delta_m_pass2)
    _safe_set(span, "esi_token_pass1", result.esi_token_pass1)
    _safe_set(span, "esi_token_pass2", result.esi_token_pass2)
    _safe_set(span, "triage_changed", result.triage_changed)
    _safe_set(span, "full_recovery", result.full_recovery)
    _safe_set(span, "partial_recovery", result.partial_recovery)
    _safe_set(span, "failed_recovery", result.failed_recovery)


# ── Context managers ───────────────────────────────────────────────────────────

@contextmanager
def vignette_trace(
    vignette_id: str,
    demographic_variant: str,
    model_version: str,
    thinking_budget: str | None,
    clinical_category: str | None = None,
) -> Generator[Span, None, None]:
    """
    Root span for a single vignette × variant mechanistic run.

    All child spans (pass1, pass2, correction_gate, crr) should be created
    inside this context so they share the same trace ID.

    Usage:
        with vignette_trace("ED_00147", "female", "gemini-2.5-pro", "medium") as span:
            # run passes inside here
    """
    with tracer.start_as_current_span(
        f"yentlguard.vignette.{demographic_variant}",
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        _safe_set(span, "yentlguard.vignette_id", vignette_id)
        _safe_set(span, "yentlguard.demographic_variant", demographic_variant)
        _safe_set(span, "yentlguard.model_version", model_version)
        _safe_set(span, "yentlguard.thinking_budget", thinking_budget)
        _safe_set(span, "yentlguard.clinical_category", clinical_category)
        yield span


@contextmanager
def pass_metrics_span(
    pass_number: int,
    delta_m_result: DeltaMResult | None,
    tar_result: TARResult | None = None,
) -> Generator[Span, None, None]:
    """
    Child span grouping all computed metrics for a single pass.

    Creates two grandchild spans (delta_m, tar) for per-metric filtering.
    """
    with tracer.start_as_current_span(
        f"yentlguard.pass{pass_number}.metrics",
        kind=trace.SpanKind.INTERNAL,
    ) as metrics_span:

        if delta_m_result is not None:
            _set_delta_m_attributes(metrics_span, delta_m_result)

        if tar_result is not None:
            _set_tar_attributes(metrics_span, tar_result)

        # ── delta_m grandchild ─────────────────────────────────────────────
        if delta_m_result is not None:
            with tracer.start_as_current_span(
                f"yentlguard.pass{pass_number}.delta_m",
                kind=trace.SpanKind.INTERNAL,
            ) as dm_span:
                _set_delta_m_attributes(dm_span, delta_m_result)
                dm_span.set_status(
                    Status(
                        StatusCode.ERROR if delta_m_result.is_low_confidence else StatusCode.OK,
                        "low_confidence" if delta_m_result.is_low_confidence else ""
                    )
                )

        # ── tar grandchild ─────────────────────────────────────────────────
        if tar_result is not None:
            with tracer.start_as_current_span(
                f"yentlguard.pass{pass_number}.tar",
                kind=trace.SpanKind.INTERNAL,
            ) as tar_span:
                _set_tar_attributes(tar_span, tar_result)
                tar_span.set_status(
                    Status(
                        StatusCode.ERROR if tar_result.is_high_friction else StatusCode.OK,
                        "high_friction" if tar_result.is_high_friction else ""
                    )
                )

        yield metrics_span


@contextmanager
def correction_gate_span(
    vignette_id: str,
    delta_m: float | None,
    threshold: float,
    demographic_trigger: bool,
    fired: bool,
) -> Generator[Span, None, None]:
    """
    Child span recording the correction gate decision and its inputs.

    Even when the gate does not fire, this span is created so you can
    analyze the full distribution of gate decisions in Phoenix.
    """
    with tracer.start_as_current_span(
        "yentlguard.correction_gate",
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        _safe_set(span, "gate.vignette_id", vignette_id)
        _safe_set(span, "gate.delta_m", delta_m)
        _safe_set(span, "gate.threshold", threshold)
        _safe_set(span, "gate.demographic_trigger", demographic_trigger)
        _safe_set(span, "gate.fired", fired)
        _safe_set(span, "gate.low_confidence", delta_m is not None and delta_m < threshold)
        span.set_status(Status(StatusCode.OK))
        yield span


@contextmanager
def mcp_lookup_span(
    vignette_id: str,
    variant: str,
    baseline_delta_m: float | None,
    success: bool,
    error: str | None = None,
) -> Generator[Span, None, None]:
    """
    Child span recording the Phoenix MCP baseline lookup result.
    """
    with tracer.start_as_current_span(
        "yentlguard.mcp.baseline_lookup",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        _safe_set(span, "mcp.vignette_id", vignette_id)
        _safe_set(span, "mcp.variant", variant)
        _safe_set(span, "mcp.baseline_delta_m", baseline_delta_m)
        _safe_set(span, "mcp.success", success)
        _safe_set(span, "mcp.error", error)
        span.set_status(
            Status(StatusCode.OK if success else StatusCode.ERROR, error if not success else "")
        )
        yield span


@contextmanager
def crr_span(result: CRRResult) -> Generator[Span, None, None]:
    """
    Child span carrying the full CRR computation result.
    """
    with tracer.start_as_current_span(
        "yentlguard.crr",
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        _set_crr_attributes(span, result)
        span.set_status(Status(StatusCode.OK))
        yield span


# ── Standalone enrichment (for the auto-instrumented generation spans) ─────────

def enrich_generation_span(
    span: Span,
    vignette_id: str,
    demographic_variant: str,
    model_version: str,
    thinking_budget: str | None,
    pass_number: int,
    delta_m_result: DeltaMResult | None = None,
    tar_result: TARResult | None = None,
    clinical_category: str | None = None,
) -> None:
    """
    Enrich the active OpenInference generation span with YentlGuard metadata.

    Call this immediately after a generate_content() call returns, while the
    OpenInference span is still active. Writes all computed metrics directly
    onto the generation span so Phoenix shows them alongside the raw LLM output.

    Parameters
    ----------
    span:
        The currently active OTel span (obtain via trace.get_current_span()).
    pass_number:
        1 or 2 — which pass this generation belongs to.
    """
    _safe_set(span, "yentlguard.vignette_id", vignette_id)
    _safe_set(span, "yentlguard.demographic_variant", demographic_variant)
    _safe_set(span, "yentlguard.model_version", model_version)
    _safe_set(span, "yentlguard.thinking_budget", thinking_budget)
    _safe_set(span, "yentlguard.pass_number", pass_number)
    _safe_set(span, "yentlguard.clinical_category", clinical_category)

    if delta_m_result is not None:
        _set_delta_m_attributes(span, delta_m_result, prefix="yentlguard")

    if tar_result is not None:
        _set_tar_attributes(span, tar_result, prefix="yentlguard")
