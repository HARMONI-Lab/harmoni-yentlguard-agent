"""
YentlGuard — Mechanistic interpretability layer for YentlBench.

Instruments Gemini 2.5 Pro vs 3.1 Pro triage runs with token-level logprob
extraction, Thought Allocation Ratio tracking, and a two-pass corrective
re-prompting loop triggered by demographic-linked confidence drops.

Published by HARMONI Lab (harmonilab.org)
"""

from yentlguard.agent.runner import YentlGuardRunner
from yentlguard.metrics.crr import compute_crr
from yentlguard.metrics.delta_m import compute_delta_m
from yentlguard.metrics.tar import compute_tar
from yentlguard.telemetry.phoenix import setup_phoenix_tracing

__version__ = "0.1.0"
__author__ = "Inna Campo"
__lab__ = "HARMONI Lab"

__all__ = [
    "compute_delta_m",
    "compute_tar",
    "compute_crr",
    "setup_phoenix_tracing",
    "YentlGuardRunner",
]
