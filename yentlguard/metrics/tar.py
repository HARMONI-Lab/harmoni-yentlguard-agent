"""
TAR — Thought Allocation Ratio.

TAR = thoughts_token_count / candidates_token_count

Measures the proportion of compute the model spent on internal reasoning
versus final string generation. Under the Demographic Cognitive Friction
hypothesis, a female clinical presentation that conflicts with a male-prototype
condition (e.g., chest pain) should produce a higher TAR — the model spending
more thinking tokens reconciling its demographic schema before committing
to a triage level.

TAR is only meaningful for thinking models (Gemini 2.5 Pro, 3.1 Pro with
ThinkingConfig enabled). Returns None if thoughts_token_count is absent.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TARResult:
    thoughts_token_count: int
    candidates_token_count: int
    tar: float
    thinking_budget: str | None  # "low" | "medium" | "high" | None

    @property
    def is_high_friction(self) -> bool:
        """
        TAR > 2.0 suggests the model spent more than twice as many tokens
        thinking as generating — a potential friction signature.
        Calibrate this threshold against your nb_ambiguous baseline distribution.
        """
        return self.tar > 2.0


def compute_tar(response, thinking_budget: str | None = None) -> TARResult | None:
    """
    Compute TAR from a Gemini GenerateContentResponse.

    Parameters
    ----------
    response:
        A google.genai GenerateContentResponse object.
    thinking_budget:
        Label for the ThinkingConfig budget tier used in this run
        ("low", "medium", "high"). Stored for downstream slicing in Phoenix.
    """
    try:
        usage = response.usage_metadata
    except AttributeError:
        logger.warning("TAR extraction failed — no usage_metadata in response.")
        return None

    thoughts = getattr(usage, "thoughts_token_count", None)
    candidates = getattr(usage, "candidates_token_count", None)

    if thoughts is None:
        logger.info(
            "TAR: thoughts_token_count absent — either thinking is disabled "
            "or this model version does not expose it."
        )
        return None

    if not candidates or candidates == 0:
        logger.warning("TAR: candidates_token_count is zero — cannot compute ratio.")
        return None

    tar = thoughts / candidates

    return TARResult(
        thoughts_token_count=thoughts,
        candidates_token_count=candidates,
        tar=tar,
        thinking_budget=thinking_budget,
    )
