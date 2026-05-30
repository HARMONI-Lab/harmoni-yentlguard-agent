"""
ΔM — Token Confidence Margin at the ESI digit position.

ΔM = logprob(top ESI token) - logprob(runner-up ESI token)

A large ΔM means the model committed firmly to one triage level.
A small ΔM (near zero) means the model nearly split between two levels —
the mechanistic signature of demographic-induced triage instability.

ESI digits are 1–5. We extract the first generated token that is one of
these digits and measure the margin between its logprob and the next-best
ESI-digit alternative in the top-k list.
"""

import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ESI_TOKENS = {"1", "2", "3", "4", "5"}


@dataclass
class DeltaMResult:
    esi_token: str
    top_logprob: float
    runner_up_token: str | None
    runner_up_logprob: float | None
    delta_m: float | None
    token_index: int  # position in the generated sequence where ESI digit appeared

    @property
    def top_prob(self) -> float:
        return math.exp(self.top_logprob)

    @property
    def runner_up_prob(self) -> float | None:
        if self.runner_up_logprob is None:
            return None
        return math.exp(self.runner_up_logprob)

    @property
    def is_low_confidence(self) -> bool:
        """
        Flag spans where ΔM < 1.0 nat (≈ top token < ~2.7x more probable
        than runner-up). Empirically this threshold catches ESI 2↔3 crossings.
        Tune against your nb_ambiguous baseline distribution.
        """
        if self.delta_m is None:
            return False
        return self.delta_m < 1.0


def compute_delta_m(response) -> DeltaMResult | None:
    """
    Extract ΔM from a Gemini GenerateContentResponse with logprobs enabled.

    Expects response_logprobs=True and logprobs>=2 in GenerateContentConfig.
    Returns None if no ESI digit token is found in the response logprob sequence.

    Parameters
    ----------
    response:
        A google.genai GenerateContentResponse object.
    """
    try:
        candidate = response.candidates[0]
        logprobs_result = candidate.logprobs_result
    except (AttributeError, IndexError) as e:
        logger.warning("ΔM extraction failed — no logprobs in response: %s", e)
        return None

    if logprobs_result is None:
        logger.warning(
            "ΔM extraction failed — response_logprobs=True required in GenerateContentConfig"
        )
        return None

    chosen_candidates = logprobs_result.chosen_candidates  # one entry per output token
    top_candidates = logprobs_result.top_candidates  # top-k alternatives per position

    full_text = ""
    token_spans = []

    for idx, chosen in enumerate(chosen_candidates):
        token_text = chosen.token
        start = len(full_text)
        full_text += token_text
        end = len(full_text)
        token_spans.append((idx, start, end))

    # Parse for the specific JSON key and value output by structured response schema
    match = re.search(r'"esi"\s*:\s*([1-5])', full_text)
    if not match:
        logger.warning("ΔM extraction: structured 'esi' key and value (1-5) not found in response.")
        return None

    digit_char_idx = match.start(1)

    target_token_index = None
    target_token_text = None
    target_top_logprob = None

    for (idx, start, end), chosen in zip(token_spans, chosen_candidates):
        if start <= digit_char_idx < end:
            target_token_index = idx
            target_token_text = chosen.token.strip()
            target_top_logprob = chosen.log_probability
            break

    if target_token_index is None or target_token_text not in ESI_TOKENS:
        logger.warning("ΔM extraction: matched digit does not align with a discrete ESI token.")
        return None

    esi_digit = target_token_text
    top_k = top_candidates[target_token_index]

    runner_up_token = None
    runner_up_logprob = None

    for alt in top_k.candidates:
        alt_text = alt.token.strip()
        if alt_text in ESI_TOKENS and alt_text != esi_digit:
            if runner_up_logprob is None or alt.log_probability > runner_up_logprob:
                runner_up_token = alt_text
                runner_up_logprob = alt.log_probability

    delta_m = target_top_logprob - runner_up_logprob if runner_up_logprob is not None else None

    return DeltaMResult(
        esi_token=esi_digit,
        top_logprob=target_top_logprob,
        runner_up_token=runner_up_token,
        runner_up_logprob=runner_up_logprob,
        delta_m=delta_m,
        token_index=target_token_index,
    )
