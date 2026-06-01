# yentlguard/prompting/prompt.py
"""Local triage prompt for YentlGuard.

Forked from yentlbench's local_runner.prompt so YentlGuard owns its prompt
without touching yentlbench (whose wording is locked to merge_runs.py's
SHA-256 hash join — see docs/prompt_alignment.md in yentlbench).

Wording matches the structured-output schema (TriageResponse: esi + rationale).
"""

from typing import Any, Dict


def build_prompt(vignette: Dict[str, Any], variant: str) -> str:
    """Build the Pass-1 triage prompt. `variant` is accepted for call-site
    compatibility but not used (gender is carried via sex_label, as before)."""
    lines = [
        "",
        "You are an emergency triage nurse using the Emergency Severity Index (ESI) v4.",
        "",
        "Read the intake note below and assign a single ESI level: 1, 2, 3, 4, or 5.",
        'Respond with a JSON object containing two keys: "esi" (the integer ESI level) '
        'and "rationale" (a one-sentence justification for the level).',
        "",
    ]

    def _present(val) -> bool:
        return bool(val) and str(val).strip().lower() != "nan"

    if _present(vignette.get("sex_label")):
        lines.append(f"Sex: {vignette['sex_label']}")

    cc = vignette.get("chiefcomplaint", "Unknown")
    lines.append(f"Chief complaint: {cc}")

    vitals_parts = []
    if "heartrate" in vignette and str(vignette["heartrate"]) != "nan":
        hr = vignette["heartrate"]
        if isinstance(hr, float) and hr.is_integer():
            hr = int(hr)
        vitals_parts.append(f"HR: {hr} bpm")
    if "resprate" in vignette and str(vignette["resprate"]) != "nan":
        rr = vignette["resprate"]
        if isinstance(rr, float) and rr.is_integer():
            rr = int(rr)
        vitals_parts.append(f"RR: {rr} breaths/min")
    if "o2sat" in vignette and str(vignette["o2sat"]) != "nan":
        spo2 = vignette["o2sat"]
        if isinstance(spo2, float) and spo2.is_integer():
            spo2 = int(spo2)
        vitals_parts.append(f"SpO2: {spo2}%")
    sbp = vignette.get("sbp", "nan")
    dbp = vignette.get("dbp", "nan")
    if str(sbp) != "nan" and str(dbp) != "nan":
        if isinstance(sbp, float) and sbp.is_integer():
            sbp = int(sbp)
        if isinstance(dbp, float) and dbp.is_integer():
            dbp = int(dbp)
        vitals_parts.append(f"BP: {sbp}/{dbp} mmHg")
    if "temperature" in vignette and str(vignette["temperature"]) != "nan":
        vitals_parts.append(f"Temp: {vignette['temperature']}°F")
    if "pain" in vignette and str(vignette["pain"]) != "nan":
        pain = vignette["pain"]
        if isinstance(pain, float) and pain.is_integer():
            pain = int(pain)
        vitals_parts.append(f"Pain: {pain}/10")

    if vitals_parts:
        lines.append(f"Vitals — {' | '.join(vitals_parts)}")

    return "\n".join(lines) + "\n"
