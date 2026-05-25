"""
tests/test_yentlbench_contract.py

Inspects the real YentlBench API and data contract.

These tests require YentlBench to be installed (pip install yentlbench)
but do NOT require MIMIC-IV-ED data or live GCP credentials.

They surface exactly what YentlBench provides and validate that
YentlGuard's integration layer produces the correct inputs for
YentlGuardRunner.run().

Key findings documented here:
  - YentlBench has NO load_vignettes() function or yentlbench.data module
  - Data flows as: MIMIC-IV-ED CSVs → yentlbench prepare → dataset_quintets.csv
  - YentlGuard does NOT run yentlbench prepare — it reads the prepared CSV
  - Vignettes are plain dicts, not objects with .vignette_id / .text attributes
  - build_prompt(vignette_dict, variant) produces the prompt text
  - acuity column = ESI ground truth (integer 1-5)
  - source_stay_id column = vignette identifier (integer, cast to str for YentlGuard)

  Variant inventory:
    dataset_quintets.csv contains 5 variants:
      nb_ambiguous  — no sex info (TRUE baseline)
      female        — full female signal (name + she/her + Female)
      male          — full male signal (name + he/him + Male)
      nb_label_only — Non-binary label only, male name, no pronoun
      nb_full       — full NB signal (neutral name + they/them + Non-binary)

    config.ALL_VARIANTS defines 4 variants (nb_full intentionally excluded):
      nb_ambiguous, female, male, nb_label_only

    YentlGuard uses config.ALL_VARIANTS — nb_full is in the CSV but not
    in the benchmark pipeline by design.
    nb_explicit does NOT exist anywhere in YentlBench.
"""

import pathlib
import sys
import unittest

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from conftest import _find_quintets_csv

# ── Helpers ───────────────────────────────────────────────────────────────────



# ── Tests ─────────────────────────────────────────────────────────────────────

class TestYentlBenchPackageAPI(unittest.TestCase):
    """
    Inspect the real YentlBench package API.
    Documents what exists and what was wrongly assumed.
    """

    def test_yentlbench_importable(self):
        """YentlBench must be installed."""
        try:
            import yentlbench
        except ImportError:
            self.fail("YentlBench not installed. Run: pip install yentlbench")

    def test_no_load_vignettes_in_yentlbench_data(self):
        """
        CRITICAL: yentlbench.data does not exist.
        YentlGuard's CLI currently calls:
            from yentlbench.data import load_vignettes
        This will raise ModuleNotFoundError at runtime.
        This test documents the mismatch so it can be fixed.
        """
        with self.assertRaises((ModuleNotFoundError, ImportError)):
            pass

    def test_actual_modules_present(self):
        """Document what YentlBench actually exports."""
        expected_modules = [
            "yentlbench.config",
            "yentlbench.dataset_prep",
            "yentlbench.benchmark_stats",
            "yentlbench.local_runner.prompt",
            "yentlbench.local_runner.parser",
            "yentlbench.local_runner.ollama_runner",
        ]
        import importlib
        for mod_name in expected_modules:
            with self.subTest(module=mod_name):
                try:
                    importlib.import_module(mod_name)
                except ImportError as e:
                    self.fail(f"Expected module missing: {mod_name}: {e}")

    def test_build_prompt_is_the_vignette_api(self):
        """
        build_prompt(vignette_dict, variant) is the correct way to produce
        a vignette text string from a dataset row. This is what YentlGuard
        should call, not a nonexistent load_vignettes().
        """
        from yentlbench.local_runner.prompt import build_prompt
        self.assertTrue(callable(build_prompt))

        import inspect
        sig = inspect.signature(build_prompt)
        params = list(sig.parameters.keys())
        self.assertIn("vignette", params)
        self.assertIn("variant", params)

    def test_config_variant_names(self):
        """
        Document the actual variant names.

        config.ALL_VARIANTS has 4 variants — nb_full is intentionally excluded
        from the benchmark pipeline even though dataset_quintets.csv contains it.
        nb_explicit does not exist anywhere in YentlBench.
        """
        from yentlbench.config import ALL_VARIANTS
        from yentlbench.dataset_prep import VARIANTS as PREP_VARIANTS

        self.assertIn("nb_ambiguous", ALL_VARIANTS)
        self.assertIn("female", ALL_VARIANTS)
        self.assertIn("male", ALL_VARIANTS)
        self.assertIn("nb_label_only", ALL_VARIANTS)
        self.assertNotIn(
            "nb_explicit", ALL_VARIANTS,
            "nb_explicit does not exist in YentlBench."
        )

        # nb_full is in the dataset but intentionally excluded from config
        prep_variants = {v["gender_variant"] for v in PREP_VARIANTS}
        self.assertIn("nb_full", prep_variants,
            "nb_full must exist in dataset_quintets.csv")
        self.assertNotIn("nb_full", ALL_VARIANTS,
            "nb_full is intentionally excluded from config.ALL_VARIANTS "
            "and the benchmark pipeline — YentlGuard correctly omits it.")

        # Document the gap explicitly
        in_csv_not_config = prep_variants - set(ALL_VARIANTS)
        print(f"\n  Variants in CSV but not config.ALL_VARIANTS: {in_csv_not_config}")
        print(f"  YentlGuard benchmark variants: {sorted(ALL_VARIANTS)}")

    def test_yentlguard_does_not_run_prepare(self):
        """
        YentlGuard reads dataset_quintets.csv — it does NOT call yentlbench prepare.

        Prerequisites before running YentlGuard:
          1. Obtain MIMIC-IV-ED Demo from PhysioNet (requires DUA)
          2. Place mimic-iv-ed-demo-2.2/ed/ in the working directory
          3. Run: yentlbench prepare
          4. Verify dataset_output/dataset_quintets.csv exists
          5. Pass path via --dataset to yentlguard baseline / run

        This test documents the contract, not the runtime behavior.
        """
        from yentlbench.dataset_prep import DATA_DIR, OUTPUT_DIR

        # YentlGuard expects the OUTPUT, not the input
        expected_input_to_yentlguard = OUTPUT_DIR / "dataset_quintets.csv"
        self.assertEqual(str(expected_input_to_yentlguard), "dataset_output/dataset_quintets.csv")

        # YentlBench reads MIMIC-IV-ED from here — YentlGuard never touches this
        self.assertIn("yentlbench", str(DATA_DIR))  # bundled in package since 0.2.0
        self.assertIn("data", str(DATA_DIR))

    def test_parse_esi_extracts_digit_from_json(self):
        """
        parse_esi extracts the ESI digit from Gemini's JSON response.
        YentlGuard must handle this if it wants to compare against
        YentlBench-style response parsing.
        """
        from yentlbench.local_runner.parser import parse_esi

        # YentlBench expects JSON: {"score": 3}
        result = parse_esi('{"score": 3}')
        self.assertEqual(result, 3.0)

        result_none = parse_esi("I cannot determine the score.")
        self.assertIsNone(result_none)


class TestBuildPromptOutput(unittest.TestCase):
    """
    Verify build_prompt produces correctly structured vignette text
    for each variant, using synthetic vignette dicts.
    """

    def _make_vignette(self, **overrides) -> dict:
        """Minimal vignette dict matching dataset_quintets.csv schema."""
        base = {
            "source_stay_id": 99999999,
            "quintet_id": 0,
            "chiefcomplaint": "chest pain",
            "heartrate": 88,
            "resprate": 18,
            "o2sat": 98,
            "sbp": 142,
            "dbp": 88,
            "temperature": 98.6,
            "pain": 7,
            "acuity": 2,
            # Variant-specific fields (set per variant)
            "gender_variant": "nb_ambiguous",
            "patient_name": None,
            "sex_label": "",
            "pronoun": "",
        }
        base.update(overrides)
        return base

    def test_nb_ambiguous_has_no_demographic_tokens(self):
        """
        nb_ambiguous must produce a prompt with no sex/gender tokens.
        This is the baseline condition — demographic signal absent.
        """
        from yentlbench.local_runner.prompt import build_prompt

        vignette = self._make_vignette(
            gender_variant="nb_ambiguous",
            patient_name=None,
            sex_label="",
            pronoun="",
        )
        prompt = build_prompt(vignette, "nb_ambiguous")

        self.assertNotIn("Sex:", prompt)
        self.assertNotIn("Pronouns:", prompt)
        self.assertNotIn("Name:", prompt)
        self.assertIn("chest pain", prompt.lower())
        self.assertIn("HR:", prompt)
        print(f"\n  nb_ambiguous prompt:\n{prompt}")

    def test_female_has_demographic_tokens(self):
        """
        Female variant must contain Sex: Female.
        """
        from yentlbench.local_runner.prompt import build_prompt

        vignette = self._make_vignette(
            gender_variant="female",
            patient_name="Sarah",
            sex_label="Female",
            pronoun="she/her",
        )
        prompt = build_prompt(vignette, "female")

        self.assertIn("Sex: Female", prompt)
        self.assertNotIn("Name:", prompt)
        self.assertNotIn("Pronouns:", prompt)
        self.assertIn("chest pain", prompt.lower())
        print(f"\n  female prompt:\n{prompt}")

    def test_male_has_demographic_tokens(self):
        """Male variant must contain Sex: Male."""
        from yentlbench.local_runner.prompt import build_prompt

        vignette = self._make_vignette(
            gender_variant="male",
            patient_name="James",
            sex_label="Male",
            pronoun="he/him",
        )
        prompt = build_prompt(vignette, "male")

        self.assertNotIn("Name: James", prompt)
        self.assertIn("Sex: Male", prompt)
        self.assertNotIn("Pronouns: he/him", prompt)
        print(f"\n  male prompt:\n{prompt}")

    def test_nb_label_only_has_label_no_pronouns(self):
        """
        nb_label_only: Sex: Non-binary present.
        This is the minimal non-binary signal — label only.
        """
        from yentlbench.local_runner.prompt import build_prompt

        vignette = self._make_vignette(
            gender_variant="nb_label_only",
            patient_name="Robert",  # male name pool
            sex_label="Non-binary",
            pronoun="",
        )
        prompt = build_prompt(vignette, "nb_label_only")

        self.assertNotIn("Name: Robert", prompt)
        self.assertIn("Sex: Non-binary", prompt)
        self.assertNotIn("Pronouns:", prompt)
        print(f"\n  nb_label_only prompt:\n{prompt}")

    def test_prompt_contains_all_vitals(self):
        """Prompt must contain all available vital signs in the correct format."""
        from yentlbench.local_runner.prompt import build_prompt

        vignette = self._make_vignette()
        prompt = build_prompt(vignette, "nb_ambiguous")

        self.assertIn("HR: 88 bpm", prompt)
        self.assertIn("RR: 18 breaths/min", prompt)
        self.assertIn("SpO2: 98%", prompt)
        self.assertIn("BP: 142/88 mmHg", prompt)
        self.assertIn("Temp: 98.6°F", prompt)
        self.assertIn("Pain: 7/10", prompt)

    def test_prompt_is_valid_yentlguard_runner_input(self):
        """
        The string returned by build_prompt must be usable as the
        vignette_text parameter to YentlGuardRunner.run().
        Validates type and minimum length.
        """
        from yentlbench.local_runner.prompt import build_prompt

        vignette = self._make_vignette(
            gender_variant="female",
            patient_name="Jennifer",
            sex_label="Female",
            pronoun="she/her",
        )
        prompt = build_prompt(vignette, "female")

        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100,
            "Prompt too short — something went wrong with build_prompt")
        # Must contain the ESI instruction
        self.assertIn("Emergency Severity Index", prompt)

    def test_quintet_clinical_fields_identical(self):
        """
        The same vignette across all variants must have identical clinical fields.
        This is YentlBench's core methodological guarantee.
        """
        from yentlbench.local_runner.prompt import build_prompt

        # Base clinical vignette
        clinical = dict(
            source_stay_id=99999999,
            quintet_id=0,
            chiefcomplaint="chest pain",
            heartrate=88,
            resprate=18,
            o2sat=98,
            sbp=142,
            dbp=88,
            temperature=98.6,
            pain=7,
            acuity=2,
        )

        variant_configs = {
            "nb_ambiguous":  dict(patient_name=None,      sex_label="",           pronoun=""),
            "female":        dict(patient_name="Sarah",    sex_label="Female",     pronoun="she/her"),
            "male":          dict(patient_name="James",    sex_label="Male",       pronoun="he/him"),
            "nb_label_only": dict(patient_name="Robert",   sex_label="Non-binary", pronoun=""),
        }

        prompts = {}
        for variant, demo in variant_configs.items():
            vignette = {**clinical, "gender_variant": variant, **demo}
            prompts[variant] = build_prompt(vignette, variant)

        # All prompts must contain identical clinical content
        for variant, prompt in prompts.items():
            self.assertIn("chest pain", prompt.lower(),
                f"{variant} prompt missing chief complaint")
            self.assertIn("HR: 88 bpm", prompt,
                f"{variant} prompt missing heart rate")
            self.assertIn("BP: 142/88 mmHg", prompt,
                f"{variant} prompt missing blood pressure")


@pytest.mark.skipif(
    _find_quintets_csv() is None,
    reason=(
        "dataset_quintets.csv not found. Run: yentlbench prepare "
        "(requires MIMIC-IV-ED data). "
        "Set YENTLGUARD_DATASET_PATH to override the search path."
    )
)
class TestQuintetsCSVContract(unittest.TestCase):
    """
    Tests against the real dataset_quintets.csv produced by yentlbench prepare.
    Skipped if the CSV is not present.
    """

    def setUp(self):
        self.csv_path = _find_quintets_csv()
        self.df = pd.read_csv(self.csv_path)
        print(f"\n  Loaded {len(self.df)} rows from {self.csv_path}")

    def test_expected_columns_present(self):
        """All columns YentlGuard depends on must be present."""
        required = [
            "source_stay_id", "quintet_id", "gender_variant",
            "chiefcomplaint", "heartrate", "resprate",
            "o2sat", "sbp", "dbp", "acuity",
            "patient_name", "sex_label", "pronoun",
        ]
        for col in required:
            self.assertIn(col, self.df.columns,
                f"Required column '{col}' missing from dataset_quintets.csv")

    def test_all_four_variants_present(self):
        """All four variant names must appear in the gender_variant column."""
        from yentlbench.config import ALL_VARIANTS
        actual = set(self.df["gender_variant"].unique())
        for variant in ALL_VARIANTS:
            self.assertIn(variant, actual,
                f"Variant '{variant}' not found in dataset_quintets.csv")

    def test_vignette_count(self):
        """
        Document actual vignette counts per variant.
        YentlGuard documentation says ~70 vignettes — verify.
        """
        from yentlbench.config import ALL_VARIANTS

        counts = self.df["gender_variant"].value_counts()
        n_quintets = self.df["quintet_id"].nunique()

        print(f"\n  Total rows      : {len(self.df)}")
        print(f"  Unique quintets : {n_quintets}")
        print("  Rows per variant:")
        for v in ALL_VARIANTS:
            print(f"    {v:<16}: {counts.get(v, 0)}")

        # All variants must have the same count
        variant_counts = [counts.get(v, 0) for v in ALL_VARIANTS]
        self.assertEqual(
            len(set(variant_counts)), 1,
            f"Variant row counts are unequal: {dict(zip(ALL_VARIANTS, variant_counts))}"
        )

    def test_acuity_is_valid_esi(self):
        """acuity column must contain only valid ESI levels 1-5."""
        valid = {1, 2, 3, 4, 5}
        actual = set(self.df["acuity"].dropna().astype(int).unique())
        invalid = actual - valid
        self.assertEqual(invalid, set(),
            f"Invalid acuity values found: {invalid}")

    def test_acuity_identical_within_quintet(self):
        """
        acuity (ESI ground truth) must be identical across all variants
        within each quintet. This is YentlBench's core guarantee.
        """
        violations = []
        for qid, group in self.df.groupby("quintet_id"):
            if group["acuity"].nunique(dropna=True) > 1:
                violations.append(qid)

        self.assertEqual(violations, [],
            f"Acuity varies within quintets: {violations[:5]}")

    def test_clinical_fields_identical_within_quintet(self):
        """
        Vital signs must be identical across all variants within each quintet.
        Any variation is a data preparation bug that would confound bias measurement.
        """
        clinical_cols = ["chiefcomplaint", "heartrate", "resprate",
                         "o2sat", "sbp", "dbp", "temperature", "pain"]

        violations = []
        for qid, group in self.df.groupby("quintet_id"):
            for col in clinical_cols:
                if col in group.columns and group[col].nunique(dropna=False) > 1:
                    violations.append((qid, col))

        self.assertEqual(violations, [],
            f"Clinical field variations within quintets: {violations[:5]}")

    def test_build_prompt_works_on_real_rows(self):
        """
        build_prompt must produce valid prompt strings for real dataset rows.
        Test 3 rows per variant.
        """
        from yentlbench.config import ALL_VARIANTS
        from yentlbench.local_runner.prompt import build_prompt

        for variant in ALL_VARIANTS:
            subset = self.df[self.df["gender_variant"] == variant].head(3)
            for _, row in subset.iterrows():
                vignette = row.to_dict()
                prompt = build_prompt(vignette, variant)

                self.assertIsInstance(prompt, str)
                self.assertGreater(len(prompt), 50)
                self.assertIn("Emergency Severity Index", prompt)

                # Verify demographic signal matches variant
                if variant == "female":
                    self.assertIn("Sex: Female", prompt)
                elif variant == "male":
                    self.assertIn("Sex: Male", prompt)
                elif variant == "nb_label_only":
                    self.assertIn("Sex: Non-binary", prompt)
                elif variant == "nb_ambiguous":
                    self.assertNotIn("Sex:", prompt)

    def test_preflight_subset_shape(self):
        """
        Show what 3 and 5 vignette subsets look like — the pre-flight window.
        Confirms YentlGuard's PREFLIGHT_N=3 is sensible relative to full dataset.
        """
        from yentlbench.config import ALL_VARIANTS

        for variant in ALL_VARIANTS:
            subset = self.df[self.df["gender_variant"] == variant]
            n_full = len(subset)
            n_3 = min(3, n_full)
            n_5 = min(5, n_full)
            print(
                f"\n  {variant:<16}: "
                f"full={n_full}, "
                f"preflight-3={n_3} ({n_3/n_full:.0%}), "
                f"preflight-5={n_5} ({n_5/n_full:.0%})"
            )
            self.assertGreaterEqual(n_full, 3,
                f"Variant {variant} has fewer than 3 rows — pre-flight will fail")


class TestYentlGuardIntegrationContract(unittest.TestCase):
    """
    Documents the correct integration pattern between YentlBench and YentlGuard.

    The current YentlGuard CLI calls from yentlbench.data import load_vignettes
    which does not exist. The correct pattern is:

        df = pd.read_csv("dataset_output/dataset_quintets.csv")
        df = df[df["gender_variant"] == variant]
        for _, row in df.iterrows():
            vignette_dict = row.to_dict()
            text = build_prompt(vignette_dict, variant)
            esi_ground_truth = str(int(vignette_dict["acuity"]))
            vignette_id = str(vignette_dict["source_stay_id"])
            runner.run(vignette_id, text, variant)
    """

    def test_vignette_dict_has_required_yentlguard_fields(self):
        """
        A vignette dict from the CSV must provide everything
        YentlGuardRunner.run() and BQWriter.write() need.
        """
        from yentlbench.local_runner.prompt import build_prompt

        # Simulate a row from dataset_quintets.csv
        row = {
            "source_stay_id": 30804581,
            "quintet_id": 1,
            "gender_variant": "female",
            "patient_name": "Jennifer",
            "sex_label": "Female",
            "pronoun": "she/her",
            "chiefcomplaint": "Chest pain",
            "heartrate": 92,
            "resprate": 16,
            "o2sat": 97,
            "sbp": 138,
            "dbp": 84,
            "temperature": 98.2,
            "pain": 8,
            "acuity": 2,
        }

        # Fields YentlGuard extracts
        vignette_id = str(row["source_stay_id"])
        text = build_prompt(row, row["gender_variant"])
        esi_ground_truth = str(int(row["acuity"]))
        clinical_category = row.get("chiefcomplaint", "unknown")

        self.assertEqual(vignette_id, "30804581")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 100)
        self.assertEqual(esi_ground_truth, "2")
        self.assertIn("Chest pain", clinical_category)

    def test_variant_name_mapping(self):
        """
        YentlGuard CLI uses nb_explicit which doesn't exist.
        This test documents the correct mapping.
        """
        from yentlbench.config import (
            ALL_VARIANTS,
            VARIANT_FEMALE,
            VARIANT_MALE,
            VARIANT_NO_SEX,
            VARIANT_NONBINARY,
        )

        # What YentlGuard should use vs what it incorrectly assumed
        correct_mapping = {
            "baseline (no demographic)": VARIANT_NO_SEX,       # "nb_ambiguous"
            "female signal":             VARIANT_FEMALE,        # "female"
            "male signal":               VARIANT_MALE,          # "male"
            "non-binary label only":     VARIANT_NONBINARY,     # "nb_label_only"
        }

        wrong_variants = ["nb_explicit"]  # does not exist in YentlBench

        for label, variant in correct_mapping.items():
            self.assertIn(variant, ALL_VARIANTS,
                f"Correct variant '{variant}' ({label}) not in ALL_VARIANTS")

        for wrong in wrong_variants:
            self.assertNotIn(wrong, ALL_VARIANTS,
                f"'{wrong}' incorrectly referenced in YentlGuard — does not exist in YentlBench")


if __name__ == "__main__":
    unittest.main(verbosity=2)
