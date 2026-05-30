import unittest
from unittest.mock import MagicMock, patch

from yentlguard.eval.agent_builder import AgentBuilderEvalLayer


class TestAgentBuilderEvalLayer(unittest.TestCase):
    """Tests for AgentBuilderEvalLayer."""

    @patch("yentlguard.eval.agent_builder.bigquery")
    def test_templated_runs_table_is_formatted(self, mock_bigquery):
        """
        Verifies AgentBuilderEvalLayer.__init__ correctly formats a templated RUNS_TABLE.
        """
        with (
            patch("yentlguard.eval.agent_builder.GCP_PROJECT_ID", "test-project"),
            patch("yentlguard.eval.agent_builder.BQ_DATASET_ID", "test-dataset"),
            patch("yentlguard.eval.agent_builder.RUNS_TABLE", "`{project}.{dataset}.runs`"),
        ):
            # Mock the BigQuery client so we don't need ADC
            mock_bigquery.Client.return_value = MagicMock()

            # Instantiate the layer
            layer = AgentBuilderEvalLayer()

            # Verify the table name was formatted correctly
            self.assertEqual(layer._runs_table, "`test-project.test-dataset.runs`")


if __name__ == "__main__":
    unittest.main()
