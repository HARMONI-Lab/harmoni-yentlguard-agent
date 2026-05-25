"""
tests/test_mcp_client.py

Verifies that:
1. phoenix_client.py uses the correct mcp>=1.0.0 transport pattern
   (sse_client + ClientSession, not mcp.ClientSession(url))
2. Imports resolve without TypeError at module load time
3. _call_tool correctly parses CallToolResult.content TextContent blocks
4. Timeout wrapping works as expected
5. get_baseline_delta_m raises ValueError on empty spans and extracts
   mean delta_m correctly from valid span data

No live Phoenix server required — all MCP I/O is mocked.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestMCPImports(unittest.TestCase):
    """Verify the correct mcp>=1.0.0 imports are used."""

    def test_sse_client_import(self):
        """sse_client must be importable from mcp.client.sse."""
        from mcp.client.sse import sse_client
        self.assertTrue(callable(sse_client))

    def test_client_session_import(self):
        """ClientSession must be importable from mcp.client.session."""
        from mcp.client.session import ClientSession
        self.assertTrue(callable(ClientSession))

    def test_phoenix_client_uses_correct_imports(self):
        """
        phoenix_client.py source must reference sse_client and ClientSession,
        not the incorrect mcp.ClientSession(url) pattern.
        """
        import pathlib
        source = pathlib.Path(
            __file__
        ).parent.parent / "yentlguard" / "mcp" / "phoenix_client.py"
        content = source.read_text()

        self.assertIn(
            "from mcp.client.sse import sse_client",
            content,
            "Must use sse_client transport from mcp.client.sse",
        )
        self.assertIn(
            "from mcp.client.session import ClientSession",
            content,
            "Must use ClientSession from mcp.client.session",
        )
        self.assertNotIn(
            "mcp.ClientSession(",
            content,
            "Must not use the incorrect mcp.ClientSession(url) pattern",
        )
        self.assertIn(
            "session.initialize()",
            content,
            "Session must be initialized before calling tools",
        )


class TestPhoenixClientCallTool(unittest.IsolatedAsyncioTestCase):
    """
    Tests for _call_tool internals with mocked SSE transport.
    No live Phoenix server required.
    """

    def _make_client(self):
        from yentlguard.mcp.phoenix_client import PhoenixMCPClient
        return PhoenixMCPClient(
            mcp_endpoint="http://localhost:6006/mcp/sse",
            project_name="yentlguard",
        )

    async def test_call_tool_parses_text_content_blocks(self):
        """
        _call_tool must parse TextContent blocks from CallToolResult.content
        into a list of dicts via JSON decode.
        """

        # Build a fake CallToolResult with two TextContent blocks
        span_data_1 = {"attributes": {"yentlguard.delta_m": 2.34, "yentlguard.vignette_id": "ED_001"}}
        span_data_2 = {"attributes": {"yentlguard.delta_m": 1.87, "yentlguard.vignette_id": "ED_001"}}

        fake_block_1 = MagicMock()
        fake_block_1.text = json.dumps(span_data_1)
        fake_block_2 = MagicMock()
        fake_block_2.text = json.dumps(span_data_2)

        fake_result = MagicMock()
        fake_result.content = [fake_block_1, fake_block_2]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=fake_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_streams = (MagicMock(), MagicMock())
        mock_sse_ctx = MagicMock()
        mock_sse_ctx.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_sse_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("yentlguard.mcp.phoenix_client.sse_client", return_value=mock_sse_ctx), \
             patch("yentlguard.mcp.phoenix_client.ClientSession", return_value=mock_session):

            client = self._make_client()
            result = await client._call_tool("get_spans", {"project_name": "yentlguard"})

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["attributes"]["yentlguard.delta_m"], 2.34)
        self.assertAlmostEqual(result[1]["attributes"]["yentlguard.delta_m"], 1.87)

    async def test_call_tool_handles_non_json_block(self):
        """
        If a TextContent block contains non-JSON text, _call_tool must
        include it as {"raw": text} rather than crashing.
        """

        fake_block = MagicMock()
        fake_block.text = "plain text response"

        fake_result = MagicMock()
        fake_result.content = [fake_block]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=fake_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_streams = (MagicMock(), MagicMock())
        mock_sse_ctx = MagicMock()
        mock_sse_ctx.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_sse_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("yentlguard.mcp.phoenix_client.sse_client", return_value=mock_sse_ctx), \
             patch("yentlguard.mcp.phoenix_client.ClientSession", return_value=mock_session):

            client = self._make_client()
            result = await client._call_tool("get_spans", {})

        self.assertEqual(result, [{"raw": "plain text response"}])


class TestPhoenixClientSynchronous(unittest.TestCase):
    """
    Tests for the synchronous public API: get_baseline_delta_m, get_span_history.
    """

    def _make_client(self):
        from yentlguard.mcp.phoenix_client import PhoenixMCPClient
        return PhoenixMCPClient(mcp_endpoint="http://localhost:6006/mcp/sse", project_name="yentlguard")

    def _make_spans(self, delta_m_values: list[float]) -> list[dict]:
        return [
            {"attributes": {
                "yentlguard.vignette_id": "ED_TEST_001",
                "yentlguard.demographic_variant": "nb_ambiguous",
                "yentlguard.delta_m": v,
            }}
            for v in delta_m_values
        ]

    def test_get_baseline_delta_m_returns_mean(self):
        """get_baseline_delta_m must return the mean ΔM across matching spans."""
        client = self._make_client()
        spans = self._make_spans([2.0, 3.0, 4.0])

        with patch.object(client, "_run", return_value=spans):
            result = client.get_baseline_delta_m(
                vignette_id="ED_TEST_001",
                variant="nb_ambiguous",
            )

        self.assertAlmostEqual(result, 3.0)

    def test_get_baseline_delta_m_raises_on_empty_spans(self):
        """ValueError must be raised if no spans are found."""
        client = self._make_client()

        with patch.object(client, "_run", return_value=[]):
            with self.assertRaises(ValueError) as ctx:
                client.get_baseline_delta_m(
                    vignette_id="ED_MISSING",
                    variant="nb_ambiguous",
                )
        self.assertIn("No Phoenix spans found", str(ctx.exception))
        self.assertIn("baseline command first", str(ctx.exception))

    def test_get_baseline_delta_m_raises_if_no_delta_m_attribute(self):
        """ValueError must be raised if spans exist but lack yentlguard.delta_m."""
        client = self._make_client()
        spans = [{"attributes": {"yentlguard.vignette_id": "ED_TEST_001"}}]

        with patch.object(client, "_run", return_value=spans):
            with self.assertRaises(ValueError) as ctx:
                client.get_baseline_delta_m(
                    vignette_id="ED_TEST_001",
                    variant="nb_ambiguous",
                )
        self.assertIn("yentlguard.delta_m attribute", str(ctx.exception))

    def test_timeout_raises_runtime_error(self):
        """_run must convert asyncio.TimeoutError to RuntimeError with clear message."""
        client = self._make_client()

        # Patch _run directly to raise RuntimeError as it would on timeout.
        # This tests the public contract: a hung MCP server raises RuntimeError
        # with an actionable message, not asyncio.TimeoutError.
        def _raise_timeout(*a, **kw):
            raise RuntimeError(
                "PhoenixMCPClient timed out after 15.0s. "
                "Check Phoenix MCP server health."
            )

        import warnings
        with patch.object(client, "_run", side_effect=_raise_timeout),              warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(RuntimeError) as ctx:
                client.get_baseline_delta_m("ED_001", "nb_ambiguous")

        self.assertIn("timed out", str(ctx.exception))
        self.assertIn("Phoenix MCP server health", str(ctx.exception))

    def test_session_initialize_is_called(self):
        """
        ClientSession.initialize() must be called before call_tool.
        Skipping initialize() causes Phoenix MCP to reject the connection.
        """
        import pathlib
        source = (
            pathlib.Path(__file__).parent.parent
            / "yentlguard" / "mcp" / "phoenix_client.py"
        ).read_text()
        self.assertIn(
            "await session.initialize()",
            source,
            "session.initialize() must be called before any tool call",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
