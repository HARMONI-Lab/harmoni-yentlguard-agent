import argparse
import asyncio
import logging
import secrets
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("yentlguard.cli")


def cmd_agent(args: argparse.Namespace) -> None:
    if args.query:
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        from yentlguard.agent.yentlguard_agent.agent import root_agent

        async def _run_single_turn(query: str) -> None:
            runner = InMemoryRunner(agent=root_agent, app_name="yentlguard")
            session_id = secrets.token_hex(8)
            await runner.session_service.create_session(
                app_name="yentlguard",
                user_id="cli_user",
                session_id=session_id,
            )
            async for event in runner.run_async(
                user_id="cli_user",
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=query)]),
            ):
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            print(part.text, end="", flush=True)
            print()

        asyncio.run(_run_single_turn(args.query))
    else:
        agent_dir = str((Path(__file__).parent.parent / "agent" / "yentlguard_agent").resolve())
        logger.info("Launching adk web → %s", agent_dir)
        result = subprocess.run(
            [sys.executable, "-m", "google.adk.cli", "web", agent_dir],
            check=False,
        )
        sys.exit(result.returncode)
