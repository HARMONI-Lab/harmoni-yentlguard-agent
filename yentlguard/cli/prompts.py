import argparse
import logging

logger = logging.getLogger("yentlguard.cli")


def cmd_prompts(args: argparse.Namespace) -> None:
    from yentlguard.mcp.phoenix_manager import PhoenixPromptManager

    logger.info("Pushing default YentlGuard prompts to Phoenix...")
    mgr = PhoenixPromptManager()
    mgr.push_all_defaults()
    logger.info("Done seeding prompts.")
