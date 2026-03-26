import argparse
import asyncio

from app.core.config import logger
from app.schemas.chat import ChatTurn
from app.services.graph_service import run_self_rag


async def cli_main() -> None:
    logger.info("=" * 80)
    logger.info("Self-RAG CLI started")
    logger.info("Type 'quit' or 'exit' to stop")
    logger.info("=" * 80)

    history: list[ChatTurn] = []

    while True:
        user_question = input("\nYou: ").strip()
        if user_question.lower() in ["quit", "exit", "종료"]:
            logger.info("CLI session ended by user")
            break
        if not user_question:
            continue

        try:
            result = await asyncio.to_thread(run_self_rag, user_question, history)
            answer = result.get("generation", "")
            history.append({"role": "user", "content": user_question})
            history.append({"role": "assistant", "content": answer})
            logger.info("CLI answer   | answer_chars=%d | history_turns=%d", len(answer), len(history))
            print(f"\nAI: {answer}")
            print("-" * 80)
        except Exception as exc:
            logger.exception("CLI runtime error")
            print(f"\n[Runtime Error] {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-RAG demo for Python 3.11")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()
