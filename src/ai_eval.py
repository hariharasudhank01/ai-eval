"""
This Script will be the entry point for this project
"""
import argparse
import os
from utils import eval
from utils.helpers.logger import get_logger, timed, LOG_DIR
import sys

from utils.helpers import validater

logger = get_logger("ai_eval")

def argparser():
    parser = argparse.ArgumentParser(
        description="AI Eval - Tool to validate your AI Model"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Provide the model to be tested (model name as understood by the selected provider)",
        default="llama3.2:3b"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai", "anthropic"],
        help="Which provider hosts the model under test",
        default="ollama"
    )

    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Provide the user prompt for the model. Instruction on how you expect the model to work.",
        required=True
    )

    parser.add_argument(
        "-f", "--filename",
        type=str,
        help="Path to the input target file",
        required=True
    )

    parser.add_argument(
        "-n", "--iteration",
        type=int,
        help="Input the number of iteration the model should run",
        required=True
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to store output file (default: the project's output/ dir, same as the API uses)",
        default=LOG_DIR + os.sep
    )
    
    return parser.parse_args()


def main():
    user_input = argparser()
    if not validater.validate(user_input):
        logger.error("Input validation failed - aborting.")
        sys.exit(0)

    try:
        with timed(logger, "Full AI Eval run"):
            gen_id = eval.run(user_input)
    except Exception:
        logger.exception("AI Eval run failed")
        sys.exit(1)

    logger.info(f"Run succeeded - gen_id={gen_id}")


if __name__ == "__main__":
    main()