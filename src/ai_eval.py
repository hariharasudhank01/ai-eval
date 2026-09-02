"""
This Script will be the entry point for this project
"""
import argparse
from utils import eval, validater
import sys

def argparser():
    parser = argparse.ArgumentParser(
        description="AI Eval - Tool to validate your AI Model"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Provide the local ollama model to be tested",
        default="llama3.2:3b"
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
        help="Path to store output file",
        required=True
    )
    
    return parser.parse_args()


def main():
    user_input = argparser()
    if not validater.validate(user_input):
        sys.exit(0)
    eval.run(user_input)
    
    

if __name__ == "__main__":
    main()