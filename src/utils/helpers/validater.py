"""
This Script is used to validate the user input via the argparser
"""
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import os
import subprocess
from utils.helpers.logger import get_logger, timed

logger = get_logger("validater")

def is_valid_txt(filename):
    if not os.path.exists(filename):
        logger.error(f"File {filename} not found")
        return False
    try:
        with open(filename, "r") as r:
            content = r.readlines()
    except Exception:
        logger.exception(f"Failed to read file {filename}")
        return False
    if not len(content) > 0:
        logger.error(f"Invalid File, number of pages is less than minimum number {len(content)}")
        return False
    return True

def is_valid_pdf(filename):
    try:
        reader = PdfReader(filename)
        if len(reader.pages) <= 0:
            logger.error(f"Invalid PDF, number of pages is less than minimum number {len(reader.pages)}")
            return False
        for page in reader.pages:
            if "/images" in page.images or len(page.images) > 0:
                logger.error("PDF contains Image. AI Eval v1 supports only Text")
                return False
        return True
    except PdfReadError as e:
        logger.error(f"Error reading PDF: {e}")
        return False

def validate_file(filename):
    if filename.endswith(".txt"):
        return is_valid_txt(filename)
    elif filename.endswith(".pdf"):
        return is_valid_pdf(filename)
    else:
        logger.error("Invalid file, supports only .txt or .pdf")
        return False

def validate_iteration(n):
    if not n > 0:
        logger.error(f"Invalid Iteration {n}, minimum value is 1")
        return False
    return True

PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

def validate_ollama_model(m):
    try:
        with timed(logger, "ollama list"):
            result = subprocess.run([
                "ollama",
                "list"
            ], capture_output=True, text=True)
    except Exception:
        logger.exception("Failed to run 'ollama list' - is Ollama installed and on PATH?")
        return False

    output = result.stdout.strip().split("\n")[1:]
    available_local_models = {model.split()[0] for model in output}
    if m not in available_local_models:
        logger.error(f"{m} not found in available local Ollama models. Run 'ollama pull {m}'")
        return False
    return True

def validate_model(m, provider="ollama"):
    if provider == "ollama":
        return validate_ollama_model(m)

    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if env_var is None:
        logger.error(f"Unknown provider '{provider}'")
        return False
    if not os.getenv(env_var):
        logger.error(f"{env_var} is not set - required to use provider '{provider}'")
        return False
    return True

def validate_prompt(p):
    if not len(p) > 0:
        logger.error("Invalid Prompt")
        return False
    return True

def validate(user_input):
    if validate_file(user_input.filename) and validate_iteration(user_input.iteration) and validate_model(user_input.model, user_input.provider) and validate_prompt(user_input.prompt):
        return True
    return False
