"""
Shared text-splitting helpers used by eval.py and test scenarios
"""
import re
from pypdf import PdfReader
from utils.helpers.logger import get_logger, timed

logger = get_logger("text_split")

LIST_MARKER_PATTERN = re.compile(r'(?:^|\s)(?:[•◦‣▪*-]|\d{1,3}[.)])\s+')

def split_into_cases(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines

    markers = [m.strip() for m in LIST_MARKER_PATTERN.split(text) if m.strip()]
    if len(markers) > 1:
        return markers

    text = text.strip()
    return [text] if text else []

def read_file(filename):
    try:
        with timed(logger, f"Read source file '{filename}'"):
            if filename.endswith(".txt"):
                with open(filename, "r") as r:
                    lines = [line for line in r.readlines() if line.strip()]
            else:
                reader = PdfReader(filename)
                lines = []
                for page in reader.pages:
                    lines = split_into_cases(page.extract_text())
                    break
    except Exception:
        logger.exception(f"Failed to read source file '{filename}'")
        raise

    logger.debug(f"Read {len(lines)} case(s) from '{filename}'")
    return lines
