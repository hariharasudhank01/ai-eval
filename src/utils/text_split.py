"""
Shared text-splitting helpers used by eval.py and test scenarios
"""
import re
from pypdf import PdfReader

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
    if filename.endswith(".txt"):
        with open(filename, "r") as r:
            return r.readlines()
    else:
        reader = PdfReader(filename)
        for page in reader.pages:
            return split_into_cases(page.extract_text())
