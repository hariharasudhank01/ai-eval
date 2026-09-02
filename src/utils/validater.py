"""
This Script is used to validate the user input via the argparser
"""
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import os
import ollama
import subprocess

def is_valid_txt(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return False
    with open(filename, "r") as r:
        content = r.readlines()
    if not len(content) > 0:
        print(f"Invalid File, number of pages is less than minimum number {len(content)}")
        return False
    return True

def is_valid_pdf(filename):
    try:
        reader = PdfReader(filename)
        if len(reader.pages) <= 0:
            print(f"Invalid PDF, number of pages is less than minimum number {len(reader.pages)}")
            return False
        for page in reader.pages:
            if "/images" in page.images or len(page.images) > 0:
                print("PDF contains Image. AI Eval v1 supports only Text")
                return False
        return True
    except PdfReadError as e:
        print(f"Error: {e}")
        return False

def validate_file(filename):
    if filename.endswith(".txt"):
        return is_valid_txt(filename)
    elif filename.endswith(".pdf"):
        return is_valid_pdf(filename)
    else:
        print("Invalid file, supports only .txt or .pdf")
        return False

def validate_iteration(n):
    if not n > 0:
        print(f"Invalid Ieration {n}, minimum value is 1")
        return False
    return True

def validate_model(m):
    # This Section Needs to be updated based on model support - Right now only ollama models supported
    result = subprocess.run([
        "ollama",
        "list"
    ], capture_output=True, text=True)
    output = result.stdout.strip().split("\n")[1:]
    available_local_models = {model.split()[0] for model in output}
    if m not in available_local_models:
        print(f"{m} not found in available local models. Run Ollama pull x")
        return False
    return True

def validate_prompt(p):
    if not len(p) > 0:
        print(f"Invalid Prompt")
        return False
    return True

def validate(user_input):
    if validate_file(user_input.filename) and validate_iteration(user_input.iteration) and validate_model(user_input.model) and validate_prompt(user_input.prompt):
        return True
    return False