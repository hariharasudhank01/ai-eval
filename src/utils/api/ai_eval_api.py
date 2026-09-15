import os
import shutil
import uuid
from types import SimpleNamespace
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from utils import eval
from utils.helpers import validater
from utils.helpers.logger import get_logger, run_log_path, LOG_DIR

logger = get_logger("api")

app = FastAPI(title="AI Eval API")

# Uploaded source files and generated output both live under the project's own
# output/ dir - never a client-supplied path. Avoids a network caller being able
# to point -f/-o at an arbitrary filesystem location.
UPLOAD_DIR = os.path.join(LOG_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_SOURCE_EXTENSIONS = (".txt", ".pdf")

class RunResponse(BaseModel):
    gen_id: int

@app.get("/")
def root():
    return {
        "message": "Welcome to AI Eval"
    }

@app.post("/runs", response_model=RunResponse)
def create_run(
    prompt: str = Form(...),
    model: str = Form("llama3.2:3b"),
    provider: str = Form("ollama"),
    iteration: int = Form(...),
    source_file: UploadFile = File(...),
):
    """
    Runs the full AI Eval pipeline and blocks until it completes. Returns
    gen_id only if generation itself succeeded - a failure (validation or
    generation) returns an error response instead, never a gen_id.

    Note: this is a synchronous def, not async def - FastAPI runs it in a
    worker thread automatically, since eval.run() is a long-running blocking
    call (DB, file I/O, model calls) with no async support of its own.
    """
    _, ext = os.path.splitext(source_file.filename or "")
    if ext.lower() not in ALLOWED_SOURCE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_SOURCE_EXTENSIONS)} source files are supported"
        )

    saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext.lower()}")
    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(source_file.file, f)
    except Exception:
        logger.exception(f"Failed to save uploaded source file to '{saved_path}'")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    finally:
        source_file.file.close()

    user_input = SimpleNamespace(
        prompt=prompt,
        model=model,
        provider=provider,
        iteration=iteration,
        filename=saved_path,
        output=LOG_DIR + os.sep,
    )

    if not validater.validate(user_input):
        raise HTTPException(status_code=400, detail="Input validation failed - check server logs for details")

    try:
        gen_id = eval.run(user_input)
    except Exception:
        logger.exception("Run failed")
        raise HTTPException(status_code=500, detail="Run failed - check server logs for details")

    return RunResponse(gen_id=gen_id)

@app.get("/runs/{gen_id}/logs", response_class=PlainTextResponse)
def get_run_logs(gen_id: int):
    path = run_log_path(gen_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No logs found for gen_id {gen_id}")

    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        logger.exception(f"Failed to read log file '{path}'")
        raise HTTPException(status_code=500, detail="Failed to read log file")
