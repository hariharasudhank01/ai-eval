"""
Shared logging configuration for AI Eval.
"""
import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_FILE = os.path.join(LOG_DIR, "ai_eval.log")

_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_configured = False

def _configure_root():
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger("ai_eval")
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    root_logger.addHandler(file_handler)

    _configured = True

def get_logger(name):
    _configure_root()
    return logging.getLogger(f"ai_eval.{name}")

@contextmanager
def timed(logger, label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug(f"{label} took {elapsed:.2f}s")

class _BufferingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))

def run_log_path(gen_id):
    return os.path.join(LOG_DIR, f"ai_eval_{gen_id}.log")

class RunLogCapture:
    """
    Captures every log line produced during one run into its own file, so a
    frontend can fetch/tail "the logs for gen_id N" in isolation, without
    parsing the shared ai_eval.log.

    gen_id isn't known until generation has already started, so this buffers
    in memory first; once attach_gen_id(gen_id) is called, the buffer is
    flushed to output/ai_eval_{gen_id}.log and further log lines are written
    directly to that file. This is additive - the existing console + shared
    rotating file handlers keep working unchanged.
    """
    def __init__(self):
        _configure_root()
        self._root = logging.getLogger("ai_eval")
        self._buffer_handler = _BufferingHandler()
        self._buffer_handler.setLevel(logging.DEBUG)
        self._buffer_handler.setFormatter(_FORMATTER)
        self._file_handler = None
        self._root.addHandler(self._buffer_handler)

    def attach_gen_id(self, gen_id):
        path = run_log_path(gen_id)
        with open(path, "w") as f:
            for line in self._buffer_handler.lines:
                f.write(line + "\n")

        self._file_handler = logging.FileHandler(path)
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(_FORMATTER)

        self._root.removeHandler(self._buffer_handler)
        self._root.addHandler(self._file_handler)
        return path

    def close(self):
        if self._file_handler is not None:
            self._root.removeHandler(self._file_handler)
            self._file_handler.close()
        else:
            self._root.removeHandler(self._buffer_handler)
