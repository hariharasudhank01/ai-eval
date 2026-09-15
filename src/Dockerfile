FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preserve the src/ nesting exactly as it exists locally - utils/logger.py derives
# the project root (and therefore output/) by walking up from its own file path,
# so the container's directory layout needs to match the local one.
COPY src/ ./src/
WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "utils.api.ai_eval_api:app", "--host", "0.0.0.0", "--port", "8000"]
