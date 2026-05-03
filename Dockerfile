FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[prod]"

COPY app/ ./app/

RUN mkdir -p logs

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
