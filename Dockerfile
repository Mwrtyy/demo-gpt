FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY data ./data

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

RUN mkdir -p /app/state /app/runtime

EXPOSE 8000

CMD ["sh", "-c", "uvicorn second_brain.web:app --host 0.0.0.0 --port ${PORT}"]
