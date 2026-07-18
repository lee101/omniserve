FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

COPY . /src
WORKDIR /src
RUN uv pip install --system --no-cache --no-deps .

ENV COG_PREDICT_TYPE_STUB=predict.py:Predictor
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
