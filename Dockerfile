FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts

RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[cloud,data,decode,metrics,tracking]"

ENTRYPOINT ["python", "-m", "monet_flow.train"]
