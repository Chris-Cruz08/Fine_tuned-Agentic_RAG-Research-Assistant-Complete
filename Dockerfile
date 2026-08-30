FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]