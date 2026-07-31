# Hugging Face Spaces (Docker SDK). Serves the FastAPI app on port 7860.
FROM python:3.11-slim

# HF Spaces runs as a non-root user; set a writable cache for model downloads.
ENV HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only torch first to avoid pulling CUDA wheels on the free tier.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The LoRA adapter in models/plainmed-lora is loaded at startup. If you host
# the adapter on the HF Hub instead, set an env var and load from there.

EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
