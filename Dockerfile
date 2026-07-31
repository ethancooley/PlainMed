# Hugging Face Spaces (Docker SDK). Serves the FastAPI app on port 7860.
FROM python:3.11-slim

# Install CPU-only torch first to avoid pulling CUDA wheels on the free tier.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# HF Spaces runs the container as uid 1000. Create that user and give it a
# writable home so the model cache and any runtime writes succeed.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app
COPY --chown=user . .

# The LoRA adapter in models/plainmed-lora is loaded at startup. If you host
# the adapter on the HF Hub instead, load it from there in scripts/model.py.

EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
