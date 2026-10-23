# Use a lightweight official Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

# Install build tools and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as non-root UID 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python dependencies first for optimal Docker layer caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache Hugging Face transformer models into image layers
# This prevents model download timeouts on container boot
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('BAAI/bge-reranker-base')"

# Copy the application source code and data assets
COPY --chown=user:user . .

# Run the offline indexing step during image build so indices exist inside storage/
RUN python scripts/indexer.py

# Switch to non-root user
USER user

# Hugging Face Spaces routes ingress traffic to port 7860
EXPOSE 7860

# Add container health check targeting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI service via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]