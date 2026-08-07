# NodeSense backend for Hugging Face Spaces
# Spaces requires the app to listen on port 7860

FROM python:3.11.10-slim-bookworm

WORKDIR /code

# Model caches must write to /tmp on Spaces (read only filesystem elsewhere)
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run as a non-root user — the process only ever reads artifacts/ and
# writes to /tmp, neither of which needs root.
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 7860

# python -c avoids adding curl to the image just for the healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=4)" || exit 1

# 2 workers: light concurrency headroom on free-tier CPU without each
# request queuing behind SHAP's ~120ms explain path. Each worker loads
# its own copy of the ONNX model and keeps its own rate-limit/connection
# counters — fine for a single-instance demo, not for scaling out.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
