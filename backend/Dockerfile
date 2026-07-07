# NodeSense backend for Hugging Face Spaces
# Spaces requires the app to listen on port 7860

FROM python:3.11-slim

WORKDIR /code

# Model caches must write to /tmp on Spaces (read only filesystem elsewhere)
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
