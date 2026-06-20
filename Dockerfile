# Dockerfile — a recipe. "Start with Python, install these packages, copy these files, run this command."


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# Layer 1 — torch separately, cached independently
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Layer 2 — everything else, retried independently if it fails
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    -r requirements-docker.txt

COPY fast_api.py ./
COPY model.py ./
COPY predict.py ./
COPY gradcam_utils.py ./
COPY llm_utils.py ./
COPY __init__.py ./
COPY utils/ ./utils/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Image — the result of following that recipe. A frozen snapshot.
# Container — a running instance of that image. Like double-clicking the image to bring it to life.


# ## Running with Docker
# 
# 1. Install Docker: https://docs.docker.com/get-docker/
# 2. Clone this repository
# 3. Create a `.env` file in the project root with:
#    OPENAI_API_KEY=your_key_here
# 4. Run:
#    docker compose up --build
# 5. Visit http://localhost:3000 for the UI
# 6. Backend API available at http://localhost:8000/health