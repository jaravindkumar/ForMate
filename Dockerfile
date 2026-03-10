FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN curl -L --retry 3 \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
    -o /app/pose_landmarker_lite.task

RUN mkdir -p /app/pipeline/bronze /app/pipeline/silver /app/pipeline/gold \
 && chown -R 1000:1000 /app

USER 1000
EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=500", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.fileWatcherType=none"]
