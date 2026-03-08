#!/bin/bash
# Pre-download MediaPipe pose model at build time
MODEL="pose_landmarker_lite.task"
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
if [ ! -f "$MODEL" ]; then
    echo "Downloading $MODEL..."
    curl -L --retry 3 -o "$MODEL" "$URL" && echo "OK: $(du -h $MODEL)" || echo "WARN: download failed, will retry at runtime"
else
    echo "Model cached: $(du -h $MODEL)"
fi
