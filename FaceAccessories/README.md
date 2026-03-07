# Face Accessories Detection

An AI-powered face accessories detection tool using computer vision to identify accessories worn on the face — glasses, sunglasses, face masks, hats, headbands, and earrings.

## Features

- **Multi-Accessory Detection**: Glasses, sunglasses, face masks, hats, headbands, earrings
- **Confidence Scoring**: Per-accessory confidence scores
- **Visual Annotations**: Overlay detected accessories on image with labels
- **Batch Processing**: Analyze multiple images at once
- **Streamlit UI**: Clean web interface for image upload and results

## Supported Accessories

| Accessory    | Detection Method              |
|--------------|-------------------------------|
| Glasses      | Eye-region edge analysis      |
| Sunglasses   | Lens darkness + edge frames   |
| Face Mask    | Lower-face texture uniformity |
| Hat / Cap    | Forehead-top edge boundary    |
| Headband     | Mid-forehead band detection   |
| Earrings     | Ear-lobe high-contrast region |

## How It Works

1. **Upload** an image (JPG, PNG, WEBP)
2. **Detect** 468 facial landmarks via MediaPipe Face Mesh
3. **Analyze** specific facial regions per accessory type
4. **Score** each accessory with a confidence value
5. **Visualize** annotated result + summary table

## Project Structure

```
FaceAccessories/
├── app/
│   └── app.py              # Streamlit web app
├── pipeline/
│   ├── __init__.py
│   ├── detector.py         # MediaPipe face landmark detection
│   ├── classifier.py       # Accessory classification logic
│   └── annotator.py        # Image annotation & result formatting
├── utils/
│   ├── __init__.py
│   └── image_utils.py      # Image loading / preprocessing helpers
├── models/
│   └── __init__.py
├── data/
│   └── samples/            # Sample images for testing
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## Deployment on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set main file path to `app/app.py`
5. Deploy — no API keys required!

## Requirements

- Python 3.10+
- Webcam or image files (JPG, PNG, WEBP, MP4)

## Tech Stack

| Layer            | Technology                |
|------------------|---------------------------|
| Frontend         | Streamlit                 |
| Face Detection   | MediaPipe Face Mesh       |
| Computer Vision  | OpenCV                    |
| Image Processing | Pillow, NumPy             |
| Data Display     | pandas                    |

## License

MIT License
