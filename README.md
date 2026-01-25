# 🤖 AI-Powered Exercise Form Analysis

An intelligent exercise form analysis tool that uses computer vision and AI to provide personalized coaching feedback for weightlifting exercises.

## Features

- 🎥 **Video Upload & Analysis**: Upload exercise videos for automatic form analysis
- 🤖 **AI Pose Detection**: Advanced MediaPipe pose tracking with key joint highlighting
- 📊 **Comprehensive Metrics**: Detailed scoring across multiple form criteria
- 🧠 **AI Coaching Reports**: Personalized feedback powered by GPT-4o-mini
- 📸 **Visual Issue Snapshots**: Automatic frame extraction showing form problems
- 🎯 **Multi-Exercise Support**: Currently supports deadlifts and squats

## How It Works

1. **Upload** your exercise video (MP4 format)
2. **Configure** exercise type and camera angle
3. **Run Analysis** to process through our Bronze-Silver-Gold pipeline:
   - **Bronze**: Pose extraction and overlay generation
   - **Silver**: Rep segmentation and basic metrics
   - **Gold**: Advanced scoring and issue detection
4. **Get Results**: AI coaching report with visual snapshots of form issues

## Deployment

This app can run locally or be deployed to the cloud:

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# For local LLM (optional)
ollama pull llama3.2:3b
ollama serve

# Run the app
streamlit run app/app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path to `app/app.py`
6. Add your OpenAI API key in the secrets section:
   ```
   OPENAI_API_KEY = "your-api-key-here"
   ```
7. Deploy!

## Requirements

- Python 3.10+
- OpenAI API key (for cloud deployment)
- Ollama (optional, for local LLM)

## Architecture

```
pipeline/
├── bronze/     # Raw pose extraction
├── silver/     # Rep segmentation
└── gold/       # Scoring & analysis

app/
└── app.py      # Streamlit web interface
```

## Tech Stack

- **Frontend**: Streamlit
- **Computer Vision**: MediaPipe, OpenCV
- **AI/ML**: OpenAI GPT-4o-mini, Ollama
- **Data Processing**: pandas, NumPy
- **Video Processing**: OpenCV

## Contributing

Feel free to open issues or submit pull requests to improve the analysis algorithms or add support for new exercises!

## License

MIT License - feel free to use and modify as needed.