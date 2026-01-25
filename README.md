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
6. Add one of these API keys in the secrets section:

   **Free Options:**
   ```
   HF_API_KEY = "your-huggingface-api-key"
   # OR
   TOGETHER_API_KEY = "your-together-api-key"
   ```

   **Paid Option:**
   ```
   OPENAI_API_KEY = "your-openai-api-key"
   ```

7. Deploy!

### Getting Free API Keys

#### Hugging Face (Recommended Free Option)
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for free account
3. Go to Settings → Access Tokens
4. Create new token with "Read" permissions
5. Use token as `HF_API_KEY`

#### Together AI (Alternative Free Option)
1. Go to [together.ai](https://together.ai)
2. Sign up for free account
3. Get API key from dashboard
4. Use as `TOGETHER_API_KEY`

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
- **AI/ML**: Multiple LLM providers (Ollama, OpenAI, Hugging Face, Together AI)
- **Data Processing**: pandas, NumPy
- **Video Processing**: OpenCV

## LLM Options

Choose from multiple AI providers:

### Free Options
- **Ollama** (local): Completely free, requires local installation
- **Hugging Face** (cloud): Free tier with generous limits
- **Together AI** (cloud): Free tier available

### Paid Option
- **OpenAI** (cloud): Very cheap (~$0.002 per analysis)

## Environment Variables

Set one of these API keys for cloud deployment:

```bash
# For OpenAI (paid but cheap)
OPENAI_API_KEY=your-openai-key

# For Hugging Face (free)
HF_API_KEY=your-huggingface-key

# For Together AI (free)
TOGETHER_API_KEY=your-together-key

# Or run locally with Ollama (no API key needed)
```

## Contributing

Feel free to open issues or submit pull requests to improve the analysis algorithms or add support for new exercises!

## License

MIT License - feel free to use and modify as needed.