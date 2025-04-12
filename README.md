# Multimodal Analysis

A multimodal analysis tool that uses AI to analyze stock chart images and provide technical analysis.

## Features

- Image-based technical analysis using AI models
- Technical indicator integration
- Structured output with action recommendations, confidence scores, and price targets

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/iamez88/multimodal-analysis.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file

## Usage

The main functionality is in the `src/model.py` file, which provides:

- `initialize_model()`: Sets up the image-text-to-text pipeline
- `generate_technical_analysis()`: Processes chart images and returns structured analysis

## License

MIT 