# Multimodal Technical/Macro Analysis


## Overview

This project uses the Gemma 3 multimodal model to analyze financial data from stock charts and economic indicators. It also uses the Hugging Face SFTTrainer to fine-tune the model with user generated labeled data.

![alt text](https://github.com/iamez88/multimodal-analysis/blob/master/screenshot/multimodal_streamlit_1.png "1")
![alt text](https://github.com/iamez88/multimodal-analysis/blob/master/screenshot/multimodal_streamlit_2.png "2")

## Features

- Retrieval of macro data from Fred and stock charts with technical indicators
- Streamlit interface for interactive model predictions and delivery of final recommendation 
- Image-based technical analysis combined with numerical macro indicators as input to Gemma
- Structured output with action recommendations, confidence scores, and price targets
- User generated labeled data using Hugging Face Dataset to format instruction/input/output pairs
- Fine-tuning Gemma using Hugging Face SFTTrainer (LoRA with custom configuration)

## Project Structure

```
.
├── src/
│   ├── fine_tune_gemma.py  # Fine-tuning implementation for Gemma 3
│   ├── model.py            # Model definition and inference logic
│   └── app.py              # Streamlit interface for the application, core functions to 
├── fine_tuning_data/
│   └── create_labeled_data.py  # Generates training data for fine-tuning
├── oai_formatted_dataset.json  # Generated dataset in OAI format
├── requirements.txt        # Project dependencies
└── README.md               # This documentation
```

## Tech Stack
- **Gemma 3**: Google's multimodal LLM that processes text and images
- **Hugging Face Libraries**:
  - Transformers: For model loading and inference pipeline
  - PEFT: Parameter-efficient fine-tuning with LoRA
  - TRL: SFTTrainer for fine-tuning implementation
- **PyTorch**: Deep learning framework with CUDA acceleration
- **Streamlit**: Web application framework for the user interface
- **YFinance**: Stock data retrieval from Yahoo Finance
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive financial charts and visualizations
- **TA (Technical Analysis)**: Library for financial indicators
- **PIL (Pillow)**: Image processing for model inputs

## Future Enhancements
- Currently, the generation of labeled training data is too difficult. Hard to find clean, quality technical chart patterns that directly represent the signal. Might use LLM to synthetically generate data. 
- Capability to select macroeconomic indicators on front-end.
- Evaluation of post vs. pre fine-tuning.


## License
MIT 