import os
import re
import json
import tempfile
import torch
from transformers import pipeline

def initialize_model():
    """
    Initialize and return the image-text-to-text pipeline model.
    
    Returns:
        The initialized pipeline object
    """
    # Initialize the model only once
    pipe = pipeline(  
        "image-text-to-text",  
        model="google/gemma-3-4b-it",  
        device="cuda",  # Use GPU acceleration
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        use_fast=True
    )
    return pipe

def generate_technical_analysis(img_bytes, tkr, timeframe, selected_indicators_code):
    """
    Analyze a stock chart image using AI model and return structured analysis.
    
    Parameters:
        img_bytes (bytes): The chart image as bytes
        tkr (str): The ticker symbol
        timeframe (str): The timeframe of the chart
        selected_indicators_code (list): List of technical indicators used
        
    Returns:
        dict: Analysis results as a dictionary
    """
    try:
        # Construct the analysis prompt
        analysis_prompt = (
            "You are an 10x expert Financial Analyst who focuses on Technical Analysis at a top financial institution."
            f"Analyze the stock chart for {tkr} ({timeframe} timeframe) based on its candlestick chart, volume, and the displayed technical indicators: {', '.join(selected_indicators_code)}. "
            f"Taking into account the parameters chosen for each indicator, "
            f"Provide a detailed justification of your analysis, explaining patterns, signals, and trends, explicitly mentioning the indicators and their parameter settings that lead to your conclusions. "
            "Your interpretation should take into account the time frame examined, whether it's day trading, longer-term investing, or anything in-between. "
            f"Based *only* on the chart, recommend an action ('Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell') and provide a confidence score (1-10, 10 highest confidence). "
            f"Return ONLY a valid JSON object with 'action', 'confidence_score', 'price_target', and 'justification' keys. Do not include any explanatory text outside the JSON object. Format: "
            f"```json\n{{\"action\": \"Buy\", \"confidence_score\": 8, \"price_target\": \"$X\", \"justification\": \"Detailed analysis...\"}}\n```"
        )
        
        # Get model pipeline
        pipe = initialize_model()
        
        # Save image bytes to a temporary file for the model to read
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
            img_file.write(img_bytes)
            img_path = img_file.name
        
        # Create a proper chat message format
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an expert Financial Analyst who focuses on Technical Analysis."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": analysis_prompt}
                ]
            }
        ]
        
        # Use the pipeline with the correct format
        response = pipe(
            text=messages,
            max_new_tokens=800,  # Increased from 400 to ensure complete response
            do_sample=False,     # Use greedy decoding for more predictable outputs
            return_full_text=False
        )
        
        # Clean up the temporary image file
        os.remove(img_path)
        
        # Extract the generated text from the response
        text = response[0]["generated_text"] if response and isinstance(response, list) and len(response) > 0 and "generated_text" in response[0] else ""
        
        if not text:
            raise ValueError("Empty response from AI model")
        
        # Try to extract JSON from the response text
        start_marker = "```json"
        end_marker   = "```"
        start_idx = text.find(start_marker)
        end_idx   = text.find(end_marker, start_idx + len(start_marker))
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx + len(start_marker):end_idx].strip()
        else:
            # Look for JSON-like content with curly braces
            start_brace = text.find('{')
            end_brace = text.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str = text[start_brace:end_brace+1].strip()
            else:
                json_str = text.strip()

        # If json_str is empty, create a default JSON object
        if not json_str:
            json_str = '{"action": "Error", "confidence_score": 0, "price_target": "N/A", "justification": "AI generated empty response"}'

        def clean_js(js_str):
            return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', js_str)

        cleaned_js = clean_js(json_str)
        parsed = json.loads(cleaned_js)
        return parsed

    except json.JSONDecodeError as e:
        return {
            "action": "Error",
            "confidence_score": 0,
            "price_target": "N/A",
            "justification": f"AI Analysis Error: JSON Parsing Error - {e}"
        }
    except Exception as e:
        return {
            "action": "Error",
            "confidence_score": 0,
            "price_target": "N/A",
            "justification": f"AI Analysis Error: {str(e)}"
        }
