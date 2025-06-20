"""
GenAI API utility functions for the evaluation app
"""

from openai import OpenAI


def get_headers_for_model(model_name):
    """
    Generate appropriate headers for the GenAI API based on the model being used
    """
    # Extract vendor from model name
    if "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower():
        vendor = "openai"
    elif "claude" in model_name.lower():
        vendor = "anthropic" 
    elif "gemini" in model_name.lower():
        vendor = "google"
    else:
        vendor = "openai"  # Default fallback
    
    return {
        "X-Vendor": vendor,
        "X-Model": model_name,
        "X-Source": "streamlit",
        "X-Usecase": "Cyborgs_12345_AutoEvalPipelineApp",
    }


def create_client_with_model(model_name):
    """
    Create an OpenAI client with headers set for the specific model
    """
    headers = get_headers_for_model(model_name)
    
    return OpenAI(
        base_url="http://generativeaiapi.stg.justanswer.local",
        api_key="FAKE_API_KEY",
        default_headers=headers
    )


def initialize_client(api_key=None):
    """Initialize with GenAI API client"""
    # The api_key parameter is now optional since we use a placeholder
    
    # Set up GenAI API headers for tracking
    gen_ai_api_headers = {
        "X-Vendor": "openai",  # Default vendor
        "X-Model": "gpt-4o-2024-05-13",  # Default model, can be overridden per request
        "X-Source": "streamlit_eval_app",
        "X-Usecase": "AutoEval_Pipeline_App",
    }
    
    return OpenAI(
        base_url="http://generativeaiapi.stg.justanswer.local",
        api_key="FAKE_API_KEY",  # Placeholder only - not used by proxy
        default_headers=gen_ai_api_headers
    )