"""
GenAI API utility functions for the evaluation app
"""

import os
import streamlit as st
from openai import OpenAI


def get_genai_base_url():
    """
    Get the GenAI API base URL from environment variables or Streamlit secrets
    """
    # Option 1: Try to get from Streamlit secrets first (recommended for Streamlit apps)
    try:
        return st.secrets["GENAI_BASE_URL"]
    except (KeyError, FileNotFoundError):
        pass
    
    # Option 2: Fall back to environment variable
    base_url = os.getenv("GENAI_BASE_URL")
    if base_url:
        return base_url
    
    # Option 3: Use the URL from your documentation as default
    default_url = "http://generativeaiapi.stg.justanswer.local"
    st.info(f"⚠️ Using default GenAI API URL: {default_url}")
    return default_url


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
        "X-Source": "streamlit_eval_app",
        "X-Usecase": "AutoEval_Pipeline_App",
    }


def create_client_with_model(model_name):
    """
    Create an OpenAI client with headers set for the specific model
    """
    headers = get_headers_for_model(model_name)
    base_url = get_genai_base_url()
    
    return OpenAI(
        base_url=base_url,
        api_key="FAKE_API_KEY",
        default_headers=headers
    )


def initialize_client():
    """
    Initialize GenAI API client without requiring an API key
    """
    # Set up GenAI API headers for tracking
    gen_ai_api_headers = {
        "X-Vendor": "openai",  # Default vendor
        "X-Model": "gpt-o1",  # Default model, can be overridden per request
        "X-Source": "streamlit_eval_app", 
        "X-Usecase": "AutoEval_Pipeline_App",
    }
    
    base_url = get_genai_base_url()
    
    return OpenAI(
        base_url=base_url,
        api_key="FAKE_API_KEY",  # Placeholder only - not used by proxy
        default_headers=gen_ai_api_headers
    )