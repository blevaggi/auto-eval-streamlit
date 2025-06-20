# Imports
import streamlit as st
import json
import re
from typing import Dict, List, Any
from openai import OpenAI
from collections import defaultdict
import pandas as pd
import altair as alt
import concurrent.futures
from functools import partial
import time
import re
from genai_utils import initialize_client, create_client_with_model, get_headers_for_model


# ====== TAB 1 FUNCTIONS ========
# This tab accepts the use case details, the API permissions, and generates the custom eval and prompt templates

# STEP ONE: User lands on the Streamlit page

# Set the Streamlit Page
st.set_page_config(
    page_title="Auto-Eval Selection & Evaluation Demo",
    page_icon="📊",
    layout="wide"
)

# STEP TWO: User fills out the sidebar with OpenAI permissions

# STEP THREE: User gives us information about their evaluation use case

# Process the input fields from the user
def process_input(task_summary: str, sample_input: str,
                 good_examples: List[str], bad_examples: List[str], 
                 context: str) -> Dict:
    """Process the initial user input package"""
    return {
        "task_summary": task_summary,
        "sample_input": sample_input,
        "good_examples": good_examples,
        "bad_examples": bad_examples,
        "context": context
    }

# STEP FOUR: App determines whether the use case is single-shot or multi-gambit. This will affect Tabs 3 & 4 (e.g. will the user provide input+output pairs, or an entire conversation transcript)

def detect_conversation_type(client, input_package: Dict, model: str) -> Dict:
    """
    Determine if this is a multi-turn conversation use case and set the appropriate 
    evaluation structure
    """
    system_message = """You are an AI evaluation analyzer focused on determining the nature of AI tasks.
    Your job is to identify whether a given task involves multi-turn conversation (like chatbots, 
    conversation agents, or dialogue systems) or single-turn input/output (like content generation, 
    classification, or summarization).
    
    Multi-turn conversations involve:
    - Back-and-forth exchanges between user and AI
    - Context building over multiple turns
    - Conversational continuity and flow
    - Often used for chatbots, virtual assistants, etc.
    
    Single-turn tasks involve:
    - One input leading to one output
    - No contextual memory between interactions
    - Self-contained generation tasks
    - Often used for content creation, transformation, or analysis
    """
    
    with st.spinner("Analyzing if this is a multi-turn conversation task..."):
        analysis_prompt = f"""
        Analyze the given task description and prompt to determine if this is a multi-turn conversation or a single-turn input/output task.
        
        TASK SUMMARY: {input_package['task_summary']}
        REQUIREMENTS: {input_package['requirements']}
        CONTEXT: {input_package['context']}
        SAMPLE INPUT: {input_package['sample_input']}
        
        Based on this information, analyze whether this is:
        1. A multi-turn conversation task (like a chatbot or dialogue system)
        2. A single-turn input/output task (like content generation or transformation)
        
        Return your analysis as a JSON object with these keys:
        - "is_conversation": boolean (true if this is a multi-turn conversation task)
        - "reasoning": brief explanation of your determination
        - "evaluation_structure": either "conversation" or "input_output"
        """
        
        # Create client with model-specific headers
        model_client = create_client_with_model(model)
        
        response = model_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": analysis_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        # Display results
        st.write("Task Type Analysis:")
        st.json(analysis)
        
        return analysis
    
# STEP FIVE: Based on the details of the use case, define the requirements as atomic parts

def generate_eval_metrics(client, input_package: Dict, model: str) -> List[Dict]:
    """Generate style and format based metrics (aka PRESENTATION metrics) with complete evaluation components"""
    
    system_message = """You live, breathe, and are obsessed with LLM evaluation. Ensuring that LLMs produce the best outputs is your passion. 

    You and I both know that evaluation metrics are best broken into two camps:

    (1) Presentation-based metrics (which deal with style and formatting)
    Examples:
     - Format compliance (word count, character limits, prohibited words)
    - Style adherence (tone, voice, brand guidelines)
    - Structural elements (layout, ordering of information)
    - Technical specifications (capitalization, punctuation, formatting)


    (2) Content-based metrics (which deal with the textual quality of the output itself)
    Examples:
    - Relevance (appropriate to context and user needs)
    - Accuracy (factual correctness)
    - Coherence (logical flow and consistency)
    - Fluency (quality of non-English language utilization)
    - Personalization (incorporates specific user details into the output).
    """

    json_example = '''{
        "metrics": [ 
            { 
                "metric": "Style",
                "description": "Ensure that the style of the LLM's output is consistently friendly, natural, and professional.",
                "parameters": [
                    {"friendliness":"Confirm that the output is friendly and not standoffish"},
                    {"naturalness":"Confirm that the output is natural and does not sound robotic."},
                    {"professional":"Confirm that the output is appropriately professional and does not sound overly causual."}
                ]
            },
            { 
                "metric": "Coherence",
                "description": "Assesses whether the conversation flows logically from one question to the next, maintaining a clear and sensible progression that builds on the user's responses.",
                "parameters": [
                    {"logical sense":"On a turn by turn basis, ensure that the bot's response makes sense following the user's most recent input."},
                    {"overall progression":"Ensure that the conversation maintains a clear and sensible progression over the course of the entire conversation"}
                ]
            }
        ]
        }'''
    
    # Create client with model-specific headers
    model_client = create_client_with_model(model)
    
    # Start with an initial generation
    with st.spinner("Generating customized eval metrics..."):
        prompt_1 = f"""
        Look at the use case below. I need your help to understand the core requirements, success criteria, and fail states to identify and define the most specific, atomic, and measurable eval metrics for the use case. 
        
        USE CASE:
        Task Summary: {input_package['task_summary']}
        Context: 
        {input_package['context']}
        
        Task Requirements -- read these VERY carefully, one by one:
        {chr(10).join([f"- {req}" for req in input_package['requirements']])}

        A representative sample input: {input_package['sample_input']}
        
        Good examples of expected outputs: 
        {input_package['good_examples']}
        
        Bad examples of possible outputs: 
        {input_package['bad_examples']}
        
        Now take a moment to think. 

        What are the best, most specific, most atomic, most measurable eval metrics for this use case?

        Please suggest a simple title for each, a brief description, and note any sub-metrics or parameters that ladder up to the metric. 

        Format as pure simple JSON, such as:
        {json_example}

        
        """

        response_1 = model_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_1}
            ]
        )
        
        initial_eval_metrics = response_1.choices[0].message.content
        
        # Display preliminary results
        with st.expander("Initial Eval Metrics:"):
            st.json(initial_eval_metrics)
        
        # Then remove redundancies from the preliminary list
        prompt_2 = f"""
        Look at this list of eval metrics. Remove any redundancies from the list.
        
        CURRENT LIST:
        {initial_eval_metrics}
        
        Return a revised version of the JSON package with any redundancies removed. 
        """

        response_2 = model_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_1},
                {"role": "assistant", "content": initial_eval_metrics},
                {"role": "user", "content": prompt_2}
            ]
        )
        
        final_eval_metrics = response_2.choices[0].message.content

        # Display final results
        with st.expander("Final Eval Metrics with redundancies removed"):
            st.json(final_eval_metrics)
        
        return final_eval_metrics


def generate_kshot_examples(client, input_package, metric_name, param_key, param_description, is_conversation, model):
    """
    Generate k-shot examples for a specific parameter at different score levels (1.0, 0.5, 0.0)
    based on good and bad examples provided by the user.
    """
    system_message = """You are an expert at LLM evaluation and creating realistic examples for different quality levels.
    Your task is to create representative examples of outputs at different score levels (high quality, medium quality, low quality)
    based on specific evaluation criteria and existing examples."""
    
    # Get good and bad examples from the input package
    good_examples = input_package.get("good_examples", [])
    bad_examples = input_package.get("bad_examples", [])
    sample_input = input_package.get("sample_input", "")
    
    # Ensure we have at least one example of each type
    if not good_examples:
        good_examples = ["[No good examples provided]"]
    if not bad_examples:
        bad_examples = ["[No bad examples provided]"]
    
    content_type = "conversation" if is_conversation else "output"
    
    with st.spinner(f"Generating examples for {metric_name} - {param_key}..."):
        prompt = f"""
    Based on the parameter "{param_key}: {param_description}" for the metric "{metric_name}", 
    and the sample input and examples provided, create three examples representing different score levels:
    
    1. A score of 1.0 (perfect): {content_type} that fully meets the parameter requirements
    2. A score of 0.5 (partial): {content_type} that partially meets the parameter requirements
    3. A score of 0.0 (fail): {content_type} that does not meet the parameter requirements at all
    
    Sample input: {sample_input}
    
    Good examples (for reference):
    {good_examples}
    
    Bad examples (for reference):
    {bad_examples}
    
    For each score level, modify the provided examples to create a realistic example that clearly demonstrates why it would receive that score 
    according to the specific parameter. 
    
    Respond in JSON format with keys "1.0", "0.5", "0.0" containing the example content for each score level. 
    """
    
    try:
        # Create client with model-specific headers
        model_client = create_client_with_model(model)
        
        response = model_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        examples = json.loads(response.choices[0].message.content)
        return examples
    
    except Exception as e:
        # Fallback if generation fails
        st.warning(f"Failed to generate k-shot examples: {e}")
        return {
            "1.0": "Example that fully meets the parameter requirements.",
            "0.5": "Example that partially meets the parameter requirements.",
            "0.0": "Example that does not meet the parameter requirements at all."
        }


# STEP SIX: Create the prompt templates for each LLM-as-judge eval metric
# Create two prompts per metric (one for individual eval, one for pairwise eval)
def generate_evaluation_templates(client, metrics_list, is_conversation, input_package, model):
    """
    Generate evaluation templates for both parameter and pairwise evaluations
    with k-shot examples for different score levels.
    
    Args:
        client: OpenAI client instance
        metrics_list: List of metrics with parameters
        is_conversation: Boolean indicating if this is a conversation task
        input_package: Dictionary containing task details including good/bad examples
        model: Model to use for generating examples
        
    Returns:
        tuple: (parameter_prompt_templates, pairwise_prompt_templates)
    """
    # Create templates for individual parameter evaluation prompts
    parameter_prompt_templates = {}
    pairwise_prompt_templates = {}
    
    # Add a note about the specific evaluation purpose
    evaluation_context = (
        "This is a parameter-specific evaluation to assess a single aspect of quality. "
        "Focus ONLY on the specific parameter definition when evaluating and no other parameters."
    )
    
    # Process each metric and its parameters to create templates
    for metric in metrics_list:
        metric_name = metric.get("metric", "Unnamed Metric")
        metric_description = metric.get("description", "")
        parameters = metric.get("parameters", [])
        
        # Generate a template for each parameter
        for param in parameters:
            if isinstance(param, dict) and len(param) > 0:
                # Extract the parameter key and value
                param_key = list(param.keys())[0]
                param_description = param[param_key]
                
                # Create a unique key for this parameter prompt
                prompt_key = f"{metric_name}::{param_key}"
                
                # Generate k-shot examples for this parameter
                kshot_examples = generate_kshot_examples(
                    client=client,
                    input_package=input_package,
                    metric_name=metric_name,
                    param_key=param_key,
                    param_description=param_description,
                    is_conversation=is_conversation,
                    model=model
                )
                
                # Format k-shot examples for inclusion in the prompt
                kshot_section = f"""
## Examples for Reference
Here are examples of outputs at different score levels for this specific parameter:

### Score 1.0 (Fully meets requirements)
```
{kshot_examples.get("1.0", "Example that fully meets the parameter requirements.")}
```

### Score 0.5 (Partially meets requirements)
```
{kshot_examples.get("0.5", "Example that partially meets the parameter requirements.")}
```

### Score 0.0 (Does not meet requirements)
```
{kshot_examples.get("0.0", "Example that does not meet the parameter requirements at all.")}
```
"""
                
                # Create different templates for conversation vs non-conversation
                if is_conversation:
                    # For conversation mode, we evaluate the entire conversation
                    prompt_template = f"""# Role: You live, breathe, and are obsessed with LLM evaluation. Ensuring that LLMs produce the best outputs is your passion

# Task: Evaluate the following LLM output to ensure whether it passes on this eval parameter.

# Parameter Evaluation: {metric_name} - {param_key}

## Context
{evaluation_context}
This is a CONVERSATION evaluation. The output provided contains the entire conversation transcript. Ignore the initial bot message except as an additional point of context.

## Metric Definition
{metric_description}

## Parameter to Evaluate
{param_key}: {param_description}

{kshot_section}

## Conversation to Evaluate
{{conversation}}

## Evaluation Instructions
Score this conversation ONLY on the specific parameter "{param_key}" using this rubric:
0.0: Does not meet the parameter requirements at all
0.5: Partially meets the parameter requirements
1.0: Fully meets the parameter requirements

Provide your score and a detailed justification using direct examples as evidence -- based solely on this specific parameter as JSON.
Do not consider any other aspects of quality outside this parameter's definition. Do not include the bot's welcome message in your evaluation. 
"""
                else:
                    # For non-conversation mode, we evaluate input-output pairs
                    prompt_template = f"""# Role: You live, breathe, and are obsessed with LLM evaluation. Ensuring that LLMs produce the best outputs is your passion

# Task: Evaluate the following LLM output to ensure whether it passes on this eval parameter.

# Parameter Evaluation: {metric_name} - {param_key}

## Context
{evaluation_context}
This is a SINGLE-TURN evaluation. The input and output represent one interaction.

## Metric Definition
{metric_description}

## Parameter to Evaluate
{param_key}: {param_description}

{kshot_section}

## Input
{{input}}

## Output to Evaluate
{{output}}

## Evaluation Instructions
Score this output ONLY on the specific parameter "{param_key}" using this rubric:
0.0: Does not meet the parameter requirements at all
0.5: Partially meets the parameter requirements
1.0: Fully meets the parameter requirements

Provide your score and a detailed justification using direct examples as evidence -- based solely on this specific parameter as JSON.
Do not consider any other aspects of quality outside this parameter's definition.
"""
                parameter_prompt_templates[prompt_key] = prompt_template
                
                # Create pairwise template (without k-shot examples since they're less relevant for pairwise comparison)
                if is_conversation:
                    # For conversation mode, we compare two complete conversations
                    pairwise_template = f"""# Role: You live, breathe, and are obsessed with LLM evaluation. Ensuring that LLMs produce the best outputs is your passion

# Task: Evaluate the following LLM output to ensure whether it passes on this eval parameter.

# Pairwise Parameter Evaluation: {metric_name} - {param_key}

## Context
This is a pairwise comparison focusing only on the specific parameter for CONVERSATION evaluation.

## Metric Definition
{metric_description}

## Parameter to Compare
{param_key}: {param_description}

## Conversations to Compare
Conversation A:
{{conversation_a}}

Conversation B:
{{conversation_b}}

## Evaluation Instructions
Compare these conversations ONLY on the specific parameter "{param_key}" using this criteria:
- Which conversation better satisfies this parameter? 
- Respond with "A is better", "B is better", or "Equivalent" if they are too similar to distinguish.
- Quantify how much better the winning conversation was with one of these tags: 
    * Equivalent -- if you answered Equivalent 
    * Slightly -- if you decided the winning response was only a bit better
    * Moderately -- if you decided the winning response was better but the loser wasn't significantly far behind
    * Majorly -- if the winning response was a clear winner, hands-down, and the loser was way worse.  

Provide your score and a detailed justification using direct examples as evidence -- based solely on this specific parameter as JSON.
Do not consider any other aspects of quality outside this parameter's definition.
"""
                else:
                    # For non-conversation mode, we compare two input-output pairs
                    pairwise_template = f"""# Role: You live, breathe, and are obsessed with LLM evaluation. Ensuring that LLMs produce the best outputs is your passion

# Task: Evaluate the following LLM output to ensure whether it passes on this eval parameter.

# Pairwise Parameter Evaluation: {metric_name} - {param_key}

## Context
This is a pairwise comparison focusing only on the specific parameter for SINGLE-TURN evaluation.

## Metric Definition
{metric_description}

## Parameter to Compare
{param_key}: {param_description}

## Input
{{input}}

## Outputs to Compare
Output A:
{{output_a}}

Output B:
{{output_b}}

## Evaluation Instructions
Compare these outputs ONLY on the specific parameter "{param_key}" using this criteria:
- Which output better satisfies this parameter? 
- Respond with "A is better", "B is better", or "Equivalent" if they are too similar to distinguish.
- Quantify how much better the winning conversation was as: slightly, moderately, or majorly.

Provide your score and a detailed justification using direct examples as evidence -- based solely on this specific parameter as JSON.
Do not consider any other aspects of quality outside this parameter's definition.
"""
                pairwise_prompt_templates[prompt_key] = pairwise_template
    
    return parameter_prompt_templates, pairwise_prompt_templates



# STEP SEVEN: Put a lot of these pieces together in sequence
def setup_evaluation_config(task_summary: str, sample_input: str,
                          good_examples: List[str], bad_examples: List[str], 
                          context: str, requirements: List[str],
                          client, model: str) -> Dict[str, Any]:
    """
    Sets up the evaluation configuration for Tab 1, generating metrics and creating
    evaluation prompt templates that can be used in other tabs.
    Handles both conversation and non-conversation scenarios.
    """
    # Build an input package
    input_package = {
        "task_summary": task_summary,
        "sample_input": sample_input,
        "good_examples": good_examples,
        "bad_examples": bad_examples,
        "context": context,
        "requirements": requirements
    }
    
    # First, detect if this is a conversation task
    conversation_analysis = detect_conversation_type(client, input_package, model)
    is_conversation = conversation_analysis.get("is_conversation", False)
    st.divider()
    # Generate metrics using the consolidated function
    eval_metrics_json_str = generate_eval_metrics(client, input_package, model)
    
    # Convert the JSON string to a Python object if it's a string
    if isinstance(eval_metrics_json_str, str):
        try:
            eval_metrics_json = json.loads(eval_metrics_json_str)
        except json.JSONDecodeError:
            # If it's not valid JSON, keep it as is and log a warning
            st.warning("Could not parse metrics JSON. The templates may not be generated correctly.")
            eval_metrics_json = eval_metrics_json_str
    else:
        # If it's already a Python object, use it directly
        eval_metrics_json = eval_metrics_json_str
    
    # Extract metrics list from the JSON structure
    metrics_list = []
    if isinstance(eval_metrics_json, dict) and "metrics" in eval_metrics_json:
        metrics_list = eval_metrics_json["metrics"]
    elif isinstance(eval_metrics_json, list):
        metrics_list = eval_metrics_json
    
    # Generate templates using the enhanced function
    parameter_prompt_templates, pairwise_prompt_templates = generate_evaluation_templates(
        client=client,
        metrics_list=metrics_list,
        is_conversation=is_conversation,
        input_package=input_package,
        model=model
    )
    
    # Return the complete configuration package with conversation flag
    return {
        "input": input_package,
        "conversation_analysis": conversation_analysis,
        "is_conversation": is_conversation,
        "metrics_json": eval_metrics_json,
        "parameter_prompt_templates": parameter_prompt_templates,
        "pairwise_prompt_templates": pairwise_prompt_templates
    }


    
    


# Allow users to download evaluation configs for future re-use.
def save_metrics_to_session(pipeline_results):
    """
    Save the pipeline results to session state in a JSON-serializable format
    """
    # Store the results in the session state
    st.session_state.saved_pipeline_results = pipeline_results
    
    # Convert to JSON for download
    try:
        json_data = json.dumps(pipeline_results, indent=2)
        return json_data
    except TypeError as e:
        st.error(f"Error serializing metrics to JSON: {e}")
        # Try to clean non-serializable parts
        cleaned_results = clean_for_json(pipeline_results)
        return json.dumps(cleaned_results, indent=2)


def download_metrics_button(json_data):
    """
    Creates a download button for metrics data in JSON format
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{timestamp}.json"
    
    st.download_button(
        label="💾 Download Metrics as JSON",
        data=json_data,
        file_name=filename,
        mime="application/json",
    )
def load_metrics_from_file():
    """
    Let the user upload a JSON file with saved metrics and regenerate prompt templates
    """
    uploaded_file = st.file_uploader("Upload metrics JSON file", type="json")
    
    if uploaded_file is not None:
        try:
            # Read the file
            content = uploaded_file.read()
            # Parse JSON
            loaded_metrics = json.loads(content)
            
            # Check if we need to recreate prompts
            has_parameter_templates = "parameter_prompt_templates" in loaded_metrics and loaded_metrics["parameter_prompt_templates"]
            has_pairwise_templates = "pairwise_prompt_templates" in loaded_metrics and loaded_metrics["pairwise_prompt_templates"]
            
            if not (has_parameter_templates and has_pairwise_templates):
                st.info("Regenerating evaluation prompts from the loaded metrics...")
                # Extract necessary components to regenerate prompts
                is_conversation = loaded_metrics.get("is_conversation", False)
                metrics_json = loaded_metrics.get("metrics_json", {})
                
                # Extract metrics list
                metrics_list = []
                if isinstance(metrics_json, dict) and "metrics" in metrics_json:
                    metrics_list = metrics_json["metrics"]
                elif isinstance(metrics_json, list):
                    metrics_list = metrics_json
                
                # Generate templates using the shared function
                parameter_prompt_templates, pairwise_prompt_templates = generate_evaluation_templates(
                    metrics_list, is_conversation
                )
                
                # Add regenerated templates to the loaded metrics
                loaded_metrics["parameter_prompt_templates"] = parameter_prompt_templates
                loaded_metrics["pairwise_prompt_templates"] = pairwise_prompt_templates
                
                st.success(f"Successfully regenerated {len(parameter_prompt_templates)} parameter templates and {len(pairwise_prompt_templates)} pairwise templates!")
            else:
                st.success("Metrics loaded successfully with existing templates!")
            
            return loaded_metrics
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid metrics file.")
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    
    return None

def add_prompt_zip_download_button(pipeline_results, prefix="evaluation_prompts"):
    """
    Creates a downloadable zip file containing all evaluation prompts
    """
    import datetime
    import io
    import zipfile
    
    # Debug info to help diagnose issues
    st.write("Checking available prompt templates...")
    
    # Extract templates from pipeline_results
    param_templates = pipeline_results.get("parameter_prompt_templates", {})
    pairwise_templates = pipeline_results.get("pairwise_prompt_templates", {})
    metrics_json = pipeline_results.get("metrics_json", {})
    
    # Debug: show what we found
    st.write(f"Found {len(param_templates)} parameter templates and {len(pairwise_templates)} pairwise templates")
    
    if not (param_templates or pairwise_templates):
        st.warning("No evaluation prompts available to download. This may indicate a problem with template generation.")
        # Display the keys in pipeline_results to help diagnose
        st.write("Available keys in pipeline_results:", list(pipeline_results.keys()))
        return
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add README
        readme_content = """# Evaluation Prompts
        
This zip file contains evaluation prompts generated for your specific use case.

## Folders:
- `parameter_prompts/`: Contains prompts for evaluating individual parameters
- `pairwise_prompts/`: Contains prompts for pairwise comparisons

You can use these prompts directly with API calls to your preferred LLM.
"""
        zipf.writestr("README.md", readme_content)
        
        # Create directories in the zip file
        zipf.writestr("parameter_prompts/.keep", "")
        zipf.writestr("pairwise_prompts/.keep", "")
        
        # Extract parameter prompt templates
        for key, template in param_templates.items():
            safe_key = key.replace("::", "_").replace(" ", "_")
            zipf.writestr(f"parameter_prompts/{safe_key}.txt", template)
        
        # Extract pairwise prompt templates
        for key, template in pairwise_templates.items():
            safe_key = key.replace("::", "_").replace(" ", "_")
            zipf.writestr(f"pairwise_prompts/{safe_key}.txt", template)
                
        # Add metrics definition file
        if metrics_json:
            try:
                # Convert to string if it's already a dict
                if isinstance(metrics_json, dict) or isinstance(metrics_json, list):
                    metrics_str = json.dumps(metrics_json, indent=2)
                else:
                    # It might already be a JSON string
                    metrics_str = metrics_json
                zipf.writestr("metrics_definition.json", metrics_str)
            except Exception as e:
                st.warning(f"Could not include metrics definition: {e}")
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.zip"
    
    # Get the value of the buffer and create download button
    zip_data = zip_buffer.getvalue()
    
    st.download_button(
        label="📥 Download All Prompts as ZIP",
        data=zip_data,
        file_name=filename,
        mime="application/zip",
    )

def load_metrics_from_file():
    """
    Let the user upload a JSON file with saved metrics
    """
    uploaded_file = st.file_uploader("Upload metrics JSON file", type="json")
    
    if uploaded_file is not None:
        try:
            # Read the file
            content = uploaded_file.read()
            # Parse JSON
            metrics = json.loads(content)
            st.success("Metrics loaded successfully!")
            return metrics
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid metrics file.")
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    
    return None



# ====== TAB 3 FUNCTIONS ========





# ====== MAIN ========

def main():
    # Initialize session state variables if not already set
    if "auto_save_enabled" not in st.session_state:
        st.session_state.auto_save_enabled = True
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None
    if "client" not in st.session_state:
        st.session_state.client = None
    if "model" not in st.session_state:
        st.session_state.model = "o3-mini-2025-01-31"
    # Initialize example lists if they don't exist
    if "good_examples" not in st.session_state:
        st.session_state.good_examples = [""]
    if "bad_examples" not in st.session_state:
        st.session_state.bad_examples = [""]
        
    st.title("Auto-Generated Eval Pipeline")

    st.text("This tool will help you identify the right eval metrics for your use case automatically using chained reasoning LLMs. Within this same tool you can upload outputs that you already generated somewhere else, run evaluation on them, and visualzie the results. ")

    # Sidebar: API configuration and new Evaluation Mode toggle.
    # Sidebar: API configuration 
    with st.sidebar:
        st.header("GenAI API Configuration")
        st.info("Using internal GenAI API proxy - no API key required!")
        
        # Show available models with vendor info
        model_options = [
            "gpt-4o-2024-05-13 (OpenAI)",
            "gpt-4o-mini-2024-07-18 (OpenAI)", 
            "gpt-4.1-2025-04-14 (OpenAI)",
            "gpt-4.1-mini-2025-04-14 (OpenAI)",
            "o3-mini-2025-01-31 (OpenAI)"
        ]
        
        selected_model = st.selectbox(
            "Metric Generation Model", 
            model_options, 
            index=0,
            help="Model used for generating metrics and customizing rubrics"
        )
        
        # Extract just the model name (remove vendor info)
        model = selected_model.split(" (")[0]

        if st.button("💾 Initialize GenAI Client"):
            st.session_state.client = initialize_client()
            st.session_state.model = model
            st.success("GenAI API client initialized!")
            
            # Show which headers will be used
            headers = get_headers_for_model(model)
            with st.expander("API Headers", expanded=False):
                st.json(headers)
        
        # Show connection status
        if st.session_state.get('client'):
            st.success("✅ GenAI API Connected")
            st.write(f"📋 Model: {st.session_state.get('model', 'Not set')}")
        else:
            st.warning("⚠️ GenAI API Not Connected")

    tab1, tab2, tab3, tab4  = st.tabs(["Generate Eval Metrics", "Review Each Metric", "Individual Eval", "Pairwise Eval"])
        
    with tab1:

        # Introduction section
        st.header("Describe your use case")
        st.info("Tell us everything you can about your LLM task to generate the most specific, customized evaluation metrics for you.")

        # TEXT FIELDS FOR THE INPUT PACKAGE
        # Ask for the Task Summary
        st.subheader("Context")
        task_summary = st.text_input(
            """In just one high-level sentence, describe what this use case is. It's a credit card headline, it's an intake bot, it's a follow-up SMS, etc. """, 
            help="In just one high-level sentence, describe what this use case is. It's a credit card headline, it's an intake bot, it's a follow-up SMS, etc. ", placeholder="Conversational intake conversation between a user with a problem and the JustAnswer chatbot to drive conversion to service")

        # Ask for deeper Context
        context = st.text_area(
            "Context: Describe where and how this content will be used",
            help="Describe where and how this content will be used",
            placeholder="The user has just landed on the JustAnswer website. We have greeted them with a chatbot named Pearl who invites them to talk about their issue. Once the user sends a message, the bot should respond with intake questions designed to gather more context before connecting them with a human JustAnswer professional.",
            height=100)
        
        # Ask for the task requirements (not necessarily the Propmt)
        requirements = st.text_area(
            "Task Requirements (or Prompt)", 
            help="List the requirements that the content must satisfy. In a pinch, you can use your existing prompt. ", 
            placeholder="""Expert Reference -- Always "the", NEVER "our"
Expert Reference -- Capitalized as a proper noun 
Length -- concise
Personalized Empathy -- Must start with a statement that is empathetic and personalized
Must ask diagnostic intake questions
Must never provide answers
Must only ask one question at a time
Must not claim to be human
Must not engage with inappropriate queries
Tone must be natural
Tone must be friendly
Tone must be professional
""",
            height=200)
        
        # Provide a representative sample input
        st.subheader("Sample Input and Output Examples",divider=True)
        sample_input = st.text_area("A representative sample input that would generate an output", height=100)
        
        # DYNAMIC EXAMPLES SECTION
        # Function to add a new example
        def add_good_example():
            st.session_state.good_examples.append("")
            
        def add_bad_example():
            st.session_state.bad_examples.append("")
            
        # Function to remove an example at a specific index
        def remove_good_example(index):
            if len(st.session_state.good_examples) > 1:  # Keep at least one example field
                st.session_state.good_examples.pop(index)
            
        def remove_bad_example(index):
            if len(st.session_state.bad_examples) > 1:  # Keep at least one example field
                st.session_state.bad_examples.pop(index)
        
        # Good Examples Section
        st.text("Good Examples of outputs that would pass evaluation")
        # st.caption("Examples of outputs that would pass evaluation")
        
        # For each good example in the state, create an input box with a remove button
        for i, example in enumerate(st.session_state.good_examples):
            col1, col2 = st.columns([10, 1])
            with col1:
                # Use key with index to ensure uniqueness
                st.session_state.good_examples[i] = st.text_area(
                    f"Example {i+1}", 
                    value=example,
                    key=f"good_example_{i}",
                    height=100
                )
            with col2:
                if st.button("🗑️", key=f"remove_good_{i}"):
                    remove_good_example(i)
                    st.rerun()  # Rerun to reflect the removal in the UI
        
        # Add a "plus" button to add a new good example
        if st.button("➕ Add Good Example"):
            add_good_example()
            st.rerun()  # Rerun to show the new example field
        
        # Bad Examples Section
        st.text("Bad Examples of outputs that would NOT pass evaluation")
        # st.caption("Examples of outputs that would NOT pass evaluation")
        
        for i, example in enumerate(st.session_state.bad_examples):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.session_state.bad_examples[i] = st.text_area(
                    f"Example {i+1}", 
                    value=example,
                    key=f"bad_example_{i}",
                    height=100
                )
            with col2:
                if st.button("🗑️", key=f"remove_bad_{i}"):
                    remove_bad_example(i)
                    st.rerun()
        
        if st.button("➕ Add Bad Example"):
            add_bad_example()
            st.rerun()
        
        # Extract examples for processing
        good_examples_list = [ex for ex in st.session_state.good_examples if ex.strip()]
        bad_examples_list = [ex for ex in st.session_state.bad_examples if ex.strip()]

        # Show the user the button to generate customized metrics based on the use case details

        st.subheader("Run Pipeline", divider=True)
        if st.button("Generate Customized Metrics"):
            
            # FIXED: Check if we have a connected client instead of api_key
            if not st.session_state.get('client'):
                st.error("Please initialize the GenAI client in the sidebar first.")
            elif not requirements:
                st.error("Task requirements are required.")
            else:
                # FIXED: Use the client from session state, no need to pass api_key
                client = st.session_state.client
                model_used = st.session_state.model
                
                # Parse requirements into a list
                requirements_list = [req.strip() for req in requirements.split('\n') if req.strip()]
                
                with st.spinner("Processing..."):
                    st.session_state.pipeline_results = setup_evaluation_config(
                        task_summary=task_summary,
                        requirements=requirements_list,  # FIXED: Pass as list
                        sample_input=sample_input,
                        good_examples=good_examples_list,
                        bad_examples=bad_examples_list,
                        context=context,
                        client=client,
                        model=model_used
                    )
                st.success("Metrics customized successfully!")
        st.divider()
       # Add the Save/Load Metrics expander panel
        with st.expander("Save/Load Metrics", expanded=True):
            st.info("Save your current metrics or load previously generated metrics.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Save Current Metrics")
                if st.session_state.pipeline_results:
                    # Save metrics as JSON
                    json_data = save_metrics_to_session(st.session_state.pipeline_results)
                    download_metrics_button(json_data)
                    
                    # Also provide prompts zip in the same section
                    st.markdown("---")
                    st.subheader("Download Evaluation Prompts")
                    add_prompt_zip_download_button(st.session_state.pipeline_results)
                else:
                    st.write("No metrics to save yet.")
            
            with col2:
                st.subheader("Load Saved Metrics")
                loaded_metrics = load_metrics_from_file()
                if loaded_metrics and st.button("Click to confirm"):
                    st.session_state.pipeline_results = loaded_metrics
                    st.success("Metrics loaded successfully!")
                    st.rerun()

    # Rest of your tabs implementation
    with tab2:
        st.header("Review Each Metric")
        st.info("Explore all metrics, their parameters, and evaluation prompts.")
        from tab_2 import add_tab2_content, display_metrics_and_prompts
        # Display metrics and prompts
        pipeline_results_for_display = st.session_state.pipeline_results
        add_tab2_content(pipeline_results_for_display)

    with tab3:
        st.header("Individual Evaluations")
        st.info("Upload a dataset to evaluate each output using the generated metrics")
        # Import the parallel version
        from tab_3 import add_tab3_content_parallel
        add_tab3_content_parallel()
   
    with tab4:
        st.header("Pairwise Evaluations")
        st.info("Upload a dataset with outputs from two versions, and determine which is 'better' for each eval metric")
        from tab_4 import add_tab4_content_improved
        add_tab4_content_improved()


if __name__ == "__main__":
    main()