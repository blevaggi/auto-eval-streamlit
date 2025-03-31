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

st.set_page_config(
    page_title="Auto-Eval Selection & Evaluation Demo",
    page_icon="📊",
    layout="wide"
)

# Initialize OpenAI client
def initialize_client(api_key=None):
    """Initialize with OpenAI client"""
    if not api_key:
        st.error("OpenAI API key is required")
        st.stop()
    return OpenAI(api_key=api_key)

# Initialize metrics library
def initialize_metrics_library():
    """Create a comprehensive metrics library of prevetted concepts"""
    return {
        "Coherence": {
            "description": "Checks that the output makes sense in context and is generally accurate, logical, coherent, and sensible.",
            "evaluator_type":"llm",
            "recommended_use_cases": ["general content", "customer support", "educational content"],
            "recommended_submetrics":"Recommended as a single score, no submetrics",
            "examples": [
                "The response directly addresses the question asked",
                "Information in the response doesn't contradict itself",
                "The response focuses on the right information from the input"
            ],
        },
        "Personalization": {
            "description": "Checks that the output incorporates specific and relevant details from the input to create an effect of a bespoke response or messaging. This typically means that the output literally repeats words or obvious paraphrase from the input. ",
            "evaluator_type":"llm",
            "recommended_use_cases": ["chatbots", "CC headlines", "question summaries"],
            "recommended_submetrics":"Yes, a good candidate for submetrics",
            "examples": ["Using customer's name in response",
                        "Referencing specific problems mentioned by the user"],
            "evaluation_scope": ["customer-specific details", "conversation references", "tailored content"]
        },
        "CTA": {
            "description": "Checks that the output contains a clear call to action",
            "evaluator_type":"llm",
            "recommended_use_cases": ["marketing materials", "sales pages", "headlines", "advertisements"],
            "recommended_submetrics":"Recommend as a single score, given that the metric is already singular and very straightforward",
            "examples": ["'Apply now' button is prominently displayed",
                        "Headline contains an action verb like 'Get', 'Find', 'Discover'"],
            "evaluation_scope": ["action verbs", "urgency indicators", "motivation elements"],
        },
        "Format": {
            "description": "Check that the output follows specific structural requirements. Does NOT apply to content quality or semantic style. Consider these to be heuristic rules.",
            "evaluator_type":"algorithmic",
            "recommended_use_cases": ["headlines", "templates", "character-limited content"],
            "examples": ["Headline is within character limit", 
                       "Content uses required formatting like sentence case"],
            "evaluation_scope": ["word count", "case formatting", "variable limits", "prohibited words"],
        },
        "Style": {
            "description": "Check that the output follows a specific style or tone, as outlined in the prompt",
            "evaluator_type":"llm",
            "recommended_use_cases": ["brand communications", "targeted messaging"],
            "recommended_submetrics":"Yes, a good candidate for submetrics. Break apart empathy from human-ness, for example.",
            "examples": ["Content matches brand voice guidelines",
                        "Messaging maintains consistent tone throughout",
                        "Output contains a clear call to action",
                        "If a style is specified in the prompt, the output complies"                       
                        ],
            "evaluation_scope": ["tone", "voice", "engagement", "emotional appeal"]
        },
        "Factual Accuracy": {
            "description": "Check that statements align with verified facts",
            "evaluator_type":"llm",
            "recommended_use_cases":["Providing direct answers to users' queries"],
            "examples": ["It is worrying if a small dog has eaten a lot of chocolate",
                        "Pregnancy in humans lasts 9 months"]
        },
        # "Brand Voice": {
        #     "description": "Check that output matches company's tone and style",
        #      "evaluator_type":"llm",
        #     "recommended_use_cases": ["marketing", "social media", "customer communications"],
        #     "examples": [
        #         "Language uses approved terminology",
        #         "Tone matches brand personality guidelines"
        #         ]
        # },
        # "Engagement": {
        #     "description": "Check that content is likely to produce user interaction",
        #      "evaluator_type":"llm",
        #     "recommended_use_cases": ["social media", "email marketing", "website headlines"],
        #     "examples": [
        #         "Content poses thought-provoking questions",
        #         "Content creates emotional connection"
        #     ]
        # },
        "Robustness": {
            "description": "Check that the chatbot adheres to its system prompt across multiple gambits",
             "evaluator_type":"llm",
            "recommended_use_cases": ["chatbots"],
            "examples": [
                "Chatbot still remembers the customer's name provided in the 1st message by the 9th message ",
                "Chatbot doesn't ask the customer for their location because it was already provided in an earlier gambit",
            ]
        },
        "Toxicity": {
            "description": "Check that the output is not likely to contain personal attacks, mockery, hatefulness, dismissive statements, threats, or intimidation.",
             "evaluator_type":"llm",
            "examples": [
                "Toxic: People like you should be banned from speaking. You'll regret saying things like that.Not Toxic: I'm not sure I fully understand your position. Could you provide more details?",
                "Toxic: Your opinion is worthless, and you're wasting everyone's time here. Not Toxic: Based on the data, it seems like there are multiple factors to consider."
            ]
        },
        "Question_Quality": {
            "description":"Check that the output question is especially relevant and specific to the user's particular context. That it is probing and uniquely applicable. ",
            "evaluator_type":"llm",
            "recommended_use_cases": ["chatbots"],
            "examples":[
                ""
                ]
        }
}

# Process the input package
def process_input(prompt: str, task_summary: str, sample_input: str,
                 good_examples: List[str], bad_examples: List[str], 
                 context: str) -> Dict:
    """Process the initial user input package"""
    return {
        "prompt": prompt,
        "task_summary": task_summary,
        "sample_input": sample_input,
        "good_examples": good_examples,
        "bad_examples": bad_examples,
        "context": context
    }

def retrieve_relevant_metrics(client, input_package: Dict, model: str) -> Dict:
    """Use LLM to define relevant metrics based on the input context"""
    
    system_message = """You are an AI evaluation metric system. Your purpose is to:
    1. Analyze tasks to determine what makes an output successful
    2. Identify relevant metrics that should be used to evaluate outputs
    3. Focus on metrics that are directly applicable to the specific task type
    
    The user will provide details about their task, and you should analyze the task type,
    define success criteria, and recommend appropriate evaluation metric types.
    """
    
    with st.spinner("Analyzing task type and success criteria..."):
        retrieval_prompt = f"""
        Analyze the input package and select the most appropriate evaluation metrics from our library.
        You must use chain of thought reasoning to work through each step of this analysis. 
        
        INPUT:
        Task Summary: {input_package['task_summary']}
        Context: {input_package['context']}
        Task Requirements: {input_package.get('task_requirements', 'Not provided')}
        Sample Input: {input_package['sample_input']}
        Good Example: {input_package['good_examples']}
        Bad Example: {input_package['bad_examples']}
        Prompt Template: {input_package['prompt']}
        
        First, analyze what kind of task this is and what would make an output successful given the context.

        Consider what makes the good examples good, the bad ones bad. 

        Make sure to note the rules and instructions present in the prompt template. 
        
        Return your analysis as JSON with the following structure:
        {{
        "task_analysis": "Brief analysis of the task type and domain",
        "success_criteria": "What makes an output successful for this task",
        "recommended_metric_types": ["list", "of", "general", "metric", "categories"]
        }}
        """

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": retrieval_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        analysis = json.loads(content)

    return analysis

def select_applicable_metrics(client, metrics_library, input_package: Dict, model: str):
    """Use LLM to select which metrics apply to this prompt"""

    system_message = """You are an AI evaluation metric system. Your purpose is to:
    1. Review task analysis and success criteria
    2. Select specific metrics from the provided library that best match the task requirements
    3. Choose metrics that will provide meaningful, non-overlapping evaluation dimensions
    
    Focus on selecting metrics that will provide actionable insights about the quality of outputs.
    """

    # First retrieve relevant metrics based on task analysis
    analysis = retrieve_relevant_metrics(client, input_package, model)
    st.write("Task Analysis:")
    st.json(analysis)
    
    # If we have explicit recommended metrics from the analysis, use those
    all_recommended = set(analysis.get("recommended_metric_types", []))
    
    # Present the full metrics library and the analysis to the LLM for final selection
    metrics_context = "\n\n".join([
        f"Metric: {name}\nDescription: {details['description']}\nRecommended Use Cases: {', '.join(details.get('recommended_use_cases', []))}" 
        for name, details in metrics_library.items()
    ])
    
    with st.spinner("Selecting appropriate metrics..."):
        prompt_for_llm = f"""
        Based on this analysis of the task:
        Task Analysis: {analysis['task_analysis']}
        Success Criteria: {analysis['success_criteria']}
        Recommended Metric Types: {', '.join(all_recommended)}
        
        And this library of evaluation metrics:
        {metrics_context}
        
        Select the specific metrics from the library that would be most appropriate to evaluate outputs for:
        Prompt: {input_package['prompt']}
        Task: {input_package['task_summary']}
        Context: {input_package['context']}
        
        Return only the names of the selected metrics as a JSON list with a 'metrics' key.
        Example: {{"metrics": ["Coherence", "CTA", "Format", "Personalization"]}}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_for_llm}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        metrics_json = json.loads(content)
        
        # Get final metrics
        final_metrics = metrics_json.get("metrics", [])
            
    return final_metrics, analysis

def prevent_evaluation_overlap(client, metrics_library, selected_metrics, input_package, model: str):
    """
    Analyze selected metrics to ensure no overlapping evaluation criteria
    and modify metrics if necessary to prevent double-dipping.
    """
    system_message = """You are an AI evaluation system expert in preventing metric overlap.
    Your purpose is to ensure that each metric evaluates distinct aspects of the output.
    When two metrics might evaluate the same aspect, create clear separation by:
    1. Identifying which aspects belong exclusively to which metrics
    2. Generating notes that clearly define which aspects each metric should NOT evaluate
    3. Focusing each metric on its core purpose

    For example:
    • If there is a rule about issue-related variable useage, that could be in Format and Personalization. But ultimately, Personalization is the umbrella it should live under, and therefore it must be removed from Format
    • If there is a rule about urgency indicators, that could be in CTA and Style. But ultimately, urgency and CTA are more directly related and must be removed from Style. 
    
    Specialty references belong solely to Personalization

    Context alignment in Coherence should focus on logical consistency, NOT personalization concepts
    """
    
    # Define which aspects belong exclusively to which metrics
    with st.expander("Preventing overlap between metrics", expanded=False):
        st.write("Ensuring metrics evaluate distinct aspects")
    
    exclusive_aspects = {
        "Format": ["word count", "sentence case", "capitalization", "prohibited words", 
                 "structural requirements"],
        "Coherence": ["logical sense", "context alignment", "intent matching"],
        "Personalization": ["customer-specific details", "conversation references", "variable usage", 
                           "relevancy to specific products/brands mentioned"],
        "CTA": ["action verbs", "urgency indicators", "motivation", "clear next steps"],
        "Style": ["tone", "voice", "engagement factors"]
    }
    
    # Create a copy of the metrics library to modify
    modified_metrics_library = metrics_library.copy()
    
    # Check for all potential overlaps between metrics
    pairs_to_check = []
    
    # Add Format and any other metric
    if "Format" in selected_metrics:
        for metric in selected_metrics:
            if metric != "Format" and metric in exclusive_aspects:
                pairs_to_check.append(("Format", metric))
    
    # Add CTA and Personalization
    if "CTA" in selected_metrics and "Personalization" in selected_metrics:
        pairs_to_check.append(("CTA", "Personalization"))
    
    # Add other potential overlapping pairs as needed
    
    # Handle all identified pairs
    for primary, secondary in pairs_to_check:
        primary_aspects = exclusive_aspects[primary]
        secondary_aspects = exclusive_aspects[secondary]
        
        with st.spinner(f"Adjusting {secondary} to avoid overlapping with {primary}..."):
            exclusion_prompt = f"""
            The '{secondary}' metric should NOT evaluate any of the following aspects as they
            will be handled by the '{primary}' metric:
            - {", ".join(primary_aspects)}
            
            Instead, the '{secondary}' metric should focus exclusively on these aspects:
            - {", ".join(secondary_aspects)}
            
            Generate a note to include in the customized metric description that clearly
            states that aspects of {primary} should not be part of the {secondary} evaluation.
            """
            
            # Get an exclusion note from the LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": exclusion_prompt}
                ],
                max_completion_tokens=1000
            )
            
            # We'll return this note to include in metric customization
            modified_metrics_library[secondary]["exclusion_note"] = response.choices[0].message.content
    
    return modified_metrics_library

def customize_metrics(client, metrics_library, input_package: Dict, selected_metrics: List[str], model: str) -> Dict[str, Dict]:
    """Customize each selected metric based on the input context"""
   
   # Define which aspects belong exclusively to which metrics
    with st.expander("Customizing the following metrics from the library...", expanded=True):
        st.write(selected_metrics, expanded=True) 

    system_message = """You are an AI evaluation metric customization system. Your purpose is to:
    1. Take general metric definitions and tailor them specifically to the task
    2. Create detailed parameters, success criteria, and scoring rubrics
    3. Ensure metrics are actionable, measurable, and relevant to the specific context
    
    For each metric, you will customize the description, parameters, success criteria, and scoring rubric
    to directly address the specific task type and requirements.
    """
    
    # First, prevent evaluation overlap by adding notes to metrics
    modified_metrics_library = prevent_evaluation_overlap(client, metrics_library, selected_metrics, input_package, model)
    
    customized_metrics = {}
    
    for metric in selected_metrics:
        if metric not in modified_metrics_library:
            continue
            
        metric_details = modified_metrics_library[metric]
        
        # Include exclusion note if it exists
        exclusion_note = metric_details.get("exclusion_note", "")
        
        with st.spinner(f"Customizing {metric} metric..."):
            customization_prompt = f"""
            Given the following input:
            Prompt: {input_package['prompt']}
            Task: {input_package['task_summary']}
            Context: {input_package['context']}
            Sample Input: {input_package['sample_input']}
            Good Examples: {input_package['good_examples']}
            Bad Examples: {input_package['bad_examples']}

            And this evaluation metric:
            Metric: {metric}
            Description: {metric_details['description']}
            Evaluation Scope: {metric_details.get('evaluation_scope', [])}

            {exclusion_note}

            Please customize this metric to best evaluate outputs for this specific prompt.

            Additionally, if applicable, break down the metric into constituent sub-metrics. For example, for a "Personalization" metric, generate separate sub-metrics such as "personalization.specialty" and "personalization.issue", and then provide an overall sub-metric "personalization.overall" that aggregates these evaluations.

            Only and always focus on the metric at hand! 
            Never put Empathy in the rubric when the metric is Personalization. 
            Never put Personalization in the rubric for Coherence. 
            Never consider X in the rubric for Y. 
            These must AWAYS be mutually exclusive!

            Include:
            1. A customized description specific to this task
            2. Parameters for evaluation
            3. Success criteria that defines what passes/fails this metric
            4. A scoring rubric with specific point values (0-2 scale)
            5. (Optional) A "sub_metrics" key: a dictionary where each key is a sub-metric name (e.g. "personalization.specialty") and its value is an object containing "parameters", "success_criteria", and "scoring_rubric".

            Return as JSON with keys: "customized_description", "parameters", "success_criteria", "scoring_rubric", and optionally "sub_metrics".
            """

        
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": customization_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            customized_metrics[metric] = json.loads(content)
            customized_metrics[metric]["base_description"] = metric_details["description"]
 
            # Get specific examples for this metric
            example_generation_prompt = f"""
            Generate 3 specific examples of what the '{metric}' metric would evaluate for:
            Task: {input_package['task_summary']}
            Context: {input_package['context']}
            
            Return as a JSON list with the key "examples".
            """
            
            example_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": example_generation_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            examples = json.loads(example_response.choices[0].message.content).get("examples", [])
            customized_metrics[metric]["examples"] = examples
        
    return customized_metrics

def run_pipeline(prompt: str, task_summary: str, sample_input: str,
                good_examples: List[str], bad_examples: List[str], 
                context: str, client, model: str) -> Dict[str, Any]:
    """Run the full pipeline to customize metrics"""
    
    # Initialize metrics library
    metrics_library = initialize_metrics_library()
    
    # Step 1: Process input
    input_package = process_input(prompt, task_summary, sample_input, good_examples, 
                                  bad_examples, context)
    
    # Step 2: Select applicable metrics
    selected_metrics, task_analysis = select_applicable_metrics(client, metrics_library, input_package, model)
    
    # Step 3: Customize the metrics
    customized_metrics = customize_metrics(client, metrics_library, input_package, selected_metrics, model)
    
    return {
        "input": input_package,
        "task_analysis": task_analysis,
        "selected_metrics": selected_metrics,
        "customized_metrics": customized_metrics
    }

def group_evaluation_results(evaluation_list):
    """
    Convert a flat list of evaluation results into a nested dict
    grouped by top-level metric name.
    
    evaluation_list: List of dicts, each with keys:
        - "Metric" (string like "Format::format.word_count")
        - "Score" (int or string)
        - "Summary" (string justification or summary)
        
    Returns: Dict like:
    {
      "Format": [
          { "sub_metric": "overall", "score": 1, "summary": "..." },
          { "sub_metric": "format.word_count", "score": 2, "summary": "..." },
          ...
      ],
      "Style": [
          { "sub_metric": "overall", "score": 1, "summary": "..." },
          { "sub_metric": "style.tone", "score": 1, "summary": "..." },
          ...
      ],
      ...
    }
    """
    grouped = {}
    for item in evaluation_list:
        metric_str = item["Metric"]
        score = item.get("Score")
        summary = item.get("Summary", "")
        
        # Split into top-level and sub-metric
        if "::" in metric_str:
            top_metric, sub_metric = metric_str.split("::", 1)
        else:
            top_metric = metric_str
            sub_metric = None
        
        if top_metric not in grouped:
            grouped[top_metric] = []
        
        grouped[top_metric].append({
            "sub_metric": sub_metric,
            "score": score,
            "summary": summary
        })
    
    return grouped


def display_grouped_results(formatted_results, eval_results):
    """
    Displays evaluation results in a nested format by grouping 
    top-level metrics and sub-metrics.

    :param formatted_results: A dict keyed by metric strings like 
        "Format::overall" or "Style::style.tone".
        Each value is another dict with keys "score" and "justification".
    :param eval_results: The raw string output from your LLM evaluations, 
        keyed by the same metric strings.
    """
    # 1) Build a nested structure, grouping by top-level metric
    grouped_results = defaultdict(list)
    
    for metric_key, data in formatted_results.items():
        # Parse out top-level metric vs. sub-metric
        if "::" in metric_key:
            top_metric, sub_metric = metric_key.split("::", 1)
        else:
            top_metric = metric_key
            sub_metric = None

        grouped_results[top_metric].append({
            "sub_metric": sub_metric,
            "score": data["score"],
            "justification": data["justification"],
            "raw_eval": eval_results[metric_key],  # Full LLM response
        })

    # 2) Display in a nested format
    for top_metric, entries in grouped_results.items():
        # Show the top-level metric name
        st.subheader(top_metric)

        # (Optional) Sort so that sub_metric == "overall" (or None) appears first
        entries.sort(key=lambda x: (
            # Put "overall" or None first, then alphabetical
            x["sub_metric"] is not None and "overall" not in x["sub_metric"], 
            x["sub_metric"] or ""
        ))

        for entry in entries:
            sub_label = entry["sub_metric"] if entry["sub_metric"] else "overall"
            expander_title = f"{top_metric}::{sub_label} (Score: {entry['score']})"
            with st.expander(expander_title):
                st.markdown(f"**Score:** {entry['score']}")
                st.markdown(f"**Justification:** {entry['justification']}")
                st.markdown("**Raw Evaluation:**")
                st.text_area(
                    label="",
                    value=entry["raw_eval"],
                    height=200,
                    disabled=True
                )

def display_grouped_pairwise_results(formatted_results, pairwise_results):
    """Display pairwise results grouped by metric category"""
    from collections import defaultdict
    
    # Remove special keys
    display_results = {k: v for k, v in formatted_results.items() 
                      if k not in ["OVERALL", "CATEGORIES"]}
    
    # Group by top-level metric
    grouped_results = defaultdict(list)
    for metric_key, data in display_results.items():
        if "::" in metric_key:
            top_metric, sub_metric = metric_key.split("::", 1)
        else:
            top_metric = metric_key
            sub_metric = None
            
        grouped_results[top_metric].append({
            "sub_metric": sub_metric,
            "winner": data["winner"],
            "justification": data["justification"],
            "raw_eval": pairwise_results[metric_key]
        })
    
    # Display category results
    category_results = formatted_results.get("CATEGORIES", {})
    
    # Create visual summary of category winners
    categories = list(category_results.keys())
    if categories:
        st.subheader("Results by Category")
        
        for category, result in category_results.items():
            winner = result["winner"]
            summary = result["summary"]
            
            # Choose color based on winner
            color = "#4CAF50" if winner == "A" else "#2196F3" if winner == "B" else "#9E9E9E"  # Green for A, Blue for B, Gray for Tie
            
            st.markdown(f"""
            <div style="
                padding: 10px; 
                border-left: 5px solid {color}; 
                background-color: {color}10;
                margin-bottom: 10px;
            ">
                <h3>{category}: Output {winner} is better overall</h3>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display detailed results by category
    for top_metric, entries in grouped_results.items():
        st.subheader(f"{top_metric} Metrics")
        
        # Sort entries so "overall" comes first
        entries.sort(key=lambda x: (
            0 if x["sub_metric"] is None or x["sub_metric"] == "overall" else 1,
            x["sub_metric"] or ""
        ))
        
        for entry in entries:
            sub_label = entry["sub_metric"] if entry["sub_metric"] else "overall"
            winner = entry["winner"]
            
            # Choose icon and color based on winner
            if "A is better" in winner:
                icon = "🅰️"
                color = "#4CAF50"  # Green
            elif "B is better" in winner:
                icon = "🅱️"
                color = "#2196F3"  # Blue
            else:  # Equivalent
                icon = "🔄"
                color = "#9E9E9E"  # Gray
            
            expander_title = f"{icon} {top_metric}::{sub_label} - {winner}"
            
            with st.expander(expander_title):
                st.markdown(f"""
                <div style="
                    padding: 10px; 
                    border-left: 5px solid {color}; 
                    background-color: {color}10;
                ">
                    <h3>{winner}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Justification:** {entry['justification']}")
                
                # Show raw evaluation 
                show_raw = st.checkbox(f"Show Raw Evaluation", key=f"pairwise_raw_{top_metric}_{sub_label}")
                if show_raw:
                    st.text_area("Raw Evaluation", entry["raw_eval"], height=200, disabled=True)

def generate_evaluation_prompts(result, output, model: str):
    """Generate evaluation prompts for each selected metric, skipping the top-level
    'overall' prompt if a sub-metric name includes 'overall'."""
    eval_prompts = {}
    metrics = result["customized_metrics"]
    
    # Truncate prompt and input for brevity
    shortened_prompt = (
        result["input"]["prompt"][:500] + "..."
        if len(result["input"]["prompt"]) > 500
        else result["input"]["prompt"]
    )
    shortened_input = (
        result["input"]["sample_input"][:500] + "..."
        if len(result["input"]["sample_input"]) > 500
        else result["input"]["sample_input"]
    )
    
    # Add a note about the specific input-output relationship
    input_context = (
        "This is the specific input that generated this output. "
        "Please consider this relationship when evaluating."
    )
    
    for metric_name, metric in metrics.items():
        # Check if sub-metrics exist for this metric
        if "sub_metrics" in metric:
            # Check if any sub-metric name includes "overall"
            sub_metric_names = metric["sub_metrics"].keys()
            has_overall_sub = any("overall" in s for s in sub_metric_names)

            # Only create the top-level overall prompt if we do NOT already have an 'overall' sub-metric
            if not has_overall_sub:
                # Create an overall evaluation prompt for the metric
                overall_description = metric["customized_description"]
                overall_success_criteria = metric["success_criteria"]
                overall_scoring_rubric = metric["scoring_rubric"]
                
                if isinstance(overall_scoring_rubric, dict):
                    overall_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in overall_scoring_rubric.items()
                    )
                else:
                    overall_rubric_str = overall_scoring_rubric

                overall_prompt = f"""# {metric_name} Evaluation (Overall)

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{overall_description}

Success Criteria: {overall_success_criteria}

## Output to Evaluate
{output}

## Evaluation Instructions
Score this output on the {metric_name} metric (overall) using this rubric:
{overall_rubric_str}

Provide your score and detailed justification based SOLEY on this metric and NO OTHER METRIC.
"""
                eval_prompts[f"{metric_name}::overall"] = overall_prompt

            # Now create an evaluation prompt for each sub-metric
            for sub_name, sub_details in metric["sub_metrics"].items():
                sub_parameters = sub_details.get("parameters", "N/A")
                sub_success_criteria = sub_details.get("success_criteria", "N/A")
                sub_scoring_rubric = sub_details.get("scoring_rubric", "N/A")
                
                if isinstance(sub_scoring_rubric, dict):
                    sub_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in sub_scoring_rubric.items()
                    )
                else:
                    sub_rubric_str = sub_scoring_rubric

                sub_prompt = f"""# {metric_name} Evaluation - {sub_name}

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Sub-Metric Definition
Parameters: {sub_parameters}
Success Criteria: {sub_success_criteria}

## Output to Evaluate
{output}

## Evaluation Instructions
Score this output on the sub-metric '{sub_name}' using this rubric:
{sub_rubric_str}

Provide your score and detailed justification based solely on this sub-metric.
"""
                eval_prompts[f"{metric_name}::{sub_name}"] = sub_prompt
        else:
            # For metrics without sub-metrics, use the original single prompt approach
            description = metric["customized_description"]
            success_criteria = metric["success_criteria"]
            scoring_rubric = metric["scoring_rubric"]
            
            if isinstance(scoring_rubric, dict):
                rubric_str = "\n".join(f"{k}: {v}" for k, v in scoring_rubric.items())
            else:
                rubric_str = scoring_rubric

            prompt = f"""# {metric_name} Evaluation

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{description}

Success Criteria: {success_criteria}

## Output to Evaluate
{output}

## Evaluation Instructions
Score this output on the {metric_name} metric using this rubric:
{rubric_str}

Provide your score and detailed justification based solely on this metric of {metric_name} and no other metric!

For example:
* Never consider Empathy in your evaluation when the metric is Personalization. 
* Never consider Personalization in your evaluation when the metric is Coherence. 
* Never consider Y in your evaluation when the metric is X.
"""
            eval_prompts[metric_name] = prompt
    
    return eval_prompts

def validate_evaluation_prompts(eval_prompts, metric_exclusions):
    """Verify evaluation prompts properly enforce metric boundaries"""
    
    validated_prompts = {}
    
    for metric_name, prompt in eval_prompts.items():
        # Get the base metric (before ::)
        base_metric = metric_name.split("::")[0] if "::" in metric_name else metric_name
        
        # If this metric has exclusions, add them clearly to the prompt
        if base_metric in metric_exclusions:
            # Add a dedicated section about what NOT to evaluate
            exclusion_note = metric_exclusions[base_metric]
            
            # Insert before "Evaluation Instructions" section
            prompt_parts = prompt.split("## Evaluation Instructions")
            if len(prompt_parts) > 1:
                enhanced_prompt = (
                    prompt_parts[0] + 
                    f"\n## Evaluation Boundaries\n{exclusion_note}\n\n" +
                    "## Evaluation Instructions" + 
                    prompt_parts[1]
                )
                validated_prompts[metric_name] = enhanced_prompt
            else:
                # If structure is different, append to end of metric definition
                validated_prompts[metric_name] = prompt + f"\n\n## Evaluation Boundaries\n{exclusion_note}"
        else:
            validated_prompts[metric_name] = prompt
            
    return validated_prompts

def run_evaluations(client, result, output, model: str):
    """Run evaluations for each metric"""
    system_message = """You are an AI evaluation system focused on scoring outputs.
    Your purpose is to:
    1. Evaluate the given output against the specified metric
    2. Apply the scoring rubric rigorously and consistently
    3. Provide detailed justification for your score
    4. Focus only on the aspects relevant to the specific metric
    
    Provide precise scores and thorough justifications based solely on the defined metric.
    """
    
    eval_prompts = generate_evaluation_prompts(result, output, model)
    # Get metric exclusions from the prevention step
    metric_exclusions = {
        metric: details.get("exclusion_note", "") 
        for metric, details in result["customized_metrics"].items()
        if "exclusion_note" in details
}

    # Validate prompts to prevent overlap
    validated_prompts = validate_evaluation_prompts(eval_prompts, metric_exclusions)
    eval_results = {}
    
    for metric_name, prompt in validated_prompts.items():
        with st.spinner(f"Evaluating {metric_name}..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            eval_results[metric_name] = response.choices[0].message.content
    
    return eval_results

def format_evaluation_results(eval_results):
    """Format evaluation results into a readable structure"""
    import re
    
    formatted_results = {}
    
    for metric_name, result in eval_results.items():
        # Extract score
        score_match = re.search(r'Score:?\s*(\d+(?:\.\d+)?|[\d/]+)', result, re.IGNORECASE)
        score = score_match.group(1) if score_match else "N/A"
        
        # Extract justification
        just_match = re.search(r'Justification:?\s*(.*?)(?=\n\n|\Z)', result, re.IGNORECASE | re.DOTALL)
        justification = just_match.group(1).strip() if just_match else result
        
        formatted_results[metric_name] = {
            "score": score,
            "justification": justification
        }
    
    return formatted_results


def evaluate_single_metric(client, metric_name, prompt, system_message, model):
    """
    Evaluate a single metric for an input-output pair
    
    Args:
        client: OpenAI client
        metric_name: Name of the metric to evaluate
        prompt: Evaluation prompt for this metric
        system_message: System message for the evaluation
        model: Model to use for evaluations
        
    Returns:
        Tuple of (metric_name, evaluation_result)
    """
    # Implement retry logic with exponential backoff
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return metric_name, response.choices[0].message.content
        except Exception as e:
            # If this is the last attempt, raise the exception
            if attempt == max_retries - 1:
                raise
            
            # Otherwise, wait and retry
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    
    # We should never reach here, but just in case
    raise Exception(f"Failed to evaluate metric {metric_name} after {max_retries} attempts")

def generate_evaluation_prompts(result, output, model: str):
    """Generate evaluation prompts for each selected metric, skipping the top-level
    'overall' prompt if a sub-metric name includes 'overall'."""
    eval_prompts = {}
    metrics = result["customized_metrics"]
    
    # Truncate prompt and input for brevity
    shortened_prompt = (
        result["input"]["prompt"][:500] + "..."
        if len(result["input"]["prompt"]) > 500
        else result["input"]["prompt"]
    )
    shortened_input = (
        result["input"]["sample_input"][:500] + "..."
        if len(result["input"]["sample_input"]) > 500
        else result["input"]["sample_input"]
    )
    
    # Add a note about the specific input-output relationship
    input_context = (
        "This is the specific input that generated this output. "
        "Please consider this relationship when evaluating."
    )
    
    for metric_name, metric in metrics.items():
        # Check if sub-metrics exist for this metric
        if "sub_metrics" in metric:
            # Check if any sub-metric name includes "overall"
            sub_metric_names = metric["sub_metrics"].keys()
            has_overall_sub = any("overall" in s for s in sub_metric_names)

            # Only create the top-level overall prompt if we do NOT already have an 'overall' sub-metric
            if not has_overall_sub:
                # Create an overall evaluation prompt for the metric
                overall_description = metric["customized_description"]
                overall_success_criteria = metric["success_criteria"]
                overall_scoring_rubric = metric["scoring_rubric"]
                
                if isinstance(overall_scoring_rubric, dict):
                    overall_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in overall_scoring_rubric.items()
                    )
                else:
                    overall_rubric_str = overall_scoring_rubric

                overall_prompt = f"""# {metric_name} Evaluation (Overall)

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{overall_description}

Success Criteria: {overall_success_criteria}

## Output to Evaluate
{output}

## Evaluation Instructions
Score this output on the {metric_name} metric (overall) using this rubric:
{overall_rubric_str}

Provide your score and detailed justification based SOLEY on this metric and NO OTHER METRIC.
"""
                eval_prompts[f"{metric_name}::overall"] = overall_prompt

            # Now create an evaluation prompt for each sub-metric
            for sub_name, sub_details in metric["sub_metrics"].items():
                sub_parameters = sub_details.get("parameters", "N/A")
                sub_success_criteria = sub_details.get("success_criteria", "N/A")
                sub_scoring_rubric = sub_details.get("scoring_rubric", "N/A")
                
                if isinstance(sub_scoring_rubric, dict):
                    sub_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in sub_scoring_rubric.items()
                    )
                else:
                    sub_rubric_str = sub_scoring_rubric

                sub_prompt = f"""# {metric_name} Evaluation - {sub_name}

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Sub-Metric Definition
Parameters: {sub_parameters}
Success Criteria: {sub_success_criteria}

## Output to Evaluate
{output}

## Evaluation Instructions
Score this output on the sub-metric '{sub_name}' using this rubric:
{sub_rubric_str}

Provide your score and detailed justification based solely on this sub-metric.
"""
                eval_prompts[f"{metric_name}::{sub_name}"] = sub_prompt
        else:
            # For metrics without sub-metrics, use the original single prompt approach
            description = metric["customized_description"]
            success_criteria = metric["success_criteria"]
            scoring_rubric = metric["scoring_rubric"]
            
            if isinstance(scoring_rubric, dict):
                rubric_str = "\n".join(f"{k}: {v}" for k, v in scoring_rubric.items())
            else:
                rubric_str = scoring_rubric

            prompt = f"""# {metric_name} Evaluation

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{description}

Success Criteria: {success_criteria}

## Output to Evaluate
{output}

## Evaluation Instructions
Score this output on the {metric_name} metric using this rubric:
{rubric_str}

Provide your score and detailed justification based solely on this metric of {metric_name} and no other metric!

For example:
* Never consider Empathy in your evaluation when the metric is Personalization. 
* Never consider Personalization in your evaluation when the metric is Coherence. 
* Never consider Y in your evaluation when the metric is X.
"""
            eval_prompts[metric_name] = prompt
    
    return eval_prompts

def validate_evaluation_prompts(eval_prompts, metric_exclusions):
    """Verify evaluation prompts properly enforce metric boundaries"""
    
    validated_prompts = {}
    
    for metric_name, prompt in eval_prompts.items():
        # Get the base metric (before ::)
        base_metric = metric_name.split("::")[0] if "::" in metric_name else metric_name
        
        # If this metric has exclusions, add them clearly to the prompt
        if base_metric in metric_exclusions:
            # Add a dedicated section about what NOT to evaluate
            exclusion_note = metric_exclusions[base_metric]
            
            # Insert before "Evaluation Instructions" section
            prompt_parts = prompt.split("## Evaluation Instructions")
            if len(prompt_parts) > 1:
                enhanced_prompt = (
                    prompt_parts[0] + 
                    f"\n## Evaluation Boundaries\n{exclusion_note}\n\n" +
                    "## Evaluation Instructions" + 
                    prompt_parts[1]
                )
                validated_prompts[metric_name] = enhanced_prompt
            else:
                # If structure is different, append to end of metric definition
                validated_prompts[metric_name] = prompt + f"\n\n## Evaluation Boundaries\n{exclusion_note}"
        else:
            validated_prompts[metric_name] = prompt
            
    return validated_prompts

def format_evaluation_results(eval_results):
    """Format evaluation results into a readable structure"""
    import re
    
    formatted_results = {}
    
    for metric_name, result in eval_results.items():
        # Extract score
        score_match = re.search(r'Score:?\s*(\d+(?:\.\d+)?|[\d/]+)', result, re.IGNORECASE)
        score = score_match.group(1) if score_match else "N/A"
        
        # Extract justification
        just_match = re.search(r'Justification:?\s*(.*?)(?=\n\n|\Z)', result, re.IGNORECASE | re.DOTALL)
        justification = just_match.group(1).strip() if just_match else result
        
        formatted_results[metric_name] = {
            "score": score,
            "justification": justification
        }
    
    return formatted_results

def bulk_process_evaluations_parallel(client, result, input_output_pairs, eval_model, selected_metrics, progress_bar=None, max_workers=None):
    """
    Process multiple input-output pairs for evaluation with parallel metric evaluation
    
    Args:
        client: OpenAI client
        result: Pipeline results with customized metrics
        input_output_pairs: List of dicts with 'input' and 'output' keys
        eval_model: Model to use for evaluations
        selected_metrics: Dict of metrics to evaluate
        progress_bar: Optional Streamlit progress bar
        max_workers: Maximum number of worker threads (None = auto)
    
    Returns:
        List of evaluation results, one per input-output pair
    """
    all_results = []
    
    # Filter metrics based on selection
    filtered_result = result.copy()
    filtered_metrics = {k: v for k, v in result["customized_metrics"].items() 
                      if k in selected_metrics and selected_metrics[k]}
    filtered_result["customized_metrics"] = filtered_metrics
    
    total_pairs = len(input_output_pairs)
    start_time = time.time()
    
    for i, pair in enumerate(input_output_pairs):
        # Update progress
        if progress_bar:
            progress_bar.progress((i) / total_pairs, 
                                text=f"Processing item {i+1} of {total_pairs} - " + 
                                     f"Elapsed: {int(time.time() - start_time)}s")
        
        # Create a copy of the result with this specific input
        evaluation_copy = filtered_result.copy()
        evaluation_copy["input"] = evaluation_copy["input"].copy()
        evaluation_copy["input"]["sample_input"] = pair["input"]
        
        # Run evaluation for this pair (with parallel metric evaluation)
        eval_results = run_evaluations_parallel(
            client=client,
            result=evaluation_copy,
            output=pair["output"],
            model=eval_model,
            max_workers=max_workers
        )
        
        # Format results
        formatted_results = format_evaluation_results(eval_results)
        
        # Calculate time spent and estimate remaining time
        elapsed = time.time() - start_time
        items_per_second = (i + 1) / elapsed if elapsed > 0 else 0
        remaining_items = total_pairs - (i + 1)
        estimated_remaining = remaining_items / items_per_second if items_per_second > 0 else 0
        
        # Update progress with time estimates
        if progress_bar:
            progress_bar.progress(
                (i + 1) / total_pairs,
                text=f"Processed {i+1}/{total_pairs} - " +
                     f"Elapsed: {int(elapsed)}s - " +
                     f"Est. remaining: {int(estimated_remaining)}s"
            )
        
        # Add to collection with input and output for reference
        all_results.append({
            "input": pair["input"],
            "output": pair["output"],
            "evaluations": formatted_results,
            "raw_evaluations": eval_results
        })
    
    # Complete progress
    if progress_bar:
        total_time = time.time() - start_time
        progress_bar.progress(1.0, text=f"Processing complete! Total time: {int(total_time)}s")
    
    return all_results

def run_evaluations_parallel(client, result, output, model: str, max_workers=None):
    """Run evaluations for each metric in parallel"""
    system_message = """You are an AI evaluation system focused on scoring outputs.
    Your purpose is to:
    1. Evaluate the given output against the specified metric
    2. Apply the scoring rubric rigorously and consistently
    3. Provide detailed justification for your score
    4. Focus only on the aspects relevant to the specific metric
    
    Provide precise scores and thorough justifications based solely on the defined metric.
    """
    
    eval_prompts = generate_evaluation_prompts(result, output, model)
    # Get metric exclusions from the prevention step
    metric_exclusions = {
        metric: details.get("exclusion_note", "") 
        for metric, details in result["customized_metrics"].items()
        if "exclusion_note" in details
    }

    # Validate prompts to prevent overlap
    validated_prompts = validate_evaluation_prompts(eval_prompts, metric_exclusions)
    eval_results = {}
    
    # Create a partial function with fixed arguments
    evaluate_func = partial(
        evaluate_single_metric,
        client=client,
        system_message=system_message,
        model=model
    )
    
    # Create a list of (metric_name, prompt) tuples for the executor
    tasks = [(metric_name, prompt) for metric_name, prompt in validated_prompts.items()]
    
    # Use ThreadPoolExecutor to run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_metric = {
            executor.submit(evaluate_func, metric_name=metric_name, prompt=prompt): metric_name
            for metric_name, prompt in validated_prompts.items()
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_metric):
            try:
                metric_name, result_content = future.result()
                eval_results[metric_name] = result_content
            except Exception as e:
                metric_name = future_to_metric[future]
                eval_results[metric_name] = f"Error evaluating {metric_name}: {str(e)}"
    
    return eval_results

def bulk_process_evaluations(client, result, input_output_pairs, eval_model, selected_metrics, progress_bar=None):
    """
    Process multiple input-output pairs for evaluation
    
    Args:
        client: OpenAI client
        result: Pipeline results with customized metrics
        input_output_pairs: List of dicts with 'input' and 'output' keys
        eval_model: Model to use for evaluations
        selected_metrics: Dict of metrics to evaluate
        progress_bar: Optional Streamlit progress bar
    
    Returns:
        List of evaluation results, one per input-output pair
    """
    all_results = []
    
    # Filter metrics based on selection
    filtered_result = result.copy()
    filtered_metrics = {k: v for k, v in result["customized_metrics"].items() 
                      if k in selected_metrics and selected_metrics[k]}
    filtered_result["customized_metrics"] = filtered_metrics
    
    total_pairs = len(input_output_pairs)
    
    for i, pair in enumerate(input_output_pairs):
        # Update progress
        if progress_bar:
            progress_bar.progress((i) / total_pairs, text=f"Processing item {i+1} of {total_pairs}")
        
        # Create a copy of the result with this specific input
        evaluation_copy = filtered_result.copy()
        evaluation_copy["input"] = evaluation_copy["input"].copy()
        evaluation_copy["input"]["sample_input"] = pair["input"]
        
        # Run evaluation for this pair
        eval_results = run_evaluations(
            client=client,
            result=evaluation_copy,
            output=pair["output"],
            model=eval_model
        )
        
        # Format results
        formatted_results = format_evaluation_results(eval_results)
        
        # Add to collection with input and output for reference
        all_results.append({
            "input": pair["input"],
            "output": pair["output"],
            "evaluations": formatted_results,
            "raw_evaluations": eval_results
        })
    
    # Complete progress
    if progress_bar:
        progress_bar.progress(1.0, text="Processing complete!")
    
    return all_results

def calculate_aggregate_metrics(bulk_results):
    """
    Calculate aggregate statistics across all evaluated items
    
    Args:
        bulk_results: List of evaluation results from bulk_process_evaluations
        
    Returns:
        Dict with aggregate statistics
    """
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    
    # Extract all scores by metric
    all_scores = defaultdict(list)
    
    for result in bulk_results:
        for metric, data in result["evaluations"].items():
            try:
                # Try to convert score to float or int
                score = float(data["score"])
                all_scores[metric].append(score)
            except (ValueError, TypeError):
                # Skip non-numeric scores
                pass
    
    # Calculate statistics
    aggregates = {}
    for metric, scores in all_scores.items():
        if scores:
            df = pd.Series(scores)
            aggregates[metric] = {
                "mean": df.mean(),
                "median": df.median(),
                "min": df.min(),
                "max": df.max(),
                "std": df.std(),
                "count": len(scores),
                "distribution": np.histogram(scores, bins=3)[0].tolist()
            }
    
    return aggregates

def display_bulk_results(bulk_results, aggregates):
    """
    Display results from bulk processing with visualizations
    
    Args:
        bulk_results: List of evaluation results
        aggregates: Dict with aggregate statistics
    """
    import pandas as pd
    import altair as alt
    from collections import defaultdict
    
    st.header("Bulk Evaluation Results")
    
    # 1. Overall summary statistics
    st.subheader("Summary Statistics")
    
    # Convert aggregates to a dataframe for display
    summary_data = []
    for metric, stats in aggregates.items():
        summary_data.append({
            "Metric": metric,
            "Mean Score": f"{stats['mean']:.2f}",
            "Median": f"{stats['median']:.2f}",
            "Min": f"{stats['min']:.2f}",
            "Max": f"{stats['max']:.2f}",
            "Std Dev": f"{stats['std']:.2f}",
            "Count": stats['count']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # 2. Score distribution charts
    st.subheader("Score Distributions")
    
    # Group metrics by category
    metrics_by_category = defaultdict(list)
    for metric in aggregates.keys():
        category = metric.split("::")[0] if "::" in metric else metric
        metrics_by_category[category].append(metric)
    
    # Create distribution charts by category
    for category, metrics in metrics_by_category.items():
        with st.expander(f"{category} Metrics", expanded=True):
            # Prepare data for visualization
            all_scores = []
            
            for result in bulk_results:
                for metric in metrics:
                    if metric in result["evaluations"]:
                        try:
                            score = float(result["evaluations"][metric]["score"])
                            all_scores.append({
                                "Metric": metric.split("::")[-1] if "::" in metric else metric,
                                "Score": score
                            })
                        except (ValueError, TypeError):
                            pass
            
            if all_scores:
                df = pd.DataFrame(all_scores)
                
                # Create a histogram of scores
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Score:Q', bin=True, title='Score'),
                    y=alt.Y('count()', title='Count'),
                    color=alt.Color('Metric:N', legend=alt.Legend(orient='top')),
                    tooltip=['Metric', 'Score', 'count()']
                ).properties(
                    width=600,
                    height=300,
                    title=f"{category} Score Distribution"
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Show average scores
                avg_df = df.groupby('Metric')['Score'].mean().reset_index()
                avg_df['Score'] = avg_df['Score'].round(2)
                
                avg_chart = alt.Chart(avg_df).mark_bar().encode(
                    x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 2])),
                    y=alt.Y('Metric:N', sort='-x'),
                    color=alt.Color('Score:Q', scale=alt.Scale(domain=[0, 1, 2], range=['#f8696b', '#ffeb84', '#63be7b'])),
                    tooltip=['Metric', 'Score']
                ).properties(
                    width=600, 
                    height=min(len(avg_df) * 40, 300),
                    title="Average Scores"
                )
                
                st.altair_chart(avg_chart, use_container_width=True)
    
    # 3. Detailed results table with filtering
    st.subheader("Detailed Results")
    
    # Prepare flattened data for table view
    table_data = []
    for i, result in enumerate(bulk_results):
        # Find the overall metrics (not submetrics) for this item
        overall_scores = {}
        for metric, data in result["evaluations"].items():
            # If it's a top-level metric or an "overall" submetric
            if "::" not in metric or "::overall" in metric:
                base_metric = metric.split("::")[0] if "::" in metric else metric
                try:
                    score = float(data["score"])
                    overall_scores[base_metric] = score
                except (ValueError, TypeError):
                    overall_scores[base_metric] = data["score"]
        
        # Create a row with input, output, and all metrics
        row = {
            "Item": i + 1,
            "Input": result["input"][:100] + "..." if len(result["input"]) > 100 else result["input"],
            "Output": result["output"][:100] + "..." if len(result["output"]) > 100 else result["output"],
            **overall_scores
        }
        table_data.append(row)
    
    if table_data:
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True)
    
    # 4. Individual result inspection
    st.subheader("Inspect Individual Results")
    
    item_index = st.selectbox(
        "Select item to inspect",
        options=range(1, len(bulk_results) + 1),
        format_func=lambda x: f"Item {x}"
    )
    
    if item_index:
        selected_item = bulk_results[item_index - 1]
        
        st.write("### Input")
        st.text_area("", selected_item["input"], height=100, disabled=True)
        
        st.write("### Output")
        st.text_area("", selected_item["output"], height=150, disabled=True)
        
        st.write("### Evaluation Results")
        
        # Group results by top-level metric
        grouped_results = defaultdict(list)
        
        for metric_key, data in selected_item["evaluations"].items():
            # Parse out top-level metric vs. sub-metric
            if "::" in metric_key:
                top_metric, sub_metric = metric_key.split("::", 1)
            else:
                top_metric = metric_key
                sub_metric = None

            grouped_results[top_metric].append({
                "sub_metric": sub_metric,
                "score": data["score"],
                "justification": data["justification"],
                "raw_eval": selected_item["raw_evaluations"][metric_key],
            })
        
        # Display grouped results
        for top_metric, entries in grouped_results.items():
            with st.expander(f"{top_metric} Evaluation", expanded=True):
                # Sort so "overall" comes first
                entries.sort(key=lambda x: (
                    0 if x["sub_metric"] is None or x["sub_metric"] == "overall" else 1,
                    x["sub_metric"] or ""
                ))
                
                for entry in entries:
                    sub_label = entry["sub_metric"] if entry["sub_metric"] else "overall"
                    score = entry["score"]
                    
                    # Choose color based on score
                    try:
                        score_val = float(score)
                        if score_val == 0:
                            color = "#f8696b"  # Red
                        elif score_val == 1:
                            color = "#ffeb84"  # Yellow
                        else:
                            color = "#63be7b"  # Green
                    except (ValueError, TypeError):
                        color = "#9E9E9E"  # Gray for non-numeric
                    
                    st.markdown(f"""
                    <div style="
                        padding: 10px;
                        border-left: 5px solid {color};
                        background-color: {color}10;
                        margin-bottom: 10px;
                    ">
                        <div style="display: flex; align-items: center;">
                            <div style="
                                background-color: {color};
                                color: white;
                                padding: 8px 16px;
                                border-radius: 15px;
                                font-weight: bold;
                                margin-right: 10px;
                            ">{score}</div>
                            <h3 style="margin: 0;">{sub_label}</h3>
                        </div>
                        <p style="margin-top: 10px;"><strong>Justification:</strong> {entry["justification"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show raw evaluation in a collapsible section
                    show_raw = st.checkbox(f"Show Raw Evaluation", key=f"bulk_raw_{top_metric}_{sub_label}")
                    if show_raw:
                        st.text_area("Raw Evaluation", entry["raw_eval"], height=200, disabled=True)

def parse_uploaded_csv(uploaded_file, input_column, output_column):
    """
    Parse uploaded CSV file and extract input-output pairs
    
    Args:
        uploaded_file: Streamlit uploaded file object
        input_column: Name of the column containing inputs
        output_column: Name of the column containing outputs
        
    Returns:
        List of dicts with 'input' and 'output' keys
    """
    import pandas as pd
    import io
    
    # Read CSV
    try:
        # Try to detect encoding automatically
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # If auto-detection fails, try with different encodings
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
    
    # Validate columns
    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in CSV")
    if output_column not in df.columns:
        raise ValueError(f"Output column '{output_column}' not found in CSV")
    
    # Extract input-output pairs
    pairs = []
    for _, row in df.iterrows():
        # Skip rows with missing values
        if pd.isna(row[input_column]) or pd.isna(row[output_column]):
            continue
            
        pairs.append({
            "input": str(row[input_column]),
            "output": str(row[output_column])
        })
    
    return pairs


# 1. Update the generate_pairwise_evaluation_prompts function to handle submetrics

def generate_pairwise_evaluation_prompts(result, output_a, output_b, model: str):
    """Generate pairwise evaluation prompts for each selected metric, including submetrics"""
    eval_prompts = {}
    metrics = result["customized_metrics"]
    
    # Truncate prompt and input for brevity
    shortened_prompt = result["input"]["prompt"][:500] + "..." if len(result["input"]["prompt"]) > 500 else result["input"]["prompt"]
    shortened_input = result["input"]["sample_input"][:500] + "..." if len(result["input"]["sample_input"]) > 500 else result["input"]["sample_input"]
    
    # Add a note about the specific input-output relationship
    input_context = (
        "This is the specific input that generated these outputs. "
        "Please consider this relationship when evaluating."
    )
    
    for metric_name, metric in metrics.items():
        # Check if sub-metrics exist for this metric
        if "sub_metrics" in metric:
            # Check if any sub-metric name includes "overall"
            sub_metric_names = metric["sub_metrics"].keys()
            has_overall_sub = any("overall" in s for s in sub_metric_names)

            # Only create the top-level overall prompt if we do NOT already have an 'overall' sub-metric
            if not has_overall_sub:
                # Create an overall evaluation prompt for the metric
                overall_description = metric["customized_description"]
                overall_success_criteria = metric["success_criteria"]
                overall_scoring_rubric = metric["scoring_rubric"]
                
                if isinstance(overall_scoring_rubric, dict):
                    overall_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in overall_scoring_rubric.items()
                    )
                else:
                    overall_rubric_str = overall_scoring_rubric

                overall_prompt = f"""# {metric_name} Comparative Evaluation (Overall)

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{overall_description}

Success Criteria: {overall_success_criteria}

## Outputs to Compare
Output A: {output_a}
Output B: {output_b}

## Evaluation Instructions
Compare these outputs on the {metric_name} metric (overall) using this rubric:
{overall_rubric_str}

Which output better satisfies this metric? Respond with either "A is better", "B is better", or "Equivalent" if they are too similar to meaningfully distinguish.

Don't be afraid to grade them as "Equivalent" when the delta is truly very small or negligible, it's much better to lean in that direction than to overrepresent one output as a winner.

Provide detailed justification for your comparison, focusing solely on this metric.
"""
                eval_prompts[f"{metric_name}::overall"] = overall_prompt

            # Now create an evaluation prompt for each sub-metric
            for sub_name, sub_details in metric["sub_metrics"].items():
                sub_parameters = sub_details.get("parameters", "N/A")
                sub_success_criteria = sub_details.get("success_criteria", "N/A")
                sub_scoring_rubric = sub_details.get("scoring_rubric", "N/A")
                
                if isinstance(sub_scoring_rubric, dict):
                    sub_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in sub_scoring_rubric.items()
                    )
                else:
                    sub_rubric_str = sub_scoring_rubric

                sub_prompt = f"""# {metric_name} Comparative Evaluation - {sub_name}

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Sub-Metric Definition
Parameters: {sub_parameters}
Success Criteria: {sub_success_criteria}

## Outputs to Compare
Output A: {output_a}
Output B: {output_b}

## Evaluation Instructions
Compare these outputs on the sub-metric '{sub_name}' using this rubric:
{sub_rubric_str}

Which output better satisfies this sub-metric? Respond with either "A is better", "B is better", or "Equivalent" if they are too similar to meaningfully distinguish.

Don't be afraid to grade them as "Equivalent" when the delta is truly very small or negligible, it's much better to lean in that direction than to overrepresent one output as a winner.

Provide detailed justification for your comparison, focusing solely on this sub-metric.
"""
                eval_prompts[f"{metric_name}::{sub_name}"] = sub_prompt
        else:
            # For metrics without sub-metrics, use the original approach
            description = metric["customized_description"]
            success_criteria = metric["success_criteria"]
            scoring_rubric = metric["scoring_rubric"]
            
            if isinstance(scoring_rubric, dict):
                rubric_str = "\n".join(f"{k}: {v}" for k, v in scoring_rubric.items())
            else:
                rubric_str = scoring_rubric

            prompt = f"""# {metric_name} Comparative Evaluation

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{description}

Success Criteria: {success_criteria}

## Outputs to Compare
Output A: {output_a}
Output B: {output_b}

## Evaluation Instructions
Compare these outputs on the {metric_name} metric using this rubric:
{rubric_str}

Which output better satisfies this metric? Respond with either "A is better", "B is better", or "Equivalent" if they are too similar to meaningfully distinguish.

Don't be afraid to grade them as "Equivalent" when the delta is truly very small or negligible, it's much better to lean in that direction than to overrepresent one output as a winner.

Provide detailed justification for your comparison, focusing solely on this metric.
"""
            eval_prompts[metric_name] = prompt
    
    return eval_prompts

def run_pairwise_evaluations(client, result, output_a, output_b, model: str):
    """Run pairwise evaluations for each metric"""
    system_message = """You are an AI evaluation system focused on comparing two outputs.
    Your purpose is to:
    1. Evaluate which of the two outputs better satisfies the specified metric
    2. Apply the scoring rubric rigorously and consistently
    3. Provide detailed justification for your decision
    4. Focus only on the aspects relevant to the specific metric
    
    For each comparison, clearly state which output is better (or if they are Equivalent) and explain why.
    """
    
    eval_prompts = generate_pairwise_evaluation_prompts(result, output_a, output_b, model)
    eval_results = {}
    
    for metric_name, prompt in eval_prompts.items():
        with st.spinner(f"Comparing outputs on {metric_name}..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            eval_results[metric_name] = response.choices[0].message.content
    
    return eval_results

def format_pairwise_evaluation_results(eval_results):
    """Format pairwise evaluation results into a readable structure with support for submetrics"""
    import re
    
    formatted_results = {}
    
    # Count wins for overall winner determination
    a_wins = 0
    b_wins = 0
    ties = 0
    
    # Dictionary to track wins by metric category
    category_wins = {}
    
    for metric_name, result in eval_results.items():
        # Extract winner (A, B, or Equivalent)
        winner_match = re.search(r'(A is better|B is better|Equivalent)', result, re.IGNORECASE)
        winner = winner_match.group(1) if winner_match else "Unclear"
        
        # Get the base metric (before ::)
        base_metric = metric_name.split("::")[0] if "::" in metric_name else metric_name
        
        # Initialize category if not present
        if base_metric not in category_wins:
            category_wins[base_metric] = {"A": 0, "B": 0, "Tie": 0}
        
        # Count for overall winner and by category
        if "A is better" in winner:
            if "overall" in metric_name or "::" not in metric_name:
                a_wins += 1
            category_wins[base_metric]["A"] += 1
        elif "B is better" in winner:
            if "overall" in metric_name or "::" not in metric_name:
                b_wins += 1
            category_wins[base_metric]["B"] += 1
        elif "Equivalent" in winner:
            if "overall" in metric_name or "::" not in metric_name:
                ties += 1
            category_wins[base_metric]["Tie"] += 1
            
        # Extract justification
        justification = result
        if winner_match:
            # Remove the "A is better" etc. from the beginning to get just the justification
            justification_parts = result.split(winner_match.group(1), 1)
            if len(justification_parts) > 1:
                justification = justification_parts[1].strip()
        
        formatted_results[metric_name] = {
            "winner": winner,
            "justification": justification
        }
    
    # Determine category winners
    category_results = {}
    for category, scores in category_wins.items():
        if scores["A"] > scores["B"]:
            winner = "A"
        elif scores["B"] > scores["A"]:
            winner = "B"
        else:
            winner = "Tie"
        
        category_results[category] = {
            "winner": winner,
            "summary": f"A won {scores['A']}, B won {scores['B']}, and {scores['Tie']} were tied."
        }
    
    # Add category results to formatted results
    formatted_results["CATEGORIES"] = category_results
    
    # Determine overall winner
    if a_wins > b_wins:
        overall_winner = "A"
    elif b_wins > a_wins:
        overall_winner = "B"
    else:
        overall_winner = "Tie"
    
    formatted_results["OVERALL"] = {
        "winner": overall_winner,
        "summary": f"A won {a_wins} metrics, B won {b_wins} metrics, and {ties} metrics were tied."
    }
    
    return formatted_results

def create_metric_selection_ui(metrics_library, key_prefix=""):
    """
    Create a hierarchical metric selection UI that allows selecting/deselecting
    individual submetrics
    
    Args:
        metrics_library: Dictionary of metrics from pipeline results
        key_prefix: Prefix for session state keys to avoid conflicts
        
    Returns:
        Dictionary mapping metric and submetric keys to boolean selection status
    """
    import streamlit as st
    from collections import defaultdict
    
    # Initialize selection dictionary
    selections = {}
    
    # Create expandable section for metric selection
    with st.expander("Select Metrics and Submetrics", expanded=True):
        st.info("Select the metrics and submetrics to include in your evaluation. You can expand each metric to select individual submetrics.")
        
        # For each metric in the library
        for metric_name, metric_details in metrics_library.items():
            # Create a column for this metric
            col1, col2 = st.columns([0.8, 0.2])
            
            with col1:
                # Add the metric name as a header
                st.markdown(f"**{metric_name}**")
                
            with col2:
                # Add a checkbox for the entire metric
                metric_selected = st.checkbox(
                    "Select All", 
                    value=True, 
                    key=f"{key_prefix}_{metric_name}",
                    help=f"Select/deselect all submetrics for {metric_name}"
                )
            
            # Store the selection status for the overall metric
            selections[metric_name] = metric_selected
            
            # Check if this metric has submetrics
            if "sub_metrics" in metric_details:
                # Create an indented section for submetrics
                with st.container():
                    st.markdown("<div style='margin-left: 20px;'>", unsafe_allow_html=True)
                    
                    # For each submetric
                    for sub_name in metric_details["sub_metrics"].keys():
                        # Generate a friendly display name
                        display_name = sub_name.replace(f"{metric_name.lower()}.", "").replace("_", " ").title()
                        
                        # Create a unique key for this submetric
                        submetric_key = f"{metric_name}::{sub_name}"
                        
                        # Create a checkbox for this submetric
                        submetric_selected = st.checkbox(
                            display_name,
                            value=metric_selected,  # Initially match the parent metric's selection
                            key=f"{key_prefix}_sub_{metric_name}_{sub_name}",
                            help=f"Include {display_name} in evaluation"
                        )
                        
                        # Store the selection status for this submetric
                        selections[submetric_key] = submetric_selected
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a separator between metrics
            st.markdown("---")
    
    return selections

def filter_metrics_based_on_selection(customized_metrics, selected_metrics):
    """
    Filter the metrics dictionary based on user selections
    
    Args:
        customized_metrics: Dictionary of metrics from pipeline results
        selected_metrics: Dictionary of selected metrics and submetrics
        
    Returns:
        Filtered metrics dictionary
    """
    # Create a copy to avoid modifying the original
    filtered_metrics = {}
    
    # For each metric in the original dictionary
    for metric_name, metric_details in customized_metrics.items():
        # Check if this metric is selected
        if metric_name in selected_metrics and selected_metrics[metric_name]:
            # Create a copy of this metric
            metric_copy = metric_details.copy()
            
            # Check if this metric has submetrics
            if "sub_metrics" in metric_copy:
                # Create a filtered submetrics dictionary
                filtered_submetrics = {}
                
                # For each submetric
                for sub_name, sub_details in metric_copy["sub_metrics"].items():
                    # Create the submetric key
                    submetric_key = f"{metric_name}::{sub_name}"
                    
                    # Check if this submetric is selected
                    if submetric_key in selected_metrics and selected_metrics[submetric_key]:
                        # Add this submetric to the filtered list
                        filtered_submetrics[sub_name] = sub_details
                
                # If no submetrics are selected, skip this metric
                if not filtered_submetrics:
                    continue
                    
                # Otherwise, update the submetrics
                metric_copy["sub_metrics"] = filtered_submetrics
            
            # Add this metric to the filtered dictionary
            filtered_metrics[metric_name] = metric_copy
    
    return filtered_metrics
def bulk_process_pairwise_evaluations(client, result, input_output_triplets, eval_model, selected_metrics, progress_bar=None, max_workers=None):
    """
    Process multiple input-output triplets for pairwise evaluation
    
    Args:
        client: OpenAI client
        result: Pipeline results with customized metrics
        input_output_triplets: List of dicts with 'input', 'output_a', and 'output_b' keys
        eval_model: Model to use for evaluations
        selected_metrics: Dict of metrics to evaluate
        progress_bar: Optional Streamlit progress bar
        max_workers: Maximum number of worker threads (None = auto)
    
    Returns:
        List of pairwise evaluation results, one per input-output triplet
    """
    all_results = []
    
    # Filter metrics based on selection
    filtered_result = result.copy()
    filtered_metrics = {k: v for k, v in result["customized_metrics"].items() 
                      if k in selected_metrics and selected_metrics[k]}
    filtered_result["customized_metrics"] = filtered_metrics
    
    total_triplets = len(input_output_triplets)
    start_time = time.time()
    
    for i, triplet in enumerate(input_output_triplets):
        # Update progress
        if progress_bar:
            progress_bar.progress((i) / total_triplets, 
                                text=f"Processing comparison {i+1} of {total_triplets} - " + 
                                     f"Elapsed: {int(time.time() - start_time)}s")
        
        # Create a copy of the result with this specific input
        evaluation_copy = filtered_result.copy()
        evaluation_copy["input"] = evaluation_copy["input"].copy()
        evaluation_copy["input"]["sample_input"] = triplet["input"]
        
        # Run pairwise evaluation for this triplet (with parallel metric evaluation)
        pairwise_results = run_pairwise_evaluations_parallel(
            client=client,
            result=evaluation_copy,
            output_a=triplet["output_a"],
            output_b=triplet["output_b"],
            model=eval_model,
            max_workers=max_workers
        )
        
        # Format results
        formatted_results = format_pairwise_evaluation_results(pairwise_results)
        
        # Calculate time spent and estimate remaining time
        elapsed = time.time() - start_time
        items_per_second = (i + 1) / elapsed if elapsed > 0 else 0
        remaining_items = total_triplets - (i + 1)
        estimated_remaining = remaining_items / items_per_second if items_per_second > 0 else 0
        
        # Update progress with time estimates
        if progress_bar:
            progress_bar.progress(
                (i + 1) / total_triplets,
                text=f"Processed {i+1}/{total_triplets} - " +
                     f"Elapsed: {int(elapsed)}s - " +
                     f"Est. remaining: {int(estimated_remaining)}s"
            )
        
        # Add to collection with input and outputs for reference
        all_results.append({
            "input": triplet["input"],
            "output_a": triplet["output_a"],
            "output_b": triplet["output_b"],
            "evaluations": formatted_results,
            "raw_evaluations": pairwise_results
        })
    
    # Complete progress
    if progress_bar:
        total_time = time.time() - start_time
        progress_bar.progress(1.0, text=f"Processing complete! Total time: {int(total_time)}s")
    
    return all_results

def calculate_aggregate_pairwise_metrics(bulk_results):
    """
    Calculate aggregate statistics across all pairwise evaluations
    
    Args:
        bulk_results: List of pairwise evaluation results
        
    Returns:
        Dict with aggregate statistics
    """
    from collections import defaultdict
    
    # Track overall and per-metric wins
    a_wins = 0
    b_wins = 0
    ties = 0
    
    # Track per-metric results
    metric_results = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0})
    
    # Track per-item results
    item_results = []
    
    for i, result in enumerate(bulk_results):
        # Get overall winner for this item
        overall = result["evaluations"].get("OVERALL", {})
        winner = overall.get("winner", "Tie")
        
        item_winner = None
        if winner == "A":
            a_wins += 1
            item_winner = "A"
        elif winner == "B":
            b_wins += 1
            item_winner = "B"
        else:
            ties += 1
            item_winner = "Tie"
            
        # Add to item results
        item_results.append({
            "item": i + 1,
            "winner": item_winner,
            "summary": overall.get("summary", "")
        })
        
        # Process individual metrics
        for metric_name, data in result["evaluations"].items():
            # Skip special keys
            if metric_name in ["OVERALL", "CATEGORIES"]:
                continue
                
            # Get winner for this metric
            metric_winner = data.get("winner", "")
            
            if "A is better" in metric_winner:
                metric_results[metric_name]["a_wins"] += 1
            elif "B is better" in metric_winner:
                metric_results[metric_name]["b_wins"] += 1
            else:  # Equivalent or unclear
                metric_results[metric_name]["ties"] += 1
    
    return {
        "overall": {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "total": len(bulk_results)
        },
        "metrics": metric_results,
        "items": item_results
    }

def display_grouped_pairwise_results(formatted_results, pairwise_results):
    """Display pairwise results grouped by metric category"""
    import streamlit as st
    from collections import defaultdict
    
    # Remove special keys
    display_results = {k: v for k, v in formatted_results.items() 
                      if k not in ["OVERALL", "CATEGORIES"]}
    
    # Group by top-level metric
    grouped_results = defaultdict(list)
    for metric_key, data in display_results.items():
        if "::" in metric_key:
            top_metric, sub_metric = metric_key.split("::", 1)
        else:
            top_metric = metric_key
            sub_metric = None
            
        grouped_results[top_metric].append({
            "sub_metric": sub_metric,
            "winner": data["winner"],
            "justification": data["justification"],
            "raw_eval": pairwise_results[metric_key]
        })
    
    # Display category results
    category_results = formatted_results.get("CATEGORIES", {})
    
    # Create visual summary of category winners
    categories = list(category_results.keys())
    if categories:
        st.subheader("Results by Category")
        
        for category, result in category_results.items():
            winner = result["winner"]
            summary = result["summary"]
            
            # Choose color based on winner
            color = "#4CAF50" if winner == "A" else "#2196F3" if winner == "B" else "#9E9E9E"  # Green for A, Blue for B, Gray for Tie
            
            st.markdown(f"""
            <div style="
                padding: 10px; 
                border-left: 5px solid {color}; 
                background-color: {color}10;
                margin-bottom: 10px;
            ">
                <h3>{category}: Output {winner} is better overall</h3>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display detailed results by category
    for top_metric, entries in grouped_results.items():
        st.subheader(f"{top_metric} Metrics")
        
        # Sort entries so "overall" comes first
        entries.sort(key=lambda x: (
            0 if x["sub_metric"] is None or x["sub_metric"] == "overall" else 1,
            x["sub_metric"] or ""
        ))
        
        for entry in entries:
            sub_label = entry["sub_metric"] if entry["sub_metric"] else "overall"
            winner = entry["winner"]
            
            # Choose icon and color based on winner
            if "A is better" in winner:
                icon = "🅰️"
                color = "#4CAF50"  # Green
            elif "B is better" in winner:
                icon = "🅱️"
                color = "#2196F3"  # Blue
            else:  # Equivalent
                icon = "🔄"
                color = "#9E9E9E"  # Gray
            
            expander_title = f"{icon} {top_metric}::{sub_label} - {winner}"
            
            with st.expander(expander_title):
                st.markdown(f"""
                <div style="
                    padding: 10px; 
                    border-left: 5px solid {color}; 
                    background-color: {color}10;
                ">
                    <h3>{winner}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Justification:** {entry['justification']}")
                
                # Show raw evaluation 
                show_raw = st.checkbox(f"Show Raw Evaluation", key=f"pairwise_raw_{top_metric}_{sub_label}")
                if show_raw:
                    st.text_area("Raw Evaluation", entry["raw_eval"], height=200, disabled=True)
def parse_pairwise_csv(uploaded_file, input_column, output_a_column, output_b_column):
    """
    Parse uploaded CSV file and extract input with two outputs for pairwise comparison
    
    Args:
        uploaded_file: Streamlit uploaded file object
        input_column: Name of the column containing inputs
        output_a_column: Name of the column containing first outputs (A)
        output_b_column: Name of the column containing second outputs (B)
        
    Returns:
        List of dicts with 'input', 'output_a', and 'output_b' keys
    """
    import pandas as pd
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Read CSV
    try:
        # Try to detect encoding automatically
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # If auto-detection fails, try with different encodings
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
    
    # Validate columns
    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in CSV")
    if output_a_column not in df.columns:
        raise ValueError(f"Output A column '{output_a_column}' not found in CSV")
    if output_b_column not in df.columns:
        raise ValueError(f"Output B column '{output_b_column}' not found in CSV")
    
    # Extract input-output pairs
    triplets = []
    for _, row in df.iterrows():
        # Skip rows with missing values
        if pd.isna(row[input_column]) or pd.isna(row[output_a_column]) or pd.isna(row[output_b_column]):
            continue
            
        triplets.append({
            "input": str(row[input_column]),
            "output_a": str(row[output_a_column]),
            "output_b": str(row[output_b_column])
        })
    
    return triplets

def run_pairwise_evaluations_parallel(client, result, output_a, output_b, model: str, max_workers=None):
    """Run pairwise evaluations for each metric in parallel"""
    system_message = """You are an AI evaluation system focused on comparing two outputs.
    Your purpose is to:
    1. Evaluate which of the two outputs better satisfies the specified metric
    2. Apply the scoring rubric rigorously and consistently
    3. Provide detailed justification for your decision
    4. Focus only on the aspects relevant to the specific metric
    
    For each comparison, clearly state which output is better (or if they are Equivalent) and explain why.
    """
    
    eval_prompts = generate_pairwise_evaluation_prompts(result, output_a, output_b, model)
    eval_results = {}
    
    # Create a partial function with fixed arguments
    evaluate_func = partial(
        evaluate_single_metric,
        client=client,
        system_message=system_message,
        model=model
    )
    
    # Use ThreadPoolExecutor to run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_metric = {
            executor.submit(evaluate_func, metric_name=metric_name, prompt=prompt): metric_name
            for metric_name, prompt in eval_prompts.items()
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_metric):
            try:
                metric_name, result_content = future.result()
                eval_results[metric_name] = result_content
            except Exception as e:
                metric_name = future_to_metric[future]
                eval_results[metric_name] = f"Error evaluating {metric_name}: {str(e)}"
    
    return eval_results

def generate_pairwise_evaluation_prompts(result, output_a, output_b, model: str):
    """Generate pairwise evaluation prompts for each selected metric, including submetrics"""
    eval_prompts = {}
    metrics = result["customized_metrics"]
    
    # Truncate prompt and input for brevity
    shortened_prompt = result["input"]["prompt"][:500] + "..." if len(result["input"]["prompt"]) > 500 else result["input"]["prompt"]
    shortened_input = result["input"]["sample_input"][:500] + "..." if len(result["input"]["sample_input"]) > 500 else result["input"]["sample_input"]
    
    # Add a note about the specific input-output relationship
    input_context = (
        "This is the specific input that generated these outputs. "
        "Please consider this relationship when evaluating."
    )
    
    for metric_name, metric in metrics.items():
        # Check if sub-metrics exist for this metric
        if "sub_metrics" in metric:
            # Check if any sub-metric name includes "overall"
            sub_metric_names = metric["sub_metrics"].keys()
            has_overall_sub = any("overall" in s for s in sub_metric_names)

            # Only create the top-level overall prompt if we do NOT already have an 'overall' sub-metric
            if not has_overall_sub:
                # Create an overall evaluation prompt for the metric
                overall_description = metric["customized_description"]
                overall_success_criteria = metric["success_criteria"]
                overall_scoring_rubric = metric["scoring_rubric"]
                
                if isinstance(overall_scoring_rubric, dict):
                    overall_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in overall_scoring_rubric.items()
                    )
                else:
                    overall_rubric_str = overall_scoring_rubric

                overall_prompt = f"""# {metric_name} Comparative Evaluation (Overall)

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{overall_description}

Success Criteria: {overall_success_criteria}

## Outputs to Compare
Output A: {output_a}
Output B: {output_b}

## Evaluation Instructions
Compare these outputs on the {metric_name} metric (overall) using this rubric:
{overall_rubric_str}

Which output better satisfies this metric? Respond with either "A is better", "B is better", or "Equivalent" if they are too similar to meaningfully distinguish.

Don't be afraid to grade them as "Equivalent" when the delta is truly very small or negligible, it's much better to lean in that direction than to overrepresent one output as a winner.

Provide detailed justification for your comparison, focusing solely on this metric.
"""
                eval_prompts[f"{metric_name}::overall"] = overall_prompt

            # Now create an evaluation prompt for each sub-metric
            for sub_name, sub_details in metric["sub_metrics"].items():
                sub_parameters = sub_details.get("parameters", "N/A")
                sub_success_criteria = sub_details.get("success_criteria", "N/A")
                sub_scoring_rubric = sub_details.get("scoring_rubric", "N/A")
                
                if isinstance(sub_scoring_rubric, dict):
                    sub_rubric_str = "\n".join(
                        f"{k}: {v}" for k, v in sub_scoring_rubric.items()
                    )
                else:
                    sub_rubric_str = sub_scoring_rubric

                sub_prompt = f"""# {metric_name} Comparative Evaluation - {sub_name}

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Sub-Metric Definition
Parameters: {sub_parameters}
Success Criteria: {sub_success_criteria}

## Outputs to Compare
Output A: {output_a}
Output B: {output_b}

## Evaluation Instructions
Compare these outputs on the sub-metric '{sub_name}' using this rubric:
{sub_rubric_str}

Which output better satisfies this sub-metric? Respond with either "A is better", "B is better", or "Equivalent" if they are too similar to meaningfully distinguish.

Don't be afraid to grade them as "Equivalent" when the delta is truly very small or negligible, it's much better to lean in that direction than to overrepresent one output as a winner.

Provide detailed justification for your comparison, focusing solely on this sub-metric.
"""
                eval_prompts[f"{metric_name}::{sub_name}"] = sub_prompt
        else:
            # For metrics without sub-metrics, use the original approach
            description = metric["customized_description"]
            success_criteria = metric["success_criteria"]
            scoring_rubric = metric["scoring_rubric"]
            
            if isinstance(scoring_rubric, dict):
                rubric_str = "\n".join(f"{k}: {v}" for k, v in scoring_rubric.items())
            else:
                rubric_str = scoring_rubric

            prompt = f"""# {metric_name} Comparative Evaluation

## Context
Task: {result["input"]["task_summary"]}
Prompt: {shortened_prompt}
Input: {shortened_input}
{input_context}

## Metric Definition
{description}

Success Criteria: {success_criteria}

## Outputs to Compare
Output A: {output_a}
Output B: {output_b}

## Evaluation Instructions
Compare these outputs on the {metric_name} metric using this rubric:
{rubric_str}

Which output better satisfies this metric? Respond with either "A is better", "B is better", or "Equivalent" if they are too similar to meaningfully distinguish.

Don't be afraid to grade them as "Equivalent" when the delta is truly very small or negligible, it's much better to lean in that direction than to overrepresent one output as a winner.

Provide detailed justification for your comparison, focusing solely on this metric.
"""
            eval_prompts[metric_name] = prompt
    
    return eval_prompts

def display_bulk_pairwise_results(bulk_results, aggregates):
    """
    Display results from bulk pairwise processing with visualizations
    
    Args:
        bulk_results: List of pairwise evaluation results
        aggregates: Dict with aggregate statistics
    """
    import streamlit as st
    import pandas as pd
    import altair as alt
    from collections import defaultdict
    
    st.header("Bulk Pairwise Evaluation Results")
    
    # 1. Overall summary statistics
    st.subheader("Summary Statistics")
    
    overall = aggregates["overall"]
    total = overall["total"]
    
    # Calculate percentages
    a_percent = (overall["a_wins"] / total) * 100 if total > 0 else 0
    b_percent = (overall["b_wins"] / total) * 100 if total > 0 else 0
    tie_percent = (overall["ties"] / total) * 100 if total > 0 else 0
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Output A Wins", f"{overall['a_wins']} ({a_percent:.1f}%)")
    
    with col2:
        st.metric("Output B Wins", f"{overall['b_wins']} ({b_percent:.1f}%)")
        
    with col3:
        st.metric("Ties", f"{overall['ties']} ({tie_percent:.1f}%)")
    
    # Create a pie chart of overall results
    overall_data = pd.DataFrame([
        {"category": "Output A Wins", "value": overall["a_wins"]},
        {"category": "Output B Wins", "value": overall["b_wins"]},
        {"category": "Ties", "value": overall["ties"]}
    ])
    
    pie_chart = alt.Chart(overall_data).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="category", type="nominal", 
                       scale=alt.Scale(domain=["Output A Wins", "Output B Wins", "Ties"],
                                      range=["#4CAF50", "#2196F3", "#9E9E9E"])),
        tooltip=["category", "value"]
    ).properties(
        width=400,
        height=300,
        title="Overall Results Distribution"
    )
    
    st.altair_chart(pie_chart, use_container_width=True)
    
    # 2. Results by metric
    st.subheader("Results by Metric")
    
    # Prepare data for visualization
    metrics_data = []
    for metric_name, results in aggregates["metrics"].items():
        total_metric = results["a_wins"] + results["b_wins"] + results["ties"]
        if total_metric > 0:
            metrics_data.append({
                "Metric": metric_name,
                "Output A Wins": results["a_wins"],
                "Output B Wins": results["b_wins"],
                "Ties": results["ties"],
                "A Win %": (results["a_wins"] / total_metric) * 100,
                "B Win %": (results["b_wins"] / total_metric) * 100,
                "Tie %": (results["ties"] / total_metric) * 100
            })
    
    if metrics_data:
        # Create a stacked bar chart
        metrics_df = pd.DataFrame(metrics_data)
        
        # Melt the dataframe for visualization
        melted_df = pd.melt(
            metrics_df, 
            id_vars=["Metric"], 
            value_vars=["Output A Wins", "Output B Wins", "Ties"],
            var_name="Result",
            value_name="Count"
        )
        
        # Create stacked bar chart
        chart = alt.Chart(melted_df).mark_bar().encode(
            x=alt.X("Count:Q", title="Number of Comparisons"),
            y=alt.Y("Metric:N", title=""),
            color=alt.Color("Result:N", scale=alt.Scale(
                domain=["Output A Wins", "Output B Wins", "Ties"],
                range=["#4CAF50", "#2196F3", "#9E9E9E"]
            )),
            tooltip=["Metric", "Result", "Count"]
        ).properties(
            width=600,
            height=min(len(metrics_data) * 40, 400),
            title="Results by Metric"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Display table with percentages
        st.write("Detailed Metrics Breakdown:")
        display_cols = ["Metric", "Output A Wins", "Output B Wins", "Ties", "A Win %", "B Win %", "Tie %"]
        st.dataframe(metrics_df[display_cols].sort_values("A Win %", ascending=False), use_container_width=True)
    
    # 3. Item-level results
    st.subheader("Results by Item")
    
    if aggregates["items"]:
        items_df = pd.DataFrame(aggregates["items"])
        
        # Create a color mapping for winners
        items_df["color"] = items_df["winner"].map({
            "A": "#4CAF50",
            "B": "#2196F3",
            "Tie": "#9E9E9E"
        })
        
        # Create a chart showing winners by item
        chart = alt.Chart(items_df).mark_bar().encode(
            x=alt.X("item:O", title="Item"),
            y=alt.Y("count():Q", title="Count", axis=alt.Axis(labels=False), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("winner:N", scale=alt.Scale(
                domain=["A", "B", "Tie"],
                range=["#4CAF50", "#2196F3", "#9E9E9E"]
            )),
            tooltip=["item", "winner", "summary"]
        ).properties(
            width=min(len(items_df) * 30, 800),
            height=100,
            title="Winner by Item"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Display item table
        st.write("Item Details:")
        display_items_df = items_df[["item", "winner", "summary"]]
        st.dataframe(display_items_df, use_container_width=True)
    
    # 4. Individual result inspection
    st.subheader("Inspect Individual Results")
    
    item_index = st.selectbox(
        "Select item to inspect",
        options=range(1, len(bulk_results) + 1),
        format_func=lambda x: f"Item {x} - {aggregates['items'][x-1]['winner']} won"
    )
    
    if item_index:
        selected_item = bulk_results[item_index - 1]
        
        st.write("### Input")
        st.text_area("", selected_item["input"], height=100, disabled=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Output A")
            st.text_area("", selected_item["output_a"], height=150, disabled=True)
        
        with col2:
            st.write("### Output B")
            st.text_area("", selected_item["output_b"], height=150, disabled=True)
        
        st.write("### Evaluation Results")
        
        # Show overall result first
        if "OVERALL" in selected_item["evaluations"]:
            overall = selected_item["evaluations"]["OVERALL"]
            winner = overall["winner"]
            summary = overall["summary"]
            
            # Choose color based on winner
            if winner == "A":
                color = "#4CAF50"  # Green
                icon = "🅰️"
            elif winner == "B":
                color = "#2196F3"  # Blue
                icon = "🅱️"
            else:  # Tie
                color = "#9E9E9E"  # Gray
                icon = "🔄"
            
            st.markdown(f"""
            <div style="
                padding: 20px; 
                border-radius: 5px;
                background-color: {color}25;
                text-align: center;
                margin-bottom: 20px;
            ">
                <h2 style="color: {color};">{icon} Overall Winner: Output {winner}</h2>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Group results by top-level metric
        metric_results = {}
        
        for metric_key, data in selected_item["evaluations"].items():
            # Skip special keys
            if metric_key in ["OVERALL", "CATEGORIES"]:
                continue
                
            # Get base metric (before ::)
            base_metric = metric_key.split("::")[0] if "::" in metric_key else metric_key
            
            if base_metric not in metric_results:
                metric_results[base_metric] = []
                
            metric_results[base_metric].append({
                "sub_metric": metric_key.split("::")[1] if "::" in metric_key else None,
                "winner": data["winner"],
                "justification": data["justification"],
                "raw_eval": selected_item["raw_evaluations"][metric_key]
            })
        
        # Show category results if available
        if "CATEGORIES" in selected_item["evaluations"]:
            category_results = selected_item["evaluations"]["CATEGORIES"]
            
            for category, result in category_results.items():
                winner = result["winner"]
                summary = result["summary"]
                
                # Choose color based on winner
                color = "#4CAF50" if winner == "A" else "#2196F3" if winner == "B" else "#9E9E9E"  # Green for A, Blue for B, Gray for Tie
                
                st.markdown(f"""
                <div style="
                    padding: 10px; 
                    border-left: 5px solid {color}; 
                    background-color: {color}10;
                    margin-bottom: 10px;
                ">
                    <h3>{category}: Output {winner} is better overall</h3>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display detailed metric results
        for metric_name, entries in metric_results.items():
            st.subheader(f"{metric_name} Comparison")
            
            # Sort entries so "overall" comes first
            entries.sort(key=lambda x: (
                0 if x["sub_metric"] is None or x["sub_metric"] == "overall" else 1,
                x["sub_metric"] or ""
            ))
            
            for entry in entries:
                sub_label = entry["sub_metric"] if entry["sub_metric"] else "overall"
                winner = entry["winner"]
                
                # Choose icon and color based on winner
                if "A is better" in winner:
                    icon = "🅰️"
                    color = "#4CAF50"  # Green
                elif "B is better" in winner:
                    icon = "🅱️"
                    color = "#2196F3"  # Blue
                else:  # Equivalent
                    icon = "🔄"
                    color = "#9E9E9E"  # Gray
                
                with st.expander(f"{icon} {metric_name}::{sub_label} - {winner}"):
                    st.markdown(f"""
                    <div style="
                        padding: 10px; 
                        border-left: 5px solid {color}; 
                        background-color: {color}10;
                    ">
                        <h3>{winner}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Justification:** {entry['justification']}")
                    
                    # Show raw evaluation 
                    show_raw = st.checkbox(f"Show Raw Evaluation", key=f"item_{item_index}_raw_{metric_name}_{sub_label}")
                    if show_raw:
                        st.text_area("Raw Evaluation", entry["raw_eval"], height=200, disabled=True)

def create_downloadable_csv(bulk_results, original_df, input_column, output_column=None, output_a_column=None, output_b_column=None, is_pairwise=False):
    """
    Create a downloadable CSV file with evaluation results
    
    Args:
        bulk_results: List of evaluation results
        original_df: Original pandas DataFrame from uploaded CSV
        input_column: Name of the input column
        output_column: Name of the output column (for single evaluation)
        output_a_column: Name of the output A column (for pairwise)
        output_b_column: Name of the output B column (for pairwise)
        is_pairwise: Whether this is pairwise evaluation or not
        
    Returns:
        CSV string ready for download
    """
    import pandas as pd
    import io
    from collections import defaultdict
    
    # Create a copy of the original dataframe with the specified columns
    if is_pairwise:
        result_df = original_df[[input_column, output_a_column, output_b_column]].copy()
    else:
        result_df = original_df[[input_column, output_column]].copy()
    
    # Reset index to ensure alignment
    result_df = result_df.reset_index(drop=True)
    
    # Limit to the number of items we evaluated
    result_df = result_df.iloc[:len(bulk_results)]
    
    # Add results to the dataframe
    for i, result in enumerate(bulk_results):
        # For each result, add the evaluation scores
        if is_pairwise:
            # For pairwise, add overall winner and metric-by-metric winners
            result_df.at[i, "overall_winner"] = result["evaluations"].get("OVERALL", {}).get("winner", "Tie")
            
            # Add each metric's winner
            for metric_name, data in result["evaluations"].items():
                # Skip special keys
                if metric_name in ["OVERALL", "CATEGORIES"]:
                    continue
                
                # Add the winner for this metric
                result_df.at[i, f"{metric_name}_winner"] = data.get("winner", "")
                
                # For sub-metrics, clean up the name for CSV headers
                clean_metric = metric_name.replace("::", "_")
                result_df.at[i, clean_metric] = data.get("winner", "")
        else:
            # For single evaluation, add scores for each metric
            for metric_name, data in result["evaluations"].items():
                # Clean up the metric name for CSV headers
                clean_metric = metric_name.replace("::", "_")
                result_df.at[i, clean_metric] = data.get("score", "N/A")
                
                # Also add the justification if it's not too long
                justification = data.get("justification", "")
                if len(justification) <= 500:  # Limit justification length for CSV
                    result_df.at[i, f"{clean_metric}_justification"] = justification
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def add_download_button(bulk_results, original_df, input_column, output_column=None, output_a_column=None, output_b_column=None, is_pairwise=False):
    """
    Add a download button for the evaluation results
    
    Args:
        bulk_results: List of evaluation results
        original_df: Original pandas DataFrame from uploaded CSV
        input_column: Name of the input column
        output_column: Name of the output column (for single evaluation)
        output_a_column: Name of the output A column (for pairwise)
        output_b_column: Name of the output B column (for pairwise)
        is_pairwise: Whether this is pairwise evaluation or not
    """
    import pandas as pd
    import streamlit as st
    
    # Create the CSV data
    csv_data = create_downloadable_csv(
        bulk_results, 
        original_df, 
        input_column, 
        output_column, 
        output_a_column, 
        output_b_column, 
        is_pairwise
    )
    
    # Create the download button
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="evaluation_results.csv",
        mime="text/csv",
    )
    
    # Add a preview of what's in the CSV
    with st.expander("Preview of Downloadable Results", expanded=False):
        # Convert CSV string back to DataFrame for display
        import io
        preview_df = pd.read_csv(io.StringIO(csv_data))
        
        # Limit columns if there are too many
        if preview_df.shape[1] > 15:
            preview_cols = list(preview_df.columns[:15])
            st.write(f"Showing first 15 of {preview_df.shape[1]} columns:")
            st.dataframe(preview_df[preview_cols], use_container_width=True)
        else:
            st.dataframe(preview_df, use_container_width=True)

def save_metrics_to_session(pipeline_results):
    """
    Save the pipeline results to session state in a JSON-serializable format
    """
    import json
    
    # Store the results in the session state
    st.session_state.saved_pipeline_results = pipeline_results
    
    # Also convert to JSON for download
    json_data = json.dumps(pipeline_results, indent=2)
    return json_data

def download_metrics_button(json_data):
    """
    Create a download button for the metrics
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
    Allow uploading a previously saved metrics JSON file
    """
    import json
    
    uploaded_file = st.file_uploader(
        "Upload saved metrics JSON file", 
        type=["json"],
        key="metrics_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Try to parse the JSON file
            content = uploaded_file.read().decode("utf-8")
            metrics_data = json.loads(content)
            
            # Validate that it has the expected structure
            if not all(key in metrics_data for key in ["input", "task_analysis", "selected_metrics", "customized_metrics"]):
                st.error("The uploaded file doesn't have the expected structure for metrics data.")
                return None
            
            st.success("Metrics loaded successfully!")
            return metrics_data
            
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")
            return None
    
    return None


def main():
    st.title("Auto-Generated Eval Pipeline")
    st.markdown("This tool helps you identify, customize, and apply evaluation metrics for your AI-generated content.")
    
    # Initialize auto-save session state if not exists
    if "auto_save_enabled" not in st.session_state:
        st.session_state.auto_save_enabled = True
    
    # Initialize saved metrics if not exists but we have pipeline results
    if "saved_pipeline_results" not in st.session_state and "pipeline_results" in st.session_state and st.session_state.pipeline_results:
        save_metrics_to_session(st.session_state.pipeline_results)
    
    # Load saved metrics into pipeline_results if we have saved metrics but no pipeline results
    if "saved_pipeline_results" in st.session_state and "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = st.session_state.saved_pipeline_results
    
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox(
            "Metric Generation Model", 
            ["o3-mini-2025-01-31","o1-2024-12-17", "gpt-4o-2024-05-13"], 
            index=0,
            help="Model used for generating metrics and customizing rubrics"
        )
        
        # Add auto-save toggle in sidebar
        st.subheader("Session Settings")
        auto_save = st.checkbox("Enable Auto-save", value=st.session_state.auto_save_enabled, 
                              help="Automatically save metrics in session to prevent loss")
        st.session_state.auto_save_enabled = auto_save
        
        if auto_save and "pipeline_results" in st.session_state and st.session_state.pipeline_results:
            st.success("Metrics auto-saved in session")
            
        if st.button("💾 Save Configuration"):
            if api_key:
                st.session_state.client = initialize_client(api_key)
                st.session_state.model = model
                st.success("Configuration saved!")
            else:
                st.error("Please provide an API key")
    
    # Initialize tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Input Configuration", "Customized Metrics", "Single Evaluation","Pairwise Evaluation"])
    
    # Session state to track our pipeline progress
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None
    if "model" not in st.session_state:
        st.session_state.model = model
    
    # Input tab
    with tab1:
        st.header("Task Configuration")
        
        st.info("Configure your task details to generate customized evaluation metrics. This information helps identify what makes a high-quality output for your specific use case.")
        
        task_summary = st.text_input("Task Summary", placeholder="E.g., Generate a credit card headline -- the what + the why")
        context = st.text_area("Context", placeholder="Describe where and how this content will be used", height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_area("Prompt Template", placeholder="The full prompt used to generate the content", height=150)
            sample_input = st.text_area("Generic Sample Input", placeholder="Example of the type of input that would be provided to the prompt (not tied to evaluation)", height=150)
        
        with col2:
            good_examples = st.text_area("Good Examples (one per line)", 
                                       placeholder="Examples of good outputs", 
                                       height=150)
            bad_examples = st.text_area("Bad Examples (one per line)", 
                                      placeholder="Examples of bad outputs", 
                                      height=150)
        
        # Process good/bad examples into lists
        good_examples_list = [ex.strip() for ex in good_examples.split('\n') if ex.strip()]
        bad_examples_list = [ex.strip() for ex in bad_examples.split('\n') if ex.strip()]
        
        if st.button("Generate Customized Metrics"):
            if not api_key:
                st.error("Please provide an API key in the sidebar first")
            elif not task_summary or not prompt:
                st.error("Task summary and prompt are required")
            else:
                with st.spinner("Processing..."):
                    client = initialize_client(api_key)
                    st.session_state.pipeline_results = run_pipeline(
                        prompt=prompt,
                        task_summary=task_summary,
                        sample_input=sample_input,
                        good_examples=good_examples_list,
                        bad_examples=bad_examples_list,
                        context=context,
                        client=client,
                        model=st.session_state.model
                    )
                    
                    # Auto-save the new metrics if enabled
                    if st.session_state.auto_save_enabled:
                        save_metrics_to_session(st.session_state.pipeline_results)
                        
                st.success("Metrics customized successfully!")
        
        # 3. Add the Save/Load Metrics expander panel
        with st.expander("Save/Load Metrics", expanded=False):
            st.info("Save your current metrics or load previously generated metrics.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Save Current Metrics")
                if st.session_state.pipeline_results:
                    json_data = save_metrics_to_session(st.session_state.pipeline_results)
                    download_metrics_button(json_data)
                else:
                    st.write("No metrics to save yet.")
            
            with col2:
                st.subheader("Load Saved Metrics")
                loaded_metrics = load_metrics_from_file()
                if loaded_metrics and st.button("Use Loaded Metrics"):
                    st.session_state.pipeline_results = loaded_metrics
                    # Also save to our persistent session state
                    if st.session_state.auto_save_enabled:
                        save_metrics_to_session(loaded_metrics)
                    st.rerun()
                
    # Metrics tab
    with tab2:
        if st.session_state.pipeline_results:
            st.header("Selected Metrics")
            
            result = st.session_state.pipeline_results
            metrics = result["selected_metrics"]
            
            st.write(f"Selected metrics for this task: **{', '.join(metrics)}**")
            
            for metric in metrics:
                with st.expander(f"{metric} Metric"):
                    metric_data = result["customized_metrics"][metric]
                    
                    st.markdown(f"**Description:** {metric_data['customized_description']}")
                    
                    st.markdown("**Parameters:**")
                    if isinstance(metric_data['parameters'], list):
                        for param in metric_data['parameters']:
                            st.markdown(f"- {param}")
                    elif isinstance(metric_data['parameters'], dict):
                        for k, v in metric_data['parameters'].items():
                            st.markdown(f"- **{k}**: {v}")
                    
                    st.markdown("**Success Criteria:**")
                    if isinstance(metric_data['success_criteria'], list):
                        for criteria in metric_data['success_criteria']:
                            st.markdown(f"- {criteria}")
                    else:
                        st.markdown(metric_data['success_criteria'])
                    
                    st.markdown("**Scoring Rubric:**")
                    if isinstance(metric_data['scoring_rubric'], dict):
                        for score, desc in metric_data['scoring_rubric'].items():
                            st.markdown(f"- **{score}**: {desc}")
                    else:
                        st.markdown(metric_data['scoring_rubric'])
                    
                    st.markdown("**Examples:**")
                    for example in metric_data['examples']:
                        st.markdown(f"- {example}")
        else:
            st.info("Please configure your task and generate metrics in the Input tab first.")
    
    # Single Evaluation tab
    # Single Evaluation tab
    with tab3:
        if not st.session_state.pipeline_results:
            st.info("Please configure your task and generate metrics in the Input tab first.")
        else:
            st.header("Output Evaluation")
            
            # Add tabs for single vs bulk evaluation
            eval_tab1, eval_tab2 = st.tabs(["Single Evaluation", "Bulk Evaluation (CSV)"])
            
            # Common settings for both tabs
            eval_model = st.selectbox(
                "Evaluation Model", 
                ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "o1-2024-12-17", "gpt-3.5-turbo"], 
                index=0,
                key="tab3_eval_model",
                help="Choose which model to use for running the evaluation"
            )
            
            # Metric and submetric selection
            st.subheader("Select Metrics to Evaluate")
            selected_eval_metrics = create_metric_selection_ui(
                st.session_state.pipeline_results["customized_metrics"],
                key_prefix="tab3_single_eval"
            )

            # Update session state metrics
            st.session_state.selected_metrics = {
                m: selected for m, selected in selected_eval_metrics.items() 
                if "::" not in m  # Only count top-level metrics
            }
            
            with eval_tab1:
                # Single evaluation input section
                specific_input = st.text_area(
                    "Specific Input for This Evaluation", 
                    height=100,
                    placeholder="Enter the specific input",
                    key="tab3_single_eval_input"
                )

                output = st.text_area(
                    "Specific Output for This Evaluation", 
                    height=100,
                    placeholder="Enter the specific output",
                    key="tab3_single_eval_output"
                )
                
                # Run single evaluation button
                run_single_eval = st.button("Run Single Evaluation", key="tab3_single_eval_run")
                
                if run_single_eval:
                    # Input validation
                    if not output:
                        st.error("Please provide an output to evaluate")
                    elif not any(st.session_state.selected_metrics.values()):
                        st.error("Please select at least one metric for evaluation")
                    else:
                        with st.spinner("Evaluating output against metrics..."):
                            try:
                                # Initialize client
                                client = initialize_client(api_key)
                                
                                # Prepare evaluation result
                                evaluation_result = st.session_state.pipeline_results.copy()
                                if specific_input.strip():
                                    evaluation_result["input"]["sample_input"] = specific_input
                                
                                # Filter metrics based on selection
                                filtered_metrics = filter_metrics_based_on_selection(
                                    evaluation_result["customized_metrics"],
                                    selected_eval_metrics
                                )
                                evaluation_result["customized_metrics"] = filtered_metrics
                                
                                # Run parallel evaluations
                                eval_results = run_evaluations_parallel(
                                    client=client, 
                                    result=evaluation_result, 
                                    output=output,
                                    model=eval_model,
                                    max_workers=5
                                )
                                
                                # Format and display results
                                formatted_results = format_evaluation_results(eval_results)
                                display_grouped_results(formatted_results, eval_results)
                            
                            except Exception as e:
                                st.error(f"Evaluation error: {str(e)}")
            
            with eval_tab2:
                # Bulk CSV evaluation section
                st.info("Upload a CSV file with multiple inputs and outputs to evaluate in bulk.")
                
                # Advanced settings
                with st.expander("Advanced Settings", expanded=False):
                    max_workers = st.slider(
                        "Max concurrent evaluations per item", 
                        min_value=1, 
                        max_value=10, 
                        value=5,
                        key="tab3_max_workers",
                        help="Maximum number of metrics to evaluate in parallel for each item."
                    )
                    
                    show_timing = st.checkbox(
                        "Show detailed timing information", 
                        value=True,
                        key="tab3_show_timing",
                        help="Display information about processing time for each item"
                    )
                
                # File uploader with a unique, fixed key
                uploaded_file = st.file_uploader(
                    "Upload CSV file", 
                    type=["csv"], 
                    key="tab3_bulk_csv_uploader"
                )
                
                if uploaded_file is not None:
                    try:
                        # Always reset the file pointer before reading
                        uploaded_file.seek(0)
                        
                        # Debug information
                        st.write(f"File details: {uploaded_file.name}, Size: {uploaded_file.size} bytes")
                        
                        # Read and process CSV - with explicit error handling
                        try:
                            df = pd.read_csv(uploaded_file)
                        except pd.errors.EmptyDataError:
                            st.error("The CSV file is empty.")
                            st.stop()
                        except UnicodeDecodeError:
                            # Try different encodings
                            uploaded_file.seek(0)  # Reset pointer
                            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                            df = None
                            for encoding in encodings:
                                try:
                                    uploaded_file.seek(0)  # Reset for each attempt
                                    df = pd.read_csv(uploaded_file, encoding=encoding)
                                    st.success(f"Successfully read CSV with encoding: {encoding}")
                                    break
                                except Exception:
                                    continue
                            
                            if df is None:
                                st.error(f"Could not read file with any of these encodings: {encodings}")
                                st.stop()
                        
                        # Preview data
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head(5), use_container_width=True)
                        
                        # Column selection
                        column_names = df.columns.tolist()
                        st.write(f"Available columns: {column_names}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            input_column = st.selectbox(
                                "Select input column",
                                options=column_names,
                                index=0 if column_names else None,
                                key="tab3_input_column"
                            )
                        
                        with col2:
                            output_column = st.selectbox(
                                "Select output column",
                                options=column_names,
                                index=1 if len(column_names) > 1 else None,
                                key="tab3_output_column"
                            )
                        
                        # Sample data preview
                        if input_column and output_column:
                            st.write("Sample input-output pairs:")
                            sample_pairs = []
                            
                            # Handle potential missing values safely
                            for _, row in df.head(3).iterrows():
                                if not pd.isna(row.get(input_column, None)) and not pd.isna(row.get(output_column, None)):
                                    sample_pairs.append({
                                        "input": str(row[input_column]),
                                        "output": str(row[output_column])
                                    })
                            
                            for i, pair in enumerate(sample_pairs):
                                with st.expander(f"Pair {i+1}", expanded=i==0):
                                    st.text_area(f"Input {i+1}", pair["input"], height=100, key=f"tab3_preview_input_{i}", disabled=True)
                                    st.text_area(f"Output {i+1}", pair["output"], height=100, key=f"tab3_preview_output_{i}", disabled=True)
                            
                            # Bulk processing options
                            max_items = st.slider(
                                "Maximum number of items to process", 
                                min_value=1, 
                                max_value=min(500, len(df)), 
                                value=min(20, len(df)),
                                key="tab3_max_items",
                                help="Limit rows to process to avoid excessive API usage"
                            )
                            
                            # Process button
                            if st.button("Run Bulk Evaluation", key="tab3_run_bulk"):
                                if not any(selected_eval_metrics.values()):
                                    st.error("Please select at least one metric for evaluation.")
                                else:
                                    try:
                                        # Reset file pointer again before parsing
                                        uploaded_file.seek(0)
                                        
                                        # Parse and process CSV
                                        input_output_pairs = parse_uploaded_csv(uploaded_file, input_column, output_column)
                                        
                                        if not input_output_pairs:
                                            st.error("No valid input-output pairs found in CSV.")
                                            st.stop()
                                        
                                        # Limit to max_items
                                        input_output_pairs = input_output_pairs[:max_items]
                                        st.write(f"Processing {len(input_output_pairs)} items...")
                                        
                                        # Progress tracking
                                        progress_bar = st.progress(0, text="Starting evaluation...")
                                        
                                        # Run bulk evaluation
                                        client = initialize_client(api_key)
                                        filtered_result = st.session_state.pipeline_results.copy()
                                        
                                        bulk_results = bulk_process_evaluations_parallel(
                                            client=client,
                                            result=filtered_result,
                                            input_output_pairs=input_output_pairs,
                                            eval_model=eval_model,
                                            selected_metrics=selected_eval_metrics,
                                            progress_bar=progress_bar,
                                            max_workers=max_workers
                                        )
                                        
                                        # Calculate and display results
                                        aggregates = calculate_aggregate_metrics(bulk_results)
                                        display_bulk_results(bulk_results, aggregates)
                                        
                                        # Download functionality
                                        st.subheader("Download Results")
                                        add_download_button(
                                            bulk_results=bulk_results,
                                            original_df=df,
                                            input_column=input_column,
                                            output_column=output_column,
                                            is_pairwise=False
                                        )
                                    
                                    except Exception as e:
                                        st.error(f"Error processing CSV: {str(e)}")
                                        st.exception(e)  # This shows the full traceback
                    
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.exception(e)  # This shows the full traceback

    # Pairwise Evaluation tab
    # Pairwise Evaluation tab
    with tab4:
        if not st.session_state.pipeline_results:
            st.info("Please configure your task and generate metrics in the Input tab first.")
        else:
            st.header("Pairwise Output Comparison")
            
            # Create separate tabs for single vs bulk pairwise evaluation
            pair_tab1, pair_tab2 = st.tabs(["Single Comparison", "Bulk Comparison (CSV)"])
            
            # Common settings for both tabs
            pair_eval_model = st.selectbox(
                "Evaluation Model", 
                ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "o1-2024-12-17", "gpt-3.5-turbo"], 
                index=0,
                key="tab4_eval_model",
                help="Choose which model to use for running the comparison"
            )
            
            # Metric selection
            st.subheader("Select Metrics to Compare")
            available_metrics = st.session_state.pipeline_results["selected_metrics"]
            selected_pairwise_metrics = {}
            
            # Create columns for metrics selection
            cols = st.columns(3)
            for i, metric in enumerate(available_metrics):
                col_idx = i % 3
                with cols[col_idx]:
                    selected_pairwise_metrics[metric] = st.checkbox(
                        metric, 
                        value=True, 
                        key=f"tab4_metric_{metric}"
                    )
            
            # Store selected metrics in session state with unique name
            st.session_state.selected_pairwise_metrics = {
                m: selected for m, selected in selected_pairwise_metrics.items() if selected
            }
            
            with pair_tab1:
                # Single pairwise comparison
                st.info("Compare two outputs to determine which one better satisfies each metric.")
                
                # Advanced settings
                with st.expander("Advanced Settings", expanded=False):
                    max_workers_single = st.slider(
                        "Max concurrent evaluations", 
                        min_value=1, 
                        max_value=10, 
                        value=min(10, len(available_metrics)),
                        help="Maximum number of metrics to evaluate in parallel.",
                        key="tab4_max_workers_single"
                    )
                
                # Common input context
                common_input = st.text_area(
                    "Input Text",
                    height=100,
                    placeholder="Enter the input text that was used to generate both outputs",
                    help="This provides important context for the evaluation",
                    key="tab4_common_input"
                )
                
                # Side-by-side output comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Output A")
                    output_a = st.text_area(
                        "First output to compare", 
                        height=200,
                        placeholder="Paste the first AI-generated output here",
                        key="tab4_output_a"
                    )
                    
                with col2:
                    st.subheader("Output B") 
                    output_b = st.text_area(
                        "Second output to compare",
                        height=200,
                        placeholder="Paste the second AI-generated output here",
                        key="tab4_output_b"
                    )
                
                # Run comparison button
                if st.button("Run Pairwise Comparison", key="tab4_run_comparison") and output_a and output_b:
                    # Validate metric selection
                    if not any(selected_pairwise_metrics.values()):
                        st.error("Please select at least one metric for comparison.")
                    else:
                        with st.spinner("Comparing outputs in parallel..."):
                            try:
                                # Initialize client and prepare evaluation
                                client = initialize_client(api_key)
                                
                                # Filter metrics based on selection
                                filtered_result = st.session_state.pipeline_results.copy()
                                filtered_metrics = {
                                    k: v for k, v in st.session_state.pipeline_results["customized_metrics"].items() 
                                    if k in selected_pairwise_metrics and selected_pairwise_metrics[k]
                                }
                                filtered_result["customized_metrics"] = filtered_metrics
                                
                                # Update with common input if provided
                                if common_input.strip():
                                    filtered_result["input"]["sample_input"] = common_input
                                
                                # Run parallel pairwise evaluation
                                start_time = time.time()
                                pairwise_results = run_pairwise_evaluations_parallel(
                                    client=client,
                                    result=filtered_result,
                                    output_a=output_a,
                                    output_b=output_b,
                                    model=pair_eval_model,
                                    max_workers=max_workers_single
                                )
                                
                                # Format and display results
                                formatted_results = format_pairwise_evaluation_results(pairwise_results)
                                processing_time = time.time() - start_time
                                
                                st.success(f"Comparison completed in {processing_time:.2f} seconds!")
                                
                                # Display overall winner
                                overall = formatted_results.pop("OVERALL", {"winner": "Tie", "summary": "Unable to determine a clear winner."})
                                
                                # Create a visual summary for the overall winner
                                winner = overall['winner']
                                if winner == "A":
                                    color = "#4CAF50"  # Green
                                    icon = "🅰️"
                                elif winner == "B":
                                    color = "#2196F3"  # Blue
                                    icon = "🅱️"
                                else:  # Tie
                                    color = "#9E9E9E"  # Gray
                                    icon = "🔄"
                                
                                st.markdown(f"""
                                <div style="
                                    padding: 20px; 
                                    border-radius: 5px;
                                    background-color: {color}25;
                                    text-align: center;
                                    margin-bottom: 20px;
                                ">
                                    <h1 style="color: {color};">{icon} Overall Winner: Output {winner}</h1>
                                    <p>{overall['summary']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display detailed results
                                display_grouped_pairwise_results(formatted_results, pairwise_results)
                            
                            except Exception as e:
                                st.error(f"Comparison error: {str(e)}")
                                st.exception(e)  # Show full traceback
            
            with pair_tab2:
                # Bulk pairwise comparison from CSV
                st.info("Upload a CSV file with inputs and paired outputs to compare in bulk.")
                
                # Advanced settings
                with st.expander("Advanced Settings", expanded=False):
                    max_workers_bulk = st.slider(
                        "Max concurrent evaluations per comparison", 
                        min_value=1, 
                        max_value=10, 
                        value=min(10, len(available_metrics)),
                        help="Maximum number of metrics to evaluate in parallel for each comparison.",
                        key="tab4_max_workers_bulk"
                    )
                    
                    show_timing = st.checkbox(
                        "Show detailed timing information", 
                        value=True,
                        help="Display information about processing time for each comparison",
                        key="tab4_show_timing"
                    )
                
                # File uploader with unique, consistent key
                pairwise_uploaded_file = st.file_uploader(
                    "Upload CSV file", 
                    type=["csv"],
                    key="tab4_pairwise_csv_uploader"  # Unique, consistent key
                )
                
                if pairwise_uploaded_file is not None:
                    try:
                        # Reset file pointer
                        pairwise_uploaded_file.seek(0)
                        
                        # Debug information
                        st.write(f"File details: {pairwise_uploaded_file.name}, Size: {pairwise_uploaded_file.size} bytes")
                        
                        # Read CSV with multiple encoding attempts
                        try:
                            df = pd.read_csv(pairwise_uploaded_file)
                        except pd.errors.EmptyDataError:
                            st.error("The CSV file is empty.")
                            st.stop()
                        except UnicodeDecodeError:
                            # Try different encodings
                            pairwise_uploaded_file.seek(0)  # Reset pointer
                            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                            df = None
                            for encoding in encodings:
                                try:
                                    pairwise_uploaded_file.seek(0)  # Reset for each attempt
                                    df = pd.read_csv(pairwise_uploaded_file, encoding=encoding)
                                    st.success(f"Successfully read CSV with encoding: {encoding}")
                                    break
                                except Exception:
                                    continue
                            
                            if df is None:
                                st.error(f"Could not read file with any of these encodings: {encodings}")
                                st.stop()
                        
                        # Preview data
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head(5), use_container_width=True)
                        
                        # Column selection
                        column_names = df.columns.tolist()
                        st.write(f"Available columns: {column_names}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            input_column = st.selectbox(
                                "Select input column",
                                options=column_names,
                                index=0 if column_names else None,
                                key="tab4_input_column"
                            )
                        
                        with col2:
                            output_a_column = st.selectbox(
                                "Select Output A column",
                                options=column_names,
                                index=1 if len(column_names) > 1 else None,
                                key="tab4_output_a_column"
                            )
                        
                        with col3:
                            output_b_column = st.selectbox(
                                "Select Output B column",
                                options=column_names,
                                index=2 if len(column_names) > 2 else None,
                                key="tab4_output_b_column"
                            )
                        
                        # Sample data preview
                        if input_column and output_a_column and output_b_column:
                            st.write("Sample comparison triplets:")
                            sample_triplets = []
                            
                            # Handle potential missing values safely
                            for _, row in df.head(3).iterrows():
                                if not pd.isna(row.get(input_column, None)) and not pd.isna(row.get(output_a_column, None)) and not pd.isna(row.get(output_b_column, None)):
                                    sample_triplets.append({
                                        "input": str(row[input_column]),
                                        "output_a": str(row[output_a_column]),
                                        "output_b": str(row[output_b_column])
                                    })
                            
                            for i, triplet in enumerate(sample_triplets):
                                with st.expander(f"Comparison {i+1}", expanded=i==0):
                                    st.text_area(f"Input {i+1}", triplet["input"], height=100, key=f"tab4_preview_input_{i}", disabled=True)
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.text_area(f"Output A {i+1}", triplet["output_a"], height=100, key=f"tab4_preview_output_a_{i}", disabled=True)
                                    with col2:
                                        st.text_area(f"Output B {i+1}", triplet["output_b"], height=100, key=f"tab4_preview_output_b_{i}", disabled=True)
                            
                            # Bulk processing options
                            max_items = st.slider(
                                "Maximum number of comparisons to process", 
                                min_value=1, 
                                max_value=min(500, len(df)), 
                                value=min(20, len(df)),
                                help="Limit rows to process to avoid excessive API usage",
                                key="tab4_max_items"
                            )
                            
                            # Process button
                            if st.button("Run Bulk Pairwise Evaluation", key="tab4_run_bulk"):
                                # Validate metric selection
                                if not any(selected_pairwise_metrics.values()):
                                    st.error("Please select at least one metric for comparison.")
                                    st.stop()
                                
                                try:
                                    # Reset file pointer before parsing
                                    pairwise_uploaded_file.seek(0)
                                    
                                    # Parse and process CSV
                                    input_output_triplets = parse_pairwise_csv(
                                        pairwise_uploaded_file, 
                                        input_column, 
                                        output_a_column, 
                                        output_b_column
                                    )
                                    
                                    if not input_output_triplets:
                                        st.error("No valid input-output triplets found in CSV.")
                                        st.stop()
                                    
                                    # Limit to max_items
                                    input_output_triplets = input_output_triplets[:max_items]
                                    st.write(f"Processing {len(input_output_triplets)} comparisons...")
                                    
                                    # Progress tracking
                                    progress_bar = st.progress(0, text="Starting pairwise evaluation...")
                                    
                                    # Run bulk evaluation
                                    client = initialize_client(api_key)
                                    start_time = time.time()
                                    
                                    # Prepare filtered results
                                    filtered_result = st.session_state.pipeline_results.copy()
                                    filtered_metrics = {
                                        k: v for k, v in st.session_state.pipeline_results["customized_metrics"].items() 
                                        if k in selected_pairwise_metrics and selected_pairwise_metrics[k]
                                    }
                                    filtered_result["customized_metrics"] = filtered_metrics
                                    
                                    # Bulk pairwise processing
                                    bulk_pairwise_results = bulk_process_pairwise_evaluations(
                                        client=client,
                                        result=filtered_result,
                                        input_output_triplets=input_output_triplets,
                                        eval_model=pair_eval_model,
                                        selected_metrics=selected_pairwise_metrics,
                                        progress_bar=progress_bar,
                                        max_workers=max_workers_bulk
                                    )
                                    
                                    # Calculate and display results
                                    total_time = time.time() - start_time
                                    pairwise_aggregates = calculate_aggregate_pairwise_metrics(bulk_pairwise_results)
                                    
                                    display_bulk_pairwise_results(bulk_pairwise_results, pairwise_aggregates)
                                    
                                    # Download functionality
                                    st.subheader("Download Results")
                                    add_download_button(
                                        bulk_results=bulk_pairwise_results,
                                        original_df=df,
                                        input_column=input_column,
                                        output_a_column=output_a_column,
                                        output_b_column=output_b_column,
                                        is_pairwise=True
                                    )
                                    
                                    # Performance metrics
                                    if show_timing:
                                        st.subheader("Performance Metrics")
                                        st.write(f"**Total Comparisons:** {len(bulk_pairwise_results)}")
                                        st.write(f"**Total Processing Time:** {total_time:.2f} seconds")
                                        st.write(f"**Average Time per Comparison:** {total_time / len(bulk_pairwise_results):.2f} seconds")
                                        st.write(f"**Number of Metrics:** {len(selected_pairwise_metrics)}")
                                
                                except Exception as e:
                                    st.error(f"Error processing CSV: {str(e)}")
                                    st.exception(e)  # Show full traceback
                    
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.exception(e)  # Show full traceback

if __name__ == "__main__":
    main()
