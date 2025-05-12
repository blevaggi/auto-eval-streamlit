import streamlit as st
import json

# ====== TAB 2 FUNCTIONS ========

# This tab shows the metrics and parameters and their prompts, and allows users to edit them

def display_metrics_and_prompts(pipeline_results_for_display):
    """
    Display all metrics, descriptions, and associated prompts in Tab 2
    with editing capabilities
    """
    pipeline_results = pipeline_results_for_display
    if not pipeline_results:
        st.info("No metrics available yet. Please generate or upload metrics in Tab 1 first.")
        return
    
    # Extract metrics from the results
    metrics_json = pipeline_results.get("metrics_json", {})
    param_templates = pipeline_results.get("parameter_prompt_templates", {})
    pairwise_templates = pipeline_results.get("pairwise_prompt_templates", {})
    
    # Initialize edited prompts in session state if not already present
    if "edited_parameter_prompts" not in st.session_state:
        st.session_state.edited_parameter_prompts = param_templates.copy()
    if "edited_pairwise_prompts" not in st.session_state:
        st.session_state.edited_pairwise_prompts = pairwise_templates.copy()
    
    # Determine metrics list structure
    metrics_list = []
    if isinstance(metrics_json, dict) and "metrics" in metrics_json:
        metrics_list = metrics_json["metrics"]
    elif isinstance(metrics_json, list):
        metrics_list = metrics_json
    
    if not metrics_list:
        st.warning("No metrics found in the results.")
        return
    
    # Display each metric with its parameters and templates
    st.subheader("Evaluation Metrics Overview")
    
    # Create tabs for each metric
    metric_tabs = st.tabs([metric.get("metric", f"Metric {i+1}") for i, metric in enumerate(metrics_list)])
    
    # Track if any changes were made
    changes_made = False
    
    for i, (metric_tab, metric) in enumerate(zip(metric_tabs, metrics_list)):
        metric_name = metric.get("metric", "Unnamed Metric")
        metric_description = metric.get("description", "")
        parameters = metric.get("parameters", [])
        
        with metric_tab:
            st.markdown(f"### {metric_name}")
            st.markdown(f"**Description:** {metric_description}")
            
            # Display parameters table
            st.markdown("#### Parameters")
            param_data = []
            for param in parameters:
                if isinstance(param, dict) and len(param) > 0:
                    param_key = list(param.keys())[0]
                    param_description = param[param_key]
                    param_data.append({
                        "Parameter": param_key, 
                        "Description": param_description
                    })
            
            if param_data:
                st.table(param_data)
            else:
                st.info("No parameters defined for this metric.")
            
            # Display prompt templates for each parameter with editing capability
            st.markdown("#### Evaluation Prompts")
            
            # Create tabs for each parameter
            if parameters:
                param_keys = [list(param.keys())[0] if isinstance(param, dict) and len(param) > 0 else f"Param {j+1}" 
                             for j, param in enumerate(parameters)]
                param_tabs = st.tabs(param_keys)
                
                for param_tab, param in zip(param_tabs, parameters):
                    if isinstance(param, dict) and len(param) > 0:
                        param_key = list(param.keys())[0]
                        prompt_key = f"{metric_name}::{param_key}"
                        
                        with param_tab:
                            # Two columns for the two types of templates
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### Parameter Evaluation Template")
                                if prompt_key in param_templates:
                                    # Get the current value from session state (edited or original)
                                    current_param_template = st.session_state.edited_parameter_prompts.get(prompt_key, param_templates[prompt_key])
                                    
                                    # Create a text area for editing with a unique key
                                    edited_param_template = st.text_area(
                                        "Edit Parameter Template:", 
                                        value=current_param_template,
                                        height=400,
                                        key=f"param_{prompt_key}"
                                    )
                                    
                                    # Check if changes were made
                                    if edited_param_template != current_param_template:
                                        st.session_state.edited_parameter_prompts[prompt_key] = edited_param_template
                                        changes_made = True
                                else:
                                    st.info("No parameter evaluation template available.")
                            
                            with col2:
                                st.markdown("##### Pairwise Evaluation Template")
                                if prompt_key in pairwise_templates:
                                    # Get the current value from session state (edited or original)
                                    current_pairwise_template = st.session_state.edited_pairwise_prompts.get(prompt_key, pairwise_templates[prompt_key])
                                    
                                    # Create a text area for editing with a unique key
                                    edited_pairwise_template = st.text_area(
                                        "Edit Pairwise Template:", 
                                        value=current_pairwise_template,
                                        height=400,
                                        key=f"pairwise_{prompt_key}"
                                    )
                                    
                                    # Check if changes were made
                                    if edited_pairwise_template != current_pairwise_template:
                                        st.session_state.edited_pairwise_prompts[prompt_key] = edited_pairwise_template
                                        changes_made = True
                                else:
                                    st.info("No pairwise evaluation template available.")
            else:
                st.info("No parameters defined for this metric.")
    
    # Save button at the bottom of the page
    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Save All Prompt Edits", type="primary"):
            # Update the pipeline results with edited prompts
            pipeline_results["parameter_prompt_templates"] = st.session_state.edited_parameter_prompts
            pipeline_results["pairwise_prompt_templates"] = st.session_state.edited_pairwise_prompts
            
            # Update the session state with the modified pipeline results
            st.session_state.pipeline_results = pipeline_results
            
            st.success("All prompt edits saved successfully! The updated prompts will be used in Tabs 3 and 4.")
    
    with col2:
        if changes_made:
            st.warning("You have unsaved changes. Click 'Save All Prompt Edits' to save them.")

def add_tab2_content(pipeline_results_for_display):
    display_metrics_and_prompts(pipeline_results_for_display)