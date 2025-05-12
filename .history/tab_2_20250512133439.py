import streamlit as st
import json

# ====== TAB 2 FUNCTIONS ========

# This tab shows the metrics and parameters and their prompts, and allows users to edit them one at a time

def display_metrics_and_prompts(pipeline_results_for_display):
    """
    Display all metrics, descriptions, and associated prompts in Tab 2
    with editing capabilities (one prompt at a time)
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
    
    # Initialize state for tracking which prompt is being edited
    if "currently_editing" not in st.session_state:
        st.session_state.currently_editing = None
    
    # Function to toggle edit mode for a specific prompt
    def toggle_edit_mode(prompt_id):
        if st.session_state.currently_editing == prompt_id:
            # If already editing this prompt, close it
            st.session_state.currently_editing = None
        elif st.session_state.currently_editing is None:
            # If not editing anything, start editing this prompt
            st.session_state.currently_editing = prompt_id
        else:
            # If editing another prompt, show error
            st.error("Please save or cancel your current edits before editing another prompt.")
    
    # Function to save edits for a prompt
    def save_edits(prompt_id, prompt_type, new_content):
        if prompt_type == "parameter":
            st.session_state.edited_parameter_prompts[prompt_id] = new_content
        else:  # pairwise
            st.session_state.edited_pairwise_prompts[prompt_id] = new_content
        
        # Update the pipeline results
        pipeline_results["parameter_prompt_templates"] = st.session_state.edited_parameter_prompts
        pipeline_results["pairwise_prompt_templates"] = st.session_state.edited_pairwise_prompts
        
        # Update the session state with the modified pipeline results
        st.session_state.pipeline_results = pipeline_results
        
        # Exit edit mode
        st.session_state.currently_editing = None
        
        # Force a rerun to update the display
        st.rerun()
    
    # Function to cancel edits
    def cancel_edits():
        st.session_state.currently_editing = None
        st.rerun()
    
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
                            
                            # Parameter template column
                            with col1:
                                st.markdown("##### Parameter Evaluation Template")
                                if prompt_key in param_templates:
                                    # Check if we're currently editing this template
                                    param_edit_id = f"param_{prompt_key}"
                                    is_editing_param = st.session_state.currently_editing == param_edit_id
                                    
                                    # Get the current template content
                                    current_param_template = st.session_state.edited_parameter_prompts.get(prompt_key, param_templates[prompt_key])
                                    
                                    # Show edit button or editor based on state
                                    if is_editing_param:
                                        # Show editor with Save/Cancel buttons
                                        edited_param_template = st.text_area(
                                            "Edit Parameter Template:", 
                                            value=current_param_template,
                                            height=400,
                                            key=f"editing_{param_edit_id}"
                                        )
                                        
                                        col1a, col1b = st.columns(2)
                                        with col1a:
                                            if st.button("Save Edits", key=f"save_{param_edit_id}", type="primary"):
                                                save_edits(prompt_key, "parameter", edited_param_template)
                                        with col1b:
                                            if st.button("Cancel", key=f"cancel_{param_edit_id}"):
                                                cancel_edits()
                                    else:
                                        # Show readonly display with edit button
                                        st.code(current_param_template, language="markdown")
                                        if st.button("✏️ Edit", key=f"edit_{param_edit_id}"):
                                            toggle_edit_mode(param_edit_id)
                                else:
                                    st.info("No parameter evaluation template available.")
                            
                            # Pairwise template column
                            with col2:
                                st.markdown("##### Pairwise Evaluation Template")
                                if prompt_key in pairwise_templates:
                                    # Check if we're currently editing this template
                                    pairwise_edit_id = f"pairwise_{prompt_key}"
                                    is_editing_pairwise = st.session_state.currently_editing == pairwise_edit_id
                                    
                                    # Get the current template content
                                    current_pairwise_template = st.session_state.edited_pairwise_prompts.get(prompt_key, pairwise_templates[prompt_key])
                                    
                                    # Show edit button or editor based on state
                                    if is_editing_pairwise:
                                        # Show editor with Save/Cancel buttons
                                        edited_pairwise_template = st.text_area(
                                            "Edit Pairwise Template:", 
                                            value=current_pairwise_template,
                                            height=400,
                                            key=f"editing_{pairwise_edit_id}"
                                        )
                                        
                                        col2a, col2b = st.columns(2)
                                        with col2a:
                                            if st.button("Save Edits", key=f"save_{pairwise_edit_id}", type="primary"):
                                                save_edits(prompt_key, "pairwise", edited_pairwise_template)
                                        with col2b:
                                            if st.button("Cancel", key=f"cancel_{pairwise_edit_id}"):
                                                cancel_edits()
                                    else:
                                        # Show readonly display with edit button
                                        st.code(current_pairwise_template, language="markdown")
                                        if st.button("✏️ Edit", key=f"edit_{pairwise_edit_id}"):
                                            toggle_edit_mode(pairwise_edit_id)
                                else:
                                    st.info("No pairwise evaluation template available.")
            else:
                st.info("No parameters defined for this metric.")
            
    # Show message if a prompt is being edited
    if st.session_state.currently_editing:
        st.sidebar.success("You're currently editing a prompt. Save or cancel your changes before editing another prompt.")

def add_tab2_content(pipeline_results_for_display):
    display_metrics_and_prompts(pipeline_results_for_display)