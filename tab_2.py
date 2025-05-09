import streamlit as st

# ====== TAB 2 FUNCTIONS ========

# All this tab does is show the metrics and parameters and their prompts, for closer user inspection

def display_metrics_and_prompts(pipeline_results):
    """
    Display all metrics, descriptions, and associated prompts in Tab 2
    """
    if not pipeline_results:
        st.info("No metrics available yet. Please generate metrics in Tab 1 first.")
        return
    
    # Extract metrics from the results
    metrics_json = pipeline_results.get("metrics_json", {})
    param_templates = pipeline_results.get("parameter_prompt_templates", {})
    pairwise_templates = pipeline_results.get("pairwise_prompt_templates", {})
    
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
            
            # Display prompt templates for each parameter
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
                                    st.code(param_templates[prompt_key], language="markdown")
                                else:
                                    st.info("No parameter evaluation template available.")
                            
                            with col2:
                                st.markdown("##### Pairwise Evaluation Template")
                                if prompt_key in pairwise_templates:
                                    st.code(pairwise_templates[prompt_key], language="markdown")
                                else:
                                    st.info("No pairwise evaluation template available.")
            else:
                st.info("No parameters defined for this metric.")

def add_tab2_content(st.session_state.pipeline_results):
    tab_2 = display_metrics_and_prompts(st.session_state.pipeline_results)
    return tab_2
