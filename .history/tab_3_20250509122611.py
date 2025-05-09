import openai
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ====== TAB 3 FUNCTIONS ========
# Purpose: Run eval on an uploaded dataset using the metrics/prompts generated in Tab 1


def run_evaluations_with_client(client, model, eval_prompt, data_to_evaluate):
    """
    Run a single evaluation with the provided API client
    """
    try:
        # Add explicit instruction to return JSON to the prompt
        eval_prompt_with_json = eval_prompt + "\n\nPlease provide your response in JSON format."
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation assistant who provides detailed, fair assessments of LLM outputs. Always respond with JSON."},
                {"role": "user", "content": eval_prompt_with_json}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def read_excel_file(uploaded_file):
    """
    Read an Excel (XLSX or XLS) file
    """
    import pandas as pd
    return pd.read_excel(uploaded_file)

def read_csv_file(uploaded_file):
    """
    Read a CSV file
    """
    import pandas as pd
    return pd.read_csv(uploaded_file)

def process_uploaded_file_individual(uploaded_file):
    """
    Process the uploaded file based on its type
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        df = read_csv_file(uploaded_file)
    elif file_type in ['xlsx', 'xls']:
        df = read_excel_file(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None
    
    return df

def run_batch_evaluations_multi_output(client, evaluation_model, parameter_prompts, data_rows, 
                                      input_col, output_cols, progress_bar=None):
    """
    Run batch evaluations on multiple rows of data with multiple output columns
    """
    results = {}
    
    # Calculate total evaluations for progress tracking
    total_evals = len(data_rows) * len(parameter_prompts) * len(output_cols)
    completed_evals = 0
    
    # Create a results entry for each output column
    for output_col in output_cols:
        output_results = []
        
        for i, row in enumerate(data_rows):
            row_results = {"row_index": i}
            
            # Get input and output from the row
            input_text = row[input_col] if input_col else ""
            output_text = row[output_col] if output_col else ""
            
            # Skip if missing required data
            if not input_text or not output_text:
                row_results["status"] = "skipped"
                row_results["reason"] = "Missing required data"
                output_results.append(row_results)
                continue
            
            row_results["evals"] = {}
            
            # Process each parameter prompt for this row
            for prompt_key, prompt_template in parameter_prompts.items():
                # Format the template with actual data
                formatted_prompt = prompt_template.replace("{input}", input_text).replace("{output}", output_text)
                
                # Run the evaluation
                eval_result = run_evaluations_with_client(client, evaluation_model, formatted_prompt, {
                    "input": input_text,
                    "output": output_text
                })
                
                # Store the result
                row_results["evals"][prompt_key] = eval_result
                
                # Update progress
                completed_evals += 1
                if progress_bar:
                    progress_bar.progress(completed_evals / total_evals, text=f"Evaluating {completed_evals}/{total_evals}")
            
            output_results.append(row_results)
        
        # Store all results for this output column
        results[output_col] = output_results
    
    return results

def calculate_average_scores(results, prompt_keys):
    """
    Calculate average scores for each metric and output column
    """
    averages = {}
    
    for output_col, output_results in results.items():
        averages[output_col] = {}
        
        for prompt_key in prompt_keys:
            scores = []
            
            for result in output_results:
                if result.get("status") != "skipped":
                    eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                    
                    # Parse the JSON result if possible
                    try:
                        if isinstance(eval_result, str):
                            eval_data = json.loads(eval_result)
                            if "score" in eval_data:
                                scores.append(float(eval_data["score"]))
                            else:
                                # Try to find any numeric value
                                for k, v in eval_data.items():
                                    if isinstance(v, (int, float)):
                                        scores.append(float(v))
                                        break
                    except Exception as e:
                        st.warning(f"Error parsing score for {prompt_key}: {e}")
                        # Continue processing other results
                        continue
            
            if scores:
                averages[output_col][prompt_key] = sum(scores) / len(scores)
            else:
                averages[output_col][prompt_key] = None
    
    return averages

def create_comparison_chart(avg_scores, prompt_keys, output_cols):
    """
    Create a comparison chart using Plotly
    """
    # Add debug logging
    # st.write("Debug - avg_scores structure:", avg_scores)
    # st.write("Debug - prompt_keys:", prompt_keys)
    # st.write("Debug - output_cols:", output_cols)
    
    # Check if we have any scores to display
    has_scores = False
    for output_col in output_cols:
        if output_col in avg_scores:
            for prompt_key in prompt_keys:
                if prompt_key in avg_scores[output_col] and avg_scores[output_col][prompt_key] is not None:
                    has_scores = True
                    break
            if has_scores:
                break
    
    if not has_scores:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No scores available for visualization",
            annotations=[dict(
                text="No numeric scores were found in the evaluation results. Check that your evaluation prompts return a score value.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Combine all figures into a single stacked figure
    combined_fig = go.Figure()
    
    try:
        # Let's create a grouped bar chart
        for i, output_col in enumerate(output_cols):
            scores = []
            for prompt_key in prompt_keys:
                if output_col in avg_scores and prompt_key in avg_scores[output_col]:
                    score = avg_scores[output_col][prompt_key]
                    if score is not None:
                        scores.append(score)
                    else:
                        scores.append(0)
                else:
                    scores.append(0)
            
            combined_fig.add_trace(go.Bar(
                y=prompt_keys,
                x=scores,
                name=output_col,
                orientation='h',
                text=[f"{score:.2f}" if score > 0 else "N/A" for score in scores],
                textposition='auto'
            ))
        
        # Update layout
        combined_fig.update_layout(
            title="Comparison of Output Columns Across Metrics",
            xaxis_title="Score",
            yaxis_title="Metric",
            barmode='group',
            height=300 + (50 * len(prompt_keys)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        # Create a basic error figure
        combined_fig = go.Figure()
        combined_fig.update_layout(
            title="Error creating visualization",
            annotations=[dict(
                text=f"An error occurred: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
    
    return combined_fig

def display_evaluation_results_multi_output(results, prompt_keys, output_cols):
    """
    Display evaluation results for multiple output columns in a formatted way
    """
    st.subheader("Evaluation Results")
    
    # Custom tab name list based on the number of columns
    if len(output_cols) > 1:
        tab_names = ["Comparison"] + [f"Results: {col}" for col in output_cols]
    else:
        tab_names = [f"Results: {col}" for col in output_cols] + ["Comparison"]
    
    # Store current tab selection in session state
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = 0
    
    # Create the actual tabs
    tabs = st.tabs(tab_names)
    
    # Calculate average scores for visualization
    avg_scores = calculate_average_scores(results, prompt_keys)
    
    # Determine the position of the comparison tab
    comparison_index = 0 if len(output_cols) > 2 else len(output_cols)
    
    # Display the comparison tab
    with tabs[comparison_index]:
        st.subheader("Comparison of Average Scores")
        fig = create_comparison_chart(avg_scores, prompt_keys, output_cols)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display individual result tabs
    for i, output_col in enumerate(output_cols):
        # Adjust the tab index based on our tab order
        tab_index = i + 1 if len(output_cols) > 2 else i
        
        with tabs[tab_index]:
            output_results = results[output_col]
            
            # Create a dataframe from the results
            rows = []
            for result in output_results:
                row_data = {"Row": result.get("row_index", "N/A")}
                
                if result.get("status") == "skipped":
                    row_data.update({prompt_key: "SKIPPED" for prompt_key in prompt_keys})
                    row_data["Reason"] = result.get("reason", "Unknown")
                else:
                    for prompt_key in prompt_keys:
                        eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                        
                        # Parse the JSON result if possible
                        try:
                            if isinstance(eval_result, str):
                                eval_data = json.loads(eval_result)
                                if "score" in eval_data:
                                    row_data[prompt_key] = eval_data["score"]
                                else:
                                    # Try to find any numeric value
                                    for k, v in eval_data.items():
                                        if isinstance(v, (int, float)):
                                            row_data[prompt_key] = v
                                            break
                                    else:
                                        row_data[prompt_key] = "See details"
                            else:
                                row_data[prompt_key] = "See details"
                        except:
                            row_data[prompt_key] = "See details"
                
                rows.append(row_data)
            
            if rows:
                results_df = pd.DataFrame(rows)
                st.dataframe(results_df)
                
                # Add detailed expandable sections for each result
                for j, result in enumerate(output_results):
                    if result.get("status") != "skipped":
                        with st.expander(f"Detailed Results for Row {result.get('row_index', j)}"):
                            for prompt_key, eval_result in result.get("evals", {}).items():
                                st.markdown(f"**{prompt_key}**")
                                
                                # Try to prettify JSON
                                try:
                                    if isinstance(eval_result, str):
                                        eval_data = json.loads(eval_result)
                                        st.json(eval_data)
                                    else:
                                        st.write(eval_result)
                                except:
                                    st.write(eval_result)
            else:
                st.info("No evaluation results to display for this output column.")




def export_evaluation_results_multi_output(results, filename_prefix="evaluation_results"):
    """
    Export evaluation results to downloadable formats
    """
    import datetime
    import io
    import uuid
    
    # Get a persistent session ID from session state
    if "export_session_id" not in st.session_state:
        st.session_state.export_session_id = str(uuid.uuid4())
    session_id = st.session_state.export_session_id
    
    # Create a timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export as JSON
    json_data = json.dumps(results, indent=2)
    
    # If download containers not already in session state, create them
    if "download_containers" not in st.session_state:
        st.session_state.download_containers = {}
        col1, col2 = st.columns(2)
        st.session_state.download_containers["col1"] = col1
        st.session_state.download_containers["col2"] = col2
    
    # Get the containers from session state
    col1 = st.session_state.download_containers["col1"]
    col2 = st.session_state.download_containers["col2"]
    
    # Use the containers with explicit keys for the download buttons
    with col1:
        st.download_button(
            label="💾 Download Results as JSON",
            data=json_data,
            file_name=f"{filename_prefix}_{timestamp}.json",
            mime="application/json",
            key=f"json_download_{session_id}"  # Use session ID as key
        )
    
    # Try to create a simplified Excel-friendly version
    try:
        import pandas as pd
        
        # Create a workbook with a sheet for each output column
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            # First, create a summary sheet
            summary_rows = []
            
            # Add rows for each output column and metric combination
            for output_col, output_results in results.items():
                for result in output_results:
                    if result.get("status") != "skipped":
                        row_index = result.get("row_index", "N/A")
                        
                        for prompt_key, eval_result in result.get("evals", {}).items():
                            row_data = {
                                "output_column": output_col,
                                "row_index": row_index,
                                "metric": prompt_key,
                                "raw_result": eval_result
                            }
                            
                            # Try to extract score and justification
                            try:
                                if isinstance(eval_result, str):
                                    eval_data = json.loads(eval_result)
                                    if isinstance(eval_data, dict):
                                        for k, v in eval_data.items():
                                            if k in ["score", "justification", "explanation", "reasoning"]:
                                                row_data[k] = v
                            except:
                                pass
                            
                            summary_rows.append(row_data)
            
            # Create summary DataFrame and write to Excel
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Create a sheet for each output column
            for output_col, output_results in results.items():
                sheet_rows = []
                
                for result in output_results:
                    row_index = result.get("row_index", "N/A")
                    
                    if result.get("status") == "skipped":
                        row_data = {
                            "row_index": row_index,
                            "status": "skipped",
                            "reason": result.get("reason", "Unknown")
                        }
                        sheet_rows.append(row_data)
                    else:
                        for prompt_key, eval_result in result.get("evals", {}).items():
                            row_data = {
                                "row_index": row_index,
                                "metric": prompt_key,
                                "raw_result": eval_result
                            }
                            
                            # Try to extract score and justification
                            try:
                                if isinstance(eval_result, str):
                                    eval_data = json.loads(eval_result)
                                    if isinstance(eval_data, dict):
                                        for k, v in eval_data.items():
                                            if k in ["score", "justification", "explanation", "reasoning"]:
                                                row_data[k] = v
                            except:
                                pass
                            
                            sheet_rows.append(row_data)
                
                # Create DataFrame and write to Excel
                if sheet_rows:
                    col_df = pd.DataFrame(sheet_rows)
                    
                    # Sanitize sheet name (Excel sheet names have restrictions)
                    sheet_name = str(output_col)[:31].replace(':', '_')
                    col_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        with col2:
            st.download_button(
                label="📊 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{filename_prefix}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_download_{session_id}"  # Use session ID as key
            )
    except Exception as e:
        st.warning(f"Could not create Excel export: {e}")


# Function from original code, kept for compatibility with is_conversation=True mode
def run_batch_evaluations(client, evaluation_model, parameter_prompts, data_rows, is_conversation, 
                          input_col, output_col, progress_bar=None):
    """
    Run batch evaluations on multiple rows of data
    """
    results = []
    
    total_evals = len(data_rows) * len(parameter_prompts)
    completed_evals = 0
    
    for i, row in enumerate(data_rows):
        row_results = {"row_index": i}
        
        # Get input and output from the row
        input_text = row[input_col] if input_col else ""
        output_text = row[output_col] if output_col else ""
        
        # Skip if missing required data
        if (not is_conversation and (not input_text or not output_text)) or \
           (is_conversation and not output_text):
            row_results["status"] = "skipped"
            row_results["reason"] = "Missing required data"
            results.append(row_results)
            continue
        
        row_results["evals"] = {}
        
        # Process each parameter prompt for this row
        for prompt_key, prompt_template in parameter_prompts.items():
            # Format the template with actual data
            if is_conversation:
                formatted_prompt = prompt_template.replace("{conversation}", output_text)
            else:
                formatted_prompt = prompt_template.replace("{input}", input_text).replace("{output}", output_text)
            
            # Run the evaluation
            eval_result = run_evaluations_with_client(client, evaluation_model, formatted_prompt, {
                "input": input_text,
                "output": output_text
            })
            
            # Store the result
            row_results["evals"][prompt_key] = eval_result
            
            # Update progress
            completed_evals += 1
            if progress_bar:
                progress_bar.progress(completed_evals / total_evals, text=f"Evaluating {completed_evals}/{total_evals}")
        
        results.append(row_results)
    
    return results

def create_download_section(results):
    """
    Create a separate section for downloads using Streamlit forms
    """
    import datetime
    import io
    import uuid
    import base64
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.subheader("Export Results", anchor="exports")
    
    # Create a form for the downloads
    with st.form(key="download_form"):
        col1, col2 = st.columns(2)
        
        # Store the data once to avoid recreating on every rerun
        json_data = json.dumps(results, indent=2)
        
        # Export JSON using HTML download link instead of st.download_button
        # This prevents Streamlit from rerunning when clicked
        with col1:
            json_filename = f"evaluation_results_{timestamp}.json"
            b64_json = base64.b64encode(json_data.encode()).decode()
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="{json_filename}" class="downloadButton">💾 Download Results as JSON</a>'
            st.markdown(href_json, unsafe_allow_html=True)
            
        # Create Excel data
        excel_data = None
        try:
            import pandas as pd
            
            # Create a workbook with a sheet for each output column
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                # First, create a summary sheet
                summary_rows = []
                
                # Add rows for each output column and metric combination
                for output_col, output_results in results.items():
                    for result in output_results:
                        if result.get("status") != "skipped":
                            row_index = result.get("row_index", "N/A")
                            
                            for prompt_key, eval_result in result.get("evals", {}).items():
                                row_data = {
                                    "output_column": output_col,
                                    "row_index": row_index,
                                    "metric": prompt_key,
                                    "raw_result": eval_result
                                }
                                
                                # Try to extract score and justification
                                try:
                                    if isinstance(eval_result, str):
                                        eval_data = json.loads(eval_result)
                                        if isinstance(eval_data, dict):
                                            for k, v in eval_data.items():
                                                if k in ["score", "justification", "explanation", "reasoning"]:
                                                    row_data[k] = v
                                except:
                                    pass
                                
                                summary_rows.append(row_data)
                
                # Create summary DataFrame and write to Excel
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Create a sheet for each output column
                for output_col, output_results in results.items():
                    sheet_rows = []
                    
                    for result in output_results:
                        row_index = result.get("row_index", "N/A")
                        
                        if result.get("status") == "skipped":
                            row_data = {
                                "row_index": row_index,
                                "status": "skipped",
                                "reason": result.get("reason", "Unknown")
                            }
                            sheet_rows.append(row_data)
                        else:
                            for prompt_key, eval_result in result.get("evals", {}).items():
                                row_data = {
                                    "row_index": row_index,
                                    "metric": prompt_key,
                                    "raw_result": eval_result
                                }
                                
                                # Try to extract score and justification
                                try:
                                    if isinstance(eval_result, str):
                                        eval_data = json.loads(eval_result)
                                        if isinstance(eval_data, dict):
                                            for k, v in eval_data.items():
                                                if k in ["score", "justification", "explanation", "reasoning"]:
                                                    row_data[k] = v
                                except:
                                    pass
                                
                                sheet_rows.append(row_data)
                    
                    # Create DataFrame and write to Excel
                    if sheet_rows:
                        col_df = pd.DataFrame(sheet_rows)
                        
                        # Sanitize sheet name (Excel sheet names have restrictions)
                        sheet_name = str(output_col)[:31].replace(':', '_')
                        col_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get the Excel data
            excel_data = excel_buffer.getvalue()
        except Exception as e:
            st.warning(f"Could not create Excel export: {e}")
        
        # Excel download
        with col2:
            if excel_data is not None:
                excel_filename = f"evaluation_results_{timestamp}.xlsx"
                b64_excel = base64.b64encode(excel_data).decode()
                href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{excel_filename}" class="downloadButton">📊 Download Results as Excel</a>'
                st.markdown(href_excel, unsafe_allow_html=True)
        
        # Add a dummy submit button to create the form
        submitted = st.form_submit_button("Refresh Download Links")
    
    # Add some CSS for the download links
    st.markdown("""
        <style>
        .downloadButton {
            display: inline-block;
            padding: 0.5em 1em;
            background-color: #4e8cff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            margin: 0.5em 0;
            text-align: center;
            width: 100%;
        }
        .downloadButton:hover {
            background-color: #3a7ce2;
        }
        </style>
    """, unsafe_allow_html=True)


# ========== DOWNLOAD SOLUTION ==========

def create_download_section(results):
    """
    Create a separate section for downloads using Streamlit forms
    """
    import datetime
    import io
    import uuid
    import base64
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.subheader("Export Results", anchor="exports")
    
    # Create a form for the downloads
    with st.form(key="download_form"):
        col1, col2 = st.columns(2)
        
        # Store the data once to avoid recreating on every rerun
        json_data = json.dumps(results, indent=2)
        
        # Export JSON using HTML download link instead of st.download_button
        # This prevents Streamlit from rerunning when clicked
        with col1:
            json_filename = f"evaluation_results_{timestamp}.json"
            b64_json = base64.b64encode(json_data.encode()).decode()
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="{json_filename}" class="downloadButton">💾 Download Results as JSON</a>'
            st.markdown(href_json, unsafe_allow_html=True)
            
        # Create Excel data
        excel_data = None
        try:
            import pandas as pd
            
            # Create a workbook with a sheet for each output column
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                # First, create a summary sheet
                summary_rows = []
                
                # Add rows for each output column and metric combination
                for output_col, output_results in results.items():
                    for result in output_results:
                        if result.get("status") != "skipped":
                            row_index = result.get("row_index", "N/A")
                            
                            for prompt_key, eval_result in result.get("evals", {}).items():
                                row_data = {
                                    "output_column": output_col,
                                    "row_index": row_index,
                                    "metric": prompt_key,
                                    "raw_result": eval_result
                                }
                                
                                # Try to extract score and justification
                                try:
                                    if isinstance(eval_result, str):
                                        eval_data = json.loads(eval_result)
                                        if isinstance(eval_data, dict):
                                            for k, v in eval_data.items():
                                                if k in ["score", "justification", "explanation", "reasoning"]:
                                                    row_data[k] = v
                                except:
                                    pass
                                
                                summary_rows.append(row_data)
                
                # Create summary DataFrame and write to Excel
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Create a sheet for each output column
                for output_col, output_results in results.items():
                    sheet_rows = []
                    
                    for result in output_results:
                        row_index = result.get("row_index", "N/A")
                        
                        if result.get("status") == "skipped":
                            row_data = {
                                "row_index": row_index,
                                "status": "skipped",
                                "reason": result.get("reason", "Unknown")
                            }
                            sheet_rows.append(row_data)
                        else:
                            for prompt_key, eval_result in result.get("evals", {}).items():
                                row_data = {
                                    "row_index": row_index,
                                    "metric": prompt_key,
                                    "raw_result": eval_result
                                }
                                
                                # Try to extract score and justification
                                try:
                                    if isinstance(eval_result, str):
                                        eval_data = json.loads(eval_result)
                                        if isinstance(eval_data, dict):
                                            for k, v in eval_data.items():
                                                if k in ["score", "justification", "explanation", "reasoning"]:
                                                    row_data[k] = v
                                except:
                                    pass
                                
                                sheet_rows.append(row_data)
                    
                    # Create DataFrame and write to Excel
                    if sheet_rows:
                        col_df = pd.DataFrame(sheet_rows)
                        
                        # Sanitize sheet name (Excel sheet names have restrictions)
                        sheet_name = str(output_col)[:31].replace(':', '_')
                        col_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get the Excel data
            excel_data = excel_buffer.getvalue()
        except Exception as e:
            st.warning(f"Could not create Excel export: {e}")
        
        # Excel download
        with col2:
            if excel_data is not None:
                excel_filename = f"evaluation_results_{timestamp}.xlsx"
                b64_excel = base64.b64encode(excel_data).decode()
                href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{excel_filename}" class="downloadButton">📊 Download Results as Excel</a>'
                st.markdown(href_excel, unsafe_allow_html=True)
        
        # Add a dummy submit button to create the form
        submitted = st.form_submit_button("Refresh Download Links")
    
    # Add some CSS for the download links
    st.markdown("""
        <style>
        .downloadButton {
            display: inline-block;
            padding: 0.5em 1em;
            background-color: #4e8cff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            margin: 0.5em 0;
            text-align: center;
            width: 100%;
        }
        .downloadButton:hover {
            background-color: #3a7ce2;
        }
        </style>
    """, unsafe_allow_html=True)


# ========== COMPARISON FIRST TAB SOLUTION ==========

def display_evaluation_results_multi_output(results, prompt_keys, output_cols):
    """
    Display evaluation results for multiple output columns in a formatted way
    """
    st.subheader("Evaluation Results")
    
    # Custom tab name list based on the number of columns
    if len(output_cols) > 2:
        tab_names = ["Comparison"] + [f"Results: {col}" for col in output_cols]
    else:
        tab_names = [f"Results: {col}" for col in output_cols] + ["Comparison"]
    
    # Store current tab selection in session state
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = 0
    
    # Create the actual tabs
    tabs = st.tabs(tab_names)
    
    # Calculate average scores for visualization
    avg_scores = calculate_average_scores(results, prompt_keys)
    
    # Determine the position of the comparison tab
    comparison_index = 0 if len(output_cols) > 2 else len(output_cols)
    
    # Display the comparison tab
    with tabs[comparison_index]:
        st.subheader("Comparison of Average Scores")
        fig = create_comparison_chart(avg_scores, prompt_keys, output_cols)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display individual result tabs
    for i, output_col in enumerate(output_cols):
        # Adjust the tab index based on our tab order
        tab_index = i + 1 if len(output_cols) > 2 else i
        
        with tabs[tab_index]:
            output_results = results[output_col]
            
            # Create a dataframe from the results
            rows = []
            for result in output_results:
                row_data = {"Row": result.get("row_index", "N/A")}
                
                if result.get("status") == "skipped":
                    row_data.update({prompt_key: "SKIPPED" for prompt_key in prompt_keys})
                    row_data["Reason"] = result.get("reason", "Unknown")
                else:
                    for prompt_key in prompt_keys:
                        eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                        
                        # Parse the JSON result if possible
                        try:
                            if isinstance(eval_result, str):
                                eval_data = json.loads(eval_result)
                                if "score" in eval_data:
                                    row_data[prompt_key] = eval_data["score"]
                                else:
                                    # Try to find any numeric value
                                    for k, v in eval_data.items():
                                        if isinstance(v, (int, float)):
                                            row_data[prompt_key] = v
                                            break
                                    else:
                                        row_data[prompt_key] = "See details"
                            else:
                                row_data[prompt_key] = "See details"
                        except:
                            row_data[prompt_key] = "See details"
                
                rows.append(row_data)
            
            if rows:
                results_df = pd.DataFrame(rows)
                st.dataframe(results_df)
                
                # Add detailed expandable sections for each result
                for j, result in enumerate(output_results):
                    if result.get("status") != "skipped":
                        with st.expander(f"Detailed Results for Row {result.get('row_index', j)}"):
                            for prompt_key, eval_result in result.get("evals", {}).items():
                                st.markdown(f"**{prompt_key}**")
                                
                                # Try to prettify JSON
                                try:
                                    if isinstance(eval_result, str):
                                        eval_data = json.loads(eval_result)
                                        st.json(eval_data)
                                    else:
                                        st.write(eval_result)
                                except:
                                    st.write(eval_result)
            else:
                st.info("No evaluation results to display for this output column.")


# ========== UPDATED TAB3 CONTENT ==========

def add_tab3_content():
    """
    Add the content for Tab 3 (Run Evaluations with multi-output support)
    """
    st.header("Run Evaluations on Your Data")
    st.info("Upload a spreadsheet with your data and run evaluations based on the metrics you've generated.")
    
    # Check if metrics are available
    if not st.session_state.pipeline_results:
        st.warning("Please generate metrics in Tab 1 first before running evaluations.")
        return
    
    # Get metrics and templates from session state
    pipeline_results = st.session_state.pipeline_results
    is_conversation = pipeline_results.get("is_conversation", False)
    parameter_templates = pipeline_results.get("parameter_prompt_templates", {})
    metrics_json = pipeline_results.get("metrics_json", {})
    
    # Determine metrics list structure
    metrics_list = []
    if isinstance(metrics_json, dict) and "metrics" in metrics_json:
        metrics_list = metrics_json["metrics"]
    elif isinstance(metrics_json, list):
        metrics_list = metrics_json
    
    if not metrics_list:
        st.warning("No metrics found in the results.")
        return
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV, XLSX, or XLS file", 
        type=["csv", "xlsx", "xls"],
        key="multi_eval_file_uploader"
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = process_uploaded_file_individual(uploaded_file)
        
        if df is not None:
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            
            # Show a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection based on conversation type
            st.subheader("Select which columns to run")
            
            # Get all column names
            column_names = list(df.columns)
            
            if is_conversation:
                st.info("This is a conversation evaluation. Please select one or more columns that contain entire conversation transcripts.")
                conversation_cols = st.multiselect("Conversation Column(s)", column_names)
                input_col = None  # Not used for conversation
                
                # For conversation eval, use the selected conversation columns as output columns
                output_cols = conversation_cols
            else:
                st.info("This is a single-turn evaluation. Please select the column name with the input and the column name with the output.")
                
                # Select input column
                input_col = st.selectbox("Input Column", [""] + column_names)
                
                # Select multiple output columns
                output_cols = st.multiselect(
                    "Output Column(s) - Select one or more to compare", 
                    column_names
                )
            
            # Metrics selection
            st.subheader("Select Metrics to Evaluate")
            
            # Create a checkbox for each metric
            selected_metrics = {}
            for metric in metrics_list:
                metric_name = metric.get("metric", "Unnamed Metric")
                parameters = metric.get("parameters", [])
                
                # Create an expandable section for each metric
                with st.expander(f"{metric_name} - {len(parameters)} parameters", expanded=True):
                    # Show metric description
                    st.markdown(f"**Description:** {metric.get('description', 'No description')}")
                    
                    # Create checkboxes for each parameter
                    param_checkboxes = []
                    for param in parameters:
                        if isinstance(param, dict) and len(param) > 0:
                            param_key = list(param.keys())[0]
                            param_description = param[param_key]
                            prompt_key = f"{metric_name}::{param_key}"
                            
                            # Only show checkbox if we have a template for this parameter
                            if prompt_key in parameter_templates:
                                is_selected = st.checkbox(
                                    f"{param_key}: {param_description}", 
                                    value=True
                                )
                                if is_selected:
                                    selected_metrics[prompt_key] = parameter_templates[prompt_key]
            
            # Evaluation model selection
            st.subheader("Evaluation Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                eval_model = st.selectbox(
                    "Evaluation Model", 
                    ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "o3-mini-2025-01-31"], 
                    index=0,
                    help="Model used for running evaluations"
                )
            
            with col2:
                sample_size = st.number_input(
                    "Sample Size (0 for all rows)", 
                    min_value=0, 
                    value=min(len(df), 5),
                    help="Number of rows to evaluate (0 means all rows)"
                )
            
            # Add a note about required columns
            if is_conversation and not output_cols:
                st.warning("Please select at least one conversation column.")
            elif not is_conversation and (not input_col or not output_cols):
                st.warning("Please select an input column and at least one output column.")
            elif not selected_metrics:
                st.warning("Please select at least one metric parameter to evaluate.")
            else:
                # Run evaluations form for better state management
                with st.form(key="run_evaluations_form"):
                    st.write("Click the button below to start evaluations.")
                    submitted = st.form_submit_button("Run Evaluations")
                    
                    if submitted:
                        # Check if we have an API client
                        if not st.session_state.client:
                            st.error("Please set up your API key in the sidebar first.")
                            return
                        
                        client = st.session_state.client
                        
                        # Prepare data for evaluation
                        data_to_evaluate = df.to_dict('records')
                        
                        # Sample if needed
                        if sample_size > 0 and sample_size < len(data_to_evaluate):
                            import random
                            random.seed(42)  # For reproducibility
                            data_to_evaluate = random.sample(data_to_evaluate, sample_size)
                        
                        # Store evaluation parameters in session state for later retrieval
                        st.session_state.eval_parameters = {
                            "client": client,
                            "model": eval_model,
                            "is_conversation": is_conversation,
                            "input_col": input_col,
                            "output_cols": output_cols,
                            "selected_metrics": selected_metrics,
                            "data_to_evaluate": data_to_evaluate
                        }
                
                # Check if form was submitted (will be true after the form's rerun)
                if "eval_parameters" in st.session_state:
                    # Get parameters from session state
                    eval_params = st.session_state.eval_parameters
                    client = eval_params["client"]
                    eval_model = eval_params["model"]
                    is_conversation = eval_params["is_conversation"]
                    input_col = eval_params["input_col"]
                    output_cols = eval_params["output_cols"]
                    selected_metrics = eval_params["selected_metrics"]
                    data_to_evaluate = eval_params["data_to_evaluate"]
                    
                    # Create a progress bar
                    progress_bar = st.progress(0, text="Starting evaluations...")
                    
                    # Run evaluations
                    try:
                        with st.spinner("Running evaluations..."):
                            if is_conversation:
                                # For conversation eval with multiple conversation columns
                                multi_results = {}
                                
                                # Process each conversation column individually
                                total_conv_cols = len(output_cols)
                                for i, conv_col in enumerate(output_cols):
                                    # Update progress message
                                    progress_msg = f"Processing conversation column {i+1}/{total_conv_cols}: {conv_col}"
                                    if progress_bar:
                                        progress_bar.progress((i/total_conv_cols)*0.1, text=progress_msg)
                                    
                                    # Run evaluation for this conversation column
                                    results = run_batch_evaluations(
                                        client=client,
                                        evaluation_model=eval_model,
                                        parameter_prompts=selected_metrics,
                                        data_rows=data_to_evaluate,
                                        is_conversation=is_conversation,
                                        input_col=None,
                                        output_col=conv_col,
                                        progress_bar=progress_bar
                                    )
                                    
                                    # Store results for this conversation column
                                    multi_results[conv_col] = results
                            else:
                                # For single-turn eval with multiple outputs
                                multi_results = run_batch_evaluations_multi_output(
                                    client=client,
                                    evaluation_model=eval_model,
                                    parameter_prompts=selected_metrics,
                                    data_rows=data_to_evaluate,
                                    input_col=input_col,
                                    output_cols=output_cols,
                                    progress_bar=progress_bar
                                )
                        
                        # Count total rows processed
                        total_rows = sum(len(results) for results in multi_results.values())
                        st.success(f"Evaluations completed for {total_rows} rows across {len(multi_results)} output columns!")
                        
                        # Store results in session state
                        st.session_state.multi_evaluation_results = multi_results
                        st.session_state.selected_metric_keys = list(selected_metrics.keys())
                        st.session_state.selected_output_cols = list(multi_results.keys())
                        
                        # Display results
                        display_evaluation_results_multi_output(multi_results, list(selected_metrics.keys()), list(multi_results.keys()))
                        
                        # Use new download approach that doesn't use Streamlit download buttons
                        create_download_section(multi_results)
                            
                    except Exception as e:
                        st.error(f"Error running evaluations: {e}")
                        st.exception(e)  # Show detailed error info
                    finally:
                        # Complete the progress bar
                        progress_bar.progress(1.0, text="Evaluations completed!")
                        
                        # Clear the form submission parameters to prevent auto-rerunning on page reload
                        if "eval_parameters" in st.session_state:
                            del st.session_state.eval_parameters

    # Check if we have results to show
    elif "multi_evaluation_results" in st.session_state and st.session_state.multi_evaluation_results:
        st.subheader("Previous Evaluation Results")
        st.info("Showing results from your last evaluation run. Upload a new file to run new evaluations.")
        
        # Get the saved prompt keys and output columns from session state
        selected_metric_keys = st.session_state.get("selected_metric_keys", list(parameter_templates.keys()))
        selected_output_cols = st.session_state.get("selected_output_cols", list(st.session_state.multi_evaluation_results.keys()))
        
        # Display previous results
        display_evaluation_results_multi_output(
            st.session_state.multi_evaluation_results,
            selected_metric_keys,
            selected_output_cols
        )
        
        # Use HTML-based downloads instead of Streamlit buttons
        create_download_section(st.session_state.multi_evaluation_results)
