import openai
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tab_2 import get_current_prompts
from genai_utils import create_client_with_model
import concurrent.futures
import threading
import time
from typing import Dict, List, Any

# ====== ENHANCED TAB 3 WITH PARALLEL PROCESSING + FULL VISUALIZATIONS ========

def run_single_evaluation_threadsafe(client, model, eval_prompt, data_to_evaluate, row_index, prompt_key):
    """
    Run a single evaluation in a thread-safe manner
    Returns a tuple of (row_index, prompt_key, result)
    """
    try:
        # Add explicit instruction to return JSON to the prompt
        eval_prompt_with_json = eval_prompt + "\n\nPlease provide your response in JSON format."
        
        # Create client with model-specific headers  
        model_client = create_client_with_model(model)
        
        response = model_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation assistant who provides detailed, fair assessments of LLM outputs. Always respond with JSON."},
                {"role": "user", "content": eval_prompt_with_json}
            ],
            response_format={"type": "json_object"}
        )
        
        return (row_index, prompt_key, response.choices[0].message.content)
    except Exception as e:
        return (row_index, prompt_key, {"error": str(e), "status": "failed"})


# ====== COPIED VISUALIZATION FUNCTIONS FROM ORIGINAL TAB_3.PY ========

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
                        # Continue processing other results
                        continue
            
            if scores:
                averages[output_col][prompt_key] = sum(scores) / len(scores)
            else:
                averages[output_col][prompt_key] = None
    
    return averages


def create_comparison_chart(avg_scores, prompt_keys, output_cols):
    """
    Create a comparison chart using Plotly (original bar chart)
    """
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


def create_spider_chart(avg_scores, prompt_keys, output_cols):
    """
    Create a radar chart with outlined traces for better visibility
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    # Convert structure for easier processing
    df = pd.DataFrame(index=prompt_keys)
    
    # Build a dataframe with output columns and metrics
    for output_col in output_cols:
        values = []
        for prompt_key in prompt_keys:
            if output_col in avg_scores and prompt_key in avg_scores[output_col]:
                score = avg_scores[output_col][prompt_key]
                df.loc[prompt_key, output_col] = score if score is not None else 0
            else:
                df.loc[prompt_key, output_col] = 0
    
    # Extract the metric names to make them more readable
    shortened_metrics = []
    for key in prompt_keys:
        if "::" in key:
            parts = key.split("::")
            if len(parts) > 1:
                shortened_metrics.append(parts[-1])
            else:
                shortened_metrics.append(key)
        else:
            shortened_metrics.append(key)
    
    # Create a radar chart
    fig = go.Figure()
    
    # Define colors with higher contrast
    colors = ['rgba(31, 119, 180, 1)', 'rgba(255, 127, 14, 1)', 
              'rgba(44, 160, 44, 1)', 'rgba(214, 39, 40, 1)',
              'rgba(148, 103, 189, 1)', 'rgba(140, 86, 75, 1)']
    
    # Add a trace for EACH output column
    for i, output_col in enumerate(df.columns):
        # Get values for this output column
        values = df[output_col].tolist()
        
        # Get a color for this trace
        color_idx = i % len(colors)
        color = colors[color_idx]
        
        # Add the trace with NO fill but with lines
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=shortened_metrics,
            fill=None,  # No fill
            mode='lines+markers',  # Lines and markers
            line=dict(
                color=color,
                width=3  # Thicker lines
            ),
            marker=dict(
                size=8,  # Bigger markers
                color=color
            ),
            name=output_col
        ))
    
    # Calculate max value for scaling
    max_value = 1.0
    
    # Update layout for better visibility
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value],
                tickfont=dict(size=12),
                linewidth=2,
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='black'),
                linewidth=2,
                gridwidth=1
            )
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=14),
            borderwidth=1
        ),
        height=600,
        width=800,  # Larger size
        margin=dict(l=80, r=80, t=50, b=50)  # More margin space
    )
    
    return fig


def create_evaluation_summary(avg_scores, prompt_keys, output_cols):
    """
    Create a summary of scores, maximums, and percentages
    that only counts metrics that were actually run
    """
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
    
    # First, create a dataframe with all the scores
    summary_df = pd.DataFrame(index=output_cols)
    
    # Set maximum score per item to 1 (0-1 scale)
    max_score_per_item = 1
    
    # Track which metrics were actually run for each output column
    metrics_run = {output_col: [] for output_col in output_cols}
    
    # For each metric
    for prompt_key in prompt_keys:
        metric_name = prompt_key.split("::")[-1] if "::" in prompt_key else prompt_key
        
        for output_col in output_cols:
            if output_col in avg_scores and prompt_key in avg_scores[output_col]:
                score = avg_scores[output_col][prompt_key]
                if score is not None:
                    summary_df.loc[output_col, metric_name] = score
                    # Track that this metric was actually run
                    metrics_run[output_col].append(metric_name)
                else:
                    summary_df.loc[output_col, metric_name] = 0
            else:
                summary_df.loc[output_col, metric_name] = 0
    
    # Calculate row totals - only for metrics that were actually run
    for output_col in output_cols:
        # Calculate the total score for metrics that were run
        total_score = sum(summary_df.loc[output_col, metric] for metric in metrics_run[output_col])
        summary_df.loc[output_col, "Total"] = total_score
        
        # Set max possible based on number of metrics actually run
        max_possible = len(metrics_run[output_col]) * max_score_per_item
        summary_df.loc[output_col, "Max Possible"] = max_possible
        
        # Calculate percentage
        if max_possible > 0:
            summary_df.loc[output_col, "Percentage"] = (total_score / max_possible * 100).round(1)
        else:
            summary_df.loc[output_col, "Percentage"] = 0
    
    # Create score bars visualization
    create_score_bars(summary_df, max_score_per_item)
    
    return summary_df


def create_score_bars(summary_df, max_score_per_item):
    """
    Create interactive bar charts showing scores, totals, and percentages
    with correct scale and only counting metrics that were run
    """
    import plotly.graph_objects as go
    
    # Remove utility columns for visualization
    viz_df = summary_df.drop(columns=["Max Possible", "Percentage"])
    
    # Create a second chart for totals
    fig2 = go.Figure()
    
    # Add total scores bars
    fig2.add_trace(go.Bar(
        x=summary_df.index,
        y=summary_df["Total"],
        marker_color='royalblue',
        text=summary_df["Total"].round(2),
        textposition='auto',
        name="Total Score"
    ))
    
    # Add percentage as a line
    fig2.add_trace(go.Scatter(
        x=summary_df.index,
        y=summary_df["Percentage"],
        mode='lines+markers+text',
        marker=dict(size=10, color='red'),
        line=dict(width=3, dash='dot', color='red'),
        text=summary_df["Percentage"].round(1).astype(str) + '%',
        textposition='top center',
        yaxis='y2',
        name="Percentage of Maximum"
    ))
    
    # Add a line for each model's maximum possible score
    for i, idx in enumerate(summary_df.index):
        max_val = summary_df.loc[idx, "Max Possible"]
        fig2.add_trace(go.Scatter(
            x=[idx],
            y=[max_val],
            mode='markers',
            marker=dict(
                symbol='line-ns',
                size=16,
                color='gray',
                line=dict(width=2)
            ),
            name=f"Max for {idx}" if i == 0 else None,  # Only add to legend once
            showlegend=(i == 0)
        ))
    
    # Customize layout for totals chart
    max_y_value = max(summary_df["Max Possible"].max() * 1.1, summary_df["Total"].max() * 1.2)
    
    fig2.update_layout(
        title="Total Scores and Percentages by Model (only counting metrics that were run)",
        xaxis_title="Model",
        yaxis=dict(
            title="Total Score",
            range=[0, max_y_value]
        ),
        yaxis2=dict(
            title=dict(
                text="Percentage",
                font=dict(color='red'),
            ),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right',
            range=[0, 110],
            ticksuffix='%'
        ),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Display chart
    st.plotly_chart(fig2, use_container_width=True)


def _display_tab_content(output_results, prompt_keys):
    """Helper function to display the content of a result tab with averages"""
    # Create a dataframe from the results
    rows = []
    metric_values = {prompt_key: [] for prompt_key in prompt_keys}  # To track values for averaging
    
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
                            score = float(eval_data["score"])
                            row_data[prompt_key] = score
                            metric_values[prompt_key].append(score)
                        else:
                            # Try to find any numeric value
                            found_numeric = False
                            for k, v in eval_data.items():
                                if isinstance(v, (int, float)):
                                    score = float(v)
                                    row_data[prompt_key] = score
                                    metric_values[prompt_key].append(score)
                                    found_numeric = True
                                    break
                            
                            if not found_numeric:
                                row_data[prompt_key] = "See details"
                    else:
                        row_data[prompt_key] = "See details"
                except:
                    row_data[prompt_key] = "See details"
        
        rows.append(row_data)
    
    if rows:
        # Create a new row for averages
        avg_row = {"Row": "AVERAGE"}
        
        # Calculate average for each metric
        for prompt_key, values in metric_values.items():
            if values:  # Only calculate if we have numeric values
                avg_row[prompt_key] = sum(values) / len(values)
            else:
                avg_row[prompt_key] = "N/A"
        
        # Add this row to the dataframe
        results_df = pd.DataFrame(rows)
        
        # Display the regular results table
        st.dataframe(results_df)
        
        # Display the average row in a separate, highlighted table
        avg_df = pd.DataFrame([avg_row])
        st.markdown("### Average Scores")
        st.dataframe(avg_df, use_container_width=True)
        
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


def display_evaluation_results_multi_output(results, prompt_keys, output_cols):
    """
    Display evaluation results for multiple output columns in a formatted way
    With comparison tab first for 2+ columns
    """
    st.subheader("Evaluation Results")
    
    # Calculate average scores for visualization
    avg_scores = calculate_average_scores(results, prompt_keys)
    
    # UPDATED CONDITION: For 2+ columns, show comparison first
    if len(output_cols) >= 2:
        # Create comparison tab first
        tab1 = st.tabs(["Comparison"])[0]
        with tab1:
            st.subheader("Comparison of Average Scores")
            
            # Bar chart comparison
            fig = create_comparison_chart(avg_scores, prompt_keys, output_cols)
            st.plotly_chart(fig, use_container_width=True)
            
            # Try to add spider chart with error handling
            st.subheader("Radar Chart Comparison")
            try:
                spider_fig = create_spider_chart(avg_scores, prompt_keys, output_cols)
                st.plotly_chart(spider_fig)
                
            except Exception as e:
                st.error(f"Could not create radar chart: {str(e)}")

            # Summary section with totals, maximums, and percentages
            summary_df = create_evaluation_summary(avg_scores, prompt_keys, output_cols)
        
        # Then create all the individual result tabs
        result_tabs = st.tabs([f"Results: {col}" for col in output_cols])
        for i, output_col in enumerate(output_cols):
            with result_tabs[i]:
                _display_tab_content(results[output_col], prompt_keys)
                
    else:
        # For a single column, use original ordering (result first, then comparison)
        result_tabs = st.tabs([f"Results: {col}" for col in output_cols])
        for i, output_col in enumerate(output_cols):
            with result_tabs[i]:
                _display_tab_content(results[output_col], prompt_keys)
        
        # Add comparison at the end
        comparison_tab = st.tabs(["Comparison"])[0]
        with comparison_tab:
            st.subheader("Comparison of Average Scores")
            
            # Bar chart
            fig = create_comparison_chart(avg_scores, prompt_keys, output_cols)
            st.plotly_chart(fig, use_container_width=True)
            
            # Try to add spider chart with error handling
            st.subheader("Radar Chart Comparison")
            try:
                spider_fig = create_spider_chart(avg_scores, prompt_keys, output_cols)
                st.plotly_chart(spider_fig)
                
                st.markdown("""
                **How to read this chart:**
                
                - Each axis represents a metric
                - Each colored line represents an output column
                - Points further from center have higher scores
                - Compare the shapes to see strengths/weaknesses
                """)
            except Exception as e:
                st.error(f"Could not create radar chart: {str(e)}")
            
            # Summary section with totals, maximums, and percentages
            summary_df = create_evaluation_summary(avg_scores, prompt_keys, output_cols)


# ====== PARALLEL PROCESSING FUNCTIONS ========


def run_parallel_batch_evaluations(client, evaluation_model, parameter_prompts, data_rows, 
                                  input_col, output_cols, batch_size=5, max_workers=5):
    """
    Run evaluations in parallel batches with automatic saving between batches
    
    Args:
        client: API client
        evaluation_model: Model to use for evaluations
        parameter_prompts: Dictionary of prompt templates
        data_rows: List of data rows to evaluate
        input_col: Column name for input
        output_cols: List of output column names
        batch_size: Number of rows to process per batch (default: 5)
        max_workers: Number of parallel threads (default: 5)
    """
    
    # Initialize session state for batch results if not exists
    if "batch_eval_results" not in st.session_state:
        st.session_state.batch_eval_results = {}
        st.session_state.batch_eval_progress = 0
        st.session_state.batch_eval_total = len(data_rows) * len(parameter_prompts) * len(output_cols)
    
    # Create placeholders for real-time updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    batch_summary_placeholder = st.empty()
    heartbeat_placeholder = st.empty()
    
    # Determine starting point
    completed_batches = st.session_state.get("completed_batch_count", 0)
    start_row = completed_batches * batch_size
    
    if start_row > 0:
        status_placeholder.info(f"🔄 Resuming from batch {completed_batches + 1}, row {start_row + 1}")
        time.sleep(1)
    
    total_rows = len(data_rows)
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    try:
        # Process in batches
        for batch_idx in range(completed_batches, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_rows)
            current_batch = data_rows[batch_start:batch_end]
            
            # Update status
            status_placeholder.info(f"🚀 Processing batch {batch_idx + 1}/{total_batches} (rows {batch_start + 1}-{batch_end})")
            
            # Create progress bar for current batch
            batch_progress = progress_placeholder.progress(0, text=f"Starting batch {batch_idx + 1}...")
            
            # Prepare all evaluation tasks for this batch
            evaluation_tasks = []
            
            for row_idx, row in enumerate(current_batch):
                global_row_idx = batch_start + row_idx
                
                # Get input text and convert to string safely
                input_text = str(row[input_col]) if input_col and row[input_col] is not None else ""
                
                # Process each output column
                for output_col in output_cols:
                    output_text = str(row[output_col]) if output_col and row[output_col] is not None else ""
                    
                    # Skip if missing required data (after converting to string)
                    if (input_col and not input_text.strip()) or not output_text.strip():
                        continue
                    
                    # Process each parameter prompt
                    for prompt_key, prompt_template in parameter_prompts.items():
                        # Format the template with actual data - ensure all values are strings
                        safe_input = str(input_text) if input_text is not None else ""
                        safe_output = str(output_text) if output_text is not None else ""
                        formatted_prompt = prompt_template.replace("{input}", safe_input).replace("{output}", safe_output)
                        
                        # Create task tuple
                        task = (
                            client, evaluation_model, formatted_prompt, 
                            {"input": input_text, "output": output_text},
                            global_row_idx, f"{output_col}::{prompt_key}"
                        )
                        evaluation_tasks.append(task)
            
            # Execute evaluations in parallel
            batch_results = {}
            completed_tasks = 0
            total_tasks = len(evaluation_tasks)
            
            # Update heartbeat every few seconds
            last_heartbeat = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(run_single_evaluation_threadsafe, *task): task 
                    for task in evaluation_tasks
                }
                
                # Process completed tasks as they finish
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        row_idx, prompt_key, result = future.result()
                        
                        # Store result
                        if row_idx not in batch_results:
                            batch_results[row_idx] = {"row_index": row_idx, "evals": {}}
                        
                        batch_results[row_idx]["evals"][prompt_key] = result
                        
                        completed_tasks += 1
                        
                        # Update progress
                        progress_pct = completed_tasks / total_tasks
                        batch_progress.progress(
                            progress_pct, 
                            text=f"Batch {batch_idx + 1}: {completed_tasks}/{total_tasks} evaluations complete"
                        )
                        
                        # Update heartbeat periodically
                        current_time = time.time()
                        if current_time - last_heartbeat > 3:
                            heartbeat_placeholder.text(f"⏱️ Processing... {completed_tasks}/{total_tasks}")
                            last_heartbeat = current_time
                            time.sleep(0.1)  # Small pause to let UI update
                        
                    except Exception as e:
                        st.error(f"Task failed: {e}")
                        continue
            
            # Save batch results to session state
            for row_idx, row_result in batch_results.items():
                st.session_state.batch_eval_results[row_idx] = row_result
            
            # Update progress tracking
            st.session_state.completed_batch_count = batch_idx + 1
            st.session_state.batch_eval_progress += len(batch_results)
            
            # Show batch completion summary
            batch_summary_placeholder.success(
                f"✅ Batch {batch_idx + 1} complete! Processed {len(batch_results)} rows. "
                f"Total progress: {len(st.session_state.batch_eval_results)}/{total_rows} rows"
            )
            
            # Clear heartbeat
            heartbeat_placeholder.empty()
            
            # Brief pause to let UI update and prevent overwhelming
            time.sleep(0.5)
            
            # If not the last batch, show intermediate results
            if batch_idx < total_batches - 1:
                with st.expander(f"Intermediate Results (after batch {batch_idx + 1})", expanded=False):
                    show_intermediate_results(st.session_state.batch_eval_results, parameter_prompts, output_cols)
        
        # Mark evaluation as complete
        st.session_state.evaluation_complete = True
        
        # Clear batch tracking since we're done
        st.session_state.completed_batch_count = 0
        
        # Clear placeholders
        status_placeholder.success(f"🎉 All evaluations complete! Processed {len(st.session_state.batch_eval_results)} rows")
        progress_placeholder.empty()
        batch_summary_placeholder.empty()
        heartbeat_placeholder.empty()
        
        # Convert results to the expected format
        final_results = {}
        for output_col in output_cols:
            final_results[output_col] = []
            
            for row_idx in sorted(st.session_state.batch_eval_results.keys()):
                row_result = st.session_state.batch_eval_results[row_idx]
                
                # Filter evaluations for this output column
                filtered_evals = {}
                for eval_key, eval_result in row_result.get("evals", {}).items():
                    if eval_key.startswith(f"{output_col}::"):
                        # Remove the output column prefix
                        clean_key = eval_key.replace(f"{output_col}::", "")
                        filtered_evals[clean_key] = eval_result
                
                if filtered_evals:  # Only add if we have evaluations for this column
                    final_results[output_col].append({
                        "row_index": row_idx,
                        "evals": filtered_evals
                    })
        
        return final_results
        
    except Exception as e:
        # Save current progress
        status_placeholder.error(f"❌ Error occurred: {e}")
        st.error(f"Evaluation interrupted. Progress saved: {len(st.session_state.batch_eval_results)} rows completed.")
        raise


def show_intermediate_results(batch_results, parameter_prompts, output_cols):
    """
    Show a summary of intermediate results without complex visualizations
    """
    if not batch_results:
        st.info("No results to display yet.")
        return
    
    # Create a simple summary
    total_rows = len(batch_results)
    total_evals = sum(len(row_data.get("evals", {})) for row_data in batch_results.values())
    
    st.write(f"**Progress Summary:**")
    st.write(f"- Rows processed: {total_rows}")
    st.write(f"- Total evaluations completed: {total_evals}")
    
    # Show a few sample results
    sample_rows = min(3, len(batch_results))
    if sample_rows > 0:
        st.write(f"**Sample results (first {sample_rows} rows):**")
        
        sample_data = []
        for i, (row_idx, row_data) in enumerate(list(batch_results.items())[:sample_rows]):
            row_info = {"Row": row_idx}
            
            for eval_key, eval_result in row_data.get("evals", {}).items():
                # Try to extract a score or summary
                try:
                    if isinstance(eval_result, str):
                        eval_data = json.loads(eval_result)
                        if "score" in eval_data:
                            row_info[eval_key] = f"Score: {eval_data['score']}"
                        else:
                            row_info[eval_key] = "Completed"
                    else:
                        row_info[eval_key] = "Completed"
                except:
                    row_info[eval_key] = "Completed"
            
            sample_data.append(row_info)
        
        if sample_data:
            df = pd.DataFrame(sample_data)
            st.dataframe(df)


def add_tab3_content_parallel():
    """
    Enhanced Tab 3 with parallel processing AND complete visualization suite
    - 5 parallel threads for fast processing
    - Automatic saving every 5 rows  
    - Resume capability if interrupted
    - Full charts, radar plots, and detailed analysis
    - All the visualization power of the original Tab 3
    """
    st.header("Individual Evaluations (Parallel Processing)")
    st.info("Upload a dataset to evaluate each output using the generated metrics. Processes rows in parallel with automatic saving between batches.")
    
    # Check if metrics are available
    if not st.session_state.pipeline_results:
        st.warning("Please generate metrics in Tab 1 first before running evaluations.")
        return
    
    # Get the current (potentially edited) prompts
    parameter_prompts, _ = get_current_prompts()
    
    # Display the number of available prompts
    st.write(f"Using {len(parameter_prompts)} parameter evaluation templates.")

    # Get metrics and templates from session state
    pipeline_results = st.session_state.pipeline_results
    is_conversation = pipeline_results.get("is_conversation", False)
    parameter_templates = pipeline_results.get("parameter_prompt_templates", {})
    metrics_json = pipeline_results.get("metrics_json", {})
    
    # Check if there's an evaluation in progress
    has_in_progress = "batch_eval_results" in st.session_state and st.session_state.batch_eval_results
    completed_batches = st.session_state.get("completed_batch_count", 0)
    
    if has_in_progress:
        st.warning(f"📊 **Evaluation in progress**: {len(st.session_state.batch_eval_results)} rows completed, {completed_batches} batches finished.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Resume Evaluation", type="primary"):
                # Will be handled in the main processing section
                st.rerun()
        with col2:
            if st.button("👀 View Current Results"):
                show_intermediate_results(st.session_state.batch_eval_results, parameter_prompts, ["output"])
        with col3:
            if st.button("🗑️ Reset & Start Over"):
                # Clear all evaluation state
                for key in list(st.session_state.keys()):
                    if key.startswith("batch_eval"):
                        del st.session_state[key]
                if "completed_batch_count" in st.session_state:
                    del st.session_state.completed_batch_count
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
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV, XLSX, or XLS file", 
        type=["csv", "xlsx", "xls"],
        key="parallel_eval_file_uploader"
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
            st.subheader("Select Columns")
            
            # Get all column names
            column_names = list(df.columns)
            
            if is_conversation:
                st.info("This is a conversation evaluation. Please select one or more columns that contain entire conversation transcripts.")
                conversation_cols = st.multiselect("Conversation Column(s)", column_names)
                input_col = None  # Not used for conversation
                output_cols = conversation_cols
            else:
                st.info("This is a single-turn evaluation. Please select the input column and one or more output columns.")
                
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
                    for param in parameters:
                        if isinstance(param, dict) and len(param) > 0:
                            param_key = list(param.keys())[0]
                            param_description = param[param_key]
                            prompt_key = f"{metric_name}::{param_key}"
                            
                            # Only show checkbox if we have a template for this parameter
                            if prompt_key in parameter_templates:
                                is_selected = st.checkbox(
                                    f"{param_key}: {param_description}", 
                                    value=True,
                                    key=f"parallel_{prompt_key}"
                                )
                                if is_selected:
                                    selected_metrics[prompt_key] = parameter_templates[prompt_key]
            
            # Evaluation settings
            st.subheader("Evaluation Settings")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                eval_model = st.selectbox(
                    "Evaluation Model", 
                    [
                        "gpt-4o-2024-05-13", 
                        "gpt-4o-mini-2024-07-18", 
                        "gpt-4.1-2025-04-14", 
                        "gpt-4.1-mini-2025-04-14", 
                        "o3-mini-2025-01-31"
                    ], 
                    index=1,  # Default to mini for faster processing
                    help="Model used for running evaluations"
                )
            
            with col2:
                sample_size = st.number_input(
                    "Sample Size (0 for all rows)", 
                    min_value=0, 
                    value=min(len(df), 10),
                    help="Number of rows to evaluate (0 means all rows)"
                )
            
            with col3:
                max_workers = st.selectbox(
                    "Parallel Threads",
                    [3, 5, 8, 10, 15, 20],
                    index=1,  # Default to 5
                    help="Number of parallel threads (more = faster but more resource intensive)"
                )
            
            # Show batch processing info
            st.info(f"🔄 **Parallel Processing**: {max_workers} threads will process 10 rows at a time, with results saved after each batch to prevent data loss.")
            
            # Validation
            if is_conversation and not output_cols:
                st.warning("Please select at least one conversation column.")
            elif not is_conversation and (not input_col or not output_cols):
                st.warning("Please select an input column and at least one output column.")
            elif not selected_metrics:
                st.warning("Please select at least one metric parameter to evaluate.")
            else:
                # Run evaluations button
                button_text = "🚀 Start Parallel Evaluation"
                if has_in_progress:
                    button_text = "▶️ Continue Evaluation"
                
                if st.button(button_text, use_container_width=True, type="primary"):
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
                    
                    # Run parallel evaluations
                    try:
                        st.markdown("---")
                        st.subheader("🚀 Parallel Evaluation Progress")
                        
                        results = run_parallel_batch_evaluations(
                            client=client,
                            evaluation_model=eval_model,
                            parameter_prompts=selected_metrics,
                            data_rows=data_to_evaluate,
                            input_col=input_col,
                            output_cols=output_cols,
                            batch_size=10,
                            max_workers=max_workers
                        )
                        
                        st.success(f"🎉 All evaluations completed successfully!")
                        
                        # Store results in session state for display
                        st.session_state.multi_evaluation_results = results
                        st.session_state.selected_metric_keys = list(selected_metrics.keys())
                        st.session_state.selected_output_cols = output_cols
                        
                        # Clean up batch processing state
                        for key in list(st.session_state.keys()):
                            if key.startswith("batch_eval"):
                                del st.session_state[key]
                        
                        # Display full results with all visualizations
                        st.markdown("---")
                        st.subheader("📊 Complete Evaluation Results")
                        
                        # Use the full visualization function
                        display_evaluation_results_multi_output(results, list(selected_metrics.keys()), output_cols)
                        
                    except Exception as e:
                        st.error(f"❌ Error during evaluation: {e}")
                        st.info("💾 Progress has been saved. You can resume by clicking the button above.")
                        
                        # Show partial results if available
                        if "batch_eval_results" in st.session_state and st.session_state.batch_eval_results:
                            with st.expander("View Partial Results", expanded=False):
                                show_intermediate_results(st.session_state.batch_eval_results, selected_metrics, output_cols)

    # Show previous results if available
    elif "multi_evaluation_results" in st.session_state and st.session_state.multi_evaluation_results:
        st.subheader("Previous Evaluation Results")
        st.info("Showing results from your last evaluation run. Upload a new file to run new evaluations.")
        
        # Get the saved prompt keys and output columns from session state
        selected_metric_keys = st.session_state.get("selected_metric_keys", list(parameter_templates.keys()))
        selected_output_cols = st.session_state.get("selected_output_cols", list(st.session_state.multi_evaluation_results.keys()))
        
        # Display previous results with full visualizations
        st.subheader("Previous Evaluation Results")
        st.info("Showing results from your last evaluation run with full charts and analysis.")
        
        # Use the complete visualization function
        display_evaluation_results_multi_output(
            st.session_state.multi_evaluation_results,
            selected_metric_keys,
            selected_output_cols
        )


# Helper functions for file processing (copied to avoid import issues)
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