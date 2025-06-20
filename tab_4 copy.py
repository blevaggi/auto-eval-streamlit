import openai
import streamlit as st
import pandas as pd
import json
# from tab_3 import read_csv_file, read_excel_file
from tab_2 import get_current_prompts
# TAB 4 is where the user can upload a file with LLM-generated inputs and two outputs, and see which is better for each auto-generated metric from Tab 1
from genai_utils import create_client_with_model

# NEW:
def read_csv_file(uploaded_file):
    """Read a CSV file"""
    import pandas as pd
    return pd.read_csv(uploaded_file)

def read_excel_file(uploaded_file):
    """Read an Excel file"""
    import pandas as pd
    return pd.read_excel(uploaded_file)
def process_uploaded_file_pairwise(uploaded_file):
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

def run_pairwise_evaluation_with_client(client, model, eval_prompt, data_to_evaluate):
    """
    Run a single pairwise evaluation with the provided API client
    """
    try:
        # Add explicit instruction to return JSON to the prompt
        eval_prompt_with_json = eval_prompt + "\n\nPlease provide your response in JSON format."
        
        # Create client with model-specific headers
        model_client = create_client_with_model(model)
        
        response = model_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation assistant who provides detailed, fair, comparative assessments of LLM outputs. Always respond with JSON."},
                {"role": "user", "content": eval_prompt_with_json}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def run_batch_pairwise_evaluations(client, evaluation_model, pairwise_prompts, data_rows, is_conversation, 
                                  input_col, output_a_col, output_b_col, reduce_order_bias=False, progress_bar=None):
    """
    Run batch pairwise evaluations on multiple rows of data
    If reduce_order_bias is True, run each evaluation twice (A vs B and B vs A)
    """
    results = []
    
    # Calculate total evaluations for progress tracking
    evals_per_row = len(pairwise_prompts)
    if reduce_order_bias:
        evals_per_row *= 2  # Double for A/B and B/A comparisons
    
    total_evals = len(data_rows) * evals_per_row
    completed_evals = 0
    
    for i, row in enumerate(data_rows):
        row_results = {"row_index": i}
        
        # Get input and outputs from the row
        input_text = row[input_col] if input_col else ""
        output_a_text = row[output_a_col] if output_a_col else ""
        output_b_text = row[output_b_col] if output_b_col else ""
        
        # Skip if missing required data
        if (not is_conversation and (not input_text or not output_a_text or not output_b_text)) or \
           (is_conversation and (not output_a_text or not output_b_text)):
            row_results["status"] = "skipped"
            row_results["reason"] = "Missing required data"
            results.append(row_results)
            continue
        
        row_results["evals"] = {}
        
        # Process each pairwise prompt for this row
        for prompt_key, prompt_template in pairwise_prompts.items():
            # First run: A vs B (normal order)
            if is_conversation:
                formatted_prompt = prompt_template.replace("{conversation_a}", output_a_text).replace("{conversation_b}", output_b_text)
            else:
                formatted_prompt = prompt_template.replace("{input}", input_text).replace("{output_a}", output_a_text).replace("{output_b}", output_b_text)
            
            # Run the first evaluation (A vs B)
            ab_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt, {
                "input": input_text,
                "output_a": output_a_text,
                "output_b": output_b_text
            })
            
            # Update progress
            completed_evals += 1
            if progress_bar:
                progress_bar.progress(completed_evals / total_evals, text=f"Evaluating {completed_evals}/{total_evals}")
            
            # If we're reducing order bias, run again with B vs A
            ba_result = None
            if reduce_order_bias:
                # Second run: B vs A (reversed order)
                if is_conversation:
                    formatted_prompt_reversed = prompt_template.replace("{conversation_a}", output_b_text).replace("{conversation_b}", output_a_text)
                else:
                    formatted_prompt_reversed = prompt_template.replace("{input}", input_text).replace("{output_a}", output_b_text).replace("{output_b}", output_a_text)
                
                # Run the second evaluation (B vs A)
                ba_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt_reversed, {
                    "input": input_text,
                    "output_a": output_b_text,
                    "output_b": output_a_text
                })
                
                # Update progress
                completed_evals += 1
                if progress_bar:
                    progress_bar.progress(completed_evals / total_evals, text=f"Evaluating {completed_evals}/{total_evals}")
                
                # Combine results from both runs
                combined_result = combine_pairwise_results(ab_result, ba_result)
                row_results["evals"][prompt_key] = {
                    "ab_result": ab_result,
                    "ba_result": ba_result,
                    "combined_result": combined_result
                }
            else:
                # Just store the single result
                row_results["evals"][prompt_key] = ab_result
        
        results.append(row_results)
    
    return results

def combine_pairwise_results(ab_result, ba_result):
    """
    Combine results from A vs B and B vs A evaluations to reduce order bias
    Returns: "A wins", "B wins", or "Tied" based on the combined results
    """
    # Parse JSON results if they're strings
    ab_data = ab_result
    ba_data = ba_result
    
    try:
        if isinstance(ab_result, str):
            ab_data = json.loads(ab_result)
        if isinstance(ba_result, str):
            ba_data = json.loads(ba_result)
    except:
        # If we can't parse JSON, return the raw results
        return {
            "status": "error_parsing",
            "ab_raw": ab_result,
            "ba_raw": ba_result
        }
    
    # Check for errors in either evaluation
    if isinstance(ab_data, dict) and "error" in ab_data or isinstance(ba_data, dict) and "error" in ba_data:
        return {
            "status": "error_in_eval",
            "ab_error": ab_data.get("error") if isinstance(ab_data, dict) else None,
            "ba_error": ba_data.get("error") if isinstance(ba_data, dict) else None
        }
    
    # Extract the winner from each evaluation
    ab_winner = extract_winner(ab_data)
    ba_winner = extract_winner(ba_data, reversed=True)  # Note: results are reversed
    
    # Determine the final winner
    result = "Tied"
    reason = ""
    
    if ab_winner == "A" and ba_winner == "A":
        result = "A wins"
        reason = "A won in both evaluations"
    elif ab_winner == "B" and ba_winner == "B":
        result = "B wins"
        reason = "B won in both evaluations"
    elif ab_winner == "Equivalent" and ba_winner == "Equivalent":
        result = "Tied"
        reason = "Both evaluations judged the outputs as equivalent"
    else:
        result = "Tied"
        reason = f"Mixed results: A vs B: {ab_winner}, B vs A: {ba_winner}"
    
    return {
        "final_result": result,
        "reason": reason,
        "ab_winner": ab_winner,
        "ba_winner": ba_winner
    }

def extract_winner(eval_data, reversed=False):
    """
    Extract the winner from evaluation data
    If reversed is True, swap A and B in the result
    """
    winner = "Equivalent"  # Default: no clear winner
    
    if isinstance(eval_data, dict):
        # Try different possible keys where winner might be stored
        for key in ["winner", "preference", "result", "evaluation", "comparison"]:
            if key in eval_data and isinstance(eval_data[key], str):
                value = eval_data[key].lower()
                if "a is better" in value or "a wins" in value or value == "a":
                    winner = "A"
                    break
                elif "b is better" in value or "b wins" in value or value == "b":
                    winner = "B"
                    break
                elif "equivalent" in value or "equal" in value or "tie" in value or "same" in value:
                    winner = "Equivalent"
                    break
        
        # If we didn't find in main keys, search all string values
        if winner == "Equivalent":
            for _, value in eval_data.items():
                if isinstance(value, str):
                    value = value.lower()
                    if "a is better" in value or "a wins" in value:
                        winner = "A"
                        break
                    elif "b is better" in value or "b wins" in value:
                        winner = "B"
                        break
    elif isinstance(eval_data, str):
        # Try to extract from raw string
        eval_lower = eval_data.lower()
        if "a is better" in eval_lower or "a wins" in eval_lower:
            winner = "A"
        elif "b is better" in eval_lower or "b wins" in eval_lower:
            winner = "B"
        elif "equivalent" in eval_lower or "equal" in eval_lower or "tie" in eval_lower:
            winner = "Equivalent"
    
    # If reversed, swap A and B
    if reversed:
        if winner == "A":
            return "B"
        elif winner == "B":
            return "A"
    
    return winner

def display_enhanced_pairwise_results(results, prompt_keys, reduce_order_bias=False):
    """
    Display pairwise evaluation results with enhanced visualizations including bigger donut charts
    with clearer metric titles and parameter information
    """
    import pandas as pd
    import altair as alt
    import json
    import re
    
    st.subheader("Pairwise Evaluation Results")
    
    # Early exit if no results
    if not results:
        st.info("No pairwise evaluation results to display.")
        return
    
    # Add an overall summary pie chart at the top
    st.subheader("Overall Results Summary")
    create_overall_summary_chart(results, prompt_keys, reduce_order_bias)
    
    # Create a dataframe from the results for the summary table
    rows = []
    for result in results:
        row_data = {"Row": result.get("row_index", "N/A")}
        
        if result.get("status") == "skipped":
            row_data.update({prompt_key: "SKIPPED" for prompt_key in prompt_keys})
            row_data["Reason"] = result.get("reason", "Unknown")
        else:
            for prompt_key in prompt_keys:
                eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                
                if reduce_order_bias:
                    # For reduced order bias, we display the combined result
                    try:
                        if isinstance(eval_result, dict) and "combined_result" in eval_result:
                            combined = eval_result["combined_result"]
                            if isinstance(combined, dict) and "final_result" in combined:
                                row_data[prompt_key] = combined["final_result"]
                            else:
                                row_data[prompt_key] = "See details"
                        else:
                            row_data[prompt_key] = "See details"
                    except:
                        row_data[prompt_key] = "See details"
                else:
                    # Parse the JSON result if possible (original single evaluation)
                    try:
                        if isinstance(eval_result, str):
                            eval_data = json.loads(eval_result)
                            if "winner" in eval_data:
                                row_data[prompt_key] = eval_data["winner"]
                            elif "preference" in eval_data:
                                row_data[prompt_key] = eval_data["preference"]
                            elif "result" in eval_data:
                                row_data[prompt_key] = eval_data["result"]
                            else:
                                # Look for string containing "A is better", "B is better", or "Equivalent"
                                for k, v in eval_data.items():
                                    if isinstance(v, str) and any(phrase in v for phrase in ["A is better", "B is better", "Equivalent"]):
                                        row_data[prompt_key] = v
                                        break
                                else:
                                    row_data[prompt_key] = "See details"
                        else:
                            row_data[prompt_key] = "See details"
                    except:
                        row_data[prompt_key] = "See details"
        
        rows.append(row_data)
    
    # Create summary dataframe
    if rows:
        results_df = pd.DataFrame(rows)
        
        # Show the summary table (can be hidden/shown with expander)
        with st.expander("Summary Table", expanded=True):
            st.dataframe(results_df)
        
        # Create visualization section
        st.subheader("Results Visualization", anchor="results-visualization")
        
        # Function to parse metric name and parameter from prompt key
        def parse_prompt_key(key):
            if "::" in key:
                parts = key.split("::")
                return parts[0], parts[1]
            return key, ""
        
        # Group prompt keys by metric
        metrics_dict = {}
        for key in prompt_keys:
            metric, param = parse_prompt_key(key)
            if metric not in metrics_dict:
                metrics_dict[metric] = []
            metrics_dict[metric].append((param, key))
        
        # Create aggregated data for visualization
        viz_data = []
        
        # Track metrics for each parameter
        for prompt_key in prompt_keys:
            a_wins = 0
            b_wins = 0
            tied = 0
            skipped = 0
            
            for row in rows:
                if prompt_key in row:
                    result = row[prompt_key]
                    if isinstance(result, str):
                        result_lower = result.lower()
                        if "a win" in result_lower or result_lower == "a is better" or result_lower == "a":
                            a_wins += 1
                        elif "b win" in result_lower or result_lower == "b is better" or result_lower == "b":
                            b_wins += 1
                        elif "tie" in result_lower or "equivalent" in result_lower or "equal" in result_lower:
                            tied += 1
                        elif "skipped" in result_lower:
                            skipped += 1
            
            # Add to visualization data
            metric, param = parse_prompt_key(prompt_key)
            viz_data.extend([
                {"parameter": prompt_key, "metric": metric, "param": param, "result": "A Wins", "count": a_wins},
                {"parameter": prompt_key, "metric": metric, "param": param, "result": "B Wins", "count": b_wins},
                {"parameter": prompt_key, "metric": metric, "param": param, "result": "Tied", "count": tied},
                {"parameter": prompt_key, "metric": metric, "param": param, "result": "Skipped", "count": skipped}
            ])
        
        # Convert to DataFrame for Altair
        viz_df = pd.DataFrame(viz_data)
        
        # Only continue if we have visualization data
        if not viz_df.empty and viz_df["count"].sum() > 0:
            # Create a color scale
            color_scale = alt.Scale(
                domain=["A Wins", "B Wins", "Tied", "Skipped"],
                range=["#4CAF50", "#2196F3", "#FFC107", "#9E9E9E"]
            )
            
            # Create visualization by metric group
            for metric, params in metrics_dict.items():
                st.markdown(f"## {metric}")
                st.markdown(f"*Click chart sections to see details. Hover for counts.*")
                
                num_params = len(params)
                cols_per_row = min(2, num_params)  # Max 2 charts per row for bigger charts
                
                # Create a grid layout for the charts in this metric group
                if num_params <= cols_per_row:
                    # Single row of charts
                    cols = st.columns(num_params)
                    for i, (param_name, prompt_key) in enumerate(params):
                        with cols[i]:
                            create_big_donut_chart(viz_df, prompt_key, param_name, color_scale)
                else:
                    # Multiple rows of charts
                    rows_needed = (num_params + cols_per_row - 1) // cols_per_row
                    for row in range(rows_needed):
                        cols = st.columns(cols_per_row)
                        for col in range(cols_per_row):
                            idx = row * cols_per_row + col
                            if idx < num_params:
                                with cols[col]:
                                    param_name, prompt_key = params[idx]
                                    create_big_donut_chart(viz_df, prompt_key, param_name, color_scale)
                
                st.markdown("---")
            
            # Add a legend
            legend_cols = st.columns(4)
            with legend_cols[0]:
                st.markdown("<div style='display:flex;align-items:center;'>"
                           "<div style='width:20px;height:20px;background-color:#4CAF50;margin-right:10px;'></div>"
                           "<div>A Wins</div></div>", unsafe_allow_html=True)
            with legend_cols[1]:
                st.markdown("<div style='display:flex;align-items:center;'>"
                           "<div style='width:20px;height:20px;background-color:#2196F3;margin-right:10px;'></div>"
                           "<div>B Wins</div></div>", unsafe_allow_html=True)
            with legend_cols[2]:
                st.markdown("<div style='display:flex;align-items:center;'>"
                           "<div style='width:20px;height:20px;background-color:#FFC107;margin-right:10px;'></div>"
                           "<div>Tied</div></div>", unsafe_allow_html=True)
            with legend_cols[3]:
                st.markdown("<div style='display:flex;align-items:center;'>"
                           "<div style='width:20px;height:20px;background-color:#9E9E9E;margin-right:10px;'></div>"
                           "<div>Skipped</div></div>", unsafe_allow_html=True)
            
            # Add a new download section using the HTML-based approach
            st.markdown("---")
            create_pairwise_download_section(results, "pairwise_viz_data")
            
        else:
            st.info("No data available for visualization. Run evaluations to see charts.")
        
        # Add detailed expandable sections for each result
        st.markdown("---")
        st.subheader("Detailed Results")
        for i, result in enumerate(results):
            if result.get("status") != "skipped":
                with st.expander(f"Row {result.get('row_index', i)} Details"):
                    for prompt_key, eval_result in result.get("evals", {}).items():
                        st.markdown(f"**{prompt_key}**")
                        
                        if reduce_order_bias:
                            # Display both individual results and the combined result
                            try:
                                if isinstance(eval_result, dict):
                                    # Show the combined result first
                                    if "combined_result" in eval_result:
                                        st.markdown("**Combined Result (A vs B and B vs A):**")
                                        st.json(eval_result["combined_result"])
                                    
                                    # Show individual results
                                    st.markdown("**A vs B Result:**")
                                    if "ab_result" in eval_result:
                                        try:
                                            if isinstance(eval_result["ab_result"], str):
                                                ab_data = json.loads(eval_result["ab_result"])
                                                st.json(ab_data)
                                            else:
                                                st.write(eval_result["ab_result"])
                                        except:
                                            st.write(eval_result["ab_result"])
                                    
                                    st.markdown("**B vs A Result:**")
                                    if "ba_result" in eval_result:
                                        try:
                                            if isinstance(eval_result["ba_result"], str):
                                                ba_data = json.loads(eval_result["ba_result"])
                                                st.json(ba_data)
                                            else:
                                                st.write(eval_result["ba_result"])
                                        except:
                                            st.write(eval_result["ba_result"])
                                else:
                                    st.write(eval_result)
                            except:
                                st.write(eval_result)
                        else:
                            # Display single evaluation result (original behavior)
                            try:
                                if isinstance(eval_result, str):
                                    eval_data = json.loads(eval_result)
                                    st.json(eval_data)
                                else:
                                    st.write(eval_result)
                            except:
                                st.write(eval_result)
    else:
        st.info("No pairwise evaluation results to display.")

def create_big_donut_chart(df, parameter, param_name, color_scale):
    """
    Create a bigger donut chart for a specific parameter with improved styling
    """
    import altair as alt
    import html
    
    # Filter data for this parameter
    param_data = df[df["parameter"] == parameter]
    
    # Only create chart if we have data
    if param_data.empty or param_data["count"].sum() == 0:
        st.markdown(f"### {param_name}")
        st.write(f"No data available")
        return
    
    # Get the metric name
    metric_name = param_data["metric"].iloc[0] if not param_data.empty else ""
    
    # Clean up parameter name for display and make sure it's HTML-safe
    display_name = html.escape(param_name)
    
    # Create a donut chart with Altair - bigger size
    base = alt.Chart(param_data).encode(
        theta=alt.Theta(field="count", type="quantitative", stack=True),
        color=alt.Color(
            field="result", 
            type="nominal",
            scale=color_scale,
            legend=None
        ),
        tooltip=[
            alt.Tooltip("result:N", title="Result"),
            alt.Tooltip("count:Q", title="Count"),
            alt.Tooltip("param:N", title="Parameter")
        ]
    ).properties(
        width=300,  # Increased from 200
        height=300   # Increased from 200
    )
    
    # Create the donut chart with bigger radius
    pie = base.mark_arc(outerRadius=140, innerRadius=50)
    
    # Add result labels outside the chart
    text = base.mark_text(radius=155, size=14).encode(text="result:N")
    
    # Add count values inside the chart
    value_text = base.mark_text(radius=90, size=16, fontWeight='bold').encode(text="count:Q")
    
    # Set chart properties
    chart = alt.layer(pie, text, value_text).properties(
        title={"text": display_name, "fontSize": 18, "fontWeight": "bold", "color": "#303030"}
    ).configure_view(
        strokeWidth=0
    )
    
    # Add a larger header for the parameter
    st.markdown(f"### {display_name}")
    
    # Display chart - use container width for responsiveness
    st.altair_chart(chart, use_container_width=True)
    
    # Display statistics
    total = param_data["count"].sum()
    
    # Calculate the values
    breakdown = param_data.set_index("result")["count"].to_dict()
    a_wins = breakdown.get("A Wins", 0)
    b_wins = breakdown.get("B Wins", 0)
    tied = breakdown.get("Tied", 0)
    skipped = breakdown.get("Skipped", 0)
    
    # Create a styled box for statistics
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f5f5f5; margin-bottom: 20px;">
            <div style="font-weight: bold; font-size: 16px;">Total evaluations: {total}</div>
            <div style="display: flex; flex-wrap: wrap;">
                <div style="margin-right: 15px;"><span style="color: #4CAF50;">A Wins:</span> {a_wins} ({a_wins/total*100:.1f}%)</div>
                <div style="margin-right: 15px;"><span style="color: #2196F3;">B Wins:</span> {b_wins} ({b_wins/total*100:.1f}%)</div>
                <div style="margin-right: 15px;"><span style="color: #FFC107;">Tied:</span> {tied} ({tied/total*100:.1f}%)</div>
                <div><span style="color: #9E9E9E;">Skipped:</span> {skipped} ({skipped/total*100:.1f}%)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_overall_summary_chart(results, prompt_keys, reduce_order_bias=False):
    """
    Create an overall summary pie chart showing the distribution of results
    across all parameters (percentage of A wins, B wins, and ties)
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import json
    
    # Early exit if no results
    if not results:
        st.info("No results available for summary.")
        return
    
    # Count total number of evaluations
    total_a_wins = 0
    total_b_wins = 0
    total_ties = 0
    total_skipped = 0
    
    # Process all results
    for result in results:
        if result.get("status") == "skipped":
            total_skipped += len(prompt_keys)
            continue
            
        for prompt_key in prompt_keys:
            eval_result = result.get("evals", {}).get(prompt_key, "N/A")
            
            # Determine the winner based on evaluation type
            if reduce_order_bias:
                # For reduced order bias, use the combined result
                try:
                    if isinstance(eval_result, dict) and "combined_result" in eval_result:
                        combined = eval_result["combined_result"]
                        if isinstance(combined, dict) and "final_result" in combined:
                            result_text = combined["final_result"].lower()
                            if "a win" in result_text:
                                total_a_wins += 1
                            elif "b win" in result_text:
                                total_b_wins += 1
                            elif "tie" in result_text or "equivalent" in result_text:
                                total_ties += 1
                            else:
                                total_skipped += 1  # Count as skipped if unclear
                        else:
                            total_skipped += 1
                    else:
                        total_skipped += 1
                except:
                    total_skipped += 1
            else:
                # For single evaluation, parse the result
                try:
                    winner = None
                    if isinstance(eval_result, str):
                        # Try to parse JSON
                        try:
                            eval_data = json.loads(eval_result)
                            # Look for winner field
                            if "winner" in eval_data:
                                winner = eval_data["winner"]
                            elif "preference" in eval_data:
                                winner = eval_data["preference"]
                            elif "result" in eval_data:
                                winner = eval_data["result"]
                            else:
                                # Search all string values
                                for k, v in eval_data.items():
                                    if isinstance(v, str) and any(phrase in v.lower() for phrase in ["a is better", "b is better", "equivalent"]):
                                        winner = v
                                        break
                        except:
                            # If can't parse JSON, try to find winner in raw string
                            result_lower = eval_result.lower()
                            if "a is better" in result_lower or "a wins" in result_lower:
                                winner = "A"
                            elif "b is better" in result_lower or "b wins" in result_lower:
                                winner = "B"
                            elif "equivalent" in result_lower or "tie" in result_lower:
                                winner = "Tie"
                    
                    # Count based on winner
                    if winner:
                        winner_lower = winner.lower()
                        if "a" in winner_lower and "win" in winner_lower or winner_lower == "a":
                            total_a_wins += 1
                        elif "b" in winner_lower and "win" in winner_lower or winner_lower == "b":
                            total_b_wins += 1
                        elif "tie" in winner_lower or "equivalent" in winner_lower or "equal" in winner_lower:
                            total_ties += 1
                        else:
                            total_skipped += 1
                    else:
                        total_skipped += 1
                except:
                    total_skipped += 1
    
    # Calculate total (excluding skipped for percentage calculation)
    total_valid = total_a_wins + total_b_wins + total_ties
    
    # Prepare data for pie chart
    if total_valid > 0:
        # Calculate percentages
        a_percent = (total_a_wins / total_valid) * 100
        b_percent = (total_b_wins / total_valid) * 100
        tie_percent = (total_ties / total_valid) * 100
        
        # Create the pie chart with Plotly
        labels = ['A Wins', 'B Wins', 'Tied']
        values = [total_a_wins, total_b_wins, total_ties]
        colors = ['#4CAF50', '#2196F3', '#FFC107']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,  # Make it a donut chart
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='outside',
            insidetextorientation='radial',
            pull=[0.05, 0.05, 0.05],  # Pull all slices slightly out
            texttemplate='%{label}: %{percent:.1f}%'
        )])
        
        fig.update_layout(
            title_text='Overall Distribution of Results',
            title_x=0.5,  # Center the title
            title_font=dict(size=24),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500,
            width=700,
            margin=dict(t=80, b=80, l=80, r=80)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a detailed breakdown below the chart
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "A Wins", 
                f"{total_a_wins} ({a_percent:.1f}%)",
                delta=None, 
                delta_color="normal"
            )
            
        with col2:
            st.metric(
                "B Wins", 
                f"{total_b_wins} ({b_percent:.1f}%)",
                delta=None, 
                delta_color="normal"
            )
            
        with col3:
            st.metric(
                "Ties", 
                f"{total_ties} ({tie_percent:.1f}%)",
                delta=None, 
                delta_color="normal"
            )
        
        # If there were skipped evaluations, show them too
        if total_skipped > 0:
            st.info(f"Additionally, {total_skipped} evaluations were skipped or had unclear results.")
            
        # Add text analysis of the results
        st.subheader("Summary Analysis")
        
        # Determine the overall winner
        if a_percent > b_percent and a_percent > tie_percent:
            lead_margin = a_percent - b_percent
            st.markdown(f"**Model A leads overall** with {a_percent:.1f}% of wins across all parameters, " +
                       f"leading Model B by {lead_margin:.1f} percentage points.")
        elif b_percent > a_percent and b_percent > tie_percent:
            lead_margin = b_percent - a_percent
            st.markdown(f"**Model B leads overall** with {b_percent:.1f}% of wins across all parameters, " +
                       f"leading Model A by {lead_margin:.1f} percentage points.")
        elif tie_percent > a_percent and tie_percent > b_percent:
            st.markdown(f"Most evaluations resulted in **ties** ({tie_percent:.1f}%), suggesting the models " +
                       f"perform similarly across the evaluated parameters.")
        else:
            # Very close results
            st.markdown("Results are very close between models, with no clear overall winner. " +
                       "Check individual parameters for differences in specific capabilities.")
            
    else:
        st.warning("No valid evaluation results to display in summary chart.")



def download_csv_button(df, filename="data.csv"):
    """
    Create a button to download a dataframe as CSV
    """
    import io
    
    # Convert dataframe to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="📊 Download Data (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=f"download_csv_{filename}"
    )

# Function to update the export_pairwise_evaluation_results function with unique keys
def export_pairwise_evaluation_results_with_keys(results, filename_prefix="pairwise_evaluation_results"):
    """
    Export pairwise evaluation results to downloadable formats with unique keys
    """
    import datetime
    import io
    import json
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export as JSON
    json_data = json.dumps(results, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="💾 Download Results as JSON",
            data=json_data,
            file_name=f"{filename_prefix}_{timestamp}.json",
            mime="application/json",
            key=f"pairwise_json_{timestamp}"  # Using timestamp to ensure uniqueness
        )
    
    # Try to create a simplified Excel-friendly version
    try:
        import pandas as pd
        
        # Flatten the results into rows
        rows = []
        for result in results:
            row_index = result.get("row_index", "N/A")
            
            if result.get("status") == "skipped":
                row_data = {
                    "row_index": row_index,
                    "status": "skipped",
                    "reason": result.get("reason", "Unknown")
                }
                rows.append(row_data)
            else:
                for prompt_key, eval_result in result.get("evals", {}).items():
                    # Parse metric and parameter from the prompt key
                    metric_name = prompt_key.split("::")[0] if "::" in prompt_key else ""
                    param_name = prompt_key.split("::")[-1] if "::" in prompt_key else prompt_key
                    
                    row_data = {
                        "row_index": row_index,
                        "metric": metric_name,
                        "parameter": param_name,
                        "full_key": prompt_key,
                        "raw_result": str(eval_result)[:1000]  # Truncate to avoid Excel issues
                    }
                    
                    # Try to extract winner and justification
                    try:
                        if isinstance(eval_result, dict) and "combined_result" in eval_result:
                            # For order bias reduced results
                            combined = eval_result["combined_result"]
                            if isinstance(combined, dict):
                                for k, v in combined.items():
                                    if k in ["final_result", "reason", "ab_winner", "ba_winner"]:
                                        row_data[k] = v
                        elif isinstance(eval_result, str):
                            # For regular results
                            eval_data = json.loads(eval_result)
                            if isinstance(eval_data, dict):
                                for k, v in eval_data.items():
                                    if k in ["winner", "preference", "result", "justification", "explanation", "reasoning"]:
                                        row_data[k] = v
                    except:
                        pass
                    
                    rows.append(row_data)
        
        # Create DataFrame
        results_df = pd.DataFrame(rows)
        
        # Export to Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="Pairwise Evaluation Results", index=False)
        
        with col2:
            st.download_button(
                label="📊 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{filename_prefix}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"pairwise_excel_{timestamp}"  # Using timestamp to ensure uniqueness
            )
    except Exception as e:
        st.warning(f"Could not create Excel export: {e}")

def create_donut_chart(df, parameter, color_scale):
    """
    Create a donut chart for a specific parameter
    """
    import altair as alt
    
    # Filter data for this parameter
    param_data = df[df["parameter"] == parameter]
    
    # Only create chart if we have data
    if param_data.empty or param_data["count"].sum() == 0:
        st.write(f"No data for {parameter}")
        return
    
    # Clean up parameter name for display
    display_name = parameter.split("::")[-1] if "::" in parameter else parameter
    
    # Create a donut chart with Altair
    base = alt.Chart(param_data).encode(
        theta=alt.Theta(field="count", type="quantitative", stack=True),
        color=alt.Color(
            field="result", 
            type="nominal",
            scale=color_scale,
            legend=None
        ),
        tooltip=[
            alt.Tooltip("result:N", title="Result"),
            alt.Tooltip("count:Q", title="Count")
        ]
    )
    
    # Create the donut chart by layering a transparent inner circle on a pie chart
    pie = base.mark_arc(outerRadius=100)
    text = base.mark_text(radius=120, size=14).encode(text="result:N")
    
    # Add value labels
    value_text = base.mark_text(radius=80, size=14).encode(text="count:Q")
    
    # Combine charts
    donut_chart = pie + text + value_text
    
    # Set chart properties
    chart = donut_chart.properties(
        width=200,
        height=200,
        title=display_name
    ).configure_view(
        strokeWidth=0
    )
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)
    
    # Display total
    total = param_data["count"].sum()
    st.markdown(f"**Total evaluations: {total}**")
    
    # Show breakdown as text
    breakdown = param_data.set_index("result")["count"].to_dict()
    a_wins = breakdown.get("A Wins", 0)
    b_wins = breakdown.get("B Wins", 0)
    tied = breakdown.get("Tied", 0)
    
    if total > 0:
        a_pct = (a_wins / total) * 100
        b_pct = (b_wins / total) * 100
        tie_pct = (tied / total) * 100
        
        st.markdown(f"A Wins: {a_wins} ({a_pct:.1f}%)")
        st.markdown(f"B Wins: {b_wins} ({b_pct:.1f}%)")
        st.markdown(f"Tied: {tied} ({tie_pct:.1f}%)")

def download_csv_button(df, filename="data.csv"):
    """
    Create a button to download a dataframe as CSV
    """
    import io
    
    # Convert dataframe to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="📊 Download Data (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=f"download_csv_{filename}"
    )

# Function to update the export_pairwise_evaluation_results function with unique keys
def export_pairwise_evaluation_results_with_keys(results, filename_prefix="pairwise_evaluation_results"):
    """
    Export pairwise evaluation results to downloadable formats with unique keys
    """
    import datetime
    import io
    import json
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export as JSON
    json_data = json.dumps(results, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="💾 Download Results as JSON",
            data=json_data,
            file_name=f"{filename_prefix}_{timestamp}.json",
            mime="application/json",
            key=f"pairwise_json_{timestamp}"  # Using timestamp to ensure uniqueness
        )
    
    # Try to create a simplified Excel-friendly version
    try:
        import pandas as pd
        
        # Flatten the results into rows
        rows = []
        for result in results:
            row_index = result.get("row_index", "N/A")
            
            if result.get("status") == "skipped":
                row_data = {
                    "row_index": row_index,
                    "status": "skipped",
                    "reason": result.get("reason", "Unknown")
                }
                rows.append(row_data)
            else:
                for prompt_key, eval_result in result.get("evals", {}).items():
                    row_data = {
                        "row_index": row_index,
                        "metric": prompt_key,
                        "raw_result": str(eval_result)[:1000]  # Truncate to avoid Excel issues
                    }
                    
                    # Try to extract winner and justification
                    try:
                        if isinstance(eval_result, dict) and "combined_result" in eval_result:
                            # For order bias reduced results
                            combined = eval_result["combined_result"]
                            if isinstance(combined, dict):
                                for k, v in combined.items():
                                    if k in ["final_result", "reason", "ab_winner", "ba_winner"]:
                                        row_data[k] = v
                        elif isinstance(eval_result, str):
                            # For regular results
                            eval_data = json.loads(eval_result)
                            if isinstance(eval_data, dict):
                                for k, v in eval_data.items():
                                    if k in ["winner", "preference", "result", "justification", "explanation", "reasoning"]:
                                        row_data[k] = v
                    except:
                        pass
                    
                    rows.append(row_data)
        
        # Create DataFrame
        results_df = pd.DataFrame(rows)
        
        # Export to Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="Pairwise Evaluation Results", index=False)
        
        with col2:
            st.download_button(
                label="📊 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{filename_prefix}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"pairwise_excel_{timestamp}"  # Using timestamp to ensure uniqueness
            )
    except Exception as e:
        st.warning(f"Could not create Excel export: {e}")

def display_pairwise_evaluation_results(results, prompt_keys, reduce_order_bias=False):
    """
    Display pairwise evaluation results in a formatted way
    """
    st.subheader("Pairwise Evaluation Results")
    
    # Create a dataframe from the results
    rows = []
    for result in results:
        row_data = {"Row": result.get("row_index", "N/A")}
        
        if result.get("status") == "skipped":
            row_data.update({prompt_key: "SKIPPED" for prompt_key in prompt_keys})
            row_data["Reason"] = result.get("reason", "Unknown")
        else:
            for prompt_key in prompt_keys:
                eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                
                if reduce_order_bias:
                    # For reduced order bias, we display the combined result
                    try:
                        if isinstance(eval_result, dict) and "combined_result" in eval_result:
                            combined = eval_result["combined_result"]
                            if isinstance(combined, dict) and "final_result" in combined:
                                row_data[prompt_key] = combined["final_result"]
                            else:
                                row_data[prompt_key] = "See details"
                        else:
                            row_data[prompt_key] = "See details"
                    except:
                        row_data[prompt_key] = "See details"
                else:
                    # Parse the JSON result if possible (original single evaluation)
                    try:
                        if isinstance(eval_result, str):
                            eval_data = json.loads(eval_result)
                            if "winner" in eval_data:
                                row_data[prompt_key] = eval_data["winner"]
                            elif "preference" in eval_data:
                                row_data[prompt_key] = eval_data["preference"]
                            elif "result" in eval_data:
                                row_data[prompt_key] = eval_data["result"]
                            else:
                                # Look for string containing "A is better", "B is better", or "Equivalent"
                                for k, v in eval_data.items():
                                    if isinstance(v, str) and any(phrase in v for phrase in ["A is better", "B is better", "Equivalent"]):
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
        for i, result in enumerate(results):
            if result.get("status") != "skipped":
                with st.expander(f"Detailed Results for Row {result.get('row_index', i)}"):
                    for prompt_key, eval_result in result.get("evals", {}).items():
                        st.markdown(f"**{prompt_key}**")
                        
                        if reduce_order_bias:
                            # Display both individual results and the combined result
                            try:
                                if isinstance(eval_result, dict):
                                    # Show the combined result first
                                    if "combined_result" in eval_result:
                                        st.markdown("**Combined Result (A vs B and B vs A):**")
                                        st.json(eval_result["combined_result"])
                                    
                                    # Show individual results
                                    st.markdown("**A vs B Result:**")
                                    if "ab_result" in eval_result:
                                        try:
                                            if isinstance(eval_result["ab_result"], str):
                                                ab_data = json.loads(eval_result["ab_result"])
                                                st.json(ab_data)
                                            else:
                                                st.write(eval_result["ab_result"])
                                        except:
                                            st.write(eval_result["ab_result"])
                                    
                                    st.markdown("**B vs A Result:**")
                                    if "ba_result" in eval_result:
                                        try:
                                            if isinstance(eval_result["ba_result"], str):
                                                ba_data = json.loads(eval_result["ba_result"])
                                                st.json(ba_data)
                                            else:
                                                st.write(eval_result["ba_result"])
                                        except:
                                            st.write(eval_result["ba_result"])
                                else:
                                    st.write(eval_result)
                            except:
                                st.write(eval_result)
                        else:
                            # Display single evaluation result (original behavior)
                            try:
                                if isinstance(eval_result, str):
                                    eval_data = json.loads(eval_result)
                                    st.json(eval_data)
                                else:
                                    st.write(eval_result)
                            except:
                                st.write(eval_result)
    else:
        st.info("No pairwise evaluation results to display.")

def export_pairwise_evaluation_results(results, filename_prefix="pairwise_evaluation_results"):
    """
    Export pairwise evaluation results to downloadable formats
    """
    import datetime
    import io
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export as JSON
    json_data = json.dumps(results, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="💾 Download Results as JSON",
            data=json_data,
            file_name=f"{filename_prefix}_{timestamp}.json",
            mime="application/json",
            key="pairwise_results_download"
        )
    
    # Try to create a simplified Excel-friendly version
    try:
        import pandas as pd
        
        # Flatten the results into rows
        rows = []
        for result in results:
            row_index = result.get("row_index", "N/A")
            
            if result.get("status") == "skipped":
                row_data = {
                    "row_index": row_index,
                    "status": "skipped",
                    "reason": result.get("reason", "Unknown")
                }
                rows.append(row_data)
            else:
                for prompt_key, eval_result in result.get("evals", {}).items():
                    row_data = {
                        "row_index": row_index,
                        "metric": prompt_key,
                        "raw_result": eval_result
                    }
                    
                    # Try to extract winner and justification
                    try:
                        if isinstance(eval_result, str):
                            eval_data = json.loads(eval_result)
                            if isinstance(eval_data, dict):
                                for k, v in eval_data.items():
                                    if k in ["winner", "preference", "result", "justification", "explanation", "reasoning"]:
                                        row_data[k] = v
                    except:
                        pass
                    
                    rows.append(row_data)
        
        # Create DataFrame
        results_df = pd.DataFrame(rows)
        
        # Export to Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, sheet_name="Pairwise Evaluation Results", index=False)
        
        with col2:
            st.download_button(
                label="📊 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{filename_prefix}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.warning(f"Could not create Excel export: {e}")

def create_pairwise_download_section(results, filename_prefix="pairwise_evaluation_results"):
    """
    Create a separate section for pairwise evaluation downloads using HTML links instead of Streamlit buttons
    This avoids the session state management issues that cause download buttons to fail
    """
    import datetime
    import io
    import uuid
    import base64
    import json
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.subheader("Export Results", anchor="exports")
    
    # Create a form for the downloads
    with st.form(key="pairwise_download_form"):
        col1, col2 = st.columns(2)
        
        # Store the data once to avoid recreating on every rerun
        json_data = json.dumps(results, indent=2)
        
        # Export JSON using HTML download link
        with col1:
            json_filename = f"{filename_prefix}_{timestamp}.json"
            b64_json = base64.b64encode(json_data.encode()).decode()
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="{json_filename}" class="downloadButton">💾 Download Results as JSON</a>'
            st.markdown(href_json, unsafe_allow_html=True)
            
        # Create Excel data
        excel_data = None
        try:
            import pandas as pd
            
            # Flatten the results into rows
            rows = []
            for result in results:
                row_index = result.get("row_index", "N/A")
                
                if result.get("status") == "skipped":
                    row_data = {
                        "row_index": row_index,
                        "status": "skipped",
                        "reason": result.get("reason", "Unknown")
                    }
                    rows.append(row_data)
                else:
                    for prompt_key, eval_result in result.get("evals", {}).items():
                        # Parse metric and parameter from the prompt key
                        metric_name = prompt_key.split("::")[0] if "::" in prompt_key else ""
                        param_name = prompt_key.split("::")[-1] if "::" in prompt_key else prompt_key
                        
                        row_data = {
                            "row_index": row_index,
                            "metric": metric_name,
                            "parameter": param_name,
                            "full_key": prompt_key,
                            "raw_result": str(eval_result)[:1000]  # Truncate to avoid Excel issues
                        }
                        
                        # Try to extract winner and justification
                        try:
                            if isinstance(eval_result, dict) and "combined_result" in eval_result:
                                # For order bias reduced results
                                combined = eval_result["combined_result"]
                                if isinstance(combined, dict):
                                    for k, v in combined.items():
                                        if k in ["final_result", "reason", "ab_winner", "ba_winner"]:
                                            row_data[k] = v
                            elif isinstance(eval_result, str):
                                # For regular results
                                eval_data = json.loads(eval_result)
                                if isinstance(eval_data, dict):
                                    for k, v in eval_data.items():
                                        if k in ["winner", "preference", "result", "justification", "explanation", "reasoning"]:
                                            row_data[k] = v
                        except:
                            pass
                        
                        rows.append(row_data)
            
            # Create DataFrame
            results_df = pd.DataFrame(rows)
            
            # Export to Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                results_df.to_excel(writer, sheet_name="Pairwise Evaluation Results", index=False)
            
            # Get the Excel data
            excel_data = excel_buffer.getvalue()
        except Exception as e:
            st.warning(f"Could not create Excel export: {e}")
        
        # Excel download using HTML link
        with col2:
            if excel_data is not None:
                excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
                b64_excel = base64.b64encode(excel_data).decode()
                href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{excel_filename}" class="downloadButton">📊 Download Results as Excel</a>'
                st.markdown(href_excel, unsafe_allow_html=True)
        
        # Add a dummy submit button to create the form (required for Streamlit forms)
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



def run_pairwise_batch_evaluations(client, evaluation_model, pairwise_prompts, data_rows, 
                                 is_conversation, input_col, output_a_col, output_b_col,
                                 reduce_order_bias=False, batch_size=5):
    """
    Run pairwise evaluations in small batches, updating session state and showing
    progress after each batch. Includes heartbeat to keep session alive.
    
    Parameters:
    - client: API client
    - evaluation_model: Model to use for evaluations (e.g., "gpt-4o-2024-05-13")
    - pairwise_prompts: Dictionary of prompt templates
    - data_rows: List of data rows to evaluate
    - is_conversation: Boolean indicating if this is a conversation evaluation
    - input_col: Column name for input text
    - output_a_col: Column name for output A
    - output_b_col: Column name for output B
    - reduce_order_bias: Whether to run A vs B and B vs A comparisons
    - batch_size: Number of rows to process in each batch (default: 5)
    """
    import time
    import datetime
    
    # Create placeholders for status updates
    status_placeholder = st.empty()
    batch_progress = st.empty()
    heartbeat_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # Initialize session state for batch processing
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    
    # Initialize heartbeat timestamp
    last_heartbeat = time.time()
    
    # Determine starting point based on existing progress
    if st.session_state.batch_results:
        # Resume from where we left off
        results = st.session_state.batch_results
        processed_rows = len(results)
        start_idx = processed_rows
        status_placeholder.info(f"Resuming from row {start_idx+1} of {len(data_rows)}")
    else:
        # Start from the beginning
        results = []
        start_idx = 0
    
    # Calculate total evaluations for progress tracking
    evals_per_row = len(pairwise_prompts)
    if reduce_order_bias:
        evals_per_row *= 2  # Double for A/B and B/A comparisons
    
    total_rows = len(data_rows)
    total_evals = total_rows * evals_per_row
    completed_evals = start_idx * evals_per_row
    
    # Main processing loop
    try:
        # Process in batches
        for batch_start in range(start_idx, total_rows, batch_size):
            # Update heartbeat if needed (every 60 seconds)
            current_time = time.time()
            if current_time - last_heartbeat > 60:
                heartbeat_message = f"Heartbeat: {datetime.datetime.now().strftime('%H:%M:%S')}"
                heartbeat_placeholder.info(heartbeat_message)
                last_heartbeat = current_time
            
            # Get the current batch
            batch_end = min(batch_start + batch_size, total_rows)
            current_batch = data_rows[batch_start:batch_end]
            
            # Show batch progress
            status_placeholder.info(f"Processing rows {batch_start+1}-{batch_end} of {total_rows}")
            
            # Create batch progress bar
            batch_progress_bar = batch_progress.progress(0, text=f"Starting batch {batch_start//batch_size + 1}...")
            
            # Process each row in the current batch
            batch_results = []
            for i, row in enumerate(current_batch):
                row_index = batch_start + i
                row_results = {"row_index": row_index}
                
                # Get input and outputs from the row
                input_text = row[input_col] if input_col else ""
                output_a_text = row[output_a_col] if output_a_col else ""
                output_b_text = row[output_b_col] if output_b_col else ""
                
                # Skip if missing required data
                if (not is_conversation and (not input_text or not output_a_text or not output_b_text)) or \
                   (is_conversation and (not output_a_text or not output_b_text)):
                    row_results["status"] = "skipped"
                    row_results["reason"] = "Missing required data"
                    batch_results.append(row_results)
                    continue
                
                row_results["evals"] = {}
                
                # Update batch progress
                batch_progress_bar.progress((i+1) / len(current_batch), 
                                          text=f"Row {row_index+1}/{total_rows}")
                
                # Process each prompt for this row
                for prompt_key, prompt_template in pairwise_prompts.items():
                    # First evaluation: A vs B
                    if is_conversation:
                        formatted_prompt = prompt_template.replace("{conversation_a}", output_a_text).replace("{conversation_b}", output_b_text)
                    else:
                        formatted_prompt = prompt_template.replace("{input}", input_text).replace("{output_a}", output_a_text).replace("{output_b}", output_b_text)
                    
                    # Run the first evaluation (A vs B)
                    ab_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt, {
                        "input": input_text,
                        "output_a": output_a_text,
                        "output_b": output_b_text
                    })
                    
                    # Update completed evaluations count
                    completed_evals += 1
                    
                    # Run second evaluation if reducing order bias
                    ba_result = None
                    if reduce_order_bias:
                        # B vs A (reversed order)
                        if is_conversation:
                            formatted_prompt_reversed = prompt_template.replace("{conversation_a}", output_b_text).replace("{conversation_b}", output_a_text)
                        else:
                            formatted_prompt_reversed = prompt_template.replace("{input}", input_text).replace("{output_a}", output_b_text).replace("{output_b}", output_a_text)
                        
                        # Run the second evaluation (B vs A)
                        ba_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt_reversed, {
                            "input": input_text,
                            "output_a": output_b_text,
                            "output_b": output_a_text
                        })
                        
                        # Update completed evaluations count
                        completed_evals += 1
                        
                        # Combine results
                        combined_result = combine_pairwise_results(ab_result, ba_result)
                        row_results["evals"][prompt_key] = {
                            "ab_result": ab_result,
                            "ba_result": ba_result,
                            "combined_result": combined_result
                        }
                    else:
                        # Store single evaluation result
                        row_results["evals"][prompt_key] = ab_result
                
                # Add row results to batch
                batch_results.append(row_results)
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Update session state with current results
            st.session_state.batch_results = results
            
            # Display interim results after each batch
            with results_placeholder.container():
                st.subheader("Interim Results")
                st.write(f"Processed {len(results)} of {total_rows} rows")
                
                # Display a summary of the interim results
                display_batch_summary(results, list(pairwise_prompts.keys()), reduce_order_bias)
            
            # Short pause to let UI update
            time.sleep(0.5)
        
        # Clear session state once all processing is complete
        st.session_state.batch_results = []
        
        # Clear placeholders
        status_placeholder.empty()
        batch_progress.empty()
        heartbeat_placeholder.empty()
        results_placeholder.empty()
        
        # Return final results
        return results
        
    except Exception as e:
        # Save progress in session state
        st.session_state.batch_results = results
        
        # Show error with context
        error_msg = f"Error during processing at row {row_index if 'row_index' in locals() else 'unknown'}: {str(e)}"
        status_placeholder.error(error_msg)
        
        # Re-raise exception
        raise Exception(error_msg)

def run_batch_pairwise_evaluations_with_checkpoints(client, evaluation_model, pairwise_prompts, data_rows, 
                                           is_conversation, input_col, output_a_col, output_b_col, 
                                           reduce_order_bias=False, progress_bar=None, batch_size=50):
    """
    Run batch pairwise evaluations on multiple rows of data in smaller batches with checkpointing
    to avoid Streamlit session state issues with large datasets.
    
    Parameters:
    - batch_size: Number of rows to process in each batch before saving progress
    """
    import time
    
    # Create placeholders for status updates - key for keeping session alive
    status_placeholder = st.empty()
    batch_progress = st.empty()
    heartbeat_placeholder = st.empty()
    
    # Initialize session state for results by batch if needed
    if "results_by_batch" not in st.session_state:
        st.session_state.results_by_batch = {}
    
    # Determine starting point based on existing progress
    if st.session_state.get("is_eval_in_progress", False) and st.session_state.results_by_batch:
        # Get completed batches and consolidate results
        results = []
        completed_batch_ids = sorted([int(k) for k in st.session_state.results_by_batch.keys()])
        
        if completed_batch_ids:
            for batch_id in completed_batch_ids:
                results.extend(st.session_state.results_by_batch[str(batch_id)])
            
            last_completed_batch = max(completed_batch_ids)
            start_idx = (last_completed_batch + 1) * batch_size
            start_idx = min(start_idx, len(data_rows))  # Ensure we don't go past the end
            
            status_placeholder.info(f"Resuming from batch {last_completed_batch + 2}, row {start_idx+1}")
        else:
            results = []
            start_idx = 0
    else:
        # Fresh start
        results = []
        start_idx = 0
        # Reset batch results
        st.session_state.results_by_batch = {}
        st.session_state.last_heartbeat = time.time()
    
    # Flag that evaluation is in progress
    st.session_state.is_eval_in_progress = True
    
    # Calculate total evaluations for progress tracking
    evals_per_row = len(pairwise_prompts)
    if reduce_order_bias:
        evals_per_row *= 2  # Double for A/B and B/A comparisons
    
    total_rows = len(data_rows)
    total_evals = total_rows * evals_per_row
    completed_evals = start_idx * evals_per_row
    
    # Set up heartbeat function to keep session alive
    def update_heartbeat():
        current_time = time.time()
        elapsed = current_time - st.session_state.get("last_heartbeat", current_time)
        if elapsed > 10:  # Update heartbeat every 10 seconds
            heartbeat_placeholder.text(f"Session heartbeat: active ({current_time:.0f})")
            st.session_state.last_heartbeat = current_time
            time.sleep(0.1)  # Small pause to ensure update is processed
            heartbeat_placeholder.empty()  # Clear after updating
    
    # Process in batches
    try:
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(start_idx, total_rows, batch_size), start=(start_idx // batch_size)):
            batch_id = str(batch_idx)
            batch_end = min(i + batch_size, total_rows)
            batch = data_rows[i:batch_end]
            
            # Show batch progress information
            status_placeholder.info(f"Processing batch {batch_idx+1} of {total_batches} (rows {i+1}-{batch_end} of {total_rows})")
            
            # Update progress bar for this batch
            if progress_bar:
                progress_bar.progress(i / total_rows, 
                                    text=f"Batch {batch_idx+1}/{total_batches}")
            
            # Create batch progress bar
            batch_progress_bar = batch_progress.progress(0, 
                                                        text=f"Starting batch {batch_idx+1}...")
            
            # Process each row in the current batch
            batch_results = []
            for j, row in enumerate(batch):
                # Update heartbeat to keep session alive
                update_heartbeat()
                
                row_results = {"row_index": i + j}
                
                # Get input and outputs from the row
                input_text = row[input_col] if input_col else ""
                output_a_text = row[output_a_col] if output_a_col else ""
                output_b_text = row[output_b_col] if output_b_col else ""
                
                # Skip if missing required data
                if (not is_conversation and (not input_text or not output_a_text or not output_b_text)) or \
                   (is_conversation and (not output_a_text or not output_b_text)):
                    row_results["status"] = "skipped"
                    row_results["reason"] = "Missing required data"
                    batch_results.append(row_results)
                    continue
                
                row_results["evals"] = {}
                
                # Update batch progress
                batch_progress_bar.progress((j+1) / len(batch), 
                                          text=f"Row {i+j+1}/{batch_end} in batch {batch_idx+1}")
                
                # Process each pairwise prompt for this row
                for prompt_key, prompt_template in pairwise_prompts.items():
                    # First run: A vs B (normal order)
                    if is_conversation:
                        formatted_prompt = prompt_template.replace("{conversation_a}", output_a_text).replace("{conversation_b}", output_b_text)
                    else:
                        formatted_prompt = prompt_template.replace("{input}", input_text).replace("{output_a}", output_a_text).replace("{output_b}", output_b_text)
                    
                    # Run the first evaluation (A vs B)
                    ab_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt, {
                        "input": input_text,
                        "output_a": output_a_text,
                        "output_b": output_b_text
                    })
                    
                    # Update progress
                    completed_evals += 1
                    if progress_bar:
                        progress_bar.progress(completed_evals / total_evals, 
                                             text=f"Row {i+j+1}/{total_rows}, Eval {completed_evals}/{total_evals}")
                    
                    # If we're reducing order bias, run again with B vs A
                    ba_result = None
                    if reduce_order_bias:
                        # Second run: B vs A (reversed order)
                        if is_conversation:
                            formatted_prompt_reversed = prompt_template.replace("{conversation_a}", output_b_text).replace("{conversation_b}", output_a_text)
                        else:
                            formatted_prompt_reversed = prompt_template.replace("{input}", input_text).replace("{output_a}", output_b_text).replace("{output_b}", output_a_text)
                        
                        # Run the second evaluation (B vs A)
                        ba_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt_reversed, {
                            "input": input_text,
                            "output_a": output_b_text,
                            "output_b": output_a_text
                        })
                        
                        # Update progress
                        completed_evals += 1
                        if progress_bar:
                            progress_bar.progress(completed_evals / total_evals, 
                                                 text=f"Row {i+j+1}/{total_rows}, Eval {completed_evals}/{total_evals}")
                        
                        # Combine results from both runs
                        combined_result = combine_pairwise_results(ab_result, ba_result)
                        row_results["evals"][prompt_key] = {
                            "ab_result": ab_result,
                            "ba_result": ba_result,
                            "combined_result": combined_result
                        }
                    else:
                        # Just store the single result
                        row_results["evals"][prompt_key] = ab_result
                
                batch_results.append(row_results)
                
                # Save to session state every 10 rows within a batch for extra safety
                if (j + 1) % 10 == 0 or j == len(batch) - 1:
                    # Update our combined results
                    combined_results = results.copy()
                    combined_results.extend(batch_results)
                    
                    # Save complete results to main session state
                    st.session_state.partial_eval_results = combined_results
                    
                    # Update heartbeat 
                    update_heartbeat()
            
            # Save completed batch and extend overall results 
            st.session_state.results_by_batch[batch_id] = batch_results
            results.extend(batch_results)
            
            # Show intermediate results if this isn't the final batch
            if batch_end < total_rows:
                # DO NOT use an expander here - causes nested expander issues
                # Instead use a simple header and container
                st.markdown(f"### Intermediate Results (Processed {batch_end}/{total_rows} rows)")
                
                # Create a simplified version of results display that doesn't use expanders
                display_batch_summary(results, list(pairwise_prompts.keys()), reduce_order_bias)
                
                # Give the user a chance to see progress
                status_placeholder.success(f"✅ Completed batch {batch_idx+1} of {total_batches}")
                
                # Explicitly force a UI refresh by clearing placeholders
                batch_progress.empty()
                time.sleep(0.5)  # Short pause to let the UI catch up
        
        # Mark evaluation as complete and save final results
        st.session_state.is_eval_in_progress = False
        st.session_state.pairwise_evaluation_results = results
        
        # Clean up temporary batch storage to save memory
        if hasattr(st.session_state, 'results_by_batch'):
            del st.session_state.results_by_batch
        
        # Clear placeholders
        status_placeholder.empty()
        batch_progress.empty()
        heartbeat_placeholder.empty()
        
    except Exception as e:
        # Save the current progress on error
        st.session_state.partial_eval_results = results
        
        # Report the error with more context
        error_msg = f"Error during batch {batch_idx+1}, row {i+j+1 if 'j' in locals() else i}: {str(e)}"
        status_placeholder.error(error_msg)
        
        # Keep the batch data for resuming
        raise Exception(error_msg)
    
    return results

def display_batch_summary(results, prompt_keys, reduce_order_bias=False):
    """
    Display a simple summary of batch results without using expanders
    to avoid nested expander issues with Streamlit
    """
    import pandas as pd
    import json
    
    # Create a dataframe from the results for the summary table
    rows = []
    for result in results:
        row_data = {"Row": result.get("row_index", "N/A")}
        
        if result.get("status") == "skipped":
            row_data.update({prompt_key: "SKIPPED" for prompt_key in prompt_keys})
            row_data["Reason"] = result.get("reason", "Unknown")
        else:
            for prompt_key in prompt_keys:
                eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                
                if reduce_order_bias:
                    # For reduced order bias, we display the combined result
                    try:
                        if isinstance(eval_result, dict) and "combined_result" in eval_result:
                            combined = eval_result["combined_result"]
                            if isinstance(combined, dict) and "final_result" in combined:
                                row_data[prompt_key] = combined["final_result"]
                            else:
                                row_data[prompt_key] = "See details"
                        else:
                            row_data[prompt_key] = "See details"
                    except:
                        row_data[prompt_key] = "See details"
                else:
                    # Parse the JSON result if possible (original single evaluation)
                    try:
                        if isinstance(eval_result, str):
                            eval_data = json.loads(eval_result)
                            if "winner" in eval_data:
                                row_data[prompt_key] = eval_data["winner"]
                            elif "preference" in eval_data:
                                row_data[prompt_key] = eval_data["preference"]
                            elif "result" in eval_data:
                                row_data[prompt_key] = eval_data["result"]
                            else:
                                # Look for string containing "A is better", "B is better", or "Equivalent"
                                for k, v in eval_data.items():
                                    if isinstance(v, str) and any(phrase in v for phrase in ["A is better", "B is better", "Equivalent"]):
                                        row_data[prompt_key] = v
                                        break
                                else:
                                    row_data[prompt_key] = "See details"
                        else:
                            row_data[prompt_key] = "See details"
                    except:
                        row_data[prompt_key] = "See details"
        
        rows.append(row_data)
    
    # Create summary dataframe and display it
    if rows:
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df)
        
        # Count results by category
        a_wins = 0
        b_wins = 0
        tied = 0
        skipped = 0
        
        for row in rows:
            for key in row:
                if key not in ["Row", "Reason"]:
                    result = str(row[key]).lower()
                    if "a win" in result or result == "a is better" or result == "a":
                        a_wins += 1
                    elif "b win" in result or result == "b is better" or result == "b":
                        b_wins += 1
                    elif "tie" in result or "equivalent" in result or "equal" in result:
                        tied += 1
                    else:
                        skipped += 1
        
        # Display summary
        st.markdown("#### Current Results Summary:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("A Wins", a_wins)
        col2.metric("B Wins", b_wins)
        col3.metric("Tied", tied)
        col4.metric("Skipped/Pending", skipped)
    else:
        st.info("No evaluation results to display yet.")

def add_tab4_content_with_batching():
    """
    Modified version of add_tab4_content that uses batched evaluation
    with robust session state handling for reliable processing of large datasets
    """
    import pandas as pd
    import json
    import datetime
    import base64
    import io
    st.header("Run Pairwise Evaluations on Your Data")
    st.info("Upload a spreadsheet with paired outputs and compare them using your generated metrics.")
    
    # Check if metrics are available
    if not st.session_state.pipeline_results:
        st.warning("Please generate metrics in Tab 1 first before running pairwise evaluations.")
        return
    
    # Get the current (potentially edited) prompts
    _, pairwise_prompts = get_current_prompts()
    
    # Display the number of available prompts
    st.write(f"Using {len(pairwise_prompts)} pairwise evaluation templates.")
    
    # Get metrics and templates from session state
    pipeline_results = st.session_state.pipeline_results
    is_conversation = pipeline_results.get("is_conversation", False)
    pairwise_templates = pipeline_results.get("pairwise_prompt_templates", {})
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
    
    # Check if an evaluation is in progress
    is_eval_in_progress = st.session_state.get("is_eval_in_progress", False)
    has_partial_results = ("partial_eval_results" in st.session_state and 
                            st.session_state.partial_eval_results and
                            len(st.session_state.partial_eval_results) > 0)
    
    if is_eval_in_progress and has_partial_results:
        # Create a colored info box with resume option
        st.warning(
            f"""**Evaluation In Progress**  
            You have an unfinished evaluation with {len(st.session_state.partial_eval_results)} rows processed.  
            You can continue or reset below."""
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Resume Evaluation", key="resume_eval"):
                # Will be handled in the main processing section
                st.rerun()
        with col2:
            if st.button("❌ Reset and Start New", key="reset_eval"):
                # Clear all evaluation state
                if "results_by_batch" in st.session_state:
                    del st.session_state.results_by_batch
                if "partial_eval_results" in st.session_state:
                    del st.session_state.partial_eval_results
                st.session_state.is_eval_in_progress = False
                st.rerun()
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV, XLSX, or XLS file with paired outputs", 
        type=["csv", "xlsx", "xls"],
        key="pairwise_file_uploader"  # Unique key
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = process_uploaded_file_pairwise(uploaded_file)
        
        if df is not None:
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            
            # Show a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection based on conversation type
            st.subheader("Column Mapping")
            
            # Get all column names
            column_names = list(df.columns)
            
            if is_conversation:
                st.info("This is a conversation evaluation. Please select columns for both conversation transcripts to compare.")
                col1, col2 = st.columns(2)
                with col1:
                    conversation_a_col = st.selectbox("Conversation A Column", [""] + column_names, key="conv_a")
                with col2:
                    conversation_b_col = st.selectbox("Conversation B Column", [""] + column_names, key="conv_b")
                input_col = None  # Not used for conversation
                output_a_col = conversation_a_col
                output_b_col = conversation_b_col
            else:
                st.info("This is a single-turn evaluation. Please select columns for input and both outputs to compare.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    input_col = st.selectbox("Input Column", [""] + column_names, key="input_pairwise")
                with col2:
                    output_a_col = st.selectbox("Output A Column", [""] + column_names, key="output_a")
                with col3:
                    output_b_col = st.selectbox("Output B Column", [""] + column_names, key="output_b")
            
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
                            if prompt_key in pairwise_templates:
                                is_selected = st.checkbox(
                                    f"{param_key}: {param_description}", 
                                    value=True,
                                    key=f"pairwise_{prompt_key}"
                                )
                                if is_selected:
                                    selected_metrics[prompt_key] = pairwise_templates[prompt_key]
            
            # Evaluation model selection
            st.subheader("Evaluation Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                eval_model = st.selectbox(
                    "Evaluation Model", 
                    ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14","o3-mini-2025-01-31"], 
                    index=0,
                    help="Model used for running pairwise evaluations",
                    key="pairwise_model"
                )
            
            with col2:
                sample_size = st.number_input(
                    "Sample Size (0 for all rows)", 
                    min_value=0, 
                    value=min(len(df), 5),
                    help="Number of rows to evaluate (0 means all rows)",
                    key="pairwise_sample"
                )
            
            # Batch size selector - Set conservative default (50 rows)
            batch_size = st.slider(
                "Batch Size", 
                min_value=10, 
                max_value=200, 
                value=50, 
                step=10,
                help="Number of rows to process before saving progress. Smaller batches are safer for long runs but may be slower."
            )
            
            # Add a note about optimal batch size based on dataset size
            if len(df) > 1000:
                st.info("ℹ️ For large datasets (1000+ rows), smaller batch sizes (20-30) are recommended for stability.")
            elif len(df) > 100:
                st.info("ℹ️ For medium datasets, the default batch size of 50 provides a good balance of speed and stability.")
            
            # Order bias reduction option
            reduce_order_bias = st.checkbox(
                "Reduce Order Bias (Run A vs B and B vs A)", 
                value=True,
                help="Run each evaluation twice with A and B swapped to reduce position bias. Results are combined to determine a clear winner only when consistent across both evaluations."
            )
            
            if reduce_order_bias:
                st.info("Order bias reduction will run each evaluation twice (A vs B and B vs A) and only declare a winner if results are consistent in both directions. This doubles the number of evaluations but produces more reliable results.")
            
            # Add a note about required columns
            if is_conversation and (not conversation_a_col or not conversation_b_col):
                st.warning("Please select both conversation columns.")
            elif not is_conversation and (not input_col or not output_a_col or not output_b_col):
                st.warning("Please select input, output A, and output B columns.")
            elif not selected_metrics:
                st.warning("Please select at least one metric parameter to evaluate.")
            else:
                # Run evaluations button
                eval_button_text = "Run Pairwise Evaluations"
                if is_eval_in_progress and has_partial_results:
                    eval_button_text = "Continue Evaluation"
                
                if st.button(eval_button_text, use_container_width=True, type="primary"):
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
                    
                    # Create a progress bar
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0, text="Starting pairwise evaluations...")
                    
                    # Save selected metrics to session state for resumption
                    st.session_state.current_eval_metrics = selected_metrics
                    st.session_state.current_eval_is_conversation = is_conversation
                    st.session_state.current_eval_input_col = input_col
                    st.session_state.current_eval_output_a_col = output_a_col  
                    st.session_state.current_eval_output_b_col = output_b_col
                    st.session_state.current_eval_reduce_bias = reduce_order_bias
                    
                    # Run evaluations with batching
                    try:
                        with st.spinner("Running pairwise evaluations..."):
                            results = run_batch_pairwise_evaluations_with_checkpoints(
                                client=client,
                                evaluation_model=eval_model,
                                pairwise_prompts=selected_metrics,
                                data_rows=data_to_evaluate,
                                is_conversation=is_conversation,
                                input_col=input_col,
                                output_a_col=output_a_col,
                                output_b_col=output_b_col,
                                reduce_order_bias=reduce_order_bias,
                                progress_bar=progress_bar,
                                batch_size=batch_size
                            )
                        
                        st.success(f"✅ Pairwise evaluations completed for {len(results)} rows!")
                        
                        # Store results in session state
                        st.session_state.pairwise_evaluation_results = results
                        st.session_state.pairwise_reduce_order_bias = reduce_order_bias
                        st.session_state.pairwise_selected_metrics = list(selected_metrics.keys())
                        
                        # Display results with enhanced visualization
                        display_enhanced_pairwise_results(results, list(selected_metrics.keys()), reduce_order_bias)
                        
                    except Exception as e:
                        st.error(f"Error running pairwise evaluations: {str(e)}")
                        
                        # Provide resume option
                        if "partial_eval_results" in st.session_state and len(st.session_state.partial_eval_results) > 0:
                            st.warning(f"Evaluation was interrupted after processing {len(st.session_state.partial_eval_results)} rows. You can resume from this point.")
                            
                            # Option to view partial results
                            if st.button("👁️ View Partial Results"):
                                st.subheader("Partial Evaluation Results")
                                # Use the non-expander version to avoid nesting issues
                                display_batch_summary(
                                    st.session_state.partial_eval_results, 
                                    list(selected_metrics.keys()),
                                    reduce_order_bias
                                )
                                
                                # Option to download partial results
                                if st.button("💾 Download Partial Results", key="dl_partial"):
                                    # Use the download function here
                                    create_pairwise_download_section(
                                        st.session_state.partial_eval_results, 
                                        "partial_pairwise_results"
                                    )
                    finally:
                        # Complete the progress bar
                        if 'progress_bar' in locals():
                            progress_bar.progress(1.0, text="Pairwise evaluations completed or paused!")

    # Check if we have results to show
    elif "pairwise_evaluation_results" in st.session_state and st.session_state.pairwise_evaluation_results:
        st.subheader("Previous Pairwise Evaluation Results")
        st.info("Showing results from your last pairwise evaluation run. Upload a new file to run new evaluations.")
        
        # Get the order bias setting from session state or default to False
        reduce_order_bias = st.session_state.get("pairwise_reduce_order_bias", False)
        selected_metrics = st.session_state.get("pairwise_selected_metrics", list(pairwise_templates.keys()))
        
        # Display previous results with enhanced visualization
        display_enhanced_pairwise_results(
            st.session_state.pairwise_evaluation_results, 
            selected_metrics,
            reduce_order_bias
        )


def add_tab4_content():
    """
    Add the content for Tab 4 (Run Pairwise Evaluations) with enhanced visualization,
    fixed download functionality, and overall summary pie chart
    """
    import pandas as pd
    import json
    import datetime
    import base64
    import io
    st.header("Run Pairwise Evaluations on Your Data")
    st.info("Upload a spreadsheet with paired outputs and compare them using your generated metrics.")
    
    # Check if metrics are available
    if not st.session_state.pipeline_results:
        st.warning("Please generate metrics in Tab 1 first before running pairwise evaluations.")
        return
    
    # Get the current (potentially edited) prompts
    _, pairwise_prompts = get_current_prompts()
    
    # Display the number of available prompts
    st.write(f"Using {len(pairwise_prompts)} pairwise evaluation templates.")
    

    # Get metrics and templates from session state
    pipeline_results = st.session_state.pipeline_results
    is_conversation = pipeline_results.get("is_conversation", False)
    pairwise_templates = pipeline_results.get("pairwise_prompt_templates", {})
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
        "Upload a CSV, XLSX, or XLS file with paired outputs", 
        type=["csv", "xlsx", "xls"],
        key="pairwise_file_uploader"  # Unique key
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = process_uploaded_file_pairwise(uploaded_file)
        
        if df is not None:
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            
            # Show a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection based on conversation type
            st.subheader("Column Mapping")
            
            # Get all column names
            column_names = list(df.columns)
            
            if is_conversation:
                st.info("This is a conversation evaluation. Please select columns for both conversation transcripts to compare.")
                col1, col2 = st.columns(2)
                with col1:
                    conversation_a_col = st.selectbox("Conversation A Column", [""] + column_names, key="conv_a")
                with col2:
                    conversation_b_col = st.selectbox("Conversation B Column", [""] + column_names, key="conv_b")
                input_col = None  # Not used for conversation
            else:
                st.info("This is a single-turn evaluation. Please select columns for input and both outputs to compare.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    input_col = st.selectbox("Input Column", [""] + column_names, key="input_pairwise")
                with col2:
                    output_a_col = st.selectbox("Output A Column", [""] + column_names, key="output_a")
                with col3:
                    output_b_col = st.selectbox("Output B Column", [""] + column_names, key="output_b")
                conversation_a_col = None  # Not used for single-turn
                conversation_b_col = None  # Not used for single-turn
            
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
                            if prompt_key in pairwise_templates:
                                is_selected = st.checkbox(
                                    f"{param_key}: {param_description}", 
                                    value=True,
                                    key=f"pairwise_{prompt_key}"
                                )
                                if is_selected:
                                    selected_metrics[prompt_key] = pairwise_templates[prompt_key]
            
            # Evaluation model selection
            st.subheader("Evaluation Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                eval_model = st.selectbox(
                    "Evaluation Model", 
                    ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14","o3-mini-2025-01-31"], 
                    index=0,
                    help="Model used for running pairwise evaluations",
                    key="pairwise_model"
                )
            
            with col2:
                sample_size = st.number_input(
                    "Sample Size (0 for all rows)", 
                    min_value=0, 
                    value=min(len(df), 5),
                    help="Number of rows to evaluate (0 means all rows)",
                    key="pairwise_sample"
                )
            
            # Order bias reduction option
            reduce_order_bias = st.checkbox(
                "Reduce Order Bias (Run A vs B and B vs A)", 
                value=True,
                help="Run each evaluation twice with A and B swapped to reduce position bias. Results are combined to determine a clear winner only when consistent across both evaluations."
            )
            
            if reduce_order_bias:
                st.info("Order bias reduction will run each evaluation twice (A vs B and B vs A) and only declare a winner if results are consistent in both directions. This doubles the number of evaluations but produces more reliable results.")
            
            # Add a note about required columns
            if is_conversation and (not conversation_a_col or not conversation_b_col):
                st.warning("Please select both conversation columns.")
            elif not is_conversation and (not input_col or not output_a_col or not output_b_col):
                st.warning("Please select input, output A, and output B columns.")
            elif not selected_metrics:
                st.warning("Please select at least one metric parameter to evaluate.")
            else:
                # Run evaluations button
                if st.button("Run Pairwise Evaluations"):
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
                    
                    # Create a progress bar
                    progress_bar = st.progress(0, text="Starting pairwise evaluations...")
                    
                    # Run evaluations
                    try:
                        with st.spinner("Running pairwise evaluations..."):
                            results = run_batch_pairwise_evaluations(
                                client=client,
                                evaluation_model=eval_model,
                                pairwise_prompts=selected_metrics,
                                data_rows=data_to_evaluate,
                                is_conversation=is_conversation,
                                input_col=input_col if not is_conversation else None,
                                output_a_col=output_a_col if not is_conversation else conversation_a_col,
                                output_b_col=output_b_col if not is_conversation else conversation_b_col,
                                reduce_order_bias=reduce_order_bias,
                                progress_bar=progress_bar
                            )
                        
                        st.success(f"Pairwise evaluations completed for {len(results)} rows!")
                        
                        # Store results in session state
                        st.session_state.pairwise_evaluation_results = results
                        st.session_state.pairwise_reduce_order_bias = reduce_order_bias
                        st.session_state.pairwise_selected_metrics = list(selected_metrics.keys())
                        
                        # Display results with enhanced visualization
                        display_enhanced_pairwise_results(results, list(selected_metrics.keys()), reduce_order_bias)
                        
                    except Exception as e:
                        st.error(f"Error running pairwise evaluations: {e}")
                    finally:
                        # Complete the progress bar
                        progress_bar.progress(1.0, text="Pairwise evaluations completed!")

    # Check if we have results to show
    elif "pairwise_evaluation_results" in st.session_state and st.session_state.pairwise_evaluation_results:
        st.subheader("Previous Pairwise Evaluation Results")
        st.info("Showing results from your last pairwise evaluation run. Upload a new file to run new evaluations.")
        
        # Get the order bias setting from session state or default to False
        reduce_order_bias = st.session_state.get("pairwise_reduce_order_bias", False)
        selected_metrics = st.session_state.get("pairwise_selected_metrics", list(pairwise_templates.keys()))
        
        # Display previous results with enhanced visualization
        display_enhanced_pairwise_results(
            st.session_state.pairwise_evaluation_results, 
            selected_metrics,
            reduce_order_bias
        )


def run_pairwise_batch_evaluations(client, evaluation_model, pairwise_prompts, data_rows, 
                                 is_conversation, input_col, output_a_col, output_b_col,
                                 reduce_order_bias=False, batch_size=5):
    """
    Run pairwise evaluations in small batches, updating session state and showing
    progress after each batch. Includes heartbeat to keep session alive.
    
    Parameters:
    - client: API client
    - evaluation_model: Model to use for evaluations (e.g., "gpt-4o-2024-05-13")
    - pairwise_prompts: Dictionary of prompt templates
    - data_rows: List of data rows to evaluate
    - is_conversation: Boolean indicating if this is a conversation evaluation
    - input_col: Column name for input text
    - output_a_col: Column name for output A
    - output_b_col: Column name for output B
    - reduce_order_bias: Whether to run A vs B and B vs A comparisons
    - batch_size: Number of rows to process in each batch (default: 5)
    """
    import time
    import datetime
    
    # Create placeholders for status updates
    status_placeholder = st.empty()
    batch_progress = st.empty()
    heartbeat_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # Initialize session state for batch processing
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    
    # Initialize heartbeat timestamp
    last_heartbeat = time.time()
    
    # Determine starting point based on existing progress
    if st.session_state.batch_results:
        # Resume from where we left off
        results = st.session_state.batch_results
        processed_rows = len(results)
        start_idx = processed_rows
        status_placeholder.info(f"Resuming from row {start_idx+1} of {len(data_rows)}")
    else:
        # Start from the beginning
        results = []
        start_idx = 0
    
    # Calculate total evaluations for progress tracking
    evals_per_row = len(pairwise_prompts)
    if reduce_order_bias:
        evals_per_row *= 2  # Double for A/B and B/A comparisons
    
    total_rows = len(data_rows)
    total_evals = total_rows * evals_per_row
    completed_evals = start_idx * evals_per_row
    
    # Main processing loop
    try:
        # Process in batches
        for batch_start in range(start_idx, total_rows, batch_size):
            # Update heartbeat if needed (every 60 seconds)
            current_time = time.time()
            if current_time - last_heartbeat > 60:
                heartbeat_message = f"Heartbeat: {datetime.datetime.now().strftime('%H:%M:%S')}"
                heartbeat_placeholder.info(heartbeat_message)
                last_heartbeat = current_time
            
            # Get the current batch
            batch_end = min(batch_start + batch_size, total_rows)
            current_batch = data_rows[batch_start:batch_end]
            
            # Show batch progress
            status_placeholder.info(f"Processing rows {batch_start+1}-{batch_end} of {total_rows}")
            
            # Create batch progress bar
            batch_progress_bar = batch_progress.progress(0, text=f"Starting batch {batch_start//batch_size + 1}...")
            
            # Process each row in the current batch
            batch_results = []
            for i, row in enumerate(current_batch):
                row_index = batch_start + i
                row_results = {"row_index": row_index}
                
                # Get input and outputs from the row
                input_text = row[input_col] if input_col else ""
                output_a_text = row[output_a_col] if output_a_col else ""
                output_b_text = row[output_b_col] if output_b_col else ""
                
                # Skip if missing required data
                if (not is_conversation and (not input_text or not output_a_text or not output_b_text)) or \
                   (is_conversation and (not output_a_text or not output_b_text)):
                    row_results["status"] = "skipped"
                    row_results["reason"] = "Missing required data"
                    batch_results.append(row_results)
                    continue
                
                row_results["evals"] = {}
                
                # Update batch progress
                batch_progress_bar.progress((i+1) / len(current_batch), 
                                          text=f"Row {row_index+1}/{total_rows}")
                
                # Process each prompt for this row
                for prompt_key, prompt_template in pairwise_prompts.items():
                    # First evaluation: A vs B
                    if is_conversation:
                        formatted_prompt = prompt_template.replace("{conversation_a}", output_a_text).replace("{conversation_b}", output_b_text)
                    else:
                        formatted_prompt = prompt_template.replace("{input}", input_text).replace("{output_a}", output_a_text).replace("{output_b}", output_b_text)
                    
                    # Run the first evaluation (A vs B)
                    ab_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt, {
                        "input": input_text,
                        "output_a": output_a_text,
                        "output_b": output_b_text
                    })
                    
                    # Update completed evaluations count
                    completed_evals += 1
                    
                    # Run second evaluation if reducing order bias
                    ba_result = None
                    if reduce_order_bias:
                        # B vs A (reversed order)
                        if is_conversation:
                            formatted_prompt_reversed = prompt_template.replace("{conversation_a}", output_b_text).replace("{conversation_b}", output_a_text)
                        else:
                            formatted_prompt_reversed = prompt_template.replace("{input}", input_text).replace("{output_a}", output_b_text).replace("{output_b}", output_a_text)
                        
                        # Run the second evaluation (B vs A)
                        ba_result = run_pairwise_evaluation_with_client(client, evaluation_model, formatted_prompt_reversed, {
                            "input": input_text,
                            "output_a": output_b_text,
                            "output_b": output_a_text
                        })
                        
                        # Update completed evaluations count
                        completed_evals += 1
                        
                        # Combine results
                        combined_result = combine_pairwise_results(ab_result, ba_result)
                        row_results["evals"][prompt_key] = {
                            "ab_result": ab_result,
                            "ba_result": ba_result,
                            "combined_result": combined_result
                        }
                    else:
                        # Store single evaluation result
                        row_results["evals"][prompt_key] = ab_result
                
                # Add row results to batch
                batch_results.append(row_results)
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Update session state with current results
            st.session_state.batch_results = results
            
            # Display interim results after each batch
            with results_placeholder.container():
                st.subheader("Interim Results")
                st.write(f"Processed {len(results)} of {total_rows} rows")
                
                # Display a summary of the interim results
                display_batch_summary(results, list(pairwise_prompts.keys()), reduce_order_bias)
            
            # Short pause to let UI update
            time.sleep(0.5)
        
        # Clear session state once all processing is complete
        st.session_state.batch_results = []
        
        # Clear placeholders
        status_placeholder.empty()
        batch_progress.empty()
        heartbeat_placeholder.empty()
        results_placeholder.empty()
        
        # Return final results
        return results
        
    except Exception as e:
        # Save progress in session state
        st.session_state.batch_results = results
        
        # Show error with context
        error_msg = f"Error during processing at row {row_index if 'row_index' in locals() else 'unknown'}: {str(e)}"
        status_placeholder.error(error_msg)
        
        # Re-raise exception
        raise Exception(error_msg)


def add_tab4_content_improved():
    """
    Improved Tab 4 implementation with better batch processing and UI feedback
    """
    st.header("Run Pairwise Evaluations on Your Data")
    st.info("Upload a spreadsheet with paired outputs and compare them using your generated metrics.")
    
    # Check if metrics are available
    if not st.session_state.pipeline_results:
        st.warning("Please generate metrics in Tab 1 first before running pairwise evaluations.")
        return
    
    # Get the current prompts
    _, pairwise_prompts = get_current_prompts()
    
    # Display the number of available prompts
    st.write(f"Using {len(pairwise_prompts)} pairwise evaluation templates.")
    
    # Get metrics and templates from session state
    pipeline_results = st.session_state.pipeline_results
    is_conversation = pipeline_results.get("is_conversation", False)
    pairwise_templates = pipeline_results.get("pairwise_prompt_templates", {})
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
    
    # Check if an evaluation is in progress
    has_in_progress_eval = "batch_results" in st.session_state and len(st.session_state.batch_results) > 0
    
    if has_in_progress_eval:
        # Create a colored info box with resume option
        st.warning(
            f"""**Evaluation In Progress**  
            You have an unfinished evaluation with {len(st.session_state.batch_results)} rows processed.  
            You can continue or reset below."""
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Resume Evaluation", key="resume_eval"):
                # Will be handled in the main processing section
                st.rerun()
        with col2:
            if st.button("❌ Reset and Start New", key="reset_eval"):
                # Clear evaluation state
                if "batch_results" in st.session_state:
                    del st.session_state.batch_results
                st.rerun()
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV, XLSX, or XLS file with paired outputs", 
        type=["csv", "xlsx", "xls"],
        key="pairwise_file_uploader"
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = process_uploaded_file_pairwise(uploaded_file)
        
        if df is not None:
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            
            # Show a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection based on conversation type
            st.subheader("Column Mapping")
            
            # Get all column names
            column_names = list(df.columns)
            
            if is_conversation:
                st.info("This is a conversation evaluation. Please select columns for both conversation transcripts to compare.")
                col1, col2 = st.columns(2)
                with col1:
                    conversation_a_col = st.selectbox("Conversation A Column", [""] + column_names, key="conv_a")
                with col2:
                    conversation_b_col = st.selectbox("Conversation B Column", [""] + column_names, key="conv_b")
                input_col = None  # Not used for conversation
                output_a_col = conversation_a_col
                output_b_col = conversation_b_col
            else:
                st.info("This is a single-turn evaluation. Please select columns for input and both outputs to compare.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    input_col = st.selectbox("Input Column", [""] + column_names, key="input_pairwise")
                with col2:
                    output_a_col = st.selectbox("Output A Column", [""] + column_names, key="output_a")
                with col3:
                    output_b_col = st.selectbox("Output B Column", [""] + column_names, key="output_b")
            
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
                            if prompt_key in pairwise_templates:
                                is_selected = st.checkbox(
                                    f"{param_key}: {param_description}", 
                                    value=True,
                                    key=f"pairwise_{prompt_key}"
                                )
                                if is_selected:
                                    selected_metrics[prompt_key] = pairwise_templates[prompt_key]
            
            # Evaluation model and sample size selection
            st.subheader("Evaluation Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                eval_model = st.selectbox(
                    "Evaluation Model", 
                    ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14","o3-mini-2025-01-31"], 
                    index=0,
                    help="Model used for running pairwise evaluations"
                )
            
            with col2:
                sample_size = st.number_input(
                    "Sample Size (0 for all rows)", 
                    min_value=0, 
                    value=min(len(df), 5),
                    help="Number of rows to evaluate (0 means all rows)",
                    key="tab4_sample_size_number"
                )
            
            # Batch size - fixed at 5 as per requirements
            batch_size = 5
            st.info(f"Evaluations will be processed in batches of {batch_size} rows at a time with results updated after each batch.")
            
            # Order bias reduction option
            reduce_order_bias = st.checkbox(
                "Reduce Order Bias (Run A vs B and B vs A)", 
                value=True,
                help="Run each evaluation twice with A and B swapped to reduce position bias."
            )
            
            if reduce_order_bias:
                st.info("Order bias reduction doubles the number of evaluations but produces more reliable results.")
            
            # Check for required fields
            form_valid = True
            validation_message = ""
            
            if is_conversation and (not conversation_a_col or not conversation_b_col):
                form_valid = False
                validation_message = "Please select both conversation columns."
            elif not is_conversation and (not input_col or not output_a_col or not output_b_col):
                form_valid = False
                validation_message = "Please select input, output A, and output B columns."
            elif not selected_metrics:
                form_valid = False
                validation_message = "Please select at least one metric parameter to evaluate."
            
            if not form_valid:
                st.warning(validation_message)
            else:
                # Run evaluations button
                eval_button_text = "Run Pairwise Evaluations"
                if has_in_progress_eval:
                    eval_button_text = "Continue Evaluation"
                
                if st.button(eval_button_text, use_container_width=True, type="primary"):
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
                    
                    # Store evaluation parameters in session state for resumption
                    st.session_state.eval_parameters = {
                        "is_conversation": is_conversation,
                        "input_col": input_col,
                        "output_a_col": output_a_col,
                        "output_b_col": output_b_col,
                        "reduce_order_bias": reduce_order_bias,
                        "selected_metrics": selected_metrics
                    }
                    
                    # Run evaluations with improved batching
                    try:
                        with st.spinner("Running pairwise evaluations..."):
                            results = run_pairwise_batch_evaluations(
                                client=client,
                                evaluation_model=eval_model,
                                pairwise_prompts=selected_metrics,
                                data_rows=data_to_evaluate,
                                is_conversation=is_conversation,
                                input_col=input_col,
                                output_a_col=output_a_col,
                                output_b_col=output_b_col,
                                reduce_order_bias=reduce_order_bias,
                                batch_size=batch_size
                            )
                        
                        st.success(f"✅ Pairwise evaluations completed for {len(results)} rows!")
                        
                        # Store final results in session state
                        st.session_state.pairwise_evaluation_results = results
                        st.session_state.pairwise_reduce_order_bias = reduce_order_bias
                        st.session_state.pairwise_selected_metrics = list(selected_metrics.keys())
                        
                        # Display final results
                        display_enhanced_pairwise_results(results, list(selected_metrics.keys()), reduce_order_bias)
                        
                    except Exception as e:
                        st.error(f"Error running pairwise evaluations: {str(e)}")
                        
                        # Provide details about resuming
                        if "batch_results" in st.session_state and len(st.session_state.batch_results) > 0:
                            st.info(f"You can resume the evaluation with the {len(st.session_state.batch_results)} rows already processed.")
                
    # Display previous results if available
    elif "pairwise_evaluation_results" in st.session_state and st.session_state.pairwise_evaluation_results:
        st.subheader("Previous Evaluation Results")
        st.info("Showing results from your last completed evaluation. Upload a new file to run new evaluations.")
        
        # Get settings from session state
        reduce_order_bias = st.session_state.get("pairwise_reduce_order_bias", False)
        selected_metrics = st.session_state.get("pairwise_selected_metrics", [])
        
        # Display results
        display_enhanced_pairwise_results(
            st.session_state.pairwise_evaluation_results, 
            selected_metrics,
            reduce_order_bias
        )


def display_batch_summary(results, prompt_keys, reduce_order_bias=False):
    """
    Display a simple summary of batch results
    """
    import pandas as pd
    import json
    
    # Create a dataframe from results
    rows = []
    for result in results:
        row_data = {"Row": result.get("row_index", "N/A")}
        
        if result.get("status") == "skipped":
            row_data.update({prompt_key: "SKIPPED" for prompt_key in prompt_keys})
            row_data["Reason"] = result.get("reason", "Unknown")
        else:
            for prompt_key in prompt_keys:
                eval_result = result.get("evals", {}).get(prompt_key, "N/A")
                
                # Extract result based on evaluation type
                if reduce_order_bias:
                    try:
                        if isinstance(eval_result, dict) and "combined_result" in eval_result:
                            combined = eval_result["combined_result"]
                            if isinstance(combined, dict) and "final_result" in combined:
                                row_data[prompt_key] = combined["final_result"]
                            else:
                                row_data[prompt_key] = "Processing..."
                        else:
                            row_data[prompt_key] = "Processing..."
                    except:
                        row_data[prompt_key] = "Error"
                else:
                    try:
                        if isinstance(eval_result, str):
                            # Try to parse JSON
                            try:
                                eval_data = json.loads(eval_result)
                                if "winner" in eval_data:
                                    row_data[prompt_key] = eval_data["winner"]
                                elif "preference" in eval_data:
                                    row_data[prompt_key] = eval_data["preference"]
                                elif "result" in eval_data:
                                    row_data[prompt_key] = eval_data["result"]
                                else:
                                    for k, v in eval_data.items():
                                        if isinstance(v, str) and any(phrase in v for phrase in ["A is better", "B is better", "Equivalent"]):
                                            row_data[prompt_key] = v
                                            break
                                    else:
                                        row_data[prompt_key] = "Processing..."
                            except:
                                row_data[prompt_key] = "Processing..."
                        else:
                            row_data[prompt_key] = "Processing..."
                    except:
                        row_data[prompt_key] = "Error"
        
        rows.append(row_data)
    
    # Create and display summary dataframe
    if rows:
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df)
        
        # Count results by category
        a_wins = 0
        b_wins = 0
        tied = 0
        skipped = 0
        
        for row in rows:
            for key in row:
                if key not in ["Row", "Reason"]:
                    result = str(row[key]).lower()
                    if "a win" in result or result == "a is better" or result == "a":
                        a_wins += 1
                    elif "b win" in result or result == "b is better" or result == "b":
                        b_wins += 1
                    elif "tie" in result or "equivalent" in result or "equal" in result:
                        tied += 1
                    else:
                        skipped += 1
        
        # Display metrics summary
        st.markdown("#### Current Results Summary:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("A Wins", a_wins)
        col2.metric("B Wins", b_wins)
        col3.metric("Tied", tied)
        col4.metric("Skipped/Pending", skipped)
    else:
        st.info("No evaluation results to display yet.")