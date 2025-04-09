# e:\Resume Projects\InsightForge AI - An AI powered data analysis tool\main.py
import streamlit as st
import pandas as pd
import os
import re
from decimal import Decimal
import ast
import numpy as np
import json
import logging
import io
from agents.database_agent import DatabaseAgent
from agents.web_scraping_agent import WebScrapingAgent
# IMPORTANT: Ensure VisualizationAgent methods return (path: str|None, description: str)
from agents.visualization_agent import VisualizationAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.report_generation_agent import ReportGenerationAgent
from agents.analysis_agent import AnalysisAgent
from agents.data_cleaning_agent import DataCleaningAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(page_title="InsightForge AI - Advanced Data Analysis Tool", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None # Holds DataFrame for file uploads OR temporary DB data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'selected_table' not in st.session_state:
    st.session_state.selected_table = None
if 'operation_result' not in st.session_state:
    st.session_state.operation_result = None
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = None # Store schema info
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None # Track uploaded file name

# Initialize agents
# Ensure DatabaseAgent is initialized correctly (assuming it connects)
try:
    db_agent = DatabaseAgent()
    # Try a simple query to check connection
    # Use a query that doesn't rely on specific table existence initially
    db_agent.query("SELECT 1")
    st.session_state.db_connected = True
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")
    # Allow app to continue but database features will be limited
    st.session_state.db_connected = False
    # st.stop() # Optional: Stop execution if DB connection is critical

web_agent = WebScrapingAgent()
# Assuming VisualizationAgent methods now return (path, description)
viz_agent = VisualizationAgent(output_dir="data/reports")
content_agent = ContentGenerationAgent()
report_agent = ReportGenerationAgent()
data_cleaning_agent = DataCleaningAgent()

# Create directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

# Title and description
st.title("InsightForge AI - Advanced Data Analysis Tool")
st.markdown("Unlock deep insights from your data with comprehensive statistical analysis and visualizations.")

def safe_literal_eval(val):
    """Safely evaluate a string literal, returning the original string on failure."""
    if not isinstance(val, str):
        return val
    try:
        # Basic check for potentially harmful patterns before eval
        if re.search(r"(__|os\.|sys\.|subprocess\.|eval\(|exec\()", val):
            logging.warning(f"Potentially unsafe pattern detected in literal_eval input: {val[:100]}")
            return val # Return original string if potentially unsafe
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, MemoryError, TypeError) as e:
        # If literal_eval fails, return the original string or handle appropriately
        logging.warning(f"ast.literal_eval failed for value starting with: {val[:100]}. Error: {e}")
        return val # Or potentially raise an error or return None

def get_db_data(table_name):
    """Fetches data and column names for a given table."""
    if not st.session_state.db_connected:
         return None, "Database connection is not available."
    if not table_name:
        return None, "No table name provided."
    try:
        # Enclose table name in backticks for safety
        safe_table_name = f"`{table_name}`"

        # Get column names using DESCRIBE
        column_info_raw = db_agent.query(f"DESCRIBE {safe_table_name}")
        column_info = safe_literal_eval(column_info_raw)

        # Validate DESCRIBE result format
        if not isinstance(column_info, list) or not all(isinstance(item, tuple) and len(item) >= 2 for item in column_info):
             # Attempt to handle potential string formatting issues from SQLDatabase.run
             if isinstance(column_info_raw, str) and '\n' in column_info_raw:
                 # Try parsing a simple string table format
                 lines = column_info_raw.strip().split('\n')
                 if len(lines) > 1:
                     # Assuming first line might be headers, rest are data
                     # This is a heuristic and might need adjustment based on actual output
                     headers = [h.strip() for h in lines[0].split('\t')] # Example split by tab
                     # Find the 'Field' column index (case-insensitive)
                     field_index = -1
                     for i, h in enumerate(headers):
                         if h.lower() == 'field':
                             field_index = i
                             break

                     if field_index != -1:
                         parsed_columns = [line.split('\t')[field_index].strip() for line in lines[1:] if line.strip() and len(line.split('\t')) > field_index]
                         if parsed_columns:
                             columns = parsed_columns # Use extracted column names
                             logging.info(f"Parsed column names from string: {columns}")
                         else:
                              raise ValueError(f"Could not extract column names from DESCRIBE string result: {column_info_raw}")
                     else:
                         # Fallback: Assume first column is the field name if 'Field' header not found
                         logging.warning("Could not find 'Field' header in DESCRIBE output, assuming first column.")
                         parsed_columns = [line.split('\t')[0].strip() for line in lines[1:] if line.strip()]
                         if parsed_columns:
                              columns = parsed_columns
                              logging.info(f"Parsed column names from string (fallback): {columns}")
                         else:
                              raise ValueError(f"Unexpected format for DESCRIBE string result (missing 'Field' and fallback failed): {column_info_raw}")
                 else:
                     raise ValueError(f"Unexpected format for DESCRIBE result (string): {column_info_raw}")
             else:
                 raise ValueError(f"Unexpected format for DESCRIBE result (list/tuple): {column_info_raw}")
        else:
             columns = [col[0] for col in column_info] # Standard case

        # Get data using SELECT *
        data_raw = db_agent.query(f"SELECT * FROM {safe_table_name}")
        data_list = safe_literal_eval(data_raw)

        # Validate SELECT * result format
        if not isinstance(data_list, list):
             # Handle potential string formatting issues
             if isinstance(data_raw, str) and '\n' in data_raw:
                 lines = data_raw.strip().split('\n')
                 # Simple parsing assuming tab-separated or similar, needs refinement
                 # Assume first line might be headers if it looks like it
                 potential_headers = [h.strip() for h in lines[0].split('\t')]
                 data_start_index = 0
                 if len(potential_headers) == len(columns) and not all(re.match(r"^-?\d+(\.\d+)?$", h) for h in potential_headers):
                      data_start_index = 1 # Skip header row in data

                 parsed_data = [tuple(line.split('\t')) for line in lines[data_start_index:] if line.strip()]

                 # Check column count consistency
                 if parsed_data and all(len(row) == len(columns) for row in parsed_data):
                     data_list = parsed_data
                     logging.info(f"Parsed data from string result.")
                 elif parsed_data:
                      # Attempt to fix column mismatch if possible (e.g., take first N columns)
                      logging.warning(f"Column count mismatch in SELECT * string result. Expected {len(columns)}, got varying counts. Attempting to use first {len(columns)} columns.")
                      data_list = [row[:len(columns)] for row in parsed_data]
                      if not all(len(row) == len(columns) for row in data_list):
                           raise ValueError(f"Persistent column mismatch after attempting fix for SELECT * string result: {data_raw}")
                 else:
                     raise ValueError(f"Unexpected format or column mismatch for SELECT * string result: {data_raw}")
             else:
                 raise ValueError(f"Unexpected format for SELECT * result: {data_raw}")

        # Create DataFrame
        df = pd.DataFrame(data_list, columns=columns)
        # Attempt type conversion (best effort)
        df = df.infer_objects()
        for col in df.columns:
            # Try converting to numeric, then datetime, ignore errors for others
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception: pass
            try:
                # Only attempt datetime conversion if not numeric
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='ignore')
            except Exception: pass
        return df, None

    except Exception as e:
        logging.error(f"Error fetching data for table {table_name}: {e}", exc_info=True)
        return None, f"Error fetching data for table {table_name}: {e}"

def compute_statistics(data):
    """Compute statistical summary for numeric columns in the DataFrame."""
    if data is None or data.empty:
        return {}
    # Ensure data types are appropriate before selecting numeric
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {}

    stats = {}
    for col in numeric_cols:
        col_data = pd.to_numeric(data[col], errors='coerce').dropna() # Ensure numeric conversion and drop NaNs introduced
        if len(col_data) > 0:
            stats[col] = {
                "count": int(col_data.count()),
                "mean": float(col_data.mean()) if pd.notna(col_data.mean()) else None,
                "median": float(col_data.median()) if pd.notna(col_data.median()) else None,
                "std": float(col_data.std()) if pd.notna(col_data.std()) else None,
                "min": float(col_data.min()) if pd.notna(col_data.min()) else None,
                "max": float(col_data.max()) if pd.notna(col_data.max()) else None,
                "q1": float(col_data.quantile(0.25)) if pd.notna(col_data.quantile(0.25)) else None,
                "q3": float(col_data.quantile(0.75)) if pd.notna(col_data.quantile(0.75)) else None
            }
            # Round floats for display
            for key, value in stats[col].items():
                if isinstance(value, float):
                     stats[col][key] = round(value, 4)

    return stats

def analyze_categorical_columns(data):
    """Analyze categorical columns and provide insights."""
    if data is None or data.empty:
        return {}
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        return {}

    categorical_insights = {}
    for col in categorical_cols:
        # Handle potential non-hashable types by converting to string first
        try:
            col_data_str = data[col].astype(str)
            unique_count = col_data_str.nunique()
            value_counts_series = col_data_str.value_counts()
        except TypeError as e:
            logging.warning(f"Could not analyze categorical column '{col}' due to non-hashable types: {e}")
            continue # Skip this column

        if unique_count > 100:
             top_values = value_counts_series.head(10).to_dict()
             note = " (Top 10 shown due to high cardinality)"
        else:
             top_values = value_counts_series.head(10).to_dict() # Show top 10 max
             note = ""

        categorical_insights[col] = {
            # Ensure value is string for JSON compatibility if needed later
            "top_values": [{"value": str(value), "count": int(count)} for value, count in top_values.items()],
            "note": note
        }
    return categorical_insights

def generate_data_report(data):
    """Generate a detailed report for the uploaded data or selected table data."""
    if data is None or data.empty:
        return "No data available to generate a report."

    numeric_stats = compute_statistics(data)
    categorical_insights = analyze_categorical_columns(data)
    # Use io.StringIO to handle large data previews better
    buffer = io.StringIO()
    data.head().to_string(buf=buffer)
    sample_data = buffer.getvalue()

    # Format numeric statistics
    stats_text = ""
    if numeric_stats:
        stats_text_parts = []
        for col, stat in numeric_stats.items():
            stat_lines = [f"**{col} Statistics:**"]
            for key, value in stat.items():
                 stat_lines.append(f"- {key.capitalize()}: {value if value is not None else 'N/A'}")
            stats_text_parts.append("\n".join(stat_lines))
        stats_text = "\n\n".join(stats_text_parts)
    else:
        stats_text = "No numeric columns found or suitable for statistical analysis."

    # Format categorical insights
    categorical_text = ""
    if categorical_insights:
        cat_text_parts = []
        for col, insights in categorical_insights.items():
             lines = [f"**{col} Top Values{insights.get('note', '')}:**"]
             # Safely format values that might be complex types
             lines.extend([f"- {item['value']}: {item['count']}" for item in insights['top_values']])
             cat_text_parts.append("\n".join(lines))
        categorical_text = "\n\n".join(cat_text_parts)
    else:
        categorical_text = "No categorical columns found or suitable for analysis."

    # Generate insight with LLM
    insight_prompt = (
        f"Based on the following analysis of the data:\n"
        f"**Numeric Statistics:**\n{stats_text}\n\n"
        f"**Categorical Insights:**\n{categorical_text}\n\n"
        f"**Sample Data (first 5 rows):**\n{sample_data}\n\n"
        f"Provide a concise report (2-3 paragraphs) summarizing the key findings for a business user. "
        f"Focus on potential patterns, data quality observations (like skewness suggested by mean vs median, presence of outliers suggested by min/max/std dev, dominant categories, missing data implications), and possible business implications. "
        f"Synthesize the information into actionable insights rather than just listing the stats. Avoid technical jargon where possible."
    )
    try:
        insight = content_agent.generate(insight_prompt)
    except Exception as e:
        insight = f"Could not generate LLM insight: {e}"
        logging.error(f"LLM Insight Generation Error: {e}", exc_info=True)


    return f"### Data Analysis Report\n\n" \
           f"**Numeric Column Summary:**\n{stats_text}\n\n" \
           f"**Categorical Column Summary:**\n{categorical_text}\n\n" \
           f"**AI Generated Insight:**\n{insight}"

def perform_operation(operation, data, table_name=None):
    """Perform the selected operation and provide insights. Requires data."""
    logging.info(f"Performing operation: {operation} on table: {table_name}")

    if operation == "None":
        return "No operation selected."

    # Ensure data is available
    if data is None or data.empty:
        # If data is None but a table name is provided, try fetching it
        if table_name:
            logging.info(f"Data is None for operation {operation}, attempting to fetch from table {table_name}")
            data, error_msg = get_db_data(table_name)
            if error_msg:
                return f"Error fetching data for operation: {error_msg}"
            if data is None or data.empty:
                 return f"No data found in table '{table_name}' for operation."
            # Store fetched data in session state if fetched successfully here
            st.session_state.data = data
        else:
             # No data and no table name
             return "No data available for this operation. Please upload a file or select a database table first."

    # --- Perform the actual operation ---
    try:
        if operation == "Clean Data":
            # Expect three return values now: cleaned_df, tech_log, layman_log
            cleaned_data, tech_steps, layman_steps = data_cleaning_agent.clean_data(data)

            if cleaned_data is None:
                # If cleaning failed catastrophically, tech_steps might contain the error
                return f"Data cleaning failed: {tech_steps}" # tech_steps holds error message here

            # Update session state with the cleaned data ONLY if successful
            st.session_state.data = cleaned_data
            st.success("Data cleaning applied successfully!")

            # Format the output to show both explanations
            return (
                f"**Cleaning Steps Performed (Technical Details):**\n"
                f"```\n{tech_steps}\n```\n\n"
                f"**Explanation (What We Did):**\n"
                f"{layman_steps}\n\n"
                f"**Preview of Cleaned Data:**\n"
                f"{cleaned_data.head().to_string()}"
            )

        elif operation == "Visualize Data":
            # Assuming generate_all returns a dict like: {plot_type: (path, description)}
            visualization_results = viz_agent.generate_all(data)
            result_string = "**Generated Visualizations:**\n"
            images_generated = False
            if not visualization_results:
                result_string += "Could not generate any visualizations for this data."
            else:
                for plot_type, result_tuple in visualization_results.items():
                    title = plot_type.replace('_', ' ').title()
                    # --- MODIFIED: Unpack tuple ---
                    image_path, description = None, None
                    if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                        image_path, description = result_tuple
                    else:
                        # Handle unexpected return format (log warning)
                        logging.warning(f"Unexpected return format from viz_agent.generate_all for {plot_type}: {result_tuple}")
                        description = f"Error processing result for {title}."

                    # Check if image_path is valid and points to an existing file
                    if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                        try:
                            # Use description as caption if available, otherwise use title
                            caption_text = f"{title}: {description}" if description else title
                            st.image(image_path, caption=caption_text, use_column_width=True)
                            images_generated = True
                        except Exception as img_e:
                             result_string += f"- {title}: Error displaying image {image_path}: {img_e}\n"
                             logging.error(f"Error displaying image {image_path}: {img_e}", exc_info=True)
                    elif description: # Handle cases where image failed but we have a description/error
                        result_string += f"- {title}: {description}\n"
                    else: # Handle case where path is None/empty and no description
                         result_string += f"- {title}: Failed to generate (unknown reason).\n"

            # Return the text summary only if no images were successfully displayed
            return result_string if not images_generated else ""

        elif operation == "Detailed Analysis":
            # Ensure data is a DataFrame before passing to AnalysisAgent
            if not isinstance(data, pd.DataFrame):
                 return f"Error: Data for detailed analysis is not in the expected format (DataFrame)."

            analysis_agent = AnalysisAgent(data) # Pass the DataFrame
            buffer = io.StringIO()
            try:
                data.info(buf=buffer)
                data_info = buffer.getvalue()
            except Exception as info_e:
                data_info = f"Could not get data info: {info_e}"

            data_head = data.head().to_string()
            # Limit data passed to report agent if it's very large
            data_summary_for_report = f"Data Info:\n{data_info}\n\nData Head:\n{data_head}"

            eda_results, insight = analysis_agent.analyze() # Analyze the DataFrame
            # Pass limited summary instead of full data to report agent
            report = report_agent.generate_report(data_summary_for_report, eda_results, insight)
            return f"**Detailed Analysis Report:**\n{report}"

        elif operation == "Web Research":
            # Generate a query based on data context
            research_topic_prompt = f"Based on the columns {data.columns.tolist()} and this data sample:\n{data.head().to_string()}\nSuggest a relevant web research topic (e.g., market trends for a product, competitor analysis based on categories, definitions of industry terms)."
            research_topic = content_agent.generate(research_topic_prompt)
            st.info(f"Suggested Research Topic: {research_topic}")
            research_result = web_agent.research(research_topic)
            return f"**Web Research on '{research_topic}':**\n{research_result}"

        elif operation == "Ask a specific question":
            # This operation is better handled by the chat interface directly.
            question_prompt = f"Based on the columns {data.columns.tolist()} and this data sample:\n{data.head().to_string()}\nSuggest an insightful question that can be answered from this data (e.g., 'What is the average value of X for category Y?', 'Are there outliers in column Z?', 'Which category has the highest average rating?')."
            suggested_question = content_agent.generate(question_prompt)
            return f"**Suggestion:** Try asking a specific question in the chat below, for example: '{suggested_question}'"

        else:
            return f"Operation '{operation}' not implemented correctly."

    except Exception as e:
        logging.error(f"Error during operation '{operation}': {e}", exc_info=True)
        st.error(f"An error occurred while performing '{operation}': {e}") # Show error in UI
        return f"An error occurred while performing '{operation}': {e}"


def format_database_response(raw_result, question, schema_info):
    """Convert raw database query results into detailed insights with statistical analysis."""
    logging.info(f"Formatting DB response for question: {question}")
    logging.debug(f"Raw result: {raw_result}")

    if raw_result is None:
        return "Query executed, but returned no result."
    # Handle potential errors returned as strings from db.run
    if isinstance(raw_result, str) and ("Error" in raw_result or "Traceback" in raw_result):
        logging.error(f"Database query returned an error: {raw_result}")
        return f"Database Error: {raw_result}" # Return DB errors directly

    try:
        # Attempt to parse the raw result safely
        cleaned_result = safe_literal_eval(raw_result)

        # Check if the original question was specifically about counting tables
        is_table_count_question = ("how many" in question.lower() and "table" in question.lower() and "database" in question.lower())
        is_show_tables_question = "show tables" in question.lower() or "list tables" in question.lower() or "what tables" in question.lower()

        if isinstance(cleaned_result, list):
            if len(cleaned_result) == 0:
                return "The query returned no results."

            # Handle SHOW TABLES or similar list of single-item tuples
            if all(isinstance(item, tuple) and len(item) == 1 for item in cleaned_result):
                items = [item[0] for item in cleaned_result]
                if is_show_tables_question:
                    if len(items) == 1:
                        return f"The database contains one table: '{items[0]}'."
                    else:
                        table_list = ", ".join(f"'{item}'" for item in items[:-1])
                        last_table = f"'{items[-1]}'" if items else ""
                        return f"The database contains the following tables: {table_list}{' and ' + last_table if last_table else ''}."
                # Handle single count result returned as list of tuple
                elif len(items) == 1 and isinstance(items[0], (int, float, Decimal)):
                     count_value = float(items[0]) if isinstance(items[0], Decimal) else items[0]
                     if is_table_count_question:
                         return f"There are {count_value:.0f} tables in the database."
                     else:
                         # General count result
                         return f"Result: {count_value}"
                else:
                    # General list of single items - display nicely
                    return "Query Result:\n- " + "\n- ".join(map(str, items))

            # Handle typical SELECT query results (list of tuples with multiple items)
            else:
                try:
                    # Try creating a DataFrame for better display and potential analysis
                    # We need column names - this is tricky without knowing the query
                    # Let pandas infer headers if possible, otherwise display raw
                    df_result = pd.DataFrame(cleaned_result)
                    # Try to get headers if they look like they are in the first row
                    if not df_result.empty and all(isinstance(h, str) for h in df_result.iloc[0]) and len(cleaned_result) > 1:
                         potential_headers = df_result.iloc[0]
                         # Basic check if headers seem reasonable (e.g., not all numbers)
                         if not all(re.match(r"^-?\d+(\.\d+)?$", str(h)) for h in potential_headers):
                             df_result.columns = potential_headers
                             df_result = df_result[1:].reset_index(drop=True)

                    st.dataframe(df_result)
                    return f"Query returned {len(df_result)} row(s). Displaying above."
                except Exception as df_e:
                    logging.warning(f"Could not display DB result as DataFrame: {df_e}. Falling back to string.")
                    # Fallback to string representation if DataFrame fails
                    return f"Query Result (could not format as table):\n```\n{str(cleaned_result)}\n```"


        # Handle single value result (e.g., COUNT(*)) returned directly as tuple or value
        elif isinstance(cleaned_result, tuple) and len(cleaned_result) == 1 and isinstance(cleaned_result[0], (int, float, Decimal)):
            count_value = float(cleaned_result[0]) if isinstance(cleaned_result[0], Decimal) else cleaned_result[0]
            if is_table_count_question:
                return f"There are {count_value:.0f} tables in the database."
            else: # General count result
                return f"Result: {count_value}"
        elif isinstance(cleaned_result, (int, float, Decimal)):
             count_value = float(cleaned_result) if isinstance(cleaned_result, Decimal) else cleaned_result
             if is_table_count_question:
                 return f"There are {count_value:.0f} tables in the database."
             else: # General numeric result
                 return f"Result: {count_value}"

        # Handle specific string responses (like database name)
        elif isinstance(cleaned_result, str) and "Database Name" in cleaned_result:
             # This case seems less likely now with direct query execution
             return "I am connected to the 'insightforge_db' database."

        # Fallback for other types or unhandled structures
        return f"Query Result:\n```\n{str(cleaned_result)}\n```"

    except Exception as e:
        logging.error(f"Error formatting database response: {e}", exc_info=True)
        # Return the raw result if formatting fails, as it might still be informative
        return f"Could not format the result. Raw output:\n```\n{raw_result}\n```\nError: {e}"

# --- Streamlit App Flow ---

# Data Source Selection
if st.session_state.data_source is None:
    st.subheader("1. Select Data Source")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Upload File", use_container_width=True):
            st.session_state.data_source = "Upload File"
            st.session_state.selected_table = None # Reset selections
            st.session_state.data = None
            st.session_state.chat_history = []
            st.session_state.operation_result = None
            st.session_state.uploaded_file_name = None
            st.rerun()
    with col2:
        # Disable DB button if connection failed
        db_disabled = not st.session_state.get('db_connected', False)
        if st.button("🛢️ Connect to Database", use_container_width=True, disabled=db_disabled):
            st.session_state.data_source = "Database"
            st.session_state.selected_table = None # Reset selections
            st.session_state.data = None
            st.session_state.chat_history = []
            st.session_state.operation_result = None
            st.session_state.uploaded_file_name = None
            st.rerun()
        if db_disabled:
            st.warning("Database connection failed. Please check configuration and restart.")


# Database Flow
elif st.session_state.data_source == "Database":
    st.subheader("2. Database Interaction")
    if not st.session_state.get('db_connected', False):
        st.error("Database connection is unavailable. Cannot proceed.")
    elif st.session_state.selected_table is None:
        st.markdown("Select a table to analyze.")
        try:
            tables_result = db_agent.query("SHOW TABLES")
            tables_list = safe_literal_eval(tables_result)
            # Handle list of tuples format
            if isinstance(tables_list, list) and all(isinstance(t, tuple) and len(t)==1 for t in tables_list):
                tables = [table[0] for table in tables_list]
            # Handle simple list format (sometimes returned by db.run)
            elif isinstance(tables_list, list) and all(isinstance(t, str) for t in tables_list):
                 tables = tables_list
            # Handle string format (sometimes returned by db.run)
            elif isinstance(tables_result, str) and tables_result:
                 # Filter out potential header/footer lines if any
                 tables = [line.strip() for line in tables_result.strip().split('\n') if line.strip() and not line.startswith('+--') and not line.startswith('| Tables_in_')]
            else:
                 tables = []
                 st.error(f"Could not parse table list from database response: {tables_result}")

            if tables:
                selected_table = st.selectbox("Available Tables:", tables, key="db_table_select", index=None, placeholder="Choose a table...")
                if selected_table and st.button(f"Analyze Table '{selected_table}'"):
                    st.session_state.selected_table = selected_table
                    st.session_state.data = None # Clear any previous file data
                    st.session_state.chat_history = [] # Reset chat
                    st.session_state.operation_result = None
                    st.session_state.uploaded_file_name = None # Clear file name
                    with st.spinner(f"Fetching data and schema for '{selected_table}'..."):
                        # Fetch initial data and schema for the selected table
                        df, error = get_db_data(selected_table)
                        if error:
                            st.error(error)
                            st.session_state.selected_table = None # Reset if fetch fails
                        else:
                            st.session_state.data = df # Store fetched data
                            try:
                                # Use the more detailed schema fetching method if available
                                if hasattr(db_agent, 'get_detailed_table_info'):
                                     schema_dict = db_agent.get_detailed_table_info()
                                     st.session_state.schema_info = schema_dict.get(selected_table, f"Schema for {selected_table} not found.")
                                else: # Fallback to langchain's method
                                     st.session_state.schema_info = db_agent.db.get_table_info([selected_table])
                            except Exception as schema_e:
                                 st.warning(f"Could not fetch detailed schema info: {schema_e}")
                                 st.session_state.schema_info = "Schema information unavailable."
                    st.rerun()
            elif not tables and isinstance(tables_result, str) and "Error" not in tables_result:
                 st.warning("No tables found in the database.")
            elif not tables:
                 st.warning("No tables found or could not parse table list.")


        except Exception as e:
            st.error(f"Error interacting with database: {e}")
            logging.error(f"Error fetching tables: {e}", exc_info=True)
    else:
        # Table is selected, show analysis and options
        st.success(f"Analyzing Table: **{st.session_state.selected_table}**")
        st.markdown("### Initial Data Analysis")

        # Display report if data is loaded
        if st.session_state.data is not None and not st.session_state.data.empty:
            with st.expander("View Data Sample", expanded=False):
                 st.dataframe(st.session_state.data.head())
            report_output = generate_data_report(st.session_state.data)
            st.markdown(report_output)
        elif st.session_state.data is not None and st.session_state.data.empty:
             st.warning(f"Table '{st.session_state.selected_table}' appears to be empty.")
        else:
             # Attempt to fetch data again if it's missing
             st.warning("Data not currently loaded. Attempting to fetch...")
             df, error = get_db_data(st.session_state.selected_table)
             if error:
                 st.error(f"Failed to fetch data: {error}")
             else:
                 st.session_state.data = df
                 st.rerun() # Rerun to display the fetched data and report


        st.subheader("3. Perform Operations (Optional)")
        operations = st.selectbox("Select an operation:", ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"], key="db_operation_select")
        if st.button("Run Operation"):
            if operations != "None":
                with st.spinner(f"Running {operations}..."):
                    # Data should be in st.session_state.data (either from initial load or re-fetch)
                    result = perform_operation(operations, st.session_state.data, table_name=st.session_state.selected_table)
                    st.session_state.operation_result = result
                    # No rerun here, display result directly below
            else:
                 st.session_state.operation_result = "No operation selected."

        # Display operation result immediately without rerun
        if st.session_state.operation_result:
            st.markdown("---")
            st.markdown(f"**Operation Result:**")
            # Use st.write for potentially complex outputs, markdown for text
            # Check if the result is specifically from the "Visualize Data" operation which handles its own display
            if operations == "Visualize Data" and isinstance(st.session_state.operation_result, str) and st.session_state.operation_result.startswith("**Generated Visualizations:**"):
                 st.markdown(st.session_state.operation_result, unsafe_allow_html=False) # Display text summary if no images shown
            elif isinstance(st.session_state.operation_result, str):
                 st.markdown(st.session_state.operation_result, unsafe_allow_html=False) # Avoid unsafe HTML
            else:
                 st.write(st.session_state.operation_result)
            # Clear the result after displaying to avoid showing stale results
            # st.session_state.operation_result = None # Or keep until next action if preferred


# File Upload Flow
elif st.session_state.data_source == "Upload File":
    st.subheader("2. Upload and Analyze File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    if uploaded_file is not None:
        # Check if it's a new file or data is not loaded
        if st.session_state.data is None or st.session_state.uploaded_file_name != uploaded_file.name:
             try:
                 with st.spinner(f"Loading and analyzing '{uploaded_file.name}'..."):
                     st.session_state.data = pd.read_csv(uploaded_file)
                     st.session_state.uploaded_file_name = uploaded_file.name
                     st.session_state.selected_table = None # Clear DB selection
                     st.session_state.chat_history = [] # Reset chat
                     st.session_state.operation_result = None
                     st.session_state.schema_info = None # Clear schema info
                 st.success(f"File '{uploaded_file.name}' uploaded and loaded successfully.")

                 # Automatically display analysis after upload
                 st.markdown("### Initial Data Analysis")
                 with st.expander("View Data Sample", expanded=False):
                      st.dataframe(st.session_state.data.head())
                 report_output = generate_data_report(st.session_state.data)
                 st.markdown(report_output)

             except Exception as e:
                 st.error(f"Error reading or analyzing CSV file: {e}")
                 logging.error(f"File Upload/Analysis Error: {e}", exc_info=True)
                 st.session_state.data = None
                 st.session_state.uploaded_file_name = None
                 st.session_state.chat_history = []
                 st.session_state.operation_result = None

        # Show operations only if data is loaded successfully
        if st.session_state.data is not None:
            st.subheader("3. Perform Operations (Optional)")
            operations = st.selectbox("Select an operation:", ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"], key="file_operation_select")
            if st.button("Run Operation"):
                if operations != "None":
                    with st.spinner(f"Running {operations}..."):
                        # Data is in st.session_state.data
                        result = perform_operation(operations, st.session_state.data)
                        st.session_state.operation_result = result
                        # No rerun here
                else:
                    st.session_state.operation_result = "No operation selected."

            # Display operation result immediately
            if st.session_state.operation_result:
                st.markdown("---")
                st.markdown(f"**Operation Result:**")
                # Check if the result is specifically from the "Visualize Data" operation which handles its own display
                if operations == "Visualize Data" and isinstance(st.session_state.operation_result, str) and st.session_state.operation_result.startswith("**Generated Visualizations:**"):
                    st.markdown(st.session_state.operation_result, unsafe_allow_html=False) # Display text summary if no images shown
                elif isinstance(st.session_state.operation_result, str):
                    st.markdown(st.session_state.operation_result, unsafe_allow_html=False)
                else:
                    st.write(st.session_state.operation_result)
                # st.session_state.operation_result = None # Clear result

    else:
        # Reset if file is removed or never uploaded
        if st.session_state.data is not None or st.session_state.uploaded_file_name is not None:
            st.session_state.data = None
            st.session_state.uploaded_file_name = None
            st.session_state.chat_history = []
            st.session_state.operation_result = None
            st.info("Upload a CSV file to begin analysis.")


# --- Chat Interface (Common for both DB and File flows once data/table is active) ---
if st.session_state.data is not None or st.session_state.selected_table is not None:
    st.subheader("4. Chat with your Data")
    st.markdown("Ask questions about the current data (from file or database table) or request actions.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Display the text content (description, answer, error)
            st.markdown(message["content"])
            # --- MODIFIED: Display image if path exists and is valid ---
            img_path = message.get("image")
            if img_path and isinstance(img_path, str) and os.path.exists(img_path):
                try:
                    st.image(img_path, use_column_width=True)
                except Exception as img_e:
                    st.warning(f"Could not display image {img_path}: {img_e}")
            # No need for elif here, errors/descriptions are in message["content"]


    prompt = st.chat_input("Ask a question about the data or request actions...")

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the prompt
        response_text = "" # Renamed from 'response' to avoid confusion
        image_path = None # For visualization results
        plot_description = None # To store description from viz agent

        with st.spinner("Thinking..."):
            try:
                # Determine context: DB table or uploaded file
                data_context = ""
                current_data = st.session_state.get('data') # Get current DataFrame if available
                current_table = st.session_state.get('selected_table')

                # Ensure data is loaded if a table is selected but data is None
                if current_table and current_data is None:
                     logging.info(f"Chat: Data for table '{current_table}' is None, attempting to fetch.")
                     current_data, fetch_error = get_db_data(current_table)
                     if fetch_error:
                         logging.error(f"Chat: Failed to fetch data for table '{current_table}': {fetch_error}")
                         # Respond with error and stop processing this chat message
                         response_text = f"Error: Could not fetch data for table '{current_table}' to answer your question. {fetch_error}"
                         st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": None})
                         st.rerun()
                     else:
                         st.session_state.data = current_data # Update session state

                # Build context string
                if current_data is not None:
                    columns = current_data.columns.tolist()
                    source_info = f"data loaded from file '{st.session_state.get('uploaded_file_name', 'unknown file')}'" if st.session_state.data_source == "Upload File" else f"database table '{current_table}'"
                    data_context = f"The user is currently working with {source_info}. The columns are: {columns}."
                    # Provide sample data for context
                    sample_data_head = current_data.head().to_string()
                    data_context += f"\nSample Data:\n{sample_data_head}"
                elif current_table: # Should not happen if fetch logic above works, but as fallback
                    schema_info_str = str(st.session_state.get('schema_info', 'Not available'))
                    data_context = f"The user is currently working with the database table '{current_table}'. The database is MySQL named 'insightforge_db'. Schema info: {schema_info_str}. Data is not currently loaded in memory."
                else:
                    # This case should ideally be prevented by the outer 'if' condition
                    response_text = "No data source (file or database table) is currently active. Please select one first."
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": None})
                    st.rerun()


                # --- LLM Planning Prompt ---
                # MODIFIED: Added guidance for distribution plots
                planning_prompt = f"""
                Analyze the user's request: '{prompt}'

                Context:
                {data_context}

                Determine the user's intent and classify it into ONE category:
                1. Database Query: User asks about DB structure (tables, schema) NOT related to the currently loaded data, or explicitly asks to query the database.
                2. Data Analysis (File/Table): User asks a question about the currently loaded data (from file or selected table), like columns, counts, averages, missing values, correlations, outliers, or requests a summary/insight.
                3. Visualization Request: User asks for a plot, graph, or chart of the currently loaded data.
                4. Web Research: User asks for external information (trends, definitions, etc.).
                5. General Chit-chat/Unclear: The request is conversational or doesn't fit other categories.

                Based on the category, provide a plan:

                - If Database Query: (Only if the question is explicitly about DB structure unrelated to current data, or uses SQL keywords)
                    - If asking about tables: Plan: SQL Query: SHOW TABLES
                    - If asking about schema of a SPECIFIC table 'some_other_table': Plan: SQL Query: DESCRIBE `some_other_table`
                    - If asking for data using SQL (e.g., 'SELECT * FROM x'): Plan: SQL Query: [Extract the SQL query]
                    - If asking about the database name: Plan: DB Name Request: True

                - If Data Analysis (File/Table): (Requires loaded data: {'Yes' if current_data is not None else 'No'})
                    - If asking for column names or list of columns: Plan: Analysis Type: List Columns
                    - If asking for missing values: Plan: Analysis Type: Missing Values
                    - If asking for summary statistics (describe, info): Plan: Analysis Type: Summary Stats
                    - If asking for counts/averages/sums/min/max for specific columns/conditions (e.g., 'average rating', 'count where country is US'): Plan: Analysis Type: Aggregation\nDetails: [Describe the aggregation needed, e.g., 'Average of product_star_rating where country == "US"']
                    - If asking for correlation: Plan: Analysis Type: Correlation
                    - If asking about outliers: Plan: Analysis Type: Outlier Check
                    - If asking for general insights/summary/report: Plan: Analysis Type: General Insight

                - If Visualization Request: (Requires loaded data: {'Yes' if current_data is not None else 'No'})
                    - **Guidance:** If the user asks for a 'distribution plot', 'histogram', or mentions 'skewness' for a *specific column* (e.g., 'distribution of price'), plan for 'histogram' and specify the column as 'Target'.
                    - **Guidance:** If the user asks for a 'distribution plot' or mentions 'skewness' *without* specifying a column, plan for 'histogram' but leave the 'Columns' field empty or note that the column is unspecified.
                    - Plan: Plot Type: [Specify plot: scatter, histogram, boxplot, heatmap, bar]\nColumns: [Specify columns if mentioned, e.g., X=colA, Y=colB, Target=colC, Hue=colD. If histogram/boxplot requested but column unclear, leave empty or state 'Target=Unspecified']

                - If Web Research:
                    - Plan: Topic: [Extract the research topic]

                - If General Chit-chat/Unclear:
                    - Plan: General Response: True

                Return ONLY the Category and Plan lines.
                Category: [Your Category]
                Plan: [Your Plan]
                """

                llm_response = content_agent.generate(planning_prompt)
                logging.info(f"LLM Plan Response: {llm_response}")

                # Parse the LLM's response
                category_match = re.search(r"Category:\s*(.+?)(?:\n|$)", llm_response)
                plan_match = re.search(r"Plan:\s*(.+)", llm_response, re.DOTALL)

                if not category_match or not plan_match:
                    response_text = f"I had trouble understanding how to proceed with your request. Could you please rephrase?\nDebug Info:\n{llm_response}"
                else:
                    category = category_match.group(1).strip()
                    plan = plan_match.group(1).strip()
                    logging.info(f"Parsed Category: {category}, Parsed Plan: {plan}")

                    # --- Execute Plan ---
                    if category == "Database Query":
                        if not st.session_state.get('db_connected', False):
                             response_text = "Database connection is not available to execute this query."
                        else:
                            db_name_match = re.search(r"DB Name Request:\s*True", plan)
                            sql_query_match = re.search(r"SQL Query:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                            if db_name_match:
                                response_text = "I am connected to the 'insightforge_db' database (or the one specified in config)."
                            elif sql_query_match:
                                sql_query = sql_query_match.group(1).strip()
                                # Basic safety: Check for common harmful patterns
                                if re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b", sql_query, re.IGNORECASE):
                                     response_text = "Sorry, I cannot execute queries that modify data (like DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE)."
                                else:
                                    try:
                                        with st.spinner("Executing database query..."):
                                             raw_result = db_agent.query(sql_query)
                                        response_text = format_database_response(raw_result, prompt, st.session_state.get('schema_info'))
                                    except Exception as e:
                                        response_text = f"Error executing database query: {str(e)}"
                                        logging.error(f"DB Query Error: {e}", exc_info=True)
                            else:
                                response_text = "I understood you want a database query, but I couldn't extract a specific SQL command from the plan."

                    elif category == "Data Analysis (File/Table)":
                        if current_data is None:
                            # This check might be redundant due to fetch logic above, but good safety net
                            response_text = "No data is currently loaded. Please upload a file or select and analyze a database table first."
                        else:
                            analysis_type_match = re.search(r"Analysis Type:\s*(.+?)(?:\n|$)", plan)
                            if analysis_type_match:
                                analysis_type = analysis_type_match.group(1).strip()
                                try:
                                    with st.spinner(f"Performing {analysis_type} analysis..."):

                                        if analysis_type == "List Columns":
                                            if current_data is not None:
                                                columns_list = current_data.columns.tolist()
                                                response_text = "The columns in the current dataset are:\n- " + "\n- ".join(columns_list)
                                            else:
                                                response_text = "No data loaded to list columns from."

                                        elif analysis_type == "Missing Values":
                                            missing_counts = current_data.isnull().sum()
                                            missing_filtered = missing_counts[missing_counts > 0]
                                            if missing_filtered.empty:
                                                response_text = "There are no missing values in the current dataset."
                                            else:
                                                response_text = "Missing value counts per column:\n```\n" + missing_filtered.to_string() + "\n```"
                                        elif analysis_type == "Summary Stats":
                                            buffer = io.StringIO()
                                            current_data.info(buf=buffer)
                                            info_str = buffer.getvalue()
                                            desc_str = current_data.describe(include='all').to_string()
                                            response_text = f"**Data Info:**\n```\n{info_str}\n```\n\n**Summary Statistics:**\n```\n{desc_str}\n```"
                                        elif analysis_type == "Correlation":
                                            numeric_df = current_data.select_dtypes(include=np.number)
                                            if numeric_df.shape[1] < 2:
                                                 response_text = "Need at least two numeric columns for correlation analysis."
                                            else:
                                                 # --- MODIFIED: Expect tuple from viz_agent ---
                                                 image_path, plot_description = viz_agent.generate_correlation_heatmap(current_data) # Use original data
                                                 corr_matrix = numeric_df.corr()
                                                 response_text = "**Correlation Matrix:**\n```\n" + corr_matrix.to_string() + "\n```"
                                                 if image_path and os.path.exists(image_path):
                                                     # Ensure plot_description is not None or empty, provide default if needed
                                                     desc_text = plot_description if plot_description else "Correlation heatmap."
                                                     response_text += f"\n\n{desc_text}" # Add description text
                                                     # Image path is already set, will be added to history later
                                                 elif plot_description: # Handle error messages like "No numeric columns..."
                                                     response_text += f"\n\n_Note: Could not generate heatmap: {plot_description}_"
                                                     image_path = None # Ensure no image path on failure
                                                 else: # Fallback if agent returned None for both
                                                     response_text += f"\n\n_Note: Could not generate heatmap (unknown reason)._"
                                                     image_path = None


                                        elif analysis_type == "Outlier Check":
                                            numeric_cols = current_data.select_dtypes(include=np.number).columns
                                            outlier_details = []
                                            found_outliers = False
                                            if not numeric_cols.empty:
                                                for col in numeric_cols:
                                                    col_data_numeric = pd.to_numeric(current_data[col], errors='coerce').dropna()
                                                    if col_data_numeric.empty or col_data_numeric.nunique() <= 1:
                                                        continue
                                                    Q1 = col_data_numeric.quantile(0.25)
                                                    Q3 = col_data_numeric.quantile(0.75)
                                                    IQR = Q3 - Q1
                                                    if IQR == 0: continue
                                                    lower_bound = Q1 - 1.5 * IQR
                                                    upper_bound = Q3 + 1.5 * IQR
                                                    outliers_mask = (pd.to_numeric(current_data[col], errors='coerce') < lower_bound) | (pd.to_numeric(current_data[col], errors='coerce') > upper_bound)
                                                    outlier_count = outliers_mask.sum()
                                                    if outlier_count > 0:
                                                        found_outliers = True
                                                        outlier_details.append(f"- **`{col}`:** {outlier_count} potential outlier(s) found outside the range [{lower_bound:.2f}, {upper_bound:.2f}].")

                                            if found_outliers:
                                                response_text = "Yes, potential outliers were detected based on the IQR method:\n" + "\n".join(outlier_details)
                                                response_text += "\n\n*Note: This is based on a standard statistical method (1.5 * IQR). These values might be valid but are statistically unusual compared to the bulk of the data in their respective columns. Consider using the 'Clean Data' operation or further investigation.*"
                                            elif not numeric_cols.empty:
                                                response_text = "No obvious outliers were detected in the numeric columns using the standard IQR method."
                                            else:
                                                response_text = "No numeric columns were found to check for outliers."

                                        elif analysis_type == "Aggregation":
                                            # --- MODIFIED: Implement robust aggregation ---
                                            details_match = re.search(r"Details:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                                            details_text = details_match.group(1).strip() if details_match else ""

                                            # Attempt to parse details for aggregation parameters
                                            # Example: 'Average of product_star_rating where country == "US"'
                                            agg_pattern = re.compile(r"(average|mean|sum|count|min|max|median|std)\s+of\s+`?([\w\s]+)`?(?:\s+where\s+(.*))?", re.IGNORECASE)
                                            agg_match = agg_pattern.match(details_text)

                                            if agg_match:
                                                agg_func_str = agg_match.group(1).lower()
                                                agg_col = agg_match.group(2).strip().strip('`')
                                                condition = agg_match.group(3).strip() if agg_match.group(3) else None

                                                # Validate column name
                                                if agg_col not in current_data.columns:
                                                    response_text = f"Error: Column '{agg_col}' not found in the data."
                                                else:
                                                    try:
                                                        # Apply filtering if condition exists
                                                        filtered_data = current_data
                                                        if condition:
                                                            # Use pandas.query for safe evaluation of conditions
                                                            filtered_data = current_data.query(condition)

                                                        if filtered_data.empty and condition:
                                                            response_text = f"No data matches the condition: {condition}"
                                                        elif filtered_data.empty:
                                                             response_text = f"No data available to calculate {agg_func_str} of '{agg_col}'."
                                                        else:
                                                            # Perform aggregation
                                                            target_series = filtered_data[agg_col]
                                                            result = None
                                                            if agg_func_str in ["average", "mean"]:
                                                                # Ensure column is numeric for mean
                                                                if pd.api.types.is_numeric_dtype(target_series):
                                                                    result = target_series.mean()
                                                                else:
                                                                    response_text = f"Cannot calculate the average of non-numeric column '{agg_col}'."
                                                            elif agg_func_str == "sum":
                                                                if pd.api.types.is_numeric_dtype(target_series):
                                                                    result = target_series.sum()
                                                                else:
                                                                    response_text = f"Cannot calculate the sum of non-numeric column '{agg_col}'."
                                                            elif agg_func_str == "count":
                                                                result = target_series.count() # Counts non-NA values
                                                            elif agg_func_str == "min":
                                                                result = target_series.min()
                                                            elif agg_func_str == "max":
                                                                result = target_series.max()
                                                            elif agg_func_str == "median":
                                                                if pd.api.types.is_numeric_dtype(target_series):
                                                                    result = target_series.median()
                                                                else:
                                                                    response_text = f"Cannot calculate the median of non-numeric column '{agg_col}'."
                                                            elif agg_func_str == "std":
                                                                if pd.api.types.is_numeric_dtype(target_series):
                                                                    result = target_series.std()
                                                                else:
                                                                    response_text = f"Cannot calculate the standard deviation of non-numeric column '{agg_col}'."

                                                            if result is not None:
                                                                # Format result nicely
                                                                if isinstance(result, float):
                                                                    result_str = f"{result:,.4f}" # Format floats
                                                                else:
                                                                    result_str = str(result)

                                                                condition_str = f" where {condition}" if condition else ""
                                                                response_text = f"The {agg_func_str} of **`{agg_col}`**{condition_str} is: **{result_str}**"
                                                            # else: response_text might already be set with an error message

                                                    except Exception as agg_e:
                                                        response_text = f"Error performing aggregation: {agg_e}"
                                                        logging.error(f"Aggregation Error: {agg_e}", exc_info=True)

                                            else:
                                                # Fallback if details couldn't be parsed or LLM didn't provide enough
                                                logging.warning(f"Could not parse aggregation details: {details_text}. Falling back to LLM prompt.")
                                                aggregation_prompt = f"""
                                                Context:
                                                The user is analyzing data with columns: {current_data.columns.tolist()}
                                                Sample Data:
                                                {current_data.head().to_string()}

                                                User Question: '{prompt}'

                                                Based ONLY on the provided sample data and column names, try to answer the user's aggregation question. If you can calculate it from the sample, provide the answer. If the sample is insufficient or the calculation is complex, state that you need to perform a full data calculation and suggest how the user might ask for it (e.g., 'Calculate the average of X'). Do not invent data.
                                                """
                                                response_text = content_agent.generate(aggregation_prompt)
                                            # --- End of Aggregation Modification ---

                                        elif analysis_type == "General Insight":
                                             response_text = generate_data_report(current_data) # Reuse the report function

                                        else:
                                             response_text = f"I recognized the analysis type '{analysis_type}', but I don't know how to perform it yet."

                                except Exception as e:
                                    response_text = f"Error during data analysis: {e}"
                                    logging.error(f"Data Analysis Error: {e}", exc_info=True)
                            else:
                                response_text = "I understood you want data analysis, but couldn't determine the specific type from the plan."

                    elif category == "Visualization Request":
                        if current_data is None:
                            response_text = "No data is currently loaded for visualization. Please upload a file or select and analyze a database table first."
                        else:
                            plot_type_match = re.search(r"Plot Type:\s*(.+?)(?:\n|$)", plan)
                            columns_match = re.search(r"Columns:\s*(.+?)(?:\n|$)", plan)

                            if plot_type_match:
                                plot_type = plot_type_match.group(1).strip().lower().replace(" ", "_") # Normalize plot type
                                cols_text = columns_match.group(1).strip() if columns_match else ""
                                # Basic column extraction (can be improved with regex)
                                x_col, y_col, target_col, hue_col = None, None, None, None
                                # Check for explicit unspecified target
                                unspecified_target = "Target=Unspecified" in cols_text or not cols_text

                                col_specs = re.findall(r"(\w+)\s*=\s*([\w\s`'.]+?)(?:,|$)", cols_text) # Allow quoted/backticked/spaced names
                                for key, val in col_specs:
                                    key_lower = key.lower()
                                    # Remove potential quotes/backticks from column name
                                    val_stripped = val.strip().strip('`\'"')
                                    if key_lower == 'x': x_col = val_stripped
                                    elif key_lower == 'y': y_col = val_stripped
                                    elif key_lower == 'target' or key_lower == 'column': target_col = val_stripped
                                    elif key_lower == 'hue' or key_lower == 'color': hue_col = val_stripped

                                # If only one column mentioned for histogram/box/dist, assume it's the target
                                if not target_col and not x_col and not y_col and cols_text and "Target=Unspecified" not in cols_text and plot_type in ['histogram', 'boxplot', 'distribution', 'bar']: # Added 'bar' here
                                    potential_col = cols_text.strip().strip('`\'"')
                                    if potential_col in current_data.columns:
                                         target_col = potential_col
                                         unspecified_target = False # We found it

                                # Check for missing target column for plots that require one
                                if plot_type in ["histogram", "distribution", "boxplot", "bar"] and not target_col:
                                    # Check if the LLM explicitly marked it as unspecified or if we just couldn't find one
                                    if unspecified_target or not target_col:
                                        plot_name = plot_type.replace('_', ' ')
                                        col_type = "categorical" if plot_type == "bar" else "numeric"
                                        response_text = f"Please specify which {col_type} column you want a {plot_name} for. For example: 'Show me a {plot_name} of [column name]'."
                                        # Skip the rest of the visualization logic
                                    else:
                                        # This case should be less likely now, but handle if target_col is None unexpectedly
                                        response_text = f"Could not determine the target column for the {plot_type.replace('_', ' ')}. Please specify it."
                                        # Skip the rest of the visualization logic

                                else: # Proceed with generation if target column is specified or not needed
                                    try:
                                        with st.spinner(f"Generating {plot_type} plot..."):
                                            # Expect tuple (image_path, plot_description) from agent
                                            image_path, plot_description = None, None # Reset before call

                                            if plot_type == "scatter":
                                                image_path, plot_description = viz_agent.generate_scatter(current_data, x_col=x_col, y_col=y_col, hue_col=hue_col)
                                            elif plot_type in ["histogram", "distribution"]:
                                                # Pass target_col identified above
                                                image_path, plot_description = viz_agent.generate_distribution(current_data, hist_col=target_col)
                                            elif plot_type == "boxplot":
                                                if hasattr(viz_agent, 'generate_boxplot'):
                                                    # Pass target_col identified above
                                                    image_path, plot_description = viz_agent.generate_boxplot(current_data, box_col=target_col)
                                                else:
                                                    plot_description = "Boxplot generation is not currently implemented in the Visualization Agent."
                                            elif plot_type == "heatmap":
                                                image_path, plot_description = viz_agent.generate_correlation_heatmap(current_data)
                                            elif plot_type == "bar":
                                                # --- MODIFIED: Call generate_bar ---
                                                if hasattr(viz_agent, 'generate_bar'):
                                                    # Pass target_col (assumed categorical here)
                                                    image_path, plot_description = viz_agent.generate_bar(current_data, cat_col=target_col)
                                                else:
                                                    plot_description = "Bar plot generation is not currently implemented in the Visualization Agent."
                                                # --- End of Bar Plot Modification ---
                                            else:
                                                # Handle unrecognized plot types planned by LLM
                                                plot_description = f"Sorry, I don't know how to generate a '{plot_type}' plot yet. Try scatter, histogram, boxplot, bar, or heatmap."

                                            # Process result tuple
                                            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                                                # Success: We have a valid image path
                                                # Ensure plot_description is not None or empty, provide default if needed
                                                response_text = plot_description if plot_description else f"Generated {plot_type.replace('_', ' ')} plot."
                                                # image_path is already set
                                            elif plot_description: # Handle cases where image failed but we have a description/error
                                                response_text = f"Could not generate plot: {plot_description}"
                                                image_path = None # Ensure no broken image link
                                            else: # Fallback if agent returned None for both path and description
                                                response_text = f"Failed to generate {plot_type} plot (unknown reason)."
                                                image_path = None

                                    except Exception as e:
                                        response_text = f"Error generating visualization: {e}"
                                        logging.error(f"Viz Error: {e}", exc_info=True)
                                        image_path = None
                            else:
                                response_text = "I understood you want a visualization, but couldn't determine the plot type or columns from the plan."

                    elif category == "Web Research":
                        topic_match = re.search(r"Topic:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                        if topic_match:
                            topic = topic_match.group(1).strip()
                            with st.spinner(f"Researching '{topic}'..."):
                                 response_text = web_agent.research(topic)
                        else:
                            response_text = "I understood you want web research, but I couldn't extract the topic from the plan."

                    elif category == "General Chit-chat/Unclear":
                         # Use LLM for a general response
                         response_text = content_agent.generate(f"Respond conversationally to the user's message: '{prompt}'")

                    else:
                        response_text = f"I couldn't categorize your request based on the plan ('{category}'). Please try rephrasing."

            except Exception as e:
                response_text = f"An unexpected error occurred while processing your request: {str(e)}"
                logging.error(f"Chat Processing Error: {e}", exc_info=True)
                image_path = None # Ensure no image path on general error

        # Add assistant response to chat history
        # 'response_text' contains the textual answer/description/error
        # 'image_path' contains the path to the image file, or None
        st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": image_path})
        # Rerun to display the new messages and potential image
        st.rerun()

# Add a footer or sidebar link to go back
st.sidebar.markdown("---")
if st.sidebar.button("Reset & Select New Source"):
    # Clear relevant session state variables
    st.session_state.data = None
    st.session_state.data_source = None
    st.session_state.selected_table = None
    st.session_state.uploaded_file_name = None
    st.session_state.chat_history = []
    st.session_state.operation_result = None
    st.session_state.schema_info = None
    st.rerun()

# --- END OF MAIN APP LOGIC ---