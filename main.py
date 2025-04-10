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
import io # Used for StringIO buffer

# Import Agents (ensure paths are correct relative to main.py)
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

# Initialize session state (using .get() for safer access)
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
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False # Initialize db connection status

# Initialize agents
# Ensure DatabaseAgent is initialized correctly (assuming it connects)
try:
    # Only initialize if not already connected or connection failed previously
    if not st.session_state.db_connected:
        db_agent = DatabaseAgent()
        # Try a simple query to check connection
        # Use a query that doesn't rely on specific table existence initially
        db_agent.query("SELECT 1") # This might raise ConnectionError if fails
        st.session_state.db_connected = True
        logging.info("Database Agent initialized and connection verified.")
    else:
        # If already connected, reuse the existing agent instance if stored, or re-init if needed
        # For simplicity here, we assume it's okay to re-init if needed, but ideally store agent in session state
        db_agent = DatabaseAgent() # Re-init for safety if not stored
        logging.info("Database Agent already connected.")

except ConnectionError as e:
    st.error(f"Failed to connect to the database: {e}")
    # Allow app to continue but database features will be limited
    st.session_state.db_connected = False
    db_agent = None # Ensure db_agent is None if connection fails
    logging.error(f"Database connection failed during initialization: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred during Database Agent initialization: {e}")
    st.session_state.db_connected = False
    db_agent = None
    logging.error(f"Unexpected error during DB Agent init: {e}", exc_info=True)


# Initialize other agents (can be done outside the DB try-except)
web_agent = WebScrapingAgent()
# Assuming VisualizationAgent methods now return (path, description)
viz_agent = VisualizationAgent(output_dir="data/reports")
content_agent = ContentGenerationAgent()
report_agent = ReportGenerationAgent()
data_cleaning_agent = DataCleaningAgent()
# AnalysisAgent is initialized later when needed with specific data

# Create directories if they don't exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

# Title and description
st.title("🚀 InsightForge AI")
st.markdown("Unlock deep insights from your data with AI-powered analysis and visualizations.")

# --- Helper Functions ---

def safe_literal_eval(val):
    """Safely evaluate a string literal, returning the original string on failure."""
    if not isinstance(val, str):
        return val
    try:
        # --- MODIFIED REGEX ---
        # Removed \( and \) from the blacklist to allow parsing of list/tuple strings
        if re.search(r"(__|os\.|sys\.|subprocess\.|eval\(|exec\(|import\s|lambda\s)", val):
            logging.warning(f"Potentially unsafe pattern detected in literal_eval input: {val[:100]}")
            return val # Return original string if potentially unsafe
        # --- END MODIFICATION ---

        # Further restrict to common safe types
        parsed = ast.literal_eval(val)
        # Add extra check if needed, e.g., isinstance(parsed, (list, tuple, dict, str, int, float, bool, type(None)))
        return parsed
    except (ValueError, SyntaxError, MemoryError, TypeError) as e:
        # If literal_eval fails, return the original string
        logging.warning(f"ast.literal_eval failed for value starting with: {val[:100]}. Error: {e}")
        return val

def get_db_data(table_name):
    """Fetches data and column names for a given table."""
    if not st.session_state.get('db_connected', False) or db_agent is None:
         return None, "Database connection is not available."
    if not table_name:
        return None, "No table name provided."
    try:
        # Enclose table name in backticks for safety, handle potential existing backticks
        safe_table_name = f"`{table_name.strip('`')}`"

        # --- Robust Column Name Parsing (Corrected - v2) ---
        logging.info(f"Fetching DESCRIBE info for {safe_table_name}...")
        column_info_result = db_agent.query(f"DESCRIBE {safe_table_name}") # Get result
        logging.info(f"Received DESCRIBE result (type: {type(column_info_result)}): {str(column_info_result)[:500]}...") # Log type and snippet

        columns = [] # Initialize empty
        parsed_successfully = False

        # --- Attempt 1: Direct Parsing (if result is already a list/tuple) ---
        if isinstance(column_info_result, (list, tuple)) and column_info_result:
            logging.info("Attempting direct parsing of list/tuple result...")
            try:
                # Check if items are tuples/lists and have at least one element
                if all(isinstance(item, (tuple, list)) and len(item) > 0 for item in column_info_result):
                    columns = [item[0] for item in column_info_result]
                    logging.info(f"Direct parsing successful. Columns: {columns}")
                    parsed_successfully = True
                else:
                    logging.warning("Direct parsing failed: Items in list/tuple are not valid tuples/lists or are empty.")
            except IndexError as e:
                 logging.warning(f"Direct parsing failed: IndexError ({e}). Result: {column_info_result}")
            except TypeError as e:
                 logging.warning(f"Direct parsing failed: TypeError ({e}) - result might not be iterable as expected. Result: {column_info_result}")
            except Exception as e:
                 logging.warning(f"Direct parsing failed: Unexpected error ({e}). Result: {column_info_result}", exc_info=True)

        # --- Attempt 2: Parsing via safe_literal_eval (if result is a string) ---
        if not parsed_successfully and isinstance(column_info_result, str):
            logging.info("Attempting parsing via safe_literal_eval for string result...")
            column_info_parsed = safe_literal_eval(column_info_result) # Uses corrected safe_literal_eval
            logging.info(f"Result after safe_literal_eval (type: {type(column_info_parsed)}): {str(column_info_parsed)[:500]}...")

            if isinstance(column_info_parsed, (list, tuple)) and column_info_parsed:
                try:
                    # Check again if items are tuples/lists and have at least one element
                    if all(isinstance(item, (tuple, list)) and len(item) > 0 for item in column_info_parsed):
                        columns = [item[0] for item in column_info_parsed]
                        logging.info(f"Parsing via safe_literal_eval successful. Columns: {columns}")
                        parsed_successfully = True
                    else:
                        logging.warning("Parsing via safe_literal_eval failed: Items in parsed list/tuple are not valid tuples/lists or are empty.")
                except IndexError as e:
                    logging.warning(f"Parsing via safe_literal_eval failed: IndexError ({e}). Parsed: {column_info_parsed}")
                except TypeError as e:
                    logging.warning(f"Parsing via safe_literal_eval failed: TypeError ({e}) - parsed result not iterable? Parsed: {column_info_parsed}")
                except Exception as e:
                    logging.warning(f"Parsing via safe_literal_eval failed: Unexpected error ({e}). Parsed: {column_info_parsed}", exc_info=True)
            else:
                logging.warning("Parsing via safe_literal_eval failed: Result was not a list/tuple or was empty.")

        # --- Attempt 3: Add custom string parsing if needed ---
        # elif not parsed_successfully and isinstance(column_info_result, str):
        #    # ... your custom string parsing logic here ...
        #    pass

        # --- Final Check: Raise error ONLY if columns list is STILL empty ---
        if not columns: # Check if columns list is empty (meaning parsing failed)
             logging.error(f"Failed to parse column names after all attempts. Final 'columns' variable is empty. Original Query Result: {column_info_result}")
             # Raise the error with the original raw result for better debugging
             raise ValueError(f"Could not extract column names from DESCRIBE result after all parsing attempts. Query Result: {column_info_result}")

        # --- Column names successfully parsed, continue ---
        logging.info(f"Column name parsing successful for {safe_table_name}. Columns: {columns}")
        # --- End Column Name Parsing ---


        # Get data using SELECT *
        data_raw = db_agent.query(f"SELECT * FROM {safe_table_name}")
        data_list = safe_literal_eval(data_raw)

        # --- Robust Data Parsing ---
        if not isinstance(data_list, list):
             if isinstance(data_raw, str):
                 lines = [line.strip() for line in data_raw.strip().split('\n') if line.strip()]
                 if lines:
                     delimiters = r'\s{2,}|\t' # Match 2+ spaces or tab
                     # Simple parsing assuming consistent delimiter, needs refinement for complex cases
                     parsed_data = [tuple(re.split(delimiters, line, maxsplit=len(columns)-1)) for line in lines] # maxsplit helps with trailing spaces

                     # Check column count consistency
                     if parsed_data and all(len(row) == len(columns) for row in parsed_data):
                         data_list = parsed_data
                         logging.info(f"Parsed data from string result for {safe_table_name}.")
                     elif parsed_data:
                          # Attempt to fix column mismatch if possible (e.g., take first N columns)
                          logging.warning(f"Column count mismatch in SELECT * string result for {safe_table_name}. Expected {len(columns)}, got varying counts. Attempting to use first {len(columns)} columns.")
                          data_list_fixed = [row[:len(columns)] for row in parsed_data]
                          # Pad rows that are too short (less common but possible)
                          data_list_fixed = [row + (None,) * (len(columns) - len(row)) if len(row) < len(columns) else row for row in data_list_fixed]

                          if all(len(row) == len(columns) for row in data_list_fixed):
                               data_list = data_list_fixed
                               logging.info(f"Successfully adjusted column count for {safe_table_name}.")
                          else:
                               raise ValueError(f"Persistent column mismatch after attempting fix for SELECT * string result: {data_raw}")
                     else: # No data parsed from string
                          data_list = [] # Treat as empty result
                 else: # Empty string result
                      data_list = []
             else: # Neither list nor string
                 raise ValueError(f"Unexpected format for SELECT * result: {type(data_raw)}")
        # --- End Data Parsing ---

        # Create DataFrame
        df = pd.DataFrame(data_list, columns=columns)

        # --- Improved Type Conversion ---
        logging.info(f"Attempting type conversion for table {safe_table_name}...")
        for col in df.columns:
            # Attempt numeric conversion first (handle potential errors)
            try:
                original_dtype = df[col].dtype
                # Use infer_objects to handle mixed types better initially
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='ignore') # Try converting string representations
                # If conversion happened, log it
                if df[col].dtype != original_dtype and pd.api.types.is_numeric_dtype(df[col]):
                     logging.debug(f"Column '{col}': Converted to numeric ({df[col].dtype}).")
            except Exception as e:
                logging.warning(f"Numeric conversion failed for column '{col}': {e}")

            # Attempt datetime conversion if not numeric (handle potential errors)
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    original_dtype = df[col].dtype
                    # Try converting, ignore if it fails
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                    if df[col].dtype != original_dtype and pd.api.types.is_datetime64_any_dtype(df[col]):
                         logging.debug(f"Column '{col}': Converted to datetime ({df[col].dtype}).")
                except Exception as e:
                    # Datetime conversion can be complex, log warning but continue
                    logging.warning(f"Datetime conversion attempt failed for column '{col}': {e}")

            # Convert object columns with low cardinality to category
            if pd.api.types.is_object_dtype(df[col]):
                 num_unique = df[col].nunique()
                 if num_unique / len(df) < 0.5 and num_unique < 100: # Heuristic for categorization
                     try:
                         df[col] = df[col].astype('category')
                         logging.debug(f"Column '{col}': Converted to category.")
                     except Exception as e:
                          logging.warning(f"Category conversion failed for column '{col}': {e}")

        logging.info(f"Type conversion finished for {safe_table_name}. Final dtypes:\n{df.dtypes}")
        # --- End Type Conversion ---

        return df, None

    except Exception as e:
        logging.error(f"Error fetching data for table {table_name}: {e}", exc_info=True)
        return None, f"Error fetching data for table `{table_name}`: {e}"

def compute_statistics(data):
    """Compute statistical summary for numeric columns in the DataFrame."""
    if data is None or data.empty:
        return {}
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {}

    stats = {}
    for col in numeric_cols:
        col_data = pd.to_numeric(data[col], errors='coerce').dropna()
        if not col_data.empty:
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
    # Include 'category' dtype along with 'object'
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        return {}

    categorical_insights = {}
    for col in categorical_cols:
        try:
            # Ensure consistent handling by converting to string for analysis if needed
            col_data_str = data[col].astype(str)
            unique_count = col_data_str.nunique()
            value_counts_series = col_data_str.value_counts()
        except TypeError as e:
            logging.warning(f"Could not analyze categorical column '{col}' due to non-hashable types: {e}")
            continue # Skip this column

        # Adjust threshold for showing top values if desired
        top_n = 10
        if unique_count > 50: # Example threshold
             top_values = value_counts_series.head(top_n).to_dict()
             note = f" (Top {top_n} shown due to high cardinality: {unique_count} unique values)"
        else:
             top_values = value_counts_series.head(top_n).to_dict()
             note = f" ({unique_count} unique values)"

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

    report_parts = ["### Data Analysis Report"]

    # Data Shape
    report_parts.append(f"**Shape:** {data.shape[0]} rows x {data.shape[1]} columns")

    # Data Types
    buffer_info = io.StringIO()
    data.info(buf=buffer_info)
    data_info_str = buffer_info.getvalue()
    report_parts.append(f"**Data Types & Non-Null Counts:**\n```\n{data_info_str}\n```")

    # Numeric Stats
    numeric_stats = compute_statistics(data)
    if numeric_stats:
        stats_text_parts = []
        for col, stat in numeric_stats.items():
            stat_lines = [f"**`{col}` Statistics:**"]
            for key, value in stat.items():
                 stat_lines.append(f"- {key.capitalize()}: {value if value is not None else 'N/A'}")
            stats_text_parts.append("\n".join(stat_lines))
        report_parts.append("**Numeric Column Summary:**\n" + "\n\n".join(stats_text_parts))
    else:
        report_parts.append("**Numeric Column Summary:**\nNo numeric columns found or suitable for statistical analysis.")

    # Categorical Insights
    categorical_insights = analyze_categorical_columns(data)
    if categorical_insights:
        cat_text_parts = []
        for col, insights in categorical_insights.items():
             lines = [f"**`{col}` Top Values{insights.get('note', '')}:**"]
             # Safely format values that might be complex types
             lines.extend([f"- {item['value']}: {item['count']}" for item in insights['top_values']])
             cat_text_parts.append("\n".join(lines))
        report_parts.append("**Categorical Column Summary:**\n" + "\n\n".join(cat_text_parts))
    else:
        report_parts.append("**Categorical Column Summary:**\nNo categorical columns found or suitable for analysis.")

    # Sample Data
    buffer_head = io.StringIO()
    data.head().to_string(buf=buffer_head)
    sample_data = buffer_head.getvalue()
    report_parts.append(f"**Sample Data (first 5 rows):**\n```\n{sample_data}\n```")


    # Generate insight with LLM
    # Create a concise summary for the LLM prompt
    llm_context_summary = f"Data Shape: {data.shape[0]} rows x {data.shape[1]} columns.\n"
    llm_context_summary += f"Numeric Columns Analyzed: {list(numeric_stats.keys())}\n"
    llm_context_summary += f"Categorical Columns Analyzed: {list(categorical_insights.keys())}\n"
    # Add key stats if available
    if numeric_stats:
        first_num_col = list(numeric_stats.keys())[0]
        llm_context_summary += f"Example Numeric Stats ({first_num_col}): Mean={numeric_stats[first_num_col].get('mean', 'N/A')}, Median={numeric_stats[first_num_col].get('median', 'N/A')}\n"
    if categorical_insights:
        first_cat_col = list(categorical_insights.keys())[0]
        top_val_info = categorical_insights[first_cat_col]['top_values']
        if top_val_info:
             llm_context_summary += f"Example Categorical Stats ({first_cat_col}): Top value='{top_val_info[0]['value']}' ({top_val_info[0]['count']} counts)\n"

    insight_prompt = (
        f"Based on the following analysis summary of the data:\n{llm_context_summary}\n"
        f"Provide a concise report (2-3 paragraphs) summarizing the key findings for a business user. "
        f"Focus on potential patterns, data quality observations (like skewness suggested by mean vs median, presence of outliers suggested by min/max/std dev, dominant categories, missing data implications if any were noted in info), and possible business implications. "
        f"Synthesize the information into actionable insights rather than just listing the stats. Avoid technical jargon where possible."
    )
    try:
        with st.spinner("Generating AI insight..."):
            insight = content_agent.generate(insight_prompt)
        report_parts.append(f"**AI Generated Insight:**\n{insight}")
    except Exception as e:
        insight = f"Could not generate LLM insight: {e}"
        logging.error(f"LLM Insight Generation Error: {e}", exc_info=True)
        report_parts.append(f"**AI Generated Insight:**\n_{insight}_")


    return "\n\n".join(report_parts)

def perform_operation(operation, data, table_name=None):
    """Perform the selected operation and provide insights. Requires data."""
    logging.info(f"Performing operation: {operation} on source: {st.session_state.data_source} (Table: {table_name})")

    if operation == "None":
        return "No operation selected."

    # --- Ensure data is available ---
    current_data = data # Use the passed data argument
    if current_data is None or current_data.empty:
        # If data is None but a table name is provided, try fetching it
        if table_name and st.session_state.data_source == "Database":
            logging.info(f"Data is None for operation {operation}, attempting to fetch from table {table_name}")
            fetched_data, error_msg = get_db_data(table_name)
            if error_msg:
                return f"Error fetching data for operation: {error_msg}"
            if fetched_data is None or fetched_data.empty:
                 return f"No data found in table '{table_name}' for operation."
            # Store fetched data in session state if fetched successfully here
            st.session_state.data = fetched_data
            current_data = fetched_data # Use the newly fetched data
            logging.info(f"Successfully fetched data for table '{table_name}' for operation.")
        else:
             # No data and no table name, or not a DB source
             return "No data available for this operation. Please upload a file or select a database table first."
    # --- End Data Availability Check ---


    # --- Perform the actual operation ---
    try:
        if operation == "Clean Data":
            # Expect three return values now: cleaned_df, tech_log, layman_log
            cleaned_data, tech_steps, layman_steps = data_cleaning_agent.clean_data(current_data)

            if cleaned_data is None:
                # If cleaning failed catastrophically, tech_steps might contain the error
                return f"Data cleaning failed: {tech_steps}" # tech_steps holds error message here

            # Update session state with the cleaned data ONLY if successful
            st.session_state.data = cleaned_data
            st.success("Data cleaning applied successfully! The current dataset has been updated.")

            # Format the output to show both explanations
            return (
                f"**Cleaning Steps Performed (Technical Details):**\n"
                f"```\n{tech_steps}\n```\n\n"
                f"**Explanation (What We Did):**\n"
                f"{layman_steps}\n\n"
                f"**Preview of Cleaned Data:**\n"
                f"```\n{cleaned_data.head().to_string()}\n```" # Use code block for better formatting
            )

        elif operation == "Visualize Data":
            # Assuming generate_all returns a dict like: {plot_type: (path, description)}
            visualization_results = viz_agent.generate_all(current_data)
            result_string_parts = ["**Generated Visualizations:**"]
            images_generated = False
            if not visualization_results:
                result_string_parts.append("Could not generate any visualizations for this data.")
            else:
                # Use columns for better layout if multiple images
                num_plots = len(visualization_results)
                cols = st.columns(min(num_plots, 3)) # Max 3 columns
                col_idx = 0

                for plot_type, result_tuple in visualization_results.items():
                    title = plot_type.replace('_', ' ').title()
                    image_path, description = None, None # Reset for each plot

                    # --- Robust Tuple Unpacking ---
                    if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                        image_path, description = result_tuple
                    elif isinstance(result_tuple, str): # Handle case where agent returns only error string
                         description = result_tuple
                    else:
                        # Handle unexpected return format (log warning)
                        logging.warning(f"Unexpected return format from viz_agent.generate_all for {plot_type}: {result_tuple}")
                        description = f"Error processing result for {title}."
                    # --- End Robust Tuple Unpacking ---

                    # Check if image_path is valid and points to an existing file
                    current_col = cols[col_idx % len(cols)] # Cycle through columns
                    with current_col:
                        st.markdown(f"**{title}**") # Add title above image/message
                        if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                            try:
                                # Use description as caption if available and not an error message
                                caption_text = description if description and "Error" not in description else ""
                                st.image(image_path, caption=caption_text, use_column_width=True)
                                images_generated = True
                                # Optionally add the description text below the image if it's informative
                                if description and caption_text != description:
                                     st.caption(description) # Show full description if it wasn't used as caption
                            except Exception as img_e:
                                 st.warning(f"Error displaying image {os.path.basename(image_path)}: {img_e}")
                                 result_string_parts.append(f"- {title}: Error displaying image.") # Add to text summary
                                 logging.error(f"Error displaying image {image_path}: {img_e}", exc_info=True)
                        elif description: # Handle cases where image failed but we have a description/error
                            st.warning(f"{description}") # Display error/message directly in the column
                            result_string_parts.append(f"- {title}: {description}") # Also add to text summary
                        else: # Handle case where path is None/empty and no description
                             st.warning(f"Failed to generate {title} (unknown reason).")
                             result_string_parts.append(f"- {title}: Failed to generate (unknown reason).")
                    col_idx += 1


            # Return the text summary only if no images were successfully displayed OR if there were errors
            if not images_generated or len(result_string_parts) > 1:
                 # Filter out the initial title if other messages exist
                 if len(result_string_parts) > 1:
                      return "\n".join(result_string_parts)
                 else: # Only the title means no plots/errors
                      return "No visualizations generated or encountered issues."
            else:
                 return "" # Return empty string if all images displayed successfully without errors

        elif operation == "Detailed Analysis":
            # Ensure data is a DataFrame before passing to AnalysisAgent
            if not isinstance(current_data, pd.DataFrame):
                 return f"Error: Data for detailed analysis is not in the expected format (DataFrame)."

            # Initialize AnalysisAgent with the current data
            analysis_agent = AnalysisAgent(current_data)
            buffer_info = io.StringIO()
            try:
                current_data.info(buf=buffer_info)
                data_info = buffer_info.getvalue()
            except Exception as info_e:
                data_info = f"Could not get data info: {info_e}"

            data_head = current_data.head().to_string()
            # Limit data passed to report agent if it's very large
            data_summary_for_report = f"Data Info:\n{data_info}\n\nData Head:\n{data_head}"

            eda_results, insight = analysis_agent.analyze() # Analyze the DataFrame
            # Pass limited summary instead of full data to report agent
            report = report_agent.generate_report(data_summary_for_report, eda_results, insight)
            return f"**Detailed Analysis Report:**\n{report}"

        elif operation == "Web Research":
            # Generate a query based on data context
            research_topic_prompt = f"Based on the columns {current_data.columns.tolist()} and this data sample:\n{current_data.head().to_string()}\nSuggest a relevant web research topic (e.g., market trends for a product, competitor analysis based on categories, definitions of industry terms). Return only the topic."
            research_topic = content_agent.generate(research_topic_prompt).strip()
            st.info(f"Suggested Research Topic: {research_topic}")
            research_result = web_agent.research(research_topic)
            return f"**Web Research on '{research_topic}':**\n{research_result}"

        elif operation == "Ask a specific question":
            # This operation is better handled by the chat interface directly.
            question_prompt = f"Based on the columns {current_data.columns.tolist()} and this data sample:\n{current_data.head().to_string()}\nSuggest an insightful question that can be answered from this data (e.g., 'What is the average value of X for category Y?', 'Are there outliers in column Z?', 'Which category has the highest average rating?'). Return only the suggested question."
            suggested_question = content_agent.generate(question_prompt).strip()
            return f"**Suggestion:** Try asking a specific question in the chat below, for example:\n> {suggested_question}"

        else:
            return f"Operation '{operation}' not implemented correctly."

    except Exception as e:
        logging.error(f"Error during operation '{operation}': {e}", exc_info=True)
        st.error(f"An error occurred while performing '{operation}': {e}") # Show error in UI
        return f"An error occurred while performing '{operation}': {e}"


def format_database_response(raw_result, question, schema_info):
    """Convert raw database query results into detailed insights with statistical analysis."""
    logging.info(f"Formatting DB response for question: {question}")
    logging.debug(f"Raw result type: {type(raw_result)}, value: {str(raw_result)[:500]}...") # Log type and snippet

    if raw_result is None:
        return "Query executed, but returned no result."
    # Handle potential errors returned as strings from db.run or pymysql
    if isinstance(raw_result, str) and ("Error" in raw_result or "Traceback" in raw_result or "failed" in raw_result.lower()):
        logging.error(f"Database query returned an error string: {raw_result}")
        # Try to extract a cleaner error message if possible
        error_lines = raw_result.split('\n')
        relevant_error = error_lines[-1] if error_lines else raw_result # Often the last line is most relevant
        return f"Database Error: {relevant_error}"

    try:
        # Attempt to parse the raw result safely
        # safe_literal_eval might be too restrictive for complex SQL results
        # Let's try to handle common formats directly
        cleaned_result = None
        if isinstance(raw_result, (list, tuple)):
             cleaned_result = raw_result
        elif isinstance(raw_result, str):
             try:
                 # Try ast.literal_eval first for simple list/tuple strings
                 parsed = safe_literal_eval(raw_result) # Uses corrected safe_literal_eval
                 if isinstance(parsed, (list, tuple)):
                      cleaned_result = parsed
                 else:
                      # If it parsed but isn't list/tuple, treat as string result
                      cleaned_result = raw_result
             except (ValueError, SyntaxError):
                 # If literal_eval fails, treat as a raw string result (e.g., SHOW TABLES output)
                 cleaned_result = raw_result
        else: # Handle direct numeric results etc.
             cleaned_result = raw_result


        logging.debug(f"Cleaned result type: {type(cleaned_result)}, value: {str(cleaned_result)[:500]}...")

        # Check for specific question types
        question_lower = question.lower()
        is_table_count_question = ("how many" in question_lower and "table" in question_lower)
        is_show_tables_question = "show tables" in question_lower or "list tables" in question_lower or "what tables" in question_lower
        is_describe_question = question_lower.startswith("describe ") or question_lower.startswith("desc ")

        # --- Handle List/Tuple Results ---
        if isinstance(cleaned_result, (list, tuple)):
            if len(cleaned_result) == 0:
                return "The query returned no results."

            # Handle SHOW TABLES or similar list of single-item tuples/list
            if all(isinstance(item, (tuple, list)) and len(item) == 1 for item in cleaned_result):
                items = [item[0] for item in cleaned_result]
                if is_show_tables_question:
                    if not items: return "No tables found."
                    if len(items) == 1: return f"The database contains one table: `{items[0]}`."
                    table_list = ", ".join(f"`{item}`" for item in items[:-1])
                    last_table = f"`{items[-1]}`"
                    return f"The database contains the following tables: {table_list} and {last_table}."
                # Handle single count result returned as list of tuple/list
                elif len(items) == 1 and isinstance(items[0], (int, float, Decimal)):
                     count_value = float(items[0]) if isinstance(items[0], Decimal) else items[0]
                     if is_table_count_question: return f"There are {count_value:.0f} tables in the database."
                     else: return f"Result: {count_value}" # General count
                else: # General list of single items
                    return "Query Result:\n- " + "\n- ".join(map(str, items))

            # Handle DESCRIBE results (list of tuples with multiple items)
            elif is_describe_question and all(isinstance(item, tuple) for item in cleaned_result):
                 try:
                     # Assume standard DESCRIBE columns: Field, Type, Null, Key, Default, Extra
                     df_describe = pd.DataFrame(cleaned_result, columns=['Field', 'Type', 'Null', 'Key', 'Default', 'Extra'][:len(cleaned_result[0])])
                     st.dataframe(df_describe)
                     return f"Schema description for the table displayed above."
                 except Exception as desc_e:
                      logging.warning(f"Could not format DESCRIBE result as DataFrame: {desc_e}")
                      # Fallback to string
                      return f"Schema Description:\n```\n{str(cleaned_result)}\n```"

            # Handle typical SELECT query results (list of tuples/lists with multiple items)
            else:
                try:
                    # Try creating a DataFrame - Need column names! This is the hard part.
                    # We don't reliably get column names from db.run()
                    # Display as list of tuples/lists for now.
                    # TODO: Enhance db_agent.query to potentially return headers if possible.
                    st.write(cleaned_result) # Display the raw list/tuple structure
                    return f"Query returned {len(cleaned_result)} row(s). Displaying raw result above."
                except Exception as df_e:
                    logging.warning(f"Could not display DB result: {df_e}. Falling back to string.")
                    return f"Query Result (could not format):\n```\n{str(cleaned_result)}\n```"

        # --- Handle Single Value Results ---
        elif isinstance(cleaned_result, (int, float, Decimal)):
             value = float(cleaned_result) if isinstance(cleaned_result, Decimal) else cleaned_result
             if is_table_count_question: return f"There are {value:.0f} tables in the database."
             else: return f"Result: {value}" # General numeric result

        # --- Handle String Results ---
        elif isinstance(cleaned_result, str):
             # Handle SHOW TABLES output if returned as a formatted string
             if is_show_tables_question and ('Tables_in_' in cleaned_result or '+-' in cleaned_result):
                 lines = [line.strip() for line in cleaned_result.strip().split('\n')]
                 # Filter out header/footer lines
                 table_lines = [line for line in lines if line and not line.startswith('+--') and not line.startswith('| Tables_in_')]
                 # Assume the table name is within pipes | name |
                 tables = [line.split('|')[1].strip() for line in table_lines if line.count('|') == 2]
                 if not tables: return "Could not parse table names from the output."
                 if len(tables) == 1: return f"The database contains one table: `{tables[0]}`."
                 table_list = ", ".join(f"`{item}`" for item in tables[:-1])
                 last_table = f"`{tables[-1]}`"
                 return f"The database contains the following tables: {table_list} and {last_table}."
             # Handle DESCRIBE output if returned as string
             elif is_describe_question and ('Field' in cleaned_result or '+-' in cleaned_result):
                  # Just display the formatted string
                  return f"Schema Description:\n```\n{cleaned_result}\n```"
             else:
                 # General string result
                 return f"Query Result:\n```\n{cleaned_result}\n```"

        # Fallback for other types or unhandled structures
        return f"Query Result (unrecognized format: {type(cleaned_result)}):\n```\n{str(cleaned_result)}\n```"

    except Exception as e:
        logging.error(f"Error formatting database response: {e}", exc_info=True)
        # Return the raw result if formatting fails, as it might still be informative
        return f"Could not format the result. Raw output:\n```\n{raw_result}\n```\nError during formatting: {e}"


# --- Streamlit App Flow ---

# Sidebar for Source Selection and Reset
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

# Data Source Selection (moved to sidebar)
st.sidebar.subheader("1. Data Source")
source_options = ["Select Source", "📁 Upload File", "🛢️ Database"]
selected_source_option = st.sidebar.selectbox(
    "Choose data source:",
    source_options,
    index=source_options.index(st.session_state.data_source) if st.session_state.data_source in source_options else 0,
    key="source_selector"
)

# Handle source selection change
if selected_source_option != st.session_state.data_source and selected_source_option != "Select Source":
    # Reset state when changing source
    st.session_state.data = None
    st.session_state.selected_table = None
    st.session_state.uploaded_file_name = None
    st.session_state.chat_history = []
    st.session_state.operation_result = None
    st.session_state.schema_info = None
    st.session_state.data_source = selected_source_option
    st.rerun()
elif selected_source_option == "Select Source" and st.session_state.data_source is not None:
     # If user goes back to "Select Source", reset
    st.session_state.data = None
    st.session_state.data_source = None
    st.session_state.selected_table = None
    st.session_state.uploaded_file_name = None
    st.session_state.chat_history = []
    st.session_state.operation_result = None
    st.session_state.schema_info = None
    st.rerun()


# Main Area Content based on Data Source

# --- No Source Selected ---
if st.session_state.data_source is None:
    st.markdown("### Welcome to InsightForge AI!")
    st.markdown("Please select a data source from the sidebar to begin.")
    st.info("📁 **Upload File:** Analyze data from CSV files.\n\n🛢️ **Database:** Connect to your MySQL database and analyze tables.")

# --- Database Flow ---
elif st.session_state.data_source == "🛢️ Database":
    st.header("🛢️ Database Analysis")
    if not st.session_state.get('db_connected', False):
        st.error("Database connection is unavailable. Please check configuration and restart.")
    elif st.session_state.selected_table is None:
        st.markdown("#### Select a table to analyze:")
        try:
            if db_agent: # Check if agent initialized successfully
                tables_result = db_agent.query("SHOW TABLES")
                # Use the formatting function to parse tables
                tables_formatted = format_database_response(tables_result, "show tables", None)
                # Extract table names from the formatted string (heuristic)
                tables = re.findall(r'`([^`]+)`', tables_formatted)
                if not tables and "contains the following tables:" in tables_formatted: # Fallback parsing
                     tables = [t.strip() for t in tables_formatted.split(':')[-1].split(',') if t.strip()]
                     tables = [t.replace(' and ', '').strip() for t in tables]

                if tables:
                    # Add placeholder to selectbox
                    table_options = ["Select a table..."] + tables
                    selected_table = st.selectbox(
                        "Available Tables:",
                        table_options,
                        key="db_table_select",
                        index=0 # Default to placeholder
                    )
                    if selected_table != "Select a table...":
                        if st.button(f"Analyze Table `{selected_table}`"):
                            st.session_state.selected_table = selected_table
                            st.session_state.data = None # Clear any previous file data
                            st.session_state.chat_history = [] # Reset chat
                            st.session_state.operation_result = None
                            st.session_state.uploaded_file_name = None # Clear file name
                            with st.spinner(f"Fetching data and schema for `{selected_table}`..."):
                                # Fetch initial data and schema for the selected table
                                df, error = get_db_data(selected_table) # <<< ERROR ORIGINATES HERE
                                if error:
                                    st.error(error)
                                    st.session_state.selected_table = None # Reset if fetch fails
                                else:
                                    st.session_state.data = df # Store fetched data
                                    try:
                                        # Use the more detailed schema fetching method
                                        if hasattr(db_agent, 'get_detailed_table_info'):
                                             schema_dict = db_agent.get_detailed_table_info()
                                             st.session_state.schema_info = schema_dict.get(selected_table, f"Schema for `{selected_table}` not found.")
                                        else: # Fallback to langchain's method
                                             st.session_state.schema_info = db_agent.db.get_table_info([selected_table])
                                    except Exception as schema_e:
                                         st.warning(f"Could not fetch detailed schema info: {schema_e}")
                                         st.session_state.schema_info = "Schema information unavailable."
                            st.rerun()
                elif "No tables found" in tables_formatted:
                     st.warning("No tables found in the database.")
                else: # Error parsing or other issue
                     st.error(f"Could not retrieve or parse table list from database. Response: {tables_formatted}")
            else: # db_agent is None
                 st.error("Database agent is not available.")

        except Exception as e:
            st.error(f"Error interacting with database: {e}")
            logging.error(f"Error fetching tables: {e}", exc_info=True)
    else:
        # --- Table is selected ---
        st.success(f"Analyzing Table: **`{st.session_state.selected_table}`**")
        st.markdown("---")

        # Display report if data is loaded
        if st.session_state.data is not None and not st.session_state.data.empty:
            with st.expander("View Initial Data Analysis", expanded=True):
                report_output = generate_data_report(st.session_state.data)
                st.markdown(report_output)
            with st.expander("View Current Data Sample (first 50 rows)", expanded=False):
                 st.dataframe(st.session_state.data.head(50))

        elif st.session_state.data is not None and st.session_state.data.empty:
             st.warning(f"Table `{st.session_state.selected_table}` appears to be empty.")
        else:
             # Attempt to fetch data again if it's missing
             st.warning("Data not currently loaded. Attempting to fetch...")
             df, error = get_db_data(st.session_state.selected_table)
             if error:
                 st.error(f"Failed to fetch data: {error}")
             else:
                 st.session_state.data = df
                 st.rerun() # Rerun to display the fetched data and report

        # --- Operations Expander ---
        with st.expander("Perform Operations (Optional)", expanded=False):
            operations_list = ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"]
            # Ensure key is unique if used elsewhere
            selected_op = st.selectbox("Select an operation:", operations_list, key="db_operation_select")
            if st.button("Run Operation", key="db_run_op"):
                if selected_op != "None":
                    with st.spinner(f"Running {selected_op}..."):
                        # Data should be in st.session_state.data (either from initial load or re-fetch)
                        result = perform_operation(selected_op, st.session_state.data, table_name=st.session_state.selected_table)
                        st.session_state.operation_result = result
                        # No rerun here, display result directly below
                else:
                     st.session_state.operation_result = "No operation selected."

            # Display operation result immediately without rerun
            if st.session_state.operation_result:
                st.markdown("---")
                st.markdown(f"**Operation Result:**")
                # Check if the result is specifically from the "Visualize Data" operation which handles its own display
                if selected_op == "Visualize Data" and isinstance(st.session_state.operation_result, str) and st.session_state.operation_result.startswith("**Generated Visualizations:**"):
                     st.markdown(st.session_state.operation_result, unsafe_allow_html=False) # Display text summary if no images shown
                elif isinstance(st.session_state.operation_result, str):
                     # Use markdown for potentially formatted text like cleaning logs
                     st.markdown(st.session_state.operation_result, unsafe_allow_html=False) # Avoid unsafe HTML
                else:
                     st.write(st.session_state.operation_result) # Fallback for other types
                # Clear the result after displaying to avoid showing stale results on rerun
                # st.session_state.operation_result = None # Optional: uncomment to clear immediately


# --- File Upload Flow ---
elif st.session_state.data_source == "📁 Upload File":
    st.header("📁 File Upload & Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

    if uploaded_file is not None:
        # Check if it's a new file or data is not loaded
        load_new_file = (st.session_state.data is None or
                         st.session_state.uploaded_file_name != uploaded_file.name)

        if load_new_file:
             try:
                 with st.spinner(f"Loading and analyzing '{uploaded_file.name}'..."):
                     st.session_state.data = pd.read_csv(uploaded_file)
                     st.session_state.uploaded_file_name = uploaded_file.name
                     st.session_state.selected_table = None # Clear DB selection
                     st.session_state.chat_history = [] # Reset chat
                     st.session_state.operation_result = None
                     st.session_state.schema_info = None # Clear schema info
                 st.success(f"File '{uploaded_file.name}' uploaded and loaded successfully.")
                 # Rerun to display analysis and options for the new file
                 st.rerun()

             except Exception as e:
                 st.error(f"Error reading or analyzing CSV file: {e}")
                 logging.error(f"File Upload/Analysis Error: {e}", exc_info=True)
                 # Reset state on error
                 st.session_state.data = None
                 st.session_state.uploaded_file_name = None
                 st.session_state.chat_history = []
                 st.session_state.operation_result = None
                 st.rerun()

        # --- Display analysis and options only if data is loaded ---
        if st.session_state.data is not None:
            st.success(f"Analyzing File: **{st.session_state.uploaded_file_name}**")
            st.markdown("---")

            # --- Initial Analysis Expander ---
            with st.expander("View Initial Data Analysis", expanded=True):
                report_output = generate_data_report(st.session_state.data)
                st.markdown(report_output)
            with st.expander("View Current Data Sample (first 50 rows)", expanded=False):
                 st.dataframe(st.session_state.data.head(50))

            # --- Operations Expander ---
            with st.expander("Perform Operations (Optional)", expanded=False):
                operations_list = ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"]
                selected_op = st.selectbox("Select an operation:", operations_list, key="file_operation_select")
                if st.button("Run Operation", key="file_run_op"):
                    if selected_op != "None":
                        with st.spinner(f"Running {selected_op}..."):
                            # Data is in st.session_state.data
                            result = perform_operation(selected_op, st.session_state.data)
                            st.session_state.operation_result = result
                            # No rerun here
                    else:
                        st.session_state.operation_result = "No operation selected."

                # Display operation result immediately
                if st.session_state.operation_result:
                    st.markdown("---")
                    st.markdown(f"**Operation Result:**")
                    # Check if the result is specifically from the "Visualize Data" operation which handles its own display
                    if selected_op == "Visualize Data" and isinstance(st.session_state.operation_result, str) and st.session_state.operation_result.startswith("**Generated Visualizations:**"):
                        st.markdown(st.session_state.operation_result, unsafe_allow_html=False) # Display text summary if no images shown
                    elif isinstance(st.session_state.operation_result, str):
                        st.markdown(st.session_state.operation_result, unsafe_allow_html=False)
                    else:
                        st.write(st.session_state.operation_result)
                    # st.session_state.operation_result = None # Optional: Clear result

    else:
        # Reset if file is removed or never uploaded
        if st.session_state.data is not None or st.session_state.uploaded_file_name is not None:
            st.session_state.data = None
            st.session_state.uploaded_file_name = None
            st.session_state.chat_history = []
            st.session_state.operation_result = None
            st.info("Upload a CSV file to begin analysis.")


# --- Chat Interface (Common for both DB and File flows once data/table is active) ---
# Only show chat if a source is selected and either data is loaded or a DB table is selected
if st.session_state.data_source and (st.session_state.data is not None or st.session_state.selected_table is not None):
    st.markdown("---")
    st.header("💬 Chat with InsightForge AI")
    st.markdown("Ask questions about the current data or request actions.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Display the text content (description, answer, error)
            st.markdown(message["content"])
            # --- Display image if path exists and is valid ---
            img_path = message.get("image")
            if img_path and isinstance(img_path, str) and os.path.exists(img_path):
                try:
                    st.image(img_path, use_column_width=True)
                except Exception as img_e:
                    st.warning(f"Could not display image {os.path.basename(img_path)}: {img_e}")


    # Get user input
    prompt = st.chat_input("Ask a question about the data or request actions...")

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message immediately (will be shown on rerun)

        # Process the prompt
        response_text = "" # Initialize response text
        image_path = None # For visualization results

        with st.spinner("Thinking..."):
            try:
                # --- Determine context and ensure data is loaded ---
                data_context = ""
                current_data = st.session_state.get('data') # Get current DataFrame if available
                current_table = st.session_state.get('selected_table')
                data_available = False

                # Ensure data is loaded if a table is selected but data is None
                if st.session_state.data_source == "🛢️ Database" and current_table and current_data is None:
                     logging.info(f"Chat: Data for table `{current_table}` is None, attempting to fetch.")
                     current_data, fetch_error = get_db_data(current_table)
                     if fetch_error:
                         logging.error(f"Chat: Failed to fetch data for table `{current_table}`: {fetch_error}")
                         # Respond with error and stop processing this chat message
                         response_text = f"Error: Could not fetch data for table `{current_table}` to answer your question. {fetch_error}"
                         st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": None})
                         st.rerun()
                     elif current_data is not None:
                         st.session_state.data = current_data # Update session state
                         data_available = True
                         logging.info(f"Chat: Successfully fetched data for table `{current_table}`.")
                     else: # Fetch didn't error but returned None/empty
                          response_text = f"Error: Fetched data for table `{current_table}` but it appears to be empty or invalid."
                          st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": None})
                          st.rerun()

                elif current_data is not None: # Data already loaded (from file or previous DB load)
                     data_available = True

                # --- Build context string for LLM ---
                if data_available and current_data is not None:
                    columns = current_data.columns.tolist()
                    source_info = f"data loaded from file '{st.session_state.get('uploaded_file_name', 'unknown file')}'" if st.session_state.data_source == "📁 Upload File" else f"database table `{current_table}`"
                    data_context = f"The user is currently working with {source_info}. The columns are: {columns}."
                    # Provide sample data for context
                    sample_data_head = current_data.head().to_string()
                    data_context += f"\nSample Data:\n{sample_data_head}"
                elif st.session_state.data_source == "🛢️ Database" and current_table: # DB selected, but data fetch failed/empty
                    schema_info_str = str(st.session_state.get('schema_info', 'Not available'))
                    data_context = f"The user is currently working with the database table `{current_table}`. The database is MySQL. Schema info: {schema_info_str}. Data is not currently loaded in memory or is empty."
                else:
                    # This case should ideally be prevented by the outer 'if' condition
                    response_text = "No data source (file or database table) is currently active. Please select one first."
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": None})
                    st.rerun()


                # --- LLM Planning Prompt (Refined based on previous discussion) ---
                planning_prompt = f"""
                Analyze the user's request: '{prompt}'

                Context:
                {data_context}
                Data Loaded in Memory: {'Yes' if data_available else 'No'}

                Determine the user's intent and classify it into ONE category:
                1. Database Query: User asks about DB structure (tables, schema) NOT related to the currently loaded data, or explicitly asks to query the database using SQL keywords (SELECT, SHOW, DESCRIBE).
                2. Data Analysis (File/Table): User asks a question about the currently loaded data (from file or selected table), like columns, counts, averages, missing values, correlations, outliers, data types, or requests a summary/insight. Requires Data Loaded = Yes.
                3. Visualization Request: User asks for a plot, graph, or chart of the currently loaded data. Requires Data Loaded = Yes.
                4. Web Research: User asks for external information (trends, definitions, etc.).
                5. General Chit-chat/Unclear: The request is conversational or doesn't fit other categories.

                Based on the category, provide a plan:

                - If Database Query: (Only if the question is explicitly about DB structure unrelated to current data, or uses SQL keywords)
                    - If asking about tables: Plan: SQL Query: SHOW TABLES
                    - If asking about schema of a SPECIFIC table 'some_other_table': Plan: SQL Query: DESCRIBE `some_other_table`
                    - If asking for data using SQL (e.g., 'SELECT * FROM x'): Plan: SQL Query: [Extract the SQL query]
                    - If asking about the database name: Plan: DB Name Request: True

                - If Data Analysis (File/Table): (Requires Data Loaded = Yes)
                    - If asking for column names or list of columns: Plan: Analysis Type: List Columns
                    - If asking "what are the column data types" or "list data types": Plan: Analysis Type: List Column Types
                    - If asking "which columns are integer/float/object/category/datetime/boolean": Plan: Analysis Type: List Columns by Type\nDetails: [Specify type, e.g., 'integer']
                    - If asking for missing values: Plan: Analysis Type: Missing Values
                    - If asking for summary statistics (describe, info): Plan: Analysis Type: Summary Stats
                    - If asking for counts/averages/sums/min/max/median/std/unique counts for specific columns/conditions (e.g., 'average rating', 'count where country is US', 'number of unique products'): Plan: Analysis Type: Aggregation\nDetails: [Describe the aggregation needed, e.g., 'Average of product_star_rating where country == "US"', 'Count unique values in product_name']
                    - If asking for correlation: Plan: Analysis Type: Correlation
                    - If asking about outliers: Plan: Analysis Type: Outlier Check
                    - If asking for general insights/summary/report: Plan: Analysis Type: General Insight

                - If Visualization Request: (Requires Data Loaded = Yes)
                    # Use the Visualization Agent's internal LLM via generate_plot. Just provide the prompt.
                    Plan: Visualization Prompt: {prompt}

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
                        if not st.session_state.get('db_connected', False) or db_agent is None:
                             response_text = "Database connection is not available to execute this query."
                        else:
                            db_name_match = re.search(r"DB Name Request:\s*True", plan)
                            sql_query_match = re.search(r"SQL Query:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                            if db_name_match:
                                response_text = "I am connected to the database specified in the configuration (default: 'insightforge_db')."
                            elif sql_query_match:
                                sql_query = sql_query_match.group(1).strip()
                                # Basic safety: Check for common harmful patterns
                                if re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b", sql_query, re.IGNORECASE):
                                     response_text = "Sorry, I cannot execute queries that modify data (like DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE)."
                                else:
                                    try:
                                        with st.spinner("Executing database query..."):
                                             raw_result = db_agent.query(sql_query)
                                        # Use the improved formatting function
                                        response_text = format_database_response(raw_result, prompt, st.session_state.get('schema_info'))
                                    except Exception as e:
                                        response_text = f"Error executing database query: {str(e)}"
                                        logging.error(f"DB Query Error: {e}", exc_info=True)
                            else:
                                response_text = "I understood you want a database query, but I couldn't extract a specific SQL command from the plan."

                    elif category == "Data Analysis (File/Table)":
                        if not data_available or current_data is None:
                            response_text = "No data is currently loaded. Please upload a file or select and analyze a database table first."
                        else:
                            analysis_type_match = re.search(r"Analysis Type:\s*(.+?)(?:\n|$)", plan)
                            if analysis_type_match:
                                analysis_type = analysis_type_match.group(1).strip()
                                try:
                                    with st.spinner(f"Performing {analysis_type} analysis..."):

                                        # --- Analysis Handlers ---
                                        if analysis_type == "List Columns":
                                            columns_list = current_data.columns.tolist()
                                            response_text = "The columns in the current dataset are:\n- `" + "`\n- `".join(columns_list) + "`"

                                        elif analysis_type == "List Column Types":
                                            dtypes_series = current_data.dtypes
                                            response_lines = ["**Column Data Types:**"]
                                            for col_name, dtype in dtypes_series.items():
                                                response_lines.append(f"- `{col_name}`: `{dtype}`")
                                            response_text = "\n".join(response_lines)

                                        elif analysis_type == "List Columns by Type":
                                            details_match = re.search(r"Details:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                                            requested_type_str = details_match.group(1).strip().lower() if details_match else None

                                            if requested_type_str:
                                                matching_columns = []
                                                for col in current_data.columns:
                                                    dtype = current_data[col].dtype
                                                    is_match = False
                                                    # Map requested type string to pandas type checks
                                                    type_map = {
                                                        'integer': pd.api.types.is_integer_dtype,
                                                        'int': pd.api.types.is_integer_dtype,
                                                        'int64': pd.api.types.is_integer_dtype,
                                                        'float': pd.api.types.is_float_dtype,
                                                        'floating': pd.api.types.is_float_dtype,
                                                        'float64': pd.api.types.is_float_dtype,
                                                        'number': pd.api.types.is_numeric_dtype, # Broad numeric
                                                        'numeric': pd.api.types.is_numeric_dtype, # Broad numeric
                                                        'object': lambda d: pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d),
                                                        'string': lambda d: pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d),
                                                        'text': lambda d: pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d),
                                                        'str': lambda d: pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d),
                                                        'category': pd.api.types.is_categorical_dtype,
                                                        'categorical': pd.api.types.is_categorical_dtype,
                                                        'datetime': pd.api.types.is_datetime64_any_dtype,
                                                        'date': pd.api.types.is_datetime64_any_dtype,
                                                        'time': pd.api.types.is_datetime64_any_dtype,
                                                        'timestamp': pd.api.types.is_datetime64_any_dtype,
                                                        'boolean': pd.api.types.is_bool_dtype,
                                                        'bool': pd.api.types.is_bool_dtype,
                                                    }
                                                    check_func = type_map.get(requested_type_str)
                                                    if check_func and check_func(dtype):
                                                        matching_columns.append(f"`{col}` (`{dtype}`)")

                                                if matching_columns:
                                                    response_text = f"Columns with data type matching **{requested_type_str}**:\n- " + "\n- ".join(matching_columns)
                                                else:
                                                    response_text = f"No columns found with data type matching '{requested_type_str}'."
                                            else:
                                                 response_text = "Please specify the data type you want to list columns for (e.g., 'list integer columns')."

                                        elif analysis_type == "Missing Values":
                                            missing_counts = current_data.isnull().sum()
                                            missing_filtered = missing_counts[missing_counts > 0]
                                            if missing_filtered.empty:
                                                response_text = "There are no missing values in the current dataset."
                                            else:
                                                response_text = "**Missing value counts per column:**\n```\n" + missing_filtered.to_string() + "\n```"

                                        elif analysis_type == "Summary Stats":
                                            # Reuse the generate_data_report function for consistency
                                            response_text = generate_data_report(current_data)
                                            # Remove the AI insight part if just asking for stats
                                            response_text = response_text.split("**AI Generated Insight:**")[0].strip()

                                        elif analysis_type == "Correlation":
                                            numeric_df = current_data.select_dtypes(include=np.number)
                                            if numeric_df.shape[1] < 2:
                                                 response_text = "Need at least two numeric columns for correlation analysis."
                                            else:
                                                 # Call the specific heatmap generation method
                                                 image_path, plot_description = viz_agent.generate_correlation_heatmap(current_data)
                                                 corr_matrix = numeric_df.corr()
                                                 response_text = "**Correlation Matrix:**\n```\n" + corr_matrix.to_string() + "\n```"
                                                 if image_path and os.path.exists(image_path):
                                                     desc_text = plot_description if plot_description else "Correlation heatmap."
                                                     response_text += f"\n\n{desc_text}"
                                                 elif plot_description: # Handle errors like "Not enough numeric columns..."
                                                     response_text += f"\n\n_Note: Could not generate heatmap: {plot_description}_"
                                                     image_path = None
                                                 else:
                                                     response_text += f"\n\n_Note: Could not generate heatmap (unknown reason)._"
                                                     image_path = None

                                        elif analysis_type == "Outlier Check":
                                            numeric_cols = current_data.select_dtypes(include=np.number).columns
                                            outlier_details = []
                                            found_outliers = False
                                            if not numeric_cols.empty:
                                                for col in numeric_cols:
                                                    col_data_numeric = pd.to_numeric(current_data[col], errors='coerce').dropna()
                                                    if col_data_numeric.empty or col_data_numeric.nunique() <= 1: continue
                                                    Q1 = col_data_numeric.quantile(0.25)
                                                    Q3 = col_data_numeric.quantile(0.75)
                                                    IQR = Q3 - Q1
                                                    if IQR == 0: continue
                                                    lower_bound = Q1 - 1.5 * IQR
                                                    upper_bound = Q3 + 1.5 * IQR
                                                    # Ensure comparison is done on numeric version of the column
                                                    numeric_original_col = pd.to_numeric(current_data[col], errors='coerce')
                                                    outliers_mask = (numeric_original_col < lower_bound) | (numeric_original_col > upper_bound)
                                                    outlier_count = outliers_mask.sum()
                                                    if outlier_count > 0:
                                                        found_outliers = True
                                                        outlier_details.append(f"- **`{col}`:** {outlier_count} potential outlier(s) found outside the range [{lower_bound:.2f}, {upper_bound:.2f}].")

                                            if found_outliers:
                                                response_text = "**Potential outliers detected (using 1.5 * IQR method):**\n" + "\n".join(outlier_details)
                                                response_text += "\n\n*Note: These values might be valid but are statistically unusual. Consider using the 'Clean Data' operation or further investigation.*"
                                            elif not numeric_cols.empty:
                                                response_text = "No obvious outliers were detected in the numeric columns using the standard IQR method."
                                            else:
                                                response_text = "No numeric columns were found to check for outliers."

                                        elif analysis_type == "Aggregation":
                                            details_match = re.search(r"Details:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                                            details_text = details_match.group(1).strip() if details_match and details_match.group(1).strip() else prompt

                                            agg_handled = False
                                            # --- Unique Count Pattern ---
                                            unique_pattern = re.compile(r"(?:unique\s+count|count\s+unique|number\s+of\s+unique)\s+(?:values\s+in|of)?\s+`?([\w\s.-]+)`?", re.IGNORECASE) # Allow . and - in col names
                                            unique_match = unique_pattern.search(details_text)
                                            if unique_match:
                                                agg_col = unique_match.group(1).strip().strip('`')
                                                if agg_col not in current_data.columns:
                                                    response_text = f"Error: Column '{agg_col}' not found."
                                                else:
                                                    try:
                                                        unique_count = current_data[agg_col].nunique()
                                                        response_text = f"The number of unique values in **`{agg_col}`** is: **{unique_count}**"
                                                        agg_handled = True
                                                    except Exception as unique_e:
                                                        response_text = f"Error calculating unique count for '{agg_col}': {unique_e}"
                                                        agg_handled = True

                                            # --- Standard Aggregation Pattern ---
                                            if not agg_handled:
                                                # Pattern: FUNC of COL [where CONDITION]
                                                agg_pattern = re.compile(r"(average|mean|sum|count|min|max|median|std)\s+(?:of\s+)?`?([\w\s.-]+)`?(?:\s+where\s+(.*))?", re.IGNORECASE)
                                                agg_match = agg_pattern.search(details_text)
                                                if agg_match:
                                                    agg_func_str = agg_match.group(1).lower()
                                                    agg_col = agg_match.group(2).strip().strip('`')
                                                    condition = agg_match.group(3).strip() if agg_match.group(3) else None

                                                    if agg_col not in current_data.columns:
                                                        response_text = f"Error: Column '{agg_col}' not found."
                                                    else:
                                                        try:
                                                            filtered_data = current_data
                                                            if condition:
                                                                # Use pandas.query for safe evaluation
                                                                filtered_data = current_data.query(condition)

                                                            if filtered_data.empty:
                                                                response_text = f"No data matches the condition: `{condition}`" if condition else "No data available for aggregation."
                                                            else:
                                                                target_series = filtered_data[agg_col]
                                                                result = None
                                                                error_msg = None

                                                                if agg_func_str in ["average", "mean"]:
                                                                    if pd.api.types.is_numeric_dtype(target_series): result = target_series.mean()
                                                                    else: error_msg = f"Cannot calculate average of non-numeric column '{agg_col}'."
                                                                elif agg_func_str == "sum":
                                                                    if pd.api.types.is_numeric_dtype(target_series): result = target_series.sum()
                                                                    else: error_msg = f"Cannot calculate sum of non-numeric column '{agg_col}'."
                                                                elif agg_func_str == "count": result = target_series.count() # Count non-NA
                                                                elif agg_func_str == "min": result = target_series.min()
                                                                elif agg_func_str == "max": result = target_series.max()
                                                                elif agg_func_str == "median":
                                                                    if pd.api.types.is_numeric_dtype(target_series): result = target_series.median()
                                                                    else: error_msg = f"Cannot calculate median of non-numeric column '{agg_col}'."
                                                                elif agg_func_str == "std":
                                                                    if pd.api.types.is_numeric_dtype(target_series): result = target_series.std()
                                                                    else: error_msg = f"Cannot calculate standard deviation of non-numeric column '{agg_col}'."

                                                                if error_msg:
                                                                     response_text = error_msg
                                                                elif result is not None:
                                                                    if isinstance(result, float): result_str = f"{result:,.4f}"
                                                                    else: result_str = str(result)
                                                                    condition_str = f" where `{condition}`" if condition else ""
                                                                    func_display = "count (non-missing)" if agg_func_str == "count" else agg_func_str
                                                                    response_text = f"The **{func_display}** of **`{agg_col}`**{condition_str} is: **{result_str}**"
                                                                else: # Should not happen if no error_msg
                                                                     response_text = "Could not calculate the result."

                                                        except Exception as agg_e:
                                                            response_text = f"Error performing aggregation '{agg_func_str}' on '{agg_col}': {agg_e}"
                                                            logging.error(f"Aggregation Error: {agg_e}", exc_info=True)
                                                    agg_handled = True

                                            # --- Fallback for Aggregation ---
                                            if not agg_handled:
                                                logging.warning(f"Could not parse aggregation details: '{details_text}'. Falling back to LLM prompt.")
                                                aggregation_prompt = f"""
                                                Context:
                                                The user is analyzing data with columns: {current_data.columns.tolist()}
                                                Sample Data (first 5 rows):
                                                {current_data.head().to_string()}

                                                User Question: '{prompt}'

                                                Based ONLY on the provided sample data and column names, try to answer the user's aggregation question. If you can calculate it from the sample, provide the answer. If the sample is insufficient or the calculation is complex, state that you need to perform a full data calculation and suggest how the user might ask for it (e.g., 'Calculate the average of `column_name`', or 'Count unique values in `column_name`'). Do not invent data.
                                                """
                                                response_text = content_agent.generate(aggregation_prompt)

                                        elif analysis_type == "General Insight":
                                             response_text = generate_data_report(current_data) # Reuse the report function

                                        else:
                                             response_text = f"I recognized the analysis type '{analysis_type}', but I don't know how to perform it yet."
                                # --- End Analysis Handlers ---

                                except Exception as e:
                                    response_text = f"Error during data analysis ('{analysis_type}'): {e}"
                                    logging.error(f"Data Analysis Error: {e}", exc_info=True)
                            else:
                                response_text = "I understood you want data analysis, but couldn't determine the specific type from the plan."

                    elif category == "Visualization Request":
                        if not data_available or current_data is None:
                            response_text = "No data is currently loaded for visualization. Please upload a file or select and analyze a database table first."
                        else:
                            # --- Use the Visualization Agent's generate_plot method ---
                            viz_prompt_match = re.search(r"Visualization Prompt:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                            viz_prompt = viz_prompt_match.group(1).strip() if viz_prompt_match else prompt # Use original prompt as fallback

                            try:
                                with st.spinner("Generating plot..."):
                                    # generate_plot internally uses LLM to decide plot type and columns
                                    image_path, plot_description = viz_agent.generate_plot(current_data, viz_prompt)

                                # Process result tuple
                                if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                                    # Success: We have a valid image path
                                    response_text = plot_description if plot_description else f"Generated plot based on your request."
                                elif plot_description: # Handle cases where image failed but we have a description/error
                                    response_text = f"Could not generate plot: {plot_description}"
                                    image_path = None # Ensure no broken image link
                                else: # Fallback if agent returned None for both path and description
                                    response_text = f"Failed to generate plot (unknown reason)."
                                    image_path = None

                            except Exception as e:
                                response_text = f"Error generating visualization: {e}"
                                logging.error(f"Viz Error: {e}", exc_info=True)
                                image_path = None
                            # --- End Visualization Agent Call ---

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
        st.session_state.chat_history.append({"role": "assistant", "content": response_text, "image": image_path})
        # Rerun to display the new messages and potential image
        st.rerun()

# --- Sidebar Reset Button ---
st.sidebar.markdown("---")
if st.sidebar.button("Reset & Select New Source"):
    # Clear relevant session state variables
    keys_to_reset = ['data', 'data_source', 'selected_table', 'uploaded_file_name',
                     'chat_history', 'operation_result', 'schema_info']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize defaults
    st.session_state.data = None
    st.session_state.chat_history = []
    st.session_state.data_source = None
    st.session_state.selected_table = None
    st.session_state.operation_result = None
    st.session_state.schema_info = None
    st.session_state.uploaded_file_name = None
    # Keep db_connected status as is, unless you want to force re-check on reset

    st.rerun()

# --- END OF MAIN APP LOGIC ---
