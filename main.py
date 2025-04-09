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

# Initialize agents
# Ensure DatabaseAgent is initialized correctly (assuming it connects)
try:
    db_agent = DatabaseAgent()
    # Try a simple query to check connection
    db_agent.query("SELECT 1")
    st.session_state.db_connected = True
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")
    st.stop() # Stop execution if DB connection fails

web_agent = WebScrapingAgent()
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
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, MemoryError):
        # If literal_eval fails, return the original string or handle appropriately
        # For large data strings, literal_eval might fail. Consider alternatives if needed.
        logging.warning(f"ast.literal_eval failed for value starting with: {val[:100]}")
        return val # Or potentially raise an error or return None

def get_db_data(table_name):
    """Fetches data and column names for a given table."""
    if not table_name:
        return None, "No table name provided."
    try:
        # Enclose table name in backticks for safety
        safe_table_name = f"`{table_name}`"

        # Get column names
        column_info_raw = db_agent.query(f"DESCRIBE {safe_table_name}")
        column_info = safe_literal_eval(column_info_raw)
        if not isinstance(column_info, list) or not all(isinstance(item, tuple) for item in column_info):
             raise ValueError(f"Unexpected format for DESCRIBE result: {column_info_raw}")
        columns = [col[0] for col in column_info]

        # Get data
        data_raw = db_agent.query(f"SELECT * FROM {safe_table_name}")
        data_list = safe_literal_eval(data_raw)
        if not isinstance(data_list, list):
             raise ValueError(f"Unexpected format for SELECT * result: {data_raw}")

        # Create DataFrame
        df = pd.DataFrame(data_list, columns=columns)
        return df, None
    except Exception as e:
        logging.error(f"Error fetching data for table {table_name}: {e}")
        return None, f"Error fetching data for table {table_name}: {e}"

def compute_statistics(data):
    """Compute statistical summary for numeric columns in the DataFrame."""
    if data is None or data.empty:
        return {}
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {}

    stats = {}
    for col in numeric_cols:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            # Convert Decimal to float for consistency if necessary
            stats[col] = {
                "count": int(col_data.count()),
                "mean": float(col_data.mean()) if isinstance(col_data.mean(), Decimal) else col_data.mean(),
                "median": float(col_data.median()) if isinstance(col_data.median(), Decimal) else col_data.median(),
                "std": float(col_data.std()) if isinstance(col_data.std(), Decimal) else col_data.std(),
                "min": float(col_data.min()) if isinstance(col_data.min(), Decimal) else col_data.min(),
                "max": float(col_data.max()) if isinstance(col_data.max(), Decimal) else col_data.max(),
                "q1": float(col_data.quantile(0.25)) if isinstance(col_data.quantile(0.25), Decimal) else col_data.quantile(0.25),
                "q3": float(col_data.quantile(0.75)) if isinstance(col_data.quantile(0.75), Decimal) else col_data.quantile(0.75)
            }
            # Handle potential NaN/Inf values after calculation
            for key, value in stats[col].items():
                if pd.isna(value) or np.isinf(value):
                    stats[col][key] = None # Or use a placeholder like "N/A"
                elif isinstance(value, float):
                     stats[col][key] = round(value, 4) # Round floats for display

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
        # Limit the number of unique values displayed for performance
        if data[col].nunique() > 100:
             top_values = data[col].value_counts().head(10).to_dict()
             note = " (Top 10 shown due to high cardinality)"
        else:
             top_values = data[col].value_counts().head(10).to_dict() # Show top 10 max
             note = ""

        categorical_insights[col] = {
            "top_values": [{"value": str(value), "count": count} for value, count in top_values.items()], # Ensure value is string
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
        stats_text = "No numeric columns found for statistical analysis."

    # Format categorical insights
    categorical_text = ""
    if categorical_insights:
        cat_text_parts = []
        for col, insights in categorical_insights.items():
             lines = [f"**{col} Top Values{insights.get('note', '')}:**"]
             lines.extend([f"- {item['value']}: {item['count']}" for item in insights['top_values']])
             cat_text_parts.append("\n".join(lines))
        categorical_text = "\n\n".join(cat_text_parts)
    else:
        categorical_text = "No categorical columns found for analysis."

    # Generate insight with LLM
    insight_prompt = (
        f"Based on the following analysis of the data:\n"
        f"**Numeric Statistics:**\n{stats_text}\n\n"
        f"**Categorical Insights:**\n{categorical_text}\n\n"
        f"**Sample Data (first 5 rows):**\n{sample_data}\n\n"
        f"Provide a concise report (2-3 paragraphs) summarizing the key findings. "
        f"Focus on potential patterns, data quality observations (like skewness, presence of outliers suggested by min/max/std dev, dominant categories), and possible implications. "
        f"Synthesize the information into actionable insights rather than just listing the stats."
    )
    try:
        insight = content_agent.generate(insight_prompt)
    except Exception as e:
        insight = f"Could not generate LLM insight: {e}"


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
        else:
             # No data and no table name
             return "No data available for this operation. Please upload a file or select a database table first."

    # --- Perform the actual operation ---
    try:
        if operation == "Clean Data":
            cleaned_data, cleaning_steps = data_cleaning_agent.clean_data(data)
            if cleaned_data is None:
                return f"Data cleaning failed: {cleaning_steps}" # Return the error message if cleaning failed.
            # Display summary and maybe head of cleaned data
            st.session_state.data = cleaned_data # Update session state with cleaned data
            st.success("Data cleaning applied successfully!")
            return f"**Cleaning Steps Performed:**\n```\n{cleaning_steps}\n```\n\n**Preview of Cleaned Data:**\n{cleaned_data.head().to_string()}"

        elif operation == "Visualize Data":
            visualization_results = viz_agent.generate_all(data) # Generate all standard plots
            result_string = "**Generated Visualizations:**\n"
            if not visualization_results:
                result_string += "Could not generate any visualizations for this data."
            else:
                for plot_type, image_path in visualization_results.items():
                    title = plot_type.replace('_', ' ').title()
                    if image_path and not image_path.startswith("No"):
                        st.image(image_path, caption=f"{title}", use_column_width=True)
                    else:
                        result_string += f"- {title}: {image_path}\n" # Report if a plot failed
            return result_string if len(visualization_results) == 0 else "" # Return string only if no images shown

        elif operation == "Detailed Analysis":
            analysis_agent = AnalysisAgent(data) # Pass the DataFrame
            buffer = io.StringIO()
            data.info(buf=buffer)
            data_info = buffer.getvalue()
            data_head = data.head().to_string()
            data_for_report = f"Data Info:\n{data_info}\n\nData Head:\n{data_head}"

            eda_results, insight = analysis_agent.analyze() # Analyze the DataFrame
            report = report_agent.generate_report(data_for_report, eda_results, insight)
            return f"**Detailed Analysis Report:**\n{report}"

        elif operation == "Web Research":
            # Generate a query based on data context
            research_topic_prompt = f"Based on the columns {data.columns.tolist()} and this data sample:\n{data.head().to_string()}\nSuggest a relevant web research topic (e.g., market trends for a product, competitor analysis based on categories)."
            research_topic = content_agent.generate(research_topic_prompt)
            st.info(f"Suggested Research Topic: {research_topic}")
            research_result = web_agent.research(research_topic)
            return f"**Web Research on '{research_topic}':**\n{research_result}"

        elif operation == "Ask a specific question":
            # This operation might be better handled by the chat interface directly.
            # Here, we can maybe generate a sample insightful question.
            question_prompt = f"Based on the columns {data.columns.tolist()} and this data sample:\n{data.head().to_string()}\nSuggest an insightful question that can be answered from this data (e.g., 'What is the average value of X for category Y?', 'Are there outliers in column Z?')."
            suggested_question = content_agent.generate(question_prompt)
            return f"**Suggestion:** Try asking a specific question in the chat below, for example: '{suggested_question}'"

        else:
            return f"Operation '{operation}' not implemented correctly."

    except Exception as e:
        logging.error(f"Error during operation '{operation}': {e}", exc_info=True)
        return f"An error occurred while performing '{operation}': {e}"


def format_database_response(raw_result, question, schema_info):
    """Convert raw database query results into detailed insights with statistical analysis."""
    logging.info(f"Formatting DB response for question: {question}")
    logging.debug(f"Raw result: {raw_result}")

    if raw_result is None:
        return "Query executed, but returned no result."
    if isinstance(raw_result, str) and raw_result.startswith("Error"):
        return raw_result # Return DB errors directly

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
                        last_table = f"'{items[-1]}'"
                        return f"The database contains the following tables: {table_list} and {last_table}."
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
                    # For simplicity, just display the raw list for now
                    # A more advanced approach would parse the SQL query for column names
                    # or pass column names alongside the result.
                    df_result = pd.DataFrame(cleaned_result) # Let pandas infer headers
                    st.dataframe(df_result)
                    return f"Query returned {len(cleaned_result)} row(s). Displaying above."
                except Exception:
                    # Fallback to string representation if DataFrame fails
                    return f"Query Result:\n{str(cleaned_result)}"


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
    data_source = st.radio("Choose your data source:", ["Database", "Upload File"], key="data_source_radio")
    if st.button("Confirm Data Source"):
        st.session_state.data_source = data_source
        st.session_state.selected_table = None # Reset selections when changing source
        st.session_state.data = None
        st.session_state.chat_history = []
        st.session_state.operation_result = None
        st.rerun()

# Database Flow
elif st.session_state.data_source == "Database":
    st.subheader("2. Database Interaction")
    if st.session_state.selected_table is None:
        st.markdown("Select a table to analyze.")
        try:
            tables_result = db_agent.query("SHOW TABLES")
            tables_list = safe_literal_eval(tables_result)
            if isinstance(tables_list, list) and all(isinstance(t, tuple) for t in tables_list):
                tables = [table[0] for table in tables_list]
                if not tables:
                    st.warning("No tables found in the database.")
                else:
                    selected_table = st.selectbox("Available Tables:", tables, key="db_table_select")
                    if st.button("Analyze Table"):
                        st.session_state.selected_table = selected_table
                        st.session_state.data = None # Clear any previous file data
                        st.session_state.chat_history = [] # Reset chat
                        st.session_state.operation_result = None
                        # Fetch initial data and schema for the selected table
                        df, error = get_db_data(selected_table)
                        if error:
                            st.error(error)
                            st.session_state.selected_table = None # Reset if fetch fails
                        else:
                            st.session_state.data = df # Store fetched data
                            st.session_state.schema_info = db_agent.db.get_table_info([selected_table]) # Get schema for this table
                        st.rerun()
            else:
                st.error(f"Could not parse table list: {tables_result}")

        except Exception as e:
            st.error(f"Error fetching tables: {e}")
            logging.error(f"Error fetching tables: {e}", exc_info=True)
    else:
        # Table is selected, show analysis and options
        st.success(f"Analyzing Table: **{st.session_state.selected_table}**")
        st.markdown("### Initial Data Analysis")

        # Display report if data is loaded
        if st.session_state.data is not None and not st.session_state.data.empty:
            st.dataframe(st.session_state.data.head())
            report = generate_data_report(st.session_state.data)
            st.markdown(report)
        elif st.session_state.data is not None and st.session_state.data.empty:
             st.warning(f"Table '{st.session_state.selected_table}' appears to be empty.")
        else:
             st.warning("Could not load data for the report. Try selecting the table again.")


        st.subheader("3. Perform Operations (Optional)")
        operations = st.selectbox("Select an operation:", ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"], key="db_operation_select")
        if st.button("Run Operation"):
            if operations != "None":
                with st.spinner(f"Running {operations}..."):
                    # Data should already be in st.session_state.data from the "Analyze Table" step
                    # Pass the table name for context if needed by the operation
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
            if isinstance(st.session_state.operation_result, str):
                 st.markdown(st.session_state.operation_result)
            else:
                 st.write(st.session_state.operation_result)
            # Clear the result after displaying
            # st.session_state.operation_result = None # Keep result until next action


# File Upload Flow
elif st.session_state.data_source == "Upload File":
    st.subheader("2. Upload and Analyze File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    if uploaded_file is not None:
        if st.session_state.data is None or getattr(st.session_state, 'uploaded_file_name', '') != uploaded_file.name:
             try:
                 st.session_state.data = pd.read_csv(uploaded_file)
                 st.session_state.uploaded_file_name = uploaded_file.name
                 st.session_state.selected_table = None # Clear DB selection
                 st.session_state.chat_history = [] # Reset chat
                 st.session_state.operation_result = None
                 st.success(f"File '{uploaded_file.name}' uploaded and loaded successfully.")
                 # Save the uploaded file (optional)
                 # upload_path = os.path.join("data/uploads", uploaded_file.name)
                 # with open(upload_path, "wb") as f:
                 #     f.write(uploaded_file.getbuffer())

                 # Automatically display analysis after upload
                 st.markdown("### Initial Data Analysis")
                 st.dataframe(st.session_state.data.head())
                 report = generate_data_report(st.session_state.data)
                 st.markdown(report)

             except Exception as e:
                 st.error(f"Error reading CSV file: {e}")
                 st.session_state.data = None
                 st.session_state.uploaded_file_name = None

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
                if isinstance(st.session_state.operation_result, str):
                    st.markdown(st.session_state.operation_result)
                else:
                    st.write(st.session_state.operation_result)
                # st.session_state.operation_result = None # Keep result

    else:
        # Reset if file is removed
        if st.session_state.data is not None:
            st.session_state.data = None
            st.session_state.uploaded_file_name = None
            st.session_state.chat_history = []
            st.session_state.operation_result = None
            st.info("Upload a CSV file to begin analysis.")


# --- Chat Interface (Common for both DB and File flows once data/table is active) ---
if st.session_state.data is not None or st.session_state.selected_table is not None:
    st.subheader("4. Chat with your Data")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Handle images stored in history
            if message.get("image"):
                if isinstance(message["image"], str) and not message["image"].startswith("No"):
                    try:
                        st.image(message["image"], use_column_width=True)
                    except Exception as img_e:
                        st.warning(f"Could not display image {message['image']}: {img_e}")
                elif isinstance(message["image"], str): # Handle "No plot generated" messages
                    st.markdown(f"_{message['image']}_")


    prompt = st.chat_input("Ask a question about the data or request actions...")

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the prompt
        response = ""
        image_path = None # For visualization results

        with st.spinner("Thinking..."):
            try:
                # Determine context: DB table or uploaded file
                data_context = ""
                current_data = st.session_state.get('data') # Get current DataFrame if available
                current_table = st.session_state.get('selected_table')

                if current_data is not None:
                    columns = current_data.columns.tolist()
                    data_context = f"The user is currently working with data loaded from a file. The columns are: {columns}."
                    # Provide sample data for context
                    sample_data_head = current_data.head().to_string()
                    data_context += f"\nSample Data:\n{sample_data_head}"
                elif current_table:
                    schema_info_str = str(st.session_state.get('schema_info', 'Not available'))
                    data_context = f"The user is currently working with the database table '{current_table}'. The database is MySQL named 'insightforge_db'. Schema info: {schema_info_str}."
                    # Fetch sample data for context if not already loaded
                    if current_data is None:
                         df_sample, _ = get_db_data(current_table)
                         if df_sample is not None:
                             sample_data_head = df_sample.head().to_string()
                             data_context += f"\nSample Data:\n{sample_data_head}"
                             current_data = df_sample # Use this fetched data if needed below
                         else:
                             data_context += "\nCould not fetch sample data for the table."
                    else:
                         sample_data_head = current_data.head().to_string()
                         data_context += f"\nSample Data:\n{sample_data_head}"

                else:
                    response = "No data source (file or database table) is currently active. Please select one first."
                    st.session_state.chat_history.append({"role": "assistant", "content": response, "image": None})
                    st.rerun()


                # Improved LLM Prompt for Planning
                planning_prompt = f"""
                Analyze the user's request: '{prompt}'

                Context:
                {data_context}

                Determine the user's intent and classify it into ONE category:
                1. Database Query: User asks about DB structure (tables, schema) or data within a specific DB table.
                2. Data Analysis (File/Table): User asks a question about the loaded data (from file or selected table), like counts, averages, missing values, correlations, or requests a summary/insight.
                3. Visualization Request: User asks for a plot, graph, or chart of the loaded data.
                4. Web Research: User asks for external information (trends, definitions, etc.).
                5. General Chit-chat/Unclear: The request is conversational or doesn't fit other categories.

                Based on the category, provide a plan:

                - If Database Query:
                    - If asking about tables: Plan: SQL Query: SHOW TABLES
                    - If asking about schema of '{current_table}': Plan: SQL Query: DESCRIBE `{current_table}`
                    - If asking for data from '{current_table}' (e.g., 'show first 10 rows', 'count rows', 'find where column X is Y'): Plan: SQL Query: [Generate appropriate SELECT query for `{current_table}`]
                    - If asking about a DIFFERENT table: Plan: SQL Query: [Generate query for the mentioned table, e.g., `DESCRIBE \`other_table\`` or `SELECT * FROM \`other_table\` LIMIT 5`]
                    - If asking about the database name: Plan: DB Name Request: True

                - If Data Analysis (File/Table): (Requires loaded data: {'Yes' if current_data is not None else 'No'})
                    - If asking for missing values: Plan: Analysis Type: Missing Values
                    - If asking for summary statistics: Plan: Analysis Type: Summary Stats
                    - If asking for counts/averages/sums/min/max for specific columns/conditions: Plan: Analysis Type: Aggregation\nDetails: [Describe the aggregation needed, e.g., 'Average of ColA where ColB > 10']
                    - If asking for correlation: Plan: Analysis Type: Correlation
                    - If asking for general insights: Plan: Analysis Type: General Insight

                - If Visualization Request: (Requires loaded data: {'Yes' if current_data is not None else 'No'})
                    - Plan: Plot Type: [Specify plot: scatter, histogram, boxplot, heatmap, bar]\nColumns: [Specify columns if mentioned, e.g., X=colA, Y=colB, Target=colC]

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
                    response = f"I had trouble understanding the structure of the plan. Could you rephrase?\nDebug Info:\n{llm_response}"
                else:
                    category = category_match.group(1).strip()
                    plan = plan_match.group(1).strip()
                    logging.info(f"Parsed Category: {category}, Parsed Plan: {plan}")

                    # --- Execute Plan ---
                    if category == "Database Query":
                        db_name_match = re.search(r"DB Name Request:\s*True", plan)
                        sql_query_match = re.search(r"SQL Query:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                        if db_name_match:
                            response = "I am connected to the 'insightforge_db' database."
                        elif sql_query_match:
                            sql_query = sql_query_match.group(1).strip()
                            # Basic safety: Check for common harmful patterns (can be improved)
                            if re.search(r"(DROP|DELETE|UPDATE|INSERT|ALTER)\s+", sql_query, re.IGNORECASE):
                                 response = "Sorry, I cannot execute queries that modify data (like DROP, DELETE, UPDATE, INSERT, ALTER)."
                            else:
                                try:
                                    raw_result = db_agent.query(sql_query)
                                    response = format_database_response(raw_result, prompt, st.session_state.get('schema_info'))
                                except Exception as e:
                                    response = f"Error executing database query: {str(e)}"
                                    logging.error(f"DB Query Error: {e}", exc_info=True)
                        else:
                            response = "I understood you want a database query, but I couldn't extract a specific SQL command from the plan."

                    elif category == "Data Analysis (File/Table)":
                        if current_data is None:
                            response = "No data is currently loaded. Please upload a file or select and analyze a database table first."
                        else:
                            analysis_type_match = re.search(r"Analysis Type:\s*(.+?)(?:\n|$)", plan)
                            if analysis_type_match:
                                analysis_type = analysis_type_match.group(1).strip()
                                try:
                                    if analysis_type == "Missing Values":
                                        missing_counts = current_data.isnull().sum()
                                        missing_filtered = missing_counts[missing_counts > 0]
                                        if missing_filtered.empty:
                                            response = "There are no missing values in the current dataset."
                                        else:
                                            response = "Missing value counts per column:\n" + missing_filtered.to_string()
                                    elif analysis_type == "Summary Stats":
                                        response = "Summary Statistics:\n" + current_data.describe(include='all').to_string()
                                    elif analysis_type == "Correlation":
                                        numeric_df = current_data.select_dtypes(include=np.number)
                                        if numeric_df.shape[1] < 2:
                                             response = "Need at least two numeric columns for correlation analysis."
                                        else:
                                             corr_matrix = numeric_df.corr()
                                             response = "Correlation Matrix:\n" + corr_matrix.to_string()
                                             # Optionally generate heatmap image
                                             image_path = viz_agent.generate_correlation_heatmap(current_data) # Use original data
                                             if image_path and not image_path.startswith("No"):
                                                 response += "\n\nDisplaying heatmap below."
                                             else:
                                                 response += f"\n\nCould not generate heatmap: {image_path}"

                                    elif analysis_type == "Aggregation":
                                         # This requires more complex parsing of 'Details' - using LLM again might be needed
                                         # For now, provide a generic response or ask for clarification
                                         response = f"I understand you want an aggregation: {plan}. Can you please ask using a more direct question like 'What is the average of X?' or 'Count rows where Y is Z'?"
                                         # TODO: Implement more robust aggregation handling if needed

                                    elif analysis_type == "General Insight":
                                         response = generate_data_report(current_data) # Reuse the report function

                                    else:
                                         response = f"I recognized the analysis type '{analysis_type}', but I don't know how to perform it yet."

                                except Exception as e:
                                    response = f"Error during data analysis: {e}"
                                    logging.error(f"Data Analysis Error: {e}", exc_info=True)
                            else:
                                response = "I understood you want data analysis, but couldn't determine the specific type from the plan."

                    elif category == "Visualization Request":
                        if current_data is None:
                            response = "No data is currently loaded for visualization. Please upload a file or select and analyze a database table first."
                        else:
                            plot_type_match = re.search(r"Plot Type:\s*(.+?)(?:\n|$)", plan)
                            columns_match = re.search(r"Columns:\s*(.+?)(?:\n|$)", plan)

                            if plot_type_match:
                                plot_type = plot_type_match.group(1).strip().lower()
                                cols_text = columns_match.group(1).strip() if columns_match else ""
                                # Basic column extraction (can be improved)
                                x_col, y_col, target_col = None, None, None
                                if 'x=' in cols_text.lower(): x_col = re.search(r"X=([\w\s]+)(?:,|$)", cols_text, re.IGNORECASE).group(1).strip()
                                if 'y=' in cols_text.lower(): y_col = re.search(r"Y=([\w\s]+)(?:,|$)", cols_text, re.IGNORECASE).group(1).strip()
                                if 'target=' in cols_text.lower(): target_col = re.search(r"Target=([\w\s]+)(?:,|$)", cols_text, re.IGNORECASE).group(1).strip()
                                # If only one column mentioned for histogram/box, assume it's the target
                                if not target_col and not x_col and not y_col and cols_text and plot_type in ['histogram', 'boxplot', 'distribution']:
                                    target_col = cols_text

                                try:
                                    if plot_type == "scatter":
                                        image_path = viz_agent.generate_scatter(current_data, x_col=x_col, y_col=y_col)
                                        response = f"Generating scatter plot{' of ' + x_col + ' vs ' + y_col if x_col and y_col else ''}..."
                                    elif plot_type in ["histogram", "distribution"]:
                                        image_path = viz_agent.generate_distribution(current_data, hist_col=target_col) # Use target_col
                                        response = f"Generating distribution plot{' for ' + target_col if target_col else ''}..."
                                    elif plot_type == "boxplot":
                                        image_path = viz_agent.generate_boxplot(current_data, box_col=target_col) # Use target_col
                                        response = f"Generating boxplot{' for ' + target_col if target_col else ''}..."
                                    elif plot_type == "heatmap":
                                        image_path = viz_agent.generate_correlation_heatmap(current_data)
                                        response = "Generating correlation heatmap..."
                                    elif plot_type == "bar":
                                        # Bar plots need more specific logic (e.g., counts of a category)
                                        # image_path = viz_agent.generate_bar(current_data, ...) # Needs implementation
                                        response = "Bar plots require specifying the categorical column. For example: 'bar plot of column Category'."
                                        # Placeholder - needs implementation in VisualizationAgent and logic here
                                    else:
                                        response = f"Sorry, I don't know how to generate a '{plot_type}' plot yet. Try scatter, histogram, boxplot, or heatmap."

                                    if image_path and image_path.startswith("No"):
                                        response = f"Could not generate {plot_type} plot: {image_path}" # Report failure reason
                                        image_path = None # Don't store failure message as image path
                                    elif not image_path:
                                         response = f"Failed to generate {plot_type} plot (no path returned)."


                                except Exception as e:
                                    response = f"Error generating visualization: {e}"
                                    logging.error(f"Viz Error: {e}", exc_info=True)
                                    image_path = None
                            else:
                                response = "I understood you want a visualization, but couldn't determine the plot type or columns from the plan."

                    elif category == "Web Research":
                        topic_match = re.search(r"Topic:\s*(.*)", plan, re.IGNORECASE | re.DOTALL)
                        if topic_match:
                            topic = topic_match.group(1).strip()
                            response = web_agent.research(topic)
                        else:
                            response = "I understood you want web research, but I couldn't extract the topic from the plan."

                    elif category == "General Chit-chat/Unclear":
                         # Use LLM for a general response
                         response = content_agent.generate(f"Respond conversationally to the user's message: '{prompt}'")

                    else:
                        response = f"I couldn't categorize your request based on the plan ('{category}'). Please try rephrasing."

            except Exception as e:
                response = f"An unexpected error occurred while processing your request: {str(e)}"
                logging.error(f"Chat Processing Error: {e}", exc_info=True)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response, "image": image_path})
        # Rerun to display the new messages and potential image
        st.rerun()