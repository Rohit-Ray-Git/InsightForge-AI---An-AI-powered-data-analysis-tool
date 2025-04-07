import streamlit as st
import pandas as pd
import os
import re
from decimal import Decimal
import ast
import numpy as np
import json
from agents.database_agent import DatabaseAgent
from agents.web_scraping_agent import WebScrapingAgent
from agents.visualization_agent import VisualizationAgent
from agents.content_generation_agent import ContentGenerationAgent

# Set page config
st.set_page_config(page_title="InsightForge AI - Advanced Data Analysis Tool", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize agents
db_agent = DatabaseAgent()
web_agent = WebScrapingAgent()
viz_agent = VisualizationAgent(output_dir="data/reports")
content_agent = ContentGenerationAgent()

# Create directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

# Title and description
st.title("InsightForge AI - Advanced Data Analysis Tool")
st.markdown("Unlock deep insights from your data with comprehensive statistical analysis and visualizations.")

# File upload section
st.subheader("Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.success(f"File '{uploaded_file.name}' uploaded successfully.")
    # Save the uploaded file
    upload_path = os.path.join("data/uploads", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Display the first few rows
    st.subheader("Data Preview")
    st.dataframe(st.session_state.data.head())

# Note about database and data queries
st.markdown(
    """
    **Note**: You can ask questions about fetching data from the ‘insightforge_db’ database or uploaded data.  
    Examples:  
    - "What tables are in the database?"  
    - "How many tables?"  
    - "Give me a detailed analysis of layoffs table"  
    - "Show me a histogram of layoffs table with column salary"  
    - "Show me a scatter plot of price vs income from uploaded data"
    """
)

# Chat interface
st.subheader("Ask a Question")
prompt = st.chat_input("Type your question here...")

def compute_statistics(table_name, schema_info):
    """Compute statistical summary for numeric columns in the table."""
    # Parse schema_info if it's a string
    if isinstance(schema_info, str):
        try:
            schema_info = json.loads(schema_info) if schema_info.strip().startswith('{') else ast.literal_eval(schema_info)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            print(f"Debug: Failed to parse schema_info as dictionary: {schema_info}")
            return None
    
    numeric_cols = [col for col in schema_info.get(table_name, {}).get("columns", {}) 
                   if schema_info.get(table_name, {}).get("columns", {}).get(col, {}).get("type").startswith(("INT", "DECIMAL", "FLOAT", "DOUBLE"))]
    if not numeric_cols:
        return None
    
    stats = {}
    query = f"SELECT {', '.join(numeric_cols)} FROM {table_name} LIMIT 1000"  # Limit to 1000 rows for performance
    result = db_agent.query(query)
    try:
        data = ast.literal_eval(result) if isinstance(result, str) else result
        if data and isinstance(data, list):
            df = pd.DataFrame(data, columns=numeric_cols)
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
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
    except Exception as e:
        print(f"Debug: Error computing statistics: {e}")
        return None
    return stats

def format_database_response(raw_result, question, schema_info):
    """Convert raw database query results into detailed insights with statistical analysis."""
    if not raw_result or "Error" in raw_result:
        return raw_result
    
    try:
        # Debug: Print the raw result and schema_info to inspect their format
        print(f"Debug: Raw result = {raw_result}")
        print(f"Debug: Schema info = {schema_info}")
        
        # Attempt to parse the raw result safely
        try:
            cleaned_result = ast.literal_eval(raw_result) if isinstance(raw_result, str) else raw_result
        except (ValueError, SyntaxError) as e:
            print(f"Debug: ast.literal_eval failed with error: {e}")
            try:
                cleaned_result = eval(raw_result)
            except Exception as eval_e:
                return f"Sorry, I couldn’t understand the result: {str(eval_e)}. Please try rephrasing your question or check the table schema."

        print(f"Debug: Cleaned result = {cleaned_result}")
        
        if isinstance(cleaned_result, list):
            if len(cleaned_result) == 0:
                return "There are no results to show."
            # Handle count query
            if "how many" in question.lower() and "table" in question.lower():
                if len(cleaned_result) == 1 and isinstance(cleaned_result[0], tuple) and len(cleaned_result[0]) == 1 and isinstance(cleaned_result[0][0], (int, float, Decimal)):
                    count = float(cleaned_result[0][0]) if isinstance(cleaned_result[0][0], Decimal) else cleaned_result[0][0]
                    return f"There are {count} tables in the database."
            # Handle table name listing
            elif all(isinstance(item, tuple) and len(item) == 1 for item in cleaned_result):
                items = [item[0] for item in cleaned_result]
                if "table" in question.lower() or "name" in question.lower():
                    if len(items) == 1:
                        return f"The database has one table: '{items[0]}'."
                    else:
                        table_list = ", ".join(f"'{item}'" for item in items[:-1])
                        last_table = f"'{items[-1]}'"
                        return f"The database has these tables: {table_list} and {last_table}."
            # Handle detailed analysis for variations like "give me an insight" or "detailed analysis"
            elif all(isinstance(item, tuple) and len(item) > 1 for item in cleaned_result) and ("give me an insight" in question.lower() or "detailed analysis" in question.lower() or "insight about" in question.lower()):
                if cleaned_result and isinstance(cleaned_result[0], tuple):
                    insight_dict = dict(zip(["total_rows", "average_value"], cleaned_result[0]))
                    total_rows = float(insight_dict.get("total_rows", "unknown")) if isinstance(insight_dict.get("total_rows"), Decimal) else insight_dict.get("total_rows")
                    avg_value = float(insight_dict.get("average_value", "unknown")) if isinstance(insight_dict.get("average_value"), Decimal) else insight_dict.get("average_value")

                    # Extract table name
                    table_name = None
                    table_match = re.search(r"(?:insight about|insight of|detailed analysis of)\s+([a-zA-Z_]+)\s+table", question.lower())
                    if table_match:
                        table_name = table_match.group(1)
                    else:
                        table_match = re.search(r"FROM\s+([a-zA-Z_]+)", question, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else None

                    sample_text = "no sample data available"
                    if table_name:
                        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                        sample_result = db_agent.query(sample_query)
                        try:
                            sample_data = ast.literal_eval(sample_result) if isinstance(sample_result, str) else sample_result
                        except (ValueError, SyntaxError):
                            sample_data = eval(sample_result) if isinstance(sample_result, str) else sample_result
                        sample_text = str(sample_data) if sample_data else "no sample data available"

                    # Compute detailed statistics
                    stats = compute_statistics(table_name, schema_info)
                    stats_text = "No statistical data available due to missing numeric columns." if not stats else "\n".join(
                        [f"**{col} Statistics:**\n"
                         f"- Count: {stat['count']}\n"
                         f"- Mean: {stat['mean']:.2f}\n"
                         f"- Median: {stat['median']:.2f}\n"
                         f"- Standard Deviation: {stat['std']:.2f}\n"
                         f"- Minimum: {stat['min']:.2f}\n"
                         f"- Maximum: {stat['max']:.2f}\n"
                         f"- 1st Quartile (Q1): {stat['q1']:.2f}\n"
                         f"- 3rd Quartile (Q3): {stat['q3']:.2f}"
                         for col, stat in stats.items()]
                    )

                    # Generate insight with LLM
                    insight_prompt = (
                        f"Based on the following statistics about the '{table_name}' table: 'This table has about {total_rows} entries. "
                        f"The average value of a key numeric column is around {avg_value}. Detailed statistics: {stats_text}.' "
                        f"And a sample of the data: '{sample_text}', provide a detailed paragraph (4-5 sentences) of insights in layman's terms. "
                        f"Include observations about trends, typical values, variability, and potential business implications or uses, without reproducing the raw data."
                    )
                    insight_response = content_agent.generate(insight_prompt)
                    insight_match = re.search(r"Plan:\s*(.+)", insight_response, re.DOTALL)
                    insight = insight_match.group(1).strip() if insight_match else "I couldn’t generate a detailed insight about this data."

                    return f"### Analysis of '{table_name}' Table\n" \
                           f"- **Total Entries**: {total_rows}\n" \
                           f"- **Average Value of a Numeric Column**: {avg_value:.2f}\n" \
                           f"{stats_text}\n" \
                           f"**Insight**: {insight}"
            return str(cleaned_result)
        elif isinstance(cleaned_result, tuple) and len(cleaned_result) == 1 and isinstance(cleaned_result[0], (int, float, Decimal)):
            count = float(cleaned_result[0]) if isinstance(cleaned_result[0], Decimal) else cleaned_result[0]
            if "how many" in question.lower() and "table" in question.lower():
                return f"There are {count} tables in the database."
            return str(count)
        elif isinstance(cleaned_result, str) and "Database Name" in cleaned_result:
            return "I am using a database called 'insightforge_db'."
        return str(cleaned_result)
    except Exception as e:
        return f"Sorry, I couldn’t understand the result: {str(e)}. Please try rephrasing your question or check the table schema."

if prompt:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Process the prompt
    response = ""
    image_path = None
    
    try:
        # Use ContentGenerationAgent to classify the prompt and generate a plan
        schema_info = db_agent.db.get_table_info()
        # Debug: Check and parse schema_info
        print(f"Debug: Raw schema_info = {schema_info}")
        if isinstance(schema_info, str):
            try:
                schema_info = json.loads(schema_info) if schema_info.strip().startswith('{') else ast.literal_eval(schema_info)
                print(f"Debug: Parsed schema_info = {schema_info}")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Debug: Failed to parse schema_info: {e}")
                schema_info = {}
        
        columns = st.session_state.data.columns.tolist() if st.session_state.data is not None else []
        prompt_analysis = content_agent.generate(
            f"""
            Analyze the following user prompt: '{prompt}'.
            The user has access to a MySQL database named 'insightforge_db' with the following schema:
            {schema_info}
            The user has also uploaded a CSV file with the following columns (if any): {columns}.
            
            Determine the user's intent and classify the prompt into one of the following categories:
            1. Database query (e.g., asking about tables, data in the database, or database metadata like its name)
            2. Visualization request (e.g., asking for a plot, graph, or chart)
            3. Web research (e.g., asking about trends, market analysis)
            
            Then, provide a plan to fulfill the request:
            - For database queries: If asking about the database name, return: Database Name: insightforge_db
              If asking about tables, return: SQL Query: SHOW TABLES
              If asking how many tables, return: SQL Query: SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'insightforge_db'
              If asking for a detailed analysis of a table (e.g., 'give me an insight about' or 'detailed analysis of'), return: SQL Query: SELECT COUNT(*) as total_rows, AVG([numeric_column]) as average_value FROM [table_name] WHERE [numeric_column] IS NOT NULL
              For other data queries, return: SQL Query: [your query]
            - For visualization requests: Specify the type of plot (scatter, histogram, heatmap, boxplot) and the columns to use (if mentioned) using the format: Plot Type: [type]\nX Column: [x_col]\nY Column: [y_col]\nHistogram Column: [hist_col]\nBoxplot Column: [box_col]\nTable Name: [table_name]
            - For web research: Extract the topic to research using the format: Topic: [topic]
            
            Return your response in the following format:
            Category: [category]
            Plan: [your plan]
            """
        )
        
        # Debug: Print the prompt analysis to understand what the LLM is returning
        print(f"Debug: Prompt Analysis = {prompt_analysis}")
        
        # Parse the LLM's response with more robust regular expressions
        category_match = re.search(r"Category:\s*(.+?)(?:\n|$)", prompt_analysis)
        plan_match = re.search(r"Plan:\s*(.+)", prompt_analysis, re.DOTALL)
        
        if not category_match or not plan_match:
            response = f"I couldn't understand your request. Could you please rephrase it?\n\nDebug Info:\n{prompt_analysis}"
        else:
            category = re.sub(r"^\d+\.\s*", "", category_match.group(1).strip())
            plan = plan_match.group(1).strip()
            
            # Extract table_name from plan if available
            table_name = None
            table_match = re.search(r"Table Name:\s*([a-zA-Z_]+)", plan, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
            else:
                # Fallback to extract from question
                table_match = re.search(r"(?:insight about|insight of|detailed analysis of)\s+([a-zA-Z_]+)\s+table", prompt.lower())
                table_name = table_match.group(1) if table_match else None

            if category == "Database query":
                db_name_match = re.search(r"Database Name:\s*(.+?)(?:\n|$)", plan)
                if db_name_match:
                    response = "I am using a database called 'insightforge_db'."
                else:
                    sql_query_match = re.search(r"SQL Query:\s*(.*?)(?:\n|$)", plan)
                    if sql_query_match:
                        sql_query = sql_query_match.group(1).strip()
                        # Replace [numeric_column] with a likely numeric column from schema
                        if "[numeric_column]" in sql_query and table_name:
                            if isinstance(schema_info, dict):
                                numeric_cols = [col for col in schema_info.get(table_name, {}).get("columns", {}) 
                                               if schema_info.get(table_name, {}).get("columns", {}).get(col, {}).get("type").startswith(("INT", "DECIMAL", "FLOAT", "DOUBLE"))]
                                print(f"Debug: Numeric columns detected = {numeric_cols}")
                                if numeric_cols:
                                    sql_query = sql_query.replace("[numeric_column]", numeric_cols[0])
                                else:
                                    response = "No numeric column found in the table. Please specify a numeric column (e.g., 'Detailed analysis of layoffs table with column salary')."
                            else:
                                response = "Schema information is unavailable. Please check the database connection."
                        try:
                            raw_result = db_agent.query(sql_query)
                            response = format_database_response(raw_result, prompt, schema_info)
                        except Exception as e:
                            response = f"Error querying database: {str(e)}. Please check the table schema or specify a valid numeric column."
                    else:
                        response = "I understood you want a database query, but I couldn't generate a valid SQL query. Please rephrase your question."
            
            elif category == "Visualization request":
                if st.session_state.data is None and not table_name:
                    response = "Please upload a data file or specify a database table first to proceed with visualization requests."
                else:
                    plot_type_match = re.search(r"Plot Type:\s*(.*?)(?:\n|$)", plan)
                    x_col_match = re.search(r"X Column:\s*(.*?)(?:\n|$)", plan)
                    y_col_match = re.search(r"Y Column:\s*(.*?)(?:\n|$)", plan)
                    hist_col_match = re.search(r"Histogram Column:\s*(.*?)(?:\n|$)", plan)
                    box_col_match = re.search(r"Boxplot Column:\s*(.*?)(?:\n|$)", plan)
                    
                    plot_type = plot_type_match.group(1).strip() if plot_type_match else None
                    x_col = x_col_match.group(1).strip() if x_col_match else None
                    y_col = y_col_match.group(1).strip() if y_col_match else None
                    hist_col = hist_col_match.group(1).strip() if hist_col_match else None
                    box_col = box_col_match.group(1).strip() if box_col_match else None
                    
                    data = st.session_state.data if st.session_state.data is not None else None
                    if plot_type and table_name and not data:
                        query = f"SELECT {x_col if x_col else '*'}, {y_col if y_col else '*'}, {hist_col if hist_col else '*'}, {box_col if box_col else '*'} FROM {table_name} LIMIT 1000"
                        result = db_agent.query(query)
                        try:
                            data = pd.DataFrame(ast.literal_eval(result) if isinstance(result, str) else result, columns=[x_col, y_col, hist_col, box_col] if any([x_col, y_col, hist_col, box_col]) else None)
                        except Exception as e:
                            response = f"Error fetching data for visualization: {str(e)}"

                    if plot_type and data is not None:
                        if "scatter" in plot_type.lower():
                            image_path = viz_agent.generate_scatter(data, x_col=x_col, y_col=y_col)
                            response = f"Here is the scatter plot{' of ' + x_col + ' vs ' + y_col if x_col and y_col else ''}:"
                        elif "histogram" in plot_type.lower() or "distribution" in plot_type.lower():
                            image_path = viz_agent.generate_distribution(data, hist_col=hist_col)
                            response = f"Here is the distribution plot{' of ' + hist_col if hist_col else ''}:"
                        elif "heatmap" in plot_type.lower():
                            image_path = viz_agent.generate_correlation_heatmap(data)
                            response = "Here is the correlation heatmap of the features:"
                        elif "boxplot" in plot_type.lower():
                            image_path = viz_agent.generate_boxplot(data, box_col=box_col)
                            response = f"Here is the boxplot{' of ' + box_col if box_col else ''}:"
                        else:
                            response = "I understood you want a visualization, but I couldn't determine the plot type. Please specify (e.g., 'scatter', 'histogram', 'heatmap', 'boxplot')."
                    else:
                        response = "I understood you want a visualization, but no data is available. Please upload a file or specify a table."
            
            elif category == "Web research":
                topic_match = re.search(r"Topic:\s*(.*?)(?:\n|$)", plan)
                if topic_match:
                    topic = topic_match.group(1).strip()
                    response = web_agent.research(topic)
                else:
                    response = "I understood you want to research a topic, but I couldn't determine the topic. Please rephrase your question."
            
            else:
                response = "I couldn't categorize your request. Please try rephrasing it."
    
    except Exception as e:
        response = f"Error processing your request: {str(e)}"
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response, "image": image_path})

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            if not message["image"].startswith("No"):
                st.image(message["image"], caption=message["content"], use_column_width=True)
            else:
                st.markdown(message["image"])

# Ensure the script ends here, no stray return statements