import streamlit as st
import pandas as pd
import os
import re
from decimal import Decimal
import ast
import numpy as np
import json
import logging
from agents.database_agent import DatabaseAgent
from agents.web_scraping_agent import WebScrapingAgent
from agents.visualization_agent import VisualizationAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.report_generation_agent import ReportGenerationAgent
from agents.analysis_agent import AnalysisAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(page_title="InsightForge AI - Advanced Data Analysis Tool", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'selected_table' not in st.session_state:
    st.session_state.selected_table = None
if 'operation_result' not in st.session_state:
    st.session_state.operation_result = None

# Initialize agents
db_agent = DatabaseAgent()
web_agent = WebScrapingAgent()
viz_agent = VisualizationAgent(output_dir="data/reports")
content_agent = ContentGenerationAgent()
report_agent = ReportGenerationAgent()

# Create directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

# Title and description
st.title("InsightForge AI - Advanced Data Analysis Tool")
st.markdown("Unlock deep insights from your data with comprehensive statistical analysis and visualizations.")

def compute_statistics(data):
    """Compute statistical summary for numeric columns in the DataFrame."""
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {}

    stats = {}
    for col in numeric_cols:
        col_data = data[col].dropna()
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
    return stats

def analyze_categorical_columns(data):
    """Analyze categorical columns and provide insights."""
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        return {}

    categorical_insights = {}
    for col in categorical_cols:
        top_values = data[col].value_counts().head(5).to_dict()
        categorical_insights[col] = {
            "top_values": [{"value": value, "count": count} for value, count in top_values.items()]
        }
    return categorical_insights

def generate_data_report(data):
    """Generate a detailed report for the uploaded data."""
    numeric_stats = compute_statistics(data)
    categorical_insights = analyze_categorical_columns(data)
    sample_data = data.head().to_string()

    # Format numeric statistics
    stats_text = ""
    if numeric_stats:
        stats_text = "\n".join(
            [f"**{col} Statistics:**\n"
             f"- Count: {stat['count']}\n"
             f"- Mean: {stat['mean']:.2f}\n"
             f"- Median: {stat['median']:.2f}\n"
             f"- Standard Deviation: {stat['std']:.2f}\n"
             f"- Minimum: {stat['min']:.2f}\n"
             f"- Maximum: {stat['max']:.2f}\n"
             f"- 1st Quartile (Q1): {stat['q1']:.2f}\n"
             f"- 3rd Quartile (Q3): {stat['q3']:.2f}"
             for col, stat in numeric_stats.items()]
        )
    else:
        stats_text = "No numeric columns found for statistical analysis."

    # Format categorical insights
    categorical_text = ""
    if categorical_insights:
        categorical_text = "\n".join(
            [f"**{col} Top Values:**\n" + "\n".join(
                [f"- {item['value']}: {item['count']}" for item in insights['top_values']]
            )
             for col, insights in categorical_insights.items()]
        )
    else:
        categorical_text = "No categorical columns found for analysis."

    # Generate insight with LLM
    insight_prompt = (
        f"Based on the following analysis of the uploaded data:\n"
        f"**Numeric Statistics:**\n{stats_text}\n\n"
        f"**Categorical Insights:**\n{categorical_text}\n\n"
        f"**Sample Data:**\n{sample_data}\n\n"
        f"Provide a detailed report (2-3 paragraphs) summarizing the key findings. "
        f"Include observations about trends, distributions, typical values, variability, and potential business implications or uses. "
        f"Do not reproduce the raw data, but rather synthesize the information into meaningful insights."
    )
    insight = content_agent.generate(insight_prompt)

    return f"### Analysis of Uploaded Data\n\n" \
           f"{stats_text}\n\n" \
           f"{categorical_text}\n\n" \
           f"**Insight Report:**\n{insight}"

def perform_operation(operation, data, table_name, schema_info):
    """Perform the selected operation and provide insights."""
    if operation == "None":
        return "No operation selected."
    elif operation == "Clean Data":
        # Placeholder for data cleaning logic
        cleaning_plan = content_agent.generate(f"Given the following data sample: {data.head().to_string() if data is not None else 'No data available'}, and the schema: {schema_info}, what data cleaning steps would you recommend? Provide a detailed plan.")
        # Enclose table name in backticks
        if table_name:
            sample_query = f"SELECT * FROM `{table_name}` LIMIT 5"
            sample_result = db_agent.query(sample_query)
            try:
                sample_data = ast.literal_eval(sample_result) if isinstance(sample_result, str) else sample_result
            except (ValueError, SyntaxError):
                sample_data = eval(sample_result) if isinstance(sample_result, str) else sample_result
            sample_text = str(sample_data) if sample_data else "no sample data available"
        else:
            sample_text = data.head().to_string() if data is not None else "No data available"
        cleaning_result = content_agent.generate(f"Based on the cleaning plan: {cleaning_plan}, clean the following data: {sample_text}. Provide the cleaned data and a summary of the changes made.")
        return f"**Data Cleaning Plan:**\n{cleaning_plan}\n\n**Cleaning Result:**\n{cleaning_result}"
    elif operation == "Visualize Data":
        # Placeholder for data visualization logic
        visualization_plan = content_agent.generate(f"Given the following data sample: {data.head().to_string() if data is not None else 'No data available'}, and the schema: {schema_info}, what type of visualization would be most insightful? Provide a detailed plan.")
        visualization_result = content_agent.generate(f"Based on the visualization plan: {visualization_plan}, generate the visualization. Provide the visualization and a summary of the changes made.")
        return f"**Visualization Plan:**\n{visualization_plan}\n\n**Visualization Result:**\n{visualization_result}"
    elif operation == "Detailed Analysis":
        # Placeholder for detailed analysis logic
        if table_name:
            analysis_agent = AnalysisAgent(pd.DataFrame(ast.literal_eval(db_agent.query(f"SELECT * FROM `{table_name}`"))))
            data_for_report = db_agent.query(f"SELECT * FROM `{table_name}`")
        else:
            analysis_agent = AnalysisAgent(data)
            data_for_report = data.to_string()
        eda_results, insight = analysis_agent.analyze()
        report = report_agent.generate_report(data_for_report, eda_results, insight)
        return f"**Detailed Analysis Report:**\n{report}"
    elif operation == "Web Research":
        # Placeholder for web research logic
        research_plan = content_agent.generate(f"Given the following data sample: {data.head().to_string() if data is not None else 'No data available'}, and the schema: {schema_info}, what type of web research would be most insightful? Provide a detailed plan.")
        research_result = content_agent.generate(f"Based on the research plan: {research_plan}, generate the research. Provide the research and a summary of the changes made.")
        return f"**Research Plan:**\n{research_plan}\n\n**Research Result:**\n{research_result}"
    elif operation == "Ask a specific question":
        # Placeholder for asking a specific question logic
        question_plan = content_agent.generate(f"Given the following data sample: {data.head().to_string() if data is not None else 'No data available'}, and the schema: {schema_info}, what type of question would be most insightful? Provide a detailed plan.")
        question_result = content_agent.generate(f"Based on the question plan: {question_plan}, generate the question. Provide the question and a summary of the changes made.")
        return f"**Question Plan:**\n{question_plan}\n\n**Question Result:**\n{question_result}"
    else:
        return "Invalid operation selected."

def format_database_response(raw_result, question, schema_info):
    """Convert raw database query results into detailed insights with statistical analysis."""
    if not raw_result or "Error" in raw_result:
        return raw_result

    try:
        # Attempt to parse the raw result safely
        try:
            cleaned_result = ast.literal_eval(raw_result) if isinstance(raw_result, str) else raw_result
        except (ValueError, SyntaxError) as e:
            logging.error(f"ast.literal_eval failed with error: {e}")
            try:
                cleaned_result = eval(raw_result)
            except Exception as eval_e:
                return f"Sorry, I couldn’t understand the result: {str(eval_e)}. Please try rephrasing your question or check the table schema."

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
            elif ("give me an insight" in question.lower() or "detailed analysis" in question.lower() or "insight about" in question.lower()):
                # Extract table name
                table_name = None
                # First, try to extract from the question
                table_match = re.search(r"(?:insight about|insight of|detailed analysis of)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)\s+table", question.lower())
                if table_match:
                    table_name = table_match.group(1)
                else:
                    # If not found in the question, try to extract from the query result (if it's a SELECT query)
                    if isinstance(cleaned_result, list) and len(cleaned_result) > 0 and isinstance(cleaned_result[0], tuple) and len(cleaned_result[0]) > 0:
                        table_match = re.search(r"FROM\s+`?([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)`?", question, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1)
                
                # If table name is still not found, use the selected table from session state
                if table_name is None:
                    table_name = st.session_state.selected_table
                
                if table_name is None:
                    return "Could not determine the table name for analysis."
                
                # Generate the report
                analysis_agent = AnalysisAgent(pd.DataFrame(ast.literal_eval(db_agent.query(f"SELECT * FROM `{table_name}`"))))
                data_for_report = db_agent.query(f"SELECT * FROM `{table_name}`")
                eda_results, insight = analysis_agent.analyze()
                report = report_agent.generate_report(data_for_report, eda_results, insight)
                return f"**Detailed Analysis Report:**\n{report}"
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

# Data Source Selection
if st.session_state.data_source is None:
    st.subheader("Select Data Source")
    data_source = st.radio("Choose your data source:", ["Database", "Upload File"])
    if st.button("Confirm Data Source"):
        st.session_state.data_source = data_source
        st.rerun()

# Database Flow
elif st.session_state.data_source == "Database":
    if st.session_state.selected_table is None:
        st.subheader("Select a Table from the Database")
        try:
            tables_result = db_agent.query("SHOW TABLES")
            # Correctly extract table names from the list of tuples
            tables = [table[0] for table in ast.literal_eval(tables_result)]
            selected_table = st.selectbox("Available Tables:", tables)
            if st.button("Analyze Table"):
                st.session_state.selected_table = selected_table
                st.rerun()
        except Exception as e:
            st.error(f"Error fetching tables: {e}")
    else:
        st.subheader(f"Analyzing Table: {st.session_state.selected_table}")
        # Display initial insight
        schema_info = db_agent.db.get_table_info()
        # Enclose table name in backticks
        insight_response = format_database_response(db_agent.query(f"SELECT * FROM `{st.session_state.selected_table}` LIMIT 1"), f"give me an insight about {st.session_state.selected_table} table", schema_info)
        st.markdown(insight_response)
        # Ask for operations
        st.subheader("What operations do you want to perform?")
        operations = st.selectbox("Suggested operation:", ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"])
        if st.button("Proceed with Operations"):
            st.session_state.chat_history.append({"role": "assistant", "content": f"Selected operation: {operations}"})
            if operations != "None":
                schema_info = db_agent.db.get_table_info()
                st.session_state.operation_result = perform_operation(operations, None, st.session_state.selected_table, schema_info)
                st.rerun()
            st.session_state.data = None
            
        if st.session_state.operation_result:
            st.markdown(st.session_state.operation_result)
            st.session_state.operation_result = None

# File Upload Flow
elif st.session_state.data_source == "Upload File":
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
        # Generate and display the report
        report = generate_data_report(st.session_state.data)
        st.markdown(report)
        # Ask for operations
        st.subheader("What operations do you want to perform?")
        operations = st.selectbox("Suggested operation:", ["None", "Clean Data", "Visualize Data", "Detailed Analysis", "Web Research", "Ask a specific question"])
        if st.button("Proceed with Operations"):
            st.session_state.chat_history.append({"role": "assistant", "content": f"Selected operation: {operations}"})
            if operations != "None":
                schema_info = {}  # No schema for uploaded files
                st.session_state.operation_result = perform_operation(operations, st.session_state.data, None, schema_info)
                st.rerun()
        if st.session_state.operation_result:
            st.markdown(st.session_state.operation_result)
            st.session_state.operation_result = None

# Chat interface (only if data is loaded or table is selected)
if st.session_state.data is not None or st.session_state.selected_table is not None:
    st.subheader("Ask a Question")
    prompt = st.chat_input("Type your question here...")

    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Process the prompt
        response = ""
        image_path = None

        try:
            # Use ContentGenerationAgent to classify the prompt and generate a plan
            schema_info = db_agent.db.get_table_info()

            if isinstance(schema_info, str):
                try:
                    schema_info = json.loads(schema_info) if schema_info.strip().startswith('{') else ast.literal_eval(schema_info)
                except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                    logging.error(f"Failed to parse schema_info: {e}")
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
                4. Detailed Analysis (e.g., asking for a comprehensive analysis of a table)

                Then, provide a plan to fulfill the request:
                - For database queries: If asking about the database name, return: Database Name: insightforge_db
                  If asking about tables, return: SQL Query: SHOW TABLES
                  If asking how many tables, return: SQL Query: SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'insightforge_db'
                  If asking for a detailed analysis of a table (e.g., 'give me an insight about' or 'detailed analysis of'), return: SQL Query: SELECT * FROM [table_name] LIMIT 1
                  For other data queries, return: SQL Query: [your query]
                - For visualization requests: Specify the type of plot (scatter, histogram, heatmap, boxplot) and the columns to use (if mentioned) using the format: Plot Type: [type]\nX Column: [x_col]\nY Column: [y_col]\nHistogram Column: [hist_col]\nBoxplot Column: [box_col]\nTable Name: [table_name]
                - For web research: Extract the topic to research using the format: Topic: [topic]
                - For detailed analysis: Return: Detailed Analysis: True\nTable Name: [table_name]

                Return your response in the following format:
                Category: [category]
                Plan: [your plan]
                """
            )

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
                table_match = re.search(r"Table Name:\s*([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)", plan, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                else:
                    # Fallback to extract from question
                    table_match = re.search(r"(?:insight about|insight of|detailed analysis of)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)\s+table", prompt.lower())
                    table_name = table_match.group(1) if table_match else None

                if category == "Database query":
                    db_name_match = re.search(r"Database Name:\s*(.+?)(?:\n|$)", plan)
                    if db_name_match:
                        response = "I am using a database called 'insightforge_db'."
                    else:
                        sql_query_match = re.search(r"SQL Query:\s*(.*?)(?:\n|$)", plan)
                        if sql_query_match:
                            sql_query = sql_query_match.group(1).strip()
                            # Enclose table name in backticks
                            if table_name:
                                sql_query = sql_query.replace(f"FROM {table_name}", f"FROM `{table_name}`")
                            try:
                                raw_result = db_agent.query(sql_query)
                                response = format_database_response(raw_result, prompt, schema_info)
                            except Exception as e:
                                response = f"Error querying database: {str(e)}. Please check the table schema."
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
                            # Enclose table name in backticks
                            query = f"SELECT {x_col if x_col else '*'}, {y_col if y_col else '*'}, {hist_col if hist_col else '*'}, {box_col if box_col else '*'} FROM `{table_name}` LIMIT 1000"
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
                
                elif category == "Detailed Analysis":
                    if table_name:
                        # Enclose table name in backticks
                        response = format_database_response(db_agent.query(f"SELECT * FROM `{table_name}` LIMIT 1"), prompt, schema_info)
                    else:
                        response = "I understood you want a detailed analysis, but I couldn't determine the table name. Please specify the table."

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
