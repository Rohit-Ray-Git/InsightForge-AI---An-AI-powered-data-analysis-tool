# main.py
import streamlit as st
import pandas as pd
import os
import re
from agents.database_agent import DatabaseAgent
from agents.web_scraping_agent import WebScrapingAgent
from agents.visualization_agent import VisualizationAgent
from agents.content_generation_agent import ContentGenerationAgent

# Set page config
st.set_page_config(page_title="InsightForge AI", layout="wide")

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
st.title("InsightForge AI")
st.markdown("Unlock insights from your data with AI")

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
    **Note**: You can ask questions about fetching data from the ‘insightforge_db’ database or about the uploaded data.  
    Examples: 
    - "Show me a scatter plot of price vs income."  
    - "What are the trends in the housing market?"
    """
)

# Chat interface
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
        columns = st.session_state.data.columns.tolist() if st.session_state.data is not None else []
        prompt_analysis = content_agent.generate(
            f"""
            Analyze the following user prompt: '{prompt}'.
            The user has access to a MySQL database named 'insightforge_db' with the following schema:
            {schema_info}
            The user has also uploaded a CSV file with the following columns (if any): {columns}.
            
            Determine the user's intent and classify the prompt into one of the following categories:
            1. Database query (e.g., asking about tables, data in the database)
            2. Visualization request (e.g., asking for a plot, graph, or chart)
            3. Web research (e.g., asking about trends, market analysis)
            
            Then, provide a plan to fulfill the request:
            - For database queries: Suggest an SQL query to execute using the format: SQL Query: [your query]
            - For visualization requests: Specify the type of plot (scatter, histogram, heatmap) and the columns to use (if mentioned) using the format: Plot Type: [type]\nX Column: [x_col]\nY Column: [y_col]\nHistogram Column: [hist_col]
            - For web research: Extract the topic to research using the format: Topic: [topic]
            
            Return your response in the following format:
            Category: [category]
            Plan: [your plan]
            """
        )
        
        # Debug: Print the prompt analysis to understand what the LLM is returning
        print(f"Prompt Analysis:\n{prompt_analysis}")
        
        # Parse the LLM's response with more robust regular expressions
        category_match = re.search(r"Category:\s*(.+?)(?:\n|$)", prompt_analysis)
        plan_match = re.search(r"Plan:\s*(.+)", prompt_analysis, re.DOTALL)
        
        if not category_match or not plan_match:
            response = f"I couldn't understand your request. Could you please rephrase it?\n\nDebug Info:\n{prompt_analysis}"
        else:
            # Remove any numbering (e.g., "1. ") from the category
            category = re.sub(r"^\d+\.\s*", "", category_match.group(1).strip())
            plan = plan_match.group(1).strip()
            
            if category == "Database query":
                # Handle specific database queries in a user-friendly way
                prompt_lower = prompt.lower()
                if "database" in prompt_lower and ("which" in prompt_lower or "what" in prompt_lower) and "using" in prompt_lower:
                    # For "Which database you are using?"
                    response = "I am using a database called 'insightforge_db'."
                elif "tables" in prompt_lower and "database" in prompt_lower and "how many" in prompt_lower:
                    # For "How many tables?" - Count the tables
                    sql_query_match = re.search(r"SQL Query:\s*(.*?)(?:\n|$)", plan)
                    if sql_query_match:
                        sql_query = sql_query_match.group(1).strip()
                        result = db_agent.query(sql_query)
                        # Extract the count from the result (e.g., [(2,)])
                        try:
                            count = int(result.strip("[]()").split(",")[0])
                            response = f"There are {count} tables in the database."
                        except (ValueError, IndexError):
                            response = "I couldn't determine the number of tables. Let's try listing them instead: " + db_agent.fetch_all_tables()
                    else:
                        response = "I understood you want to know how many tables are in the database, but I couldn't generate a valid SQL query. Let's try listing them instead: " + db_agent.fetch_all_tables()
                elif "tables" in prompt_lower and "database" in prompt_lower:
                    # For "What tables are in the database?" or "Name the tables"
                    response = db_agent.fetch_all_tables()
                else:
                    # For other database queries
                    sql_query_match = re.search(r"SQL Query:\s*(.*?)(?:\n|$)", plan)
                    if sql_query_match:
                        sql_query = sql_query_match.group(1).strip()
                        response = db_agent.query(sql_query)
                    else:
                        response = "I understood you want a database query, but I couldn't generate a valid SQL query. Please rephrase your question."
            
            elif category == "Visualization request":
                # Visualization requests require an uploaded file
                if st.session_state.data is None:
                    response = "Please upload a data file first to proceed with visualization requests."
                else:
                    plot_type_match = re.search(r"Plot Type:\s*(.*?)(?:\n|$)", plan)
                    x_col_match = re.search(r"X Column:\s*(.*?)(?:\n|$)", plan)
                    y_col_match = re.search(r"Y Column:\s*(.*?)(?:\n|$)", plan)
                    hist_col_match = re.search(r"Histogram Column:\s*(.*?)(?:\n|$)", plan)
                    
                    plot_type = plot_type_match.group(1).strip() if plot_type_match else None
                    x_col = x_col_match.group(1).strip() if x_col_match else None
                    y_col = y_col_match.group(1).strip() if y_col_match else None
                    hist_col = hist_col_match.group(1).strip() if hist_col_match else None
                    
                    if plot_type:
                        if "scatter" in plot_type.lower():
                            image_path = viz_agent.generate_scatter(st.session_state.data, x_col=x_col, y_col=y_col)
                            response = f"Here is the scatter plot{' of ' + x_col + ' vs ' + y_col if x_col and y_col else ''}:"
                        elif "histogram" in plot_type.lower() or "distribution" in plot_type.lower():
                            image_path = viz_agent.generate_distribution(st.session_state.data, hist_col=hist_col)
                            response = f"Here is the distribution plot{' of ' + hist_col if hist_col else ''}:"
                        elif "heatmap" in plot_type.lower():
                            image_path = viz_agent.generate_correlation_heatmap(st.session_state.data)
                            response = "Here is the correlation heatmap of the features:"
                        else:
                            response = "I understood you want a visualization, but I couldn't determine the plot type. Please specify (e.g., 'scatter', 'histogram', 'heatmap')."
                    else:
                        response = "I understood you want a visualization, but I couldn't determine the plot type. Please specify (e.g., 'scatter', 'histogram', 'heatmap')."
            
            elif category == "Web research":
                # Web research doesn't require an uploaded file
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