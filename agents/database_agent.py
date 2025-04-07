# agents/database_agent.py
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

load_dotenv()

class DatabaseAgent:
    def __init__(self):
        db_user = os.getenv("MYSQL_USER", "root")
        db_password = os.getenv("MYSQL_PASSWORD", "your_password")
        db_host = os.getenv("MYSQL_HOST", "localhost")
        db_port = os.getenv("MYSQL_PORT", "3306")
        db_name = os.getenv("MYSQL_DB", "insightforge_db")
        
        db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        try:
            self.db = SQLDatabase.from_uri(db_uri)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}")

        self.llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))

    def query(self, question: str) -> str:
        """
        Execute an SQL query based on the user's question.
        
        Args:
            question (str): The user's question or SQL query to execute.
            
        Returns:
            str: The query results or an error message.
        """
        try:
            # If the question is already an SQL query (e.g., "SHOW TABLES;"), execute it directly
            if question.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")):
                result = self.db.run(question)
                return str(result) if result else "No data found for your query."
            
            # Otherwise, use the LLM to generate an SQL query
            schema_info = self.db.get_table_info()
            prompt = (
                f"Given this database schema:\n{schema_info}\n"
                f"Generate an SQL query to answer this question: {question}\n"
                "Return only the SQL query, without any additional text or explanation."
            )
            sql_query = self.llm.invoke(prompt).content.strip()
            
            # Execute the generated SQL query
            result = self.db.run(sql_query)
            return str(result) if result else "No data found for your query."
        except Exception as e:
            return f"Error querying database: {str(e)}"

    def fetch_all_tables(self) -> str:
        """
        Fetch the list of all tables in the database and return them in a user-friendly format.
        
        Returns:
            str: A string listing the tables in layman's terms.
        """
        try:
            tables = self.db.get_usable_table_names()
            if not tables:
                return "There are no tables in the database right now."
            
            # Format the table names in a readable way
            # Replace underscores with spaces for better readability
            formatted_tables = [table.replace("_", " ") for table in tables]
            if len(formatted_tables) == 1:
                return f"The database has one table called '{formatted_tables[0]}'."
            else:
                table_list = ", ".join(f"'{table}'" for table in formatted_tables[:-1])
                last_table = f"'{formatted_tables[-1]}'"
                return f"The database has these tables: {table_list} and {last_table}."
        except Exception as e:
            return f"Sorry, I couldn't get the list of tables because of an error: {str(e)}"