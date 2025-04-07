import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import pymysql

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
            self.raw_db = pymysql.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name,
                port=int(db_port)
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}")

        self.llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

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
            
            # Use the detailed schema for better query generation
            schema_info = self.get_detailed_table_info()
            print(f"Debug (DatabaseAgent): Schema info = {schema_info}")  # Added debugging
            prompt = (
                f"Given this database schema:\n{schema_info}\n"
                f"Generate an SQL query to answer this question: {question}\n"
                "Ensure the query uses a numeric column (with types INT, DECIMAL, FLOAT, DOUBLE, TINYINT, or BIGINT) for aggregations (e.g., AVG, SUM) if required. "
                "If no numeric column is specified and multiple numeric columns exist, prefer 'salary' or the first numeric column found. "
                "Return only the SQL query, without any additional text or explanation."
            )
            sql_query = self.llm.invoke(prompt).content.strip()
            print(f"Debug (DatabaseAgent): Generated SQL query = {sql_query}")  # Added debugging
            
            # Execute the generated SQL query
            result = self.db.run(sql_query)
            return str(result) if result else "No data found for your query."
        except Exception as e:
            return f"Error querying database: {str(e)}"
        finally:
            pass

    def get_detailed_table_info(self):
        """
        Fetch detailed schema information including table names and column types.
        
        Returns:
            dict: A dictionary with table names as keys and column details as values.
        """
        schema = {}
        cursor = self.raw_db.cursor()
        try:
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            for table_name in tables:
                cursor.execute(f"DESCRIBE {table_name}")
                columns = {}
                for row in cursor.fetchall():
                    col_name, col_type, _, _, _, _ = row
                    columns[col_name] = {"type": col_type.upper()}  # Convert to uppercase for consistency
                if columns:
                    schema[table_name] = {"columns": columns}
            print(f"Debug (DatabaseAgent): Detailed schema = {schema}")  # Added debugging
            return schema
        except Exception as e:
            print(f"Debug (DatabaseAgent): Error fetching detailed table info: {e}")
            return {}
        finally:
            cursor.close()

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
            
            formatted_tables = [table.replace("_", " ") for table in tables]
            if len(formatted_tables) == 1:
                return f"The database has one table called '{formatted_tables[0]}'."
            else:
                table_list = ", ".join(f"'{table}'" for table in formatted_tables[:-1])
                last_table = f"'{formatted_tables[-1]}'"
                return f"The database has these tables: {table_list} and {last_table}."
        except Exception as e:
            return f"Sorry, I couldn't get the list of tables because of an error: {str(e)}"

    def __del__(self):
        """Ensure the raw database connection is closed when the object is destroyed."""
        if hasattr(self, 'raw_db') and self.raw_db.open:
            self.raw_db.close()

if __name__ == "__main__":
    agent = DatabaseAgent()
    print(agent.get_detailed_table_info())