import os
import logging
import pymysql
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseAgent:
    def __init__(self):
        db_user = os.getenv("MYSQL_USER", "root")
        db_password = os.getenv("MYSQL_PASSWORD", "your_password")
        db_host = os.getenv("MYSQL_HOST", "localhost")
        db_port = os.getenv("MYSQL_PORT", "3306")
        db_name = os.getenv("MYSQL_DB", "insightforge_db")  # Use env variable for db name
        
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
            logging.info("Successfully connected to MySQL database.")
        except pymysql.Error as e:
            logging.error(f"Failed to connect to MySQL: {e}")
            raise ConnectionError(f"Failed to connect to MySQL: {e}") from e

        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),  # Use env variable for model
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True,
            api_version="v1",
            client=None # Add this line
        )
        self.schema_cache = None  # Initialize schema cache

    def query(self, question: str) -> str:
        """Execute an SQL query based on the user's question."""
        try:
            if question.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")):
                result = self.db.run(question)
                return str(result) if result else "No data found for your query."

            # Use cached schema if available
            if self.schema_cache is None:
                self.schema_cache = self.get_detailed_table_info()
            schema_info = self.schema_cache

            logging.debug(f"Schema info = {schema_info}")
            prompt = (
                f"Given this database schema:\n{schema_info}\n"
                f"Generate an SQL query to answer: {question}\n"
                "Ensure the query uses a numeric column for aggregations if required. "
                "Return only the SQL query."
            )
            sql_query = self.llm.invoke(prompt).content.strip()
            logging.debug(f"Generated SQL query = {sql_query}")

            result = self.db.run(sql_query)
            return str(result) if result else "No data found for your query."
        except pymysql.Error as e:
            logging.error(f"Database query error: {e}")
            return f"Error querying database: {e}"
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    def get_detailed_table_info(self):
        """Fetch detailed schema information."""
        schema = {}
        with self.raw_db.cursor() as cursor:  # Use context manager
            try:
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                for table_name in tables:
                    cursor.execute(f"DESCRIBE `{table_name}`") # Enclose table name in backticks
                    columns = {}
                    for row in cursor.fetchall():
                        col_name, col_type, _, _, _, _ = row
                        columns[col_name] = {"type": col_type.upper()}
                    if columns:
                        schema[table_name] = {"columns": columns}
                logging.debug(f"Detailed schema = {schema}")
                return schema
            except pymysql.Error as e:
                logging.error(f"Error fetching detailed table info: {e}")
                return {}


    def __del__(self):
        """Ensure the raw database connection is closed."""
        if hasattr(self, 'raw_db') and self.raw_db.open:
            self.raw_db.close()
            logging.info("Closed raw database connection.")
