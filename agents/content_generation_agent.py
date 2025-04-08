# agents/content_generation_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

class ContentGenerationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True, api_version="v1")

    def generate(self, prompt: str) -> str:
        """
        Generate a plain text response for the given prompt using the LLM.
        
        Args:
            prompt (str): The prompt to process.
            
        Returns:
            str: The generated text response.
        """
        try:
            # Add explicit instruction to ensure the format
            full_prompt = (
                f"{prompt}\n\n"
                "Ensure your response strictly follows this format:\n"
                "Category: [category]\n"
                "Plan: [your plan]\n"
                "For the Category, use one of: 'Database query', 'Visualization request', 'Web research', 'Detailed Analysis' (without numbering).\n"
                "For the Plan:\n"
                "- If Database query, use:\n"
                "  - For table listing: SQL Query: SHOW TABLES\n"
                "  - For counting tables: SQL Query: SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'insightforge_db'\n"
                "  - For database metadata (e.g., database name): Database Name: insightforge_db\n"
                "  - For fetching all data from a table: SQL Query: SELECT * FROM [table_name]\n"
                "  - For getting insights about a table (e.g., 'tell me about'): SQL Query: SELECT COUNT(*) as total_rows, AVG([numeric_column]) as average_value FROM [table_name] WHERE [numeric_column] IS NOT NULL\n"
                "  - For other data queries: SQL Query: [your query]\n"
                "- If Visualization request, use: Plot Type: [type]\nX Column: [x_col]\nY Column: [y_col]\nHistogram Column: [hist_col]\n"
                "- If Web research, use: Topic: [topic]\n"
                "- If Detailed Analysis, use: Detailed Analysis: True\nTable Name: [table_name]\n"
                "Do not add extra text before or after this format. Replace [table_name] and [numeric_column] with appropriate values based on the schema."
            )
            response = self.llm.invoke(full_prompt).content
            if not response:
                return "Category: Unknown\nPlan: I couldn't generate a plan. Please try rephrasing your prompt."
            return response
        except Exception as e:
            return f"Category: Error\nPlan: Error generating response: {str(e)}"
