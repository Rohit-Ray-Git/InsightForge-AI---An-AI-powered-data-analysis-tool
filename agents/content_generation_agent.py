# agents/content_generation_agent.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class ContentGenerationAgent:
    def __init__(self):
        self.llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))

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
                "For the Category, use one of: 'Database query', 'Visualization request', 'Web research' (without numbering).\n"
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
                "Do not add extra text before or after this format. Replace [table_name] and [numeric_column] with appropriate values based on the schema."
            )
            response = self.llm.invoke(full_prompt).content
            if not response:
                return "Category: Unknown\nPlan: I couldn't generate a plan. Please try rephrasing your prompt."
            return response
        except Exception as e:
            return f"Category: Error\nPlan: Error generating response: {str(e)}"