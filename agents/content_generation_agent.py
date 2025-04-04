# agents/content_generation_agent.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class ContentGenerationAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

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
                "- If Database query, use: SQL Query: [your query]\n"
                "- If Visualization request, use: Plot Type: [type]\nX Column: [x_col]\nY Column: [y_col]\nHistogram Column: [hist_col]\n"
                "- If Web research, use: Topic: [topic]\n"
                "Do not add extra text before or after this format."
            )
            response = self.llm.invoke(full_prompt).content
            if not response:
                return "Category: Unknown\nPlan: I couldn't generate a plan. Please try rephrasing your prompt."
            return response
        except Exception as e:
            return f"Category: Error\nPlan: Error generating response: {str(e)}"