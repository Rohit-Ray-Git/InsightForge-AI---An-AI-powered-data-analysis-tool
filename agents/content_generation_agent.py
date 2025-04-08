# agents/content_generation_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

class ContentGenerationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=os.getenv("GOOGLE_API_KEY"), 
            convert_system_message_to_human=True, 
            api_version="v1",
            client=None # Add this line
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a plain text response for the given prompt using the LLM.
        
        Args:
            prompt (str): The prompt to process.
            
        Returns:
            str: The generated text response.
        """
        try:
            response = self.llm.invoke(prompt).content
            if not response:
                return "I couldn't generate a plan. Please try rephrasing your prompt."
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
