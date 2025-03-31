# models/llm_handler.py
import os
from groq import Groq
from config.settings import settings

class LLMHandler:
    """
    Handles interactions with Groq Cloud's LLMs for fast inference.
    """
    def __init__(self):
        """
        Initialize with Groq API key from settings.
        
        Raises:
            ValueError: If GROQ_API_KEY is not set in the environment.
        """
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in the environment")
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def get_completion(self, prompt, model="llama-3.3-70b-versatile", max_tokens=100):
        """
        Get a completion from the Groq LLM for the given prompt.
        
        Args:
            prompt (str): Input prompt for the LLM.
            model (str): Groq model to use (default: "llama-3.3-70b-versatile").
            max_tokens (int): Maximum number of tokens in the response.
            
        Returns:
            str: LLM response text.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Failed to get completion from Groq: {str(e)}")