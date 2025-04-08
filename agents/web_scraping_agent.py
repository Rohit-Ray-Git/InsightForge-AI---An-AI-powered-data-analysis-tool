# agents/web_scraping_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

class WebScrapingAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True, api_version="v1", client=None)

    def research(self, topic: str) -> str:
        prompt = f"""Research the following topic online and summarize the findings: {topic}

        Use SerpAPI to gather information. Return a concise summary of the key information found.  If SerpAPI fails, provide a brief explanation.
        """
        try:
            response = self.llm.invoke(prompt).content
            return response
        except Exception as e:
            return f"Error during web research: {str(e)}"

