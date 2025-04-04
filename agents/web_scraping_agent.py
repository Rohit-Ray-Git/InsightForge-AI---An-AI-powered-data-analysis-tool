# agents/web_scraping_agent.py
from crewai import Agent
from crewai import Task
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

class SerpSearchTool(BaseTool):
    name: str = "serp_search"
    description: str = "Search the web using SerpAPI and return a summary of results."

    def _run(self, query: str) -> str:
        try:
            params = {
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "num": 5
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])
            return "\n".join([r.get("snippet", "") for r in results])
        except Exception as e:
            return f"Error searching SerpAPI: {str(e)}"

class WebScrapingAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        
        # Instantiate the custom tool
        serp_tool = SerpSearchTool()

        self.agent = Agent(
            role="Web Researcher",
            goal="Gather and summarize information from the web",
            backstory="Skilled at finding relevant data online.",
            llm=self.llm,
            tools=[serp_tool],
            verbose=True
        )

    def research(self, topic: str) -> str:
        task = Task(
            description=f"Research this topic online and summarize findings: {topic}",
            expected_output="A detailed and structured summary of the topic.",
            agent=self.agent
        )
        return task.execute()