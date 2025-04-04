# agents/report_generation_agent.py

import os
import litellm
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from agents.content_generation_agent import ContentGenerationAgent

class ReportGenerationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        self.content_agent = ContentGenerationAgent()
        self.agent = Agent(
            role="Report Writer",
            goal="Generate a detailed and professional report based on data analysis",
            backstory="An expert in technical writing and business reporting.",
            llm=self.llm,
            tools=[],
            verbose=True
        )

    def generate_report(self, data, eda_results, insights: str):
        task = Task(
            description=f"""Using the following insights, write a clear, structured, and professional report section for a data analysis report:

    {insights}

    Make sure the report is suitable for business stakeholders.
    """,
            expected_output="A professional report section.",
            agent=self.agent
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()
        return result

