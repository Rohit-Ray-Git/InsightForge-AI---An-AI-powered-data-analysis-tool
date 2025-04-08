# agents/report_generation_agent.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.content_generation_agent import ContentGenerationAgent

class ReportGenerationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), client=None)
        self.content_agent = ContentGenerationAgent()

    def generate_report(self, data, eda_results, insights: str):
        prompt = f"""Generate a clear, structured, and professional report section for a data analysis report based on the following insights:

Data: {data}
EDA Results: {eda_results}
Insights: {insights}

Make sure the report is suitable for business stakeholders.  Do not reproduce the raw data; synthesize the information into meaningful insights.
"""
        try:
            report = self.llm.invoke(prompt).content
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"
