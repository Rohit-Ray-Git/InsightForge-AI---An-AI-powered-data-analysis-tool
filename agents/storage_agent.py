# agents/storage_agent.py
import os
from crewai import Agent
from crewai.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional

load_dotenv()  # Load environment variables from .env

class SaveFileTool(BaseTool):
    name: str = "save_file"
    description: str = "Save content to a file in the specified directory."
    output_dir: Optional[str] = None  # Declare output_dir as an optional field

    def __init__(self, output_dir: str, **kwargs):
        super().__init__(name="save_file", description="Save content to a file in the specified directory.", **kwargs)
        self.output_dir = output_dir

    def _run(self, content: str, filename: str) -> str:
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File saved at: {filepath}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

class StorageAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True, api_version="v1")
        
        # Instantiate the custom tool
        save_tool = SaveFileTool(output_dir=self.output_dir)

        self.agent = Agent(
            role="Storage Manager",
            goal="Store data and reports efficiently",
            backstory="Expert in managing file storage.",
            llm=self.llm,
            tools=[save_tool],
            verbose=True
        )

    def store(self, content: str, filename: str) -> str:
        task = f"Save the following content to a file named '{filename}':\n{content}"
        return self.agent.execute_task(task)
