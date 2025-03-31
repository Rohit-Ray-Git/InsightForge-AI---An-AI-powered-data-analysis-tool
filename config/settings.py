# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings
class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    INPUT_FILE_PATH = "data/raw/USA_Housing.csv"  # Example default path
    INPUT_FILE_TYPE = "csv"  # Default file type
    VECTOR_STORE_PATH = "data/vector_store"  # Directory for vector store

settings = Settings()