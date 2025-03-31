# utils/file_loader.py
import pandas as pd
import json
from pypdf import PdfReader

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def load_excel(file_path):
    """Load an Excel file into a pandas DataFrame."""
    return pd.read_excel(file_path)

def load_json(file_path):
    """Load a JSON file into a dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)  # Updated to use pypdf
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text