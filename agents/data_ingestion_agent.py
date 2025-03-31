# agents/data_ingestion_agent.py
from utils.file_loader import load_csv, load_excel, load_json, load_pdf

def ingest_data(file_path, file_type):
    """Ingest data based on file type."""
    if file_type == "csv":
        return load_csv(file_path)
    elif file_type == "excel":
        return load_excel(file_path)
    elif file_type == "json":
        return load_json(file_path)
    elif file_type == "pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")