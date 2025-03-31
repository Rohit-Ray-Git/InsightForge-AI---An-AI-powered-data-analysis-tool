# main.py
from config.settings import settings
from agents.data_ingestion_agent import ingest_data

def main():
    """Main function to run the InsightForge AI pipeline."""
    try:
        # Test PDF ingestion
        data = ingest_data("data/raw/Generative AI.pdf", "pdf")
        print("PDF ingested successfully!")
        print(data[:500])  # Print first 500 characters
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()