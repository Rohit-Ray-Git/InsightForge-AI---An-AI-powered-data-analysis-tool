# tests/test_data_ingestion.py
import unittest
from agents.data_ingestion_agent import ingest_data

class TestDataIngestionAgent(unittest.TestCase):
    def test_ingest_csv(self):
        # Use a sample CSV file path (create one for testing)
        file_path = "data/raw/USA_Housing.csv"
        data = ingest_data(file_path, "csv")
        self.assertTrue(hasattr(data, 'head'), "Data should be a DataFrame")

if __name__ == "__main__":
    unittest.main()