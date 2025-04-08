# agents/data_cleaning_agent.py
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re
import io

class DataCleaningAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), client=None)

    def clean_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Clean the provided DataFrame using an LLM."""
        if data is None or data.empty:
            return None, "No data provided for cleaning."

        # Analyze the data to inform the LLM prompt
        null_counts = data.isnull().sum()
        numeric_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        cleaning_prompt = f"""Given the following Pandas DataFrame:

{data.to_string()}

Data Analysis:
- Number of rows: {len(data)}
- Number of columns: {len(data.columns)}
- Numeric columns: {list(numeric_cols)}
- Categorical columns: {list(categorical_cols)}
- Missing values per column: {null_counts.to_dict()}


Perform the following data cleaning steps:

1. **Missing Value Handling:** For each column with missing values, provide instructions for handling them.  For numerical columns, suggest imputation methods (mean, median, or a specific value). For categorical columns, suggest imputation methods (mode or a specific value) or removal if appropriate.  Clearly state your reasoning.

2. **Outlier Handling:** For each numerical column, identify potential outliers using the IQR method (Interquartile Range).  Provide instructions on how to handle these outliers (e.g., capping, removal, or transformation).  Explain your reasoning.

Provide your cleaning instructions in a structured format:

**Cleaning Instructions:**
"""
        cleaning_instructions = self.llm.invoke(cleaning_prompt).content
        try:
            # Parse cleaning instructions (this part needs improvement for robustness)
            instructions = {}
            for line in cleaning_instructions.strip().split('\n'):
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    col_name, instruction = parts
                    instructions[col_name.strip()] = instruction.strip()

            cleaned_data = data.copy()
            cleaning_steps_description = ""

            for col_name, instruction in instructions.items():
                try:
                    method, rationale = instruction.split(";", 1)  # Simple split; improve for robustness
                    method = method.strip().lower()
                    rationale = rationale.strip()
                    cleaning_steps_description += f"Column '{col_name}': {method.capitalize()} - {rationale}\n"

                    if pd.api.types.is_numeric_dtype(cleaned_data[col_name]):
                        if method == "mean imputation":
                            cleaned_data[col_name] = cleaned_data[col_name].fillna(cleaned_data[col_name].mean())
                        elif method == "median imputation":
                            cleaned_data[col_name] = cleaned_data[col_name].fillna(cleaned_data[col_name].median())
                        elif method.startswith("remove"):
                            cleaned_data = cleaned_data.dropna(subset=[col_name])
                        # Add other numerical handling methods as needed

                    elif pd.api.types.is_categorical_dtype(cleaned_data[col_name]):
                        if method == "mode imputation":
                            cleaned_data[col_name] = cleaned_data[col_name].fillna(cleaned_data[col_name].mode()[0])
                        elif method.startswith("remove"):
                            cleaned_data = cleaned_data.dropna(subset=[col_name])
                        # Add other categorical handling methods as needed

                except ValueError:
                    cleaning_steps_description += f"Column '{col_name}': Could not parse instructions.\n"
                except KeyError:
                    cleaning_steps_description += f"Column '{col_name}': Column not found in DataFrame.\n"
                except Exception as e:
                    cleaning_steps_description += f"Column '{col_name}': An error occurred: {str(e)}\n"


            return cleaned_data, cleaning_steps_description

        except Exception as e:
            return None, f"Error: An unexpected error occurred during data cleaning: {str(e)}"

