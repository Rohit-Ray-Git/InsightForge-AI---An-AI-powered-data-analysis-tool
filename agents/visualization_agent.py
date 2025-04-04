# agents/visualization_agent.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from crewai import Agent
from langchain_groq import ChatGroq

class VisualizationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        self.agent = Agent(
            role="Data Visualizer",
            goal="Generate insightful visualizations from data",
            backstory="Expert in creating clear, impactful charts from any dataset.",
            llm=self.llm,
            tools=[],
            verbose=True
        )

    def _select_columns(self, data: pd.DataFrame, x_col: str = None, y_col: str = None, hue_col: str = None, hist_col: str = None) -> tuple:
        """Dynamically select columns for visualization, with optional user-specified columns."""
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Use specified columns if provided and valid, otherwise fall back to defaults
        x = x_col if x_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        y = y_col if y_col in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else x)
        hue = hue_col if hue_col in categorical_cols else (categorical_cols[0] if categorical_cols else None)
        hist = hist_col if hist_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)

        return x, y, hue, hist

    def generate_scatter(self, data: pd.DataFrame, x_col: str = None, y_col: str = None, hue_col: str = None) -> str:
        """Generate a scatter plot with optional column specification."""
        x, y, hue, _ = self._select_columns(data, x_col=x_col, y_col=y_col, hue_col=hue_col)
        if not x or not y:
            return "No numeric columns available for scatter plot."
        
        plt.figure(figsize=(8, 6))
        if hue:
            sns.scatterplot(x=x, y=y, hue=hue, data=data, palette='coolwarm')
        else:
            sns.scatterplot(x=x, y=y, data=data)
        plt.title(f'{x} vs {y}' + (f' (Hue: {hue})' if hue else ''))
        path = os.path.join(self.output_dir, f'scatter_{x}_vs_{y}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def generate_distribution(self, data: pd.DataFrame, hist_col: str = None) -> str:
        """Generate a distribution plot with optional column specification."""
        _, _, _, hist = self._select_columns(data, hist_col=hist_col)
        if not hist:
            return "No numeric columns available for distribution plot."
        
        plt.figure(figsize=(8, 6))
        sns.histplot(data[hist], bins=20, kde=True, color='purple')
        plt.title(f'Distribution of {hist}')
        path = os.path.join(self.output_dir, f'distribution_{hist}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def generate_correlation_heatmap(self, data: pd.DataFrame) -> str:
        """Generate a correlation heatmap for all numeric columns."""
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            return "No numeric columns available for correlation heatmap."
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def generate_all(self, data: pd.DataFrame) -> dict:
        if data.empty:
            return {
                'scatter': "DataFrame is empty.",
                'distribution': "DataFrame is empty.",
                'correlation_heatmap': "DataFrame is empty."
            }
        return {
            'scatter': self.generate_scatter(data),
            'distribution': self.generate_distribution(data),
            'correlation_heatmap': self.generate_correlation_heatmap(data)
        }