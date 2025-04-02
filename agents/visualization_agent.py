# agents/visualization_agent.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from models.llm_handler import LLMHandler

class VisualizationAgent:
    """
    Agent that automatically analyzes data and generates appropriate visualizations.
    """
    def __init__(self, data: pd.DataFrame, output_dir: str = "data/reports"):
        """
        Initialize with input data and an output directory for saving plots.
        
        Args:
            data (pd.DataFrame): Input data for visualization.
            output_dir (str): Directory to save generated plots (default: "data/reports").
        """
        self.data = data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set professional Seaborn style
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.family'] = 'Arial'
        
        # Data analysis
        self.numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.llm = LLMHandler()
        self.plot_decisions = self._decide_visualizations()

    def _decide_visualizations(self):
        """
        Analyze data and decide which visualizations to generate.
        
        Returns:
            list: List of tuples (plot_type, args) for each visualization.
        """
        if not self.numeric_cols:
            return []
        
        # Basic EDA
        stats = self.data[self.numeric_cols].describe()
        corr = self.data[self.numeric_cols].corr()
        
        # Prepare prompt for LLM to decide visualizations
        prompt = (
            "Given this data summary, suggest up to 3 visualization types (e.g., heatmap, scatter, histogram, box) "
            "and the columns to use for each. Provide concise reasoning. Data has these numeric columns: "
            f"{', '.join(self.numeric_cols)}\n"
            f"Statistics:\n{stats.to_string()}\n"
            f"Correlations:\n{corr.to_string()}"
        )
        llm_response = self.llm.get_completion(prompt, max_tokens=300)
        
        # Parse LLM response (simplified for now, could use regex or NLP later)
        decisions = []
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                if "heatmap" in line.lower():
                    decisions.append(("heatmap", {}))
                elif "scatter" in line.lower():
                    # Extract columns (rough heuristic, improve later)
                    cols = [col for col in self.numeric_cols if col.lower() in line.lower()]
                    if len(cols) >= 2:
                        decisions.append(("scatter", {"x_col": cols[0], "y_col": cols[1]}))
                    elif len(self.numeric_cols) >= 2:
                        decisions.append(("scatter", {"x_col": self.numeric_cols[0], "y_col": self.numeric_cols[1]}))
                elif "histogram" in line.lower():
                    cols = [col for col in self.numeric_cols if col.lower() in line.lower()]
                    decisions.append(("histogram", {"col": cols[0] if cols else self.numeric_cols[0]}))
                elif "box" in line.lower():
                    cols = [col for col in self.numeric_cols if col.lower() in line.lower()]
                    decisions.append(("box", {"col": cols[0] if cols else self.numeric_cols[0]}))
        
        # Fallback if LLM fails to suggest
        if not decisions and self.numeric_cols:
            decisions.append(("heatmap", {}))
            if len(self.numeric_cols) >= 2:
                decisions.append(("scatter", {"x_col": self.numeric_cols[0], "y_col": self.numeric_cols[1]}))
            decisions.append(("histogram", {"col": self.numeric_cols[0]}))
        
        return decisions[:3]  # Limit to 3 visualizations

    def plot_correlation_heatmap(self):
        """Generate a correlation heatmap."""
        if not self.numeric_cols:
            return None
        corr = self.data[self.numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        plt.title("Correlation Heatmap", pad=20, fontsize=16, weight='bold')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def plot_scatter(self, x_col: str, y_col: str):
        """Generate a scatter plot."""
        if x_col not in self.data.columns or y_col not in self.data.columns:
            return None
        plt.figure(figsize=(10, 6))
        sns.regplot(x=x_col, y=y_col, data=self.data, scatter_kws={'s': 50, 'alpha': 0.5, 'color': '#1f77b4'}, line_kws={'color': '#ff7f0e'})
        plt.title(f"{x_col} vs. {y_col}", pad=20, fontsize=16, weight='bold')
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{x_col}_vs_{y_col}_scatter.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def plot_histogram(self, col: str):
        """Generate a histogram with KDE."""
        if col not in self.data.columns:
            return None
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x=col, bins=30, kde=True, color='#2ca02c', edgecolor='black')
        plt.title(f"Distribution of {col}", pad=20, fontsize=16, weight='bold')
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{col}_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def plot_box(self, col: str):
        """Generate a box plot."""
        if col not in self.data.columns:
            return None
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, y=col, color='#9467bd', width=0.5)
        plt.title(f"Box Plot of {col}", pad=20, fontsize=16, weight='bold')
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{col}_box.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def generate_visualizations(self):
        """
        Automatically generate visualizations based on data analysis.
        
        Returns:
            dict: Paths to saved plot files.
        """
        viz_paths = {}
        for plot_type, args in self.plot_decisions:
            if plot_type == "heatmap":
                path = self.plot_correlation_heatmap()
                viz_paths["correlation_heatmap"] = path
            elif plot_type == "scatter":
                path = self.plot_scatter(args["x_col"], args["y_col"])
                viz_paths[f"{args['x_col']}_vs_{args['y_col']}_scatter"] = path
            elif plot_type == "histogram":
                path = self.plot_histogram(args["col"])
                viz_paths[f"{args['col']}_distribution"] = path
            elif plot_type == "box":
                path = self.plot_box(args["col"])
                viz_paths[f"{args['col']}_box"] = path
        return viz_paths