# agents/report_generation_agent.py
import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd

class ReportGenerationAgent:
    """
    Agent responsible for generating a professional PDF report from analysis and visualizations.
    """
    def __init__(self, output_dir: str = "data/reports"):
        """
        Initialize with an output directory for the report.
        
        Args:
            output_dir (str): Directory to save the report (default: "data/reports").
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader('templates'))  # Assumes a templates/ folder

    def generate_report(self, data: pd.DataFrame, eda_results: dict, insight: str, viz_paths: dict):
        """
        Generate a PDF report combining data, EDA, insights, and visualizations.
        
        Args:
            data (pd.DataFrame): Ingested dataset.
            eda_results (dict): EDA results from AnalysisAgent.
            insight (str): AI-generated insight from AnalysisAgent.
            viz_paths (dict): Paths to visualization files from VisualizationAgent.
            
        Returns:
            str: Path to the generated PDF report.
        """
        # Prepare data for the template
        data_head = data.head().to_html(index=False, classes="table table-striped", border=0)
        stats_table = eda_results['stats']['describe']
        missing_values = "\n".join([f"{k}: {v}" for k, v in eda_results['stats']['missing_values'].items()])
        correlations = eda_results['correlations']
        
        # Convert visualization paths to absolute paths for WeasyPrint
        abs_viz_paths = {k: os.path.abspath(v) if v else None for k, v in viz_paths.items()}

        # Render HTML template
        template = self.env.get_template('report_template.html')
        html_content = template.render(
            title="InsightForge AI Analysis Report",
            data_head=data_head,
            stats_table=stats_table,
            missing_values=missing_values,
            correlations=correlations,
            insight=insight,
            viz_paths=abs_viz_paths
        )

        # Write HTML to file (optional, for debugging)
        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Convert to PDF
        pdf_path = os.path.join(self.output_dir, "analysis_report.pdf")
        HTML(string=html_content).write_pdf(pdf_path)
        return pdf_path