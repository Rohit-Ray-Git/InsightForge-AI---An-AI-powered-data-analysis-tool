# agents/report_generation_agent.py
import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd
from datetime import datetime

class ReportGenerationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader('templates'))

    def _select_preview_columns(self, data: pd.DataFrame, max_columns: int = 7):
        if len(data.columns) <= max_columns:
            return data
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            variance = data[numeric_cols].var().sort_values(ascending=False)
            top_numeric = variance.index[:min(max_columns - 1, len(numeric_cols))].tolist()
        else:
            top_numeric = []
        categorical_cols = data.select_dtypes(include=['object']).columns
        top_categorical = [categorical_cols[0]] if len(categorical_cols) > 0 else []
        selected_cols = top_numeric + top_categorical
        if len(selected_cols) > max_columns:
            selected_cols = selected_cols[:max_columns]
        elif len(selected_cols) < max_columns and len(data.columns) > len(selected_cols):
            remaining = [col for col in data.columns if col not in selected_cols][:max_columns - len(selected_cols)]
            selected_cols.extend(remaining)
        return data[selected_cols]

    def generate_report(self, data: pd.DataFrame, eda_results: dict, insight: str, viz_paths: dict):
        preview_data = self._select_preview_columns(data, max_columns=7)
        data_head = preview_data.head().to_html(index=False, classes="table table-striped", border=0)
        
        stats_table = eda_results['stats']['describe']
        if isinstance(stats_table, (pd.DataFrame, pd.Series)):
            stats_table = stats_table.to_string(float_format="%.6f")
        elif not isinstance(stats_table, str):
            stats_table = str(stats_table)
        
        correlations = eda_results['correlations']
        if isinstance(correlations, (pd.DataFrame, pd.Series)):
            correlations = correlations.to_string(float_format="%.6f")
        elif not isinstance(correlations, str):
            correlations = str(correlations)
        
        missing_values = "\n".join([f"{k}: {v}" for k, v in eda_results['stats']['missing_values'].items()])
        column_note = f"Showing {len(preview_data.columns)} of {len(data.columns)} columns; full data available in source file." if len(data.columns) > 7 else ""
        
        # Use relative paths for images
        abs_viz_paths = {k: os.path.basename(v) if v else None for k, v in viz_paths.items()}
        today_date = datetime.now().strftime("%B %d, %Y")

        template = self.env.get_template('report_template.html')
        html_content = template.render(
            title="InsightForge AI Analysis Report",
            data_head=data_head,
            column_note=column_note,
            stats_table=stats_table,
            missing_values=missing_values,
            correlations=correlations,
            insight=insight,
            viz_paths=abs_viz_paths,
            today_date=today_date
        )

        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        pdf_path = os.path.join(self.output_dir, "analysis_report.pdf")
        HTML(string=html_content, base_url=self.output_dir).write_pdf(pdf_path)
        return pdf_path