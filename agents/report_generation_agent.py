# agents/report_generation_agent.py
import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd
from datetime import datetime
from models.llm_handler import LLMHandler

class ReportGenerationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.llm = LLMHandler()  # Initialize LLM for insights

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
        correlations = eda_results['correlations']
        if isinstance(correlations, (pd.DataFrame, pd.Series)):
            correlations = correlations.to_string(float_format="%.6f")
        missing_values = "\n".join([f"{k}: {v}" for k, v in eda_results['stats']['missing_values'].items()])
        column_note = f"Showing {len(preview_data.columns)} of {len(data.columns)} columns; full data available in source file." if len(data.columns) > 7 else ""

        # Generate insight with HTML formatting
        insight_prompt = (
            "Provide key insights in HTML format (use <strong> for bold, <ul><li> for lists) based on this EDA:\n"
            f"Statistics:\n{stats_table}\n"
            f"Missing Values:\n{missing_values}\n"
            f"Correlations:\n{correlations}\n"
            "Include insights on temperature, correlations, coupon redemption, direction, and the target variable Y."
        )
        insight = self.llm.get_completion(insight_prompt, max_tokens=300)

        # Prepare visualization data with descriptions
        visualizations = []
        for name, path in viz_paths.items():
            if path:
                prompt = (
                    f"Describe this visualization in HTML format (use <strong> for bold, <ul><li> for lists): "
                    f"{name.replace('_', ' ').title()} based on:\n"
                    f"Statistics:\n{stats_table}\n"
                    f"Correlations:\n{correlations}"
                )
                description = self.llm.get_completion(prompt, max_tokens=200)
                visualizations.append({
                    'name': name,
                    'path': os.path.basename(path),
                    'description': description
                })

        # Generate outcomes with HTML
        outcomes_prompt = (
            "Provide actionable outcomes in HTML format (use <strong> for bold, <ul><li> for lists) based on this EDA:\n"
            f"Statistics:\n{stats_table}\n"
            f"Missing Values:\n{missing_values}\n"
            f"Correlations:\n{correlations}\n"
            f"Insight:\n{insight}\n"
            "Focus on temperature-based strategies, data cleaning, variable optimization, and customer behavior."
        )
        outcomes = self.llm.get_completion(outcomes_prompt, max_tokens=300)

        # Generate conclusion with HTML
        conclusion_prompt = (
            "Provide a conclusion in HTML format (use <strong> for bold, <ul><li> for lists) based on this analysis:\n"
            f"Statistics:\n{stats_table}\n"
            f"Correlations:\n{correlations}\n"
            f"Insight:\n{insight}\n"
            f"Outcomes:\n{outcomes}\n"
            "Include a summary, actionable recommendations, and next steps."
        )
        conclusion = self.llm.get_completion(conclusion_prompt, max_tokens=200)

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
            outcomes=outcomes,
            visualizations=visualizations,
            conclusion=conclusion,
            today_date=today_date
        )

        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        pdf_path = os.path.join(self.output_dir, "analysis_report.pdf")
        HTML(string=html_content, base_url=self.output_dir).write_pdf(pdf_path)
        return pdf_path