# agents/report_generation_agent.py
import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd
from datetime import datetime
from models.llm_handler import LLMHandler  # Assuming this exists

class ReportGenerationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.llm = LLMHandler()  # Initialize LLM for insights

    def _select_preview_columns(self, data: pd.DataFrame, max_columns: int = 7) -> pd.DataFrame:
        """Select a subset of columns for data preview based on variance and type."""
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

    def generate_report(self, data: pd.DataFrame, eda_results: dict, insight: str, viz_paths: dict) -> dict:
        """Generate an HTML and PDF report from data and EDA results."""
        try:
            # Data preview
            preview_data = self._select_preview_columns(data, max_columns=7)
            data_head = preview_data.head().to_html(index=False, classes="table table-striped", border=0)

            # Format EDA results
            stats_table = eda_results['stats']['describe']
            if isinstance(stats_table, (pd.DataFrame, pd.Series)):
                stats_table = stats_table.to_string(float_format="%.6f")
            correlations = eda_results['correlations']
            if isinstance(correlations, (pd.DataFrame, pd.Series)):
                correlations = correlations.to_string(float_format="%.6f")
            missing_values = "\n".join([f"{k}: {v}" for k, v in eda_results['stats']['missing_values'].items()])
            column_note = (
                f"Showing {len(preview_data.columns)} of {len(data.columns)} columns; "
                f"full data available in source file." if len(data.columns) > 7 else ""
            )

            # Generate insight dynamically (replacing the placeholder)
            insight_prompt = (
                f"Provide key insights in HTML format (use <strong> for bold, <ul><li> for lists) based on this EDA:\n"
                f"Statistics:\n{stats_table}\nMissing Values:\n{missing_values}\nCorrelations:\n{correlations}\n"
                f"Focus on temperature, correlations, coupon redemption, direction, and target variable Y."
            )
            insight = self.llm.get_completion(insight_prompt, max_tokens=1000)

            # Prepare visualizations (use relative paths for web serving)
            visualizations = []
            for name, path in viz_paths.items():
                if path and os.path.exists(os.path.join(self.output_dir, path)):  # Check if file exists
                    prompt = (
                        f"Describe this visualization in HTML format (use <strong> for bold, <ul><li> for lists): "
                        f"{name.replace('_', ' ').title()} based on this data:\nStats:\n{stats_table}\nCorrelations:\n{correlations}"
                    )
                    description = self.llm.get_completion(prompt, max_tokens=1000)
                    visualizations.append({
                        'name': name,
                        'path': path,  # Use relative path directly (e.g., 'temperature_distribution.png')
                        'description': description
                    })
                else:
                    print(f"Warning: Visualization file {path} not found; skipping.")

            # Generate outcomes
            outcomes_prompt = (
                f"Provide actionable outcomes in HTML format (use <strong> for bold, <ul><li> for lists):\n"
                f"Stats:\n{stats_table}\nMissing Values:\n{missing_values}\nCorrelations:\n{correlations}\nInsight:\n{insight}\n"
                f"Focus on temperature-based strategies, data cleaning, variable optimization, and customer behavior."
            )
            outcomes = self.llm.get_completion(outcomes_prompt, max_tokens=1000)

            # Generate conclusion
            conclusion_prompt = (
                f"Provide a conclusion in HTML format (use <strong> for bold, <ul><li> for lists):\n"
                f"Stats:\n{stats_table}\nCorrelations:\n{correlations}\nInsight:\n{insight}\nOutcomes:\n{outcomes}\n"
                f"Include a summary, actionable recommendations, and next steps."
            )
            conclusion = self.llm.get_completion(conclusion_prompt, max_tokens=1000)

            # Render report
            today_date = datetime.now().strftime("%B %d, %Y")
            html_path = os.path.join(self.output_dir, "report.html")
            pdf_path = os.path.join(self.output_dir, "analysis_report.pdf")

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

            # Write HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Generate PDF
            HTML(string=html_content, base_url=self.output_dir).write_pdf(pdf_path)

            return {'html_path': html_path, 'pdf_path': pdf_path}

        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise  # Re-raise to let main.py handle the error
