# agents/visualization_agent.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI  # Import LLM
import logging
import re

# Configure logging for this module if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizationAgent:
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), client=None)  # Initialize LLM

    def _attempt_numeric(self, data: pd.DataFrame, col_name: str) -> pd.Series | None:
        """Attempts to convert a column to numeric."""
        if col_name and col_name in data.columns:
            try:
                numeric_series = pd.to_numeric(data[col_name], errors='coerce')
                if not numeric_series.isnull().all():
                    return numeric_series
            except Exception as e:
                logging.warning(f"Could not convert column '{col_name}' to numeric: {e}")
        return None

    def _select_columns(self, data: pd.DataFrame, x_col: str = None, y_col: str = None, hue_col: str = None, target_col: str = None, cat_col: str = None) -> tuple[dict, dict]:
        """
        Dynamically select columns, attempting numeric conversion.
        Returns a dictionary of final columns used and a dictionary of fallback info.
        """
        numeric_cols_original = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        final_cols = {'x': None, 'y': None, 'hue': None, 'target': None, 'cat': None}  # Use 'target' for hist/box, 'cat' for bar
        fallback_info = {}

        # --- X Column ---
        x_numeric = self._attempt_numeric(data, x_col)
        if x_numeric is not None:
            final_cols['x'] = x_col
            data[x_col] = x_numeric  # Update data copy temporarily
        else:
            default_x = numeric_cols_original[0] if numeric_cols_original else None
            if x_col and x_col != default_x:
                fallback_info['x'] = f"Could not use '{x_col}' (not numeric), falling back to '{default_x}'."
            final_cols['x'] = default_x

        # --- Y Column ---
        y_numeric = self._attempt_numeric(data, y_col)
        if y_numeric is not None:
            final_cols['y'] = y_col
            data[y_col] = y_numeric  # Update data copy temporarily
        else:
            default_y = numeric_cols_original[1] if len(numeric_cols_original) > 1 else final_cols['x']
            if y_col and y_col != default_y:
                fallback_info['y'] = f"Could not use '{y_col}' (not numeric), falling back to '{default_y}'."
            final_cols['y'] = default_y

        # --- Hue Column ---
        if hue_col and hue_col in categorical_cols:
            final_cols['hue'] = hue_col
        else:
            default_hue = categorical_cols[0] if categorical_cols else None
            if hue_col and hue_col != default_hue:
                fallback_info['hue'] = f"Could not use '{hue_col}' (not categorical/found), falling back to '{default_hue}'."
            final_cols['hue'] = default_hue

        # --- Hist/Box Column (Target) ---
        target_numeric = self._attempt_numeric(data, target_col)
        if target_numeric is not None:
            final_cols['target'] = target_col
            data[target_col] = target_numeric  # Update data copy temporarily
        else:
            default_target = numeric_cols_original[0] if numeric_cols_original else None
            if target_col and target_col != default_target:
                fallback_info['target'] = f"Could not use '{target_col}' (not numeric), falling back to '{default_target}'."
            final_cols['target'] = default_target

        # --- Bar Column (Categorical) ---
        if cat_col and cat_col in categorical_cols:
            final_cols['cat'] = cat_col
        else:
            default_cat = categorical_cols[0] if categorical_cols else None
            if cat_col and cat_col != default_cat:
                fallback_info['cat'] = f"Could not use '{cat_col}' (not categorical/found), falling back to '{default_cat}'."
            final_cols['cat'] = default_cat

        # Remove None values from final_cols for cleaner output
        final_cols = {k: v for k, v in final_cols.items() if v is not None}

        return final_cols, fallback_info

    def _generate_plot(self, plot_func, plot_data, final_cols, plot_type_name, file_prefix) -> tuple[str | None, dict]:
        """Helper function to generate and save a plot."""
        plt.figure(figsize=(8, 6))
        plot_title = f"Plot of {plot_type_name}"  # Generic title
        try:
            plot_title = plot_func(plot_data, final_cols)  # Plotting function returns the specific title
            plt.title(plot_title)
            # Construct filename from actual columns used
            filename_parts = [file_prefix] + [str(v) for k, v in final_cols.items() if v]
            filename = "_".join(filename_parts).replace(" ", "_") + ".png"
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return path, final_cols
        except Exception as e:
            plt.close()
            logging.error(f"Error generating {plot_type_name} plot: {e}", exc_info=True)
            # Return error message and the columns attempted
            return f"Error generating {plot_type_name} plot: {e}", final_cols

    def generate_scatter(self, data: pd.DataFrame, x_col: str = None, y_col: str = None, hue_col: str = None) -> tuple[str, dict]:
        """Generate scatter plot, return path and actual columns used."""
        plot_data = data.copy()
        final_cols, fallback_info = self._select_columns(plot_data, x_col=x_col, y_col=y_col, hue_col=hue_col)

        x = final_cols.get('x')
        y = final_cols.get('y')
        hue = final_cols.get('hue')

        if not x or not y:
            return "No suitable numeric columns available for scatter plot.", final_cols

        def plot_scatter(p_data, f_cols):
            x_ax, y_ax, h_ax = f_cols.get('x'), f_cols.get('y'), f_cols.get('hue')
            if h_ax:
                sns.scatterplot(x=x_ax, y=y_ax, hue=h_ax, data=p_data, palette='coolwarm')
                return f'{x_ax} vs {y_ax} (Color: {h_ax})'
            else:
                sns.scatterplot(x=x_ax, y=y_ax, data=p_data)
                return f'{x_ax} vs {y_ax}'

        path, actual_cols = self._generate_plot(plot_scatter, plot_data, final_cols, "Scatter", "scatter")

        fallback_msg = ""
        if fallback_info:
            fallback_msg = "\n*Note: " + " ".join(fallback_info.values()) + "*"

        if isinstance(path, str) and os.path.exists(path):
            return path + fallback_msg, actual_cols
        else:  # Error occurred
            return path + fallback_msg, actual_cols  # path contains error message

    def generate_distribution(self, data: pd.DataFrame, hist_col: str = None) -> tuple[str, dict]:
        """Generate distribution plot, return path and actual columns used."""
        plot_data = data.copy()
        final_cols, fallback_info = self._select_columns(plot_data, target_col=hist_col)
        target = final_cols.get('target')

        if not target:
            return "No suitable numeric columns available for distribution plot.", final_cols

        def plot_hist(p_data, f_cols):
            t_ax = f_cols.get('target')
            if pd.api.types.is_numeric_dtype(p_data[t_ax]):
                sns.histplot(data=p_data, x=t_ax, bins=20, kde=True, color='purple')
                return f'Distribution of {t_ax}'
            elif pd.api.types.is_categorical_dtype(p_data[t_ax]) or pd.api.types.is_object_dtype(p_data[t_ax]):
                sns.countplot(data=p_data, x=t_ax, color='purple')
                return f'Distribution of {t_ax}'
            else:
                return f"Column {t_ax} is not numeric or categorical."

        path, actual_cols = self._generate_plot(plot_hist, plot_data, final_cols, "Distribution", "distribution")

        fallback_msg = ""
        if fallback_info.get('target'):
            fallback_msg = "\n*Note: " + fallback_info['target'] + "*"

        if isinstance(path, str) and os.path.exists(path):
            return path + fallback_msg, actual_cols
        else:
            return path + fallback_msg, actual_cols

    def generate_boxplot(self, data: pd.DataFrame, box_col: str = None) -> tuple[str, dict]:
        """Generate box plot, return path and actual columns used."""
        plot_data = data.copy()
        final_cols, fallback_info = self._select_columns(plot_data, target_col=box_col)
        target = final_cols.get('target')

        if not target:
            return "No suitable numeric columns available for box plot.", final_cols

        def plot_box(p_data, f_cols):
            t_ax = f_cols.get('target')
            sns.boxplot(data=p_data, x=t_ax, color='skyblue')
            return f'Box Plot of {t_ax}'

        path, actual_cols = self._generate_plot(plot_box, plot_data, final_cols, "Box Plot", "boxplot")

        fallback_msg = ""
        if fallback_info.get('target'):
            fallback_msg = "\n*Note: " + fallback_info['target'] + "*"

        if isinstance(path, str) and os.path.exists(path):
            return path + fallback_msg, actual_cols
        else:
            return path + fallback_msg, actual_cols

    def generate_correlation_heatmap(self, data: pd.DataFrame) -> tuple[str, dict]:
        """Generate heatmap, return path and columns used (all numeric)."""
        numeric_data = data.copy()
        converted_cols_dict = {}  # Store original col name and converted series
        for col in numeric_data.columns:
            num_series = self._attempt_numeric(numeric_data, col)
            if num_series is not None:
                numeric_data[col] = num_series
                converted_cols_dict[col] = num_series

        numeric_cols_final = list(converted_cols_dict.keys())
        final_cols_info = {'numeric_columns': numeric_cols_final}  # Info for explanation

        if len(numeric_cols_final) < 2:
            return "Not enough numeric columns (at least 2 required) for correlation heatmap.", final_cols_info

        # Use only the successfully converted numeric columns for correlation
        numeric_data_for_corr = numeric_data[numeric_cols_final]

        plt.figure(figsize=(10, 8))
        try:
            sns.heatmap(numeric_data_for_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", center=0)
            plot_title = 'Correlation Heatmap'
            plt.title(plot_title)
            path = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return path, final_cols_info
        except Exception as e:
            plt.close()
            logging.error(f"Error generating correlation heatmap: {e}", exc_info=True)
            return f"Error generating correlation heatmap: {e}", final_cols_info

    def generate_bar(self, data: pd.DataFrame, cat_col: str = None) -> tuple[str, dict]:
        """Generate bar plot for a categorical column."""
        plot_data = data.copy()
        final_cols, fallback_info = self._select_columns(plot_data, cat_col=cat_col)
        cat = final_cols.get('cat')

        if not cat:
            return "No suitable categorical columns available for bar plot.", final_cols

        def plot_bar(p_data, f_cols):
            c_ax = f_cols.get('cat')
            sns.countplot(data=p_data, x=c_ax, color='lightgreen')
            return f'Bar Plot of {c_ax}'

        path, actual_cols = self._generate_plot(plot_bar, plot_data, final_cols, "Bar Plot", "bar")

        fallback_msg = ""
        if fallback_info.get('cat'):
            fallback_msg = "\n*Note: " + fallback_info['cat'] + "*"

        if isinstance(path, str) and os.path.exists(path):
            return path + fallback_msg, actual_cols
        else:
            return path + fallback_msg, actual_cols

    def generate_plot(self, data: pd.DataFrame, prompt: str) -> tuple[str | None, str]:
        """
        Generates the most appropriate plot based on the user's prompt.
        Returns:
            tuple[str | None, str]: The path to the generated plot (or None if an error occurred) and a description of the plot.
        """
        # --- LLM Planning Prompt ---
        planning_prompt = f"""
        Analyze the user's request: '{prompt}'

        Determine the user's intent and classify it into ONE category:
        1. Scatter Plot: User asks about the relationship between two numeric columns.
        2. Distribution Plot: User asks about the distribution of a numeric column or categorical column.
        3. Box Plot: User asks to compare a numeric column across categories.
        4. Bar Plot: User asks about the distribution of a categorical column.
        5. Correlation Heatmap: User asks about the correlation between numeric columns.
        6. Unclear: The request is unclear or doesn't fit other categories.

        Based on the category, provide a plan:

        - If Scatter Plot:
            - Plan: Plot Type: scatter\nColumns: [Specify columns, e.g., X=colA, Y=colB, Hue=colC]

        - If Distribution Plot:
            - Plan: Plot Type: distribution\nColumns: [Specify column, e.g., Target=colA]

        - If Box Plot:
            - Plan: Plot Type: boxplot\nColumns: [Specify column, e.g., Target=colA]

        - If Bar Plot:
            - Plan: Plot Type: bar\nColumns: [Specify column, e.g., Target=colA]

        - If Correlation Heatmap:
            - Plan: Plot Type: heatmap

        - If Unclear:
            - Plan: Unclear

        Return ONLY the Category and Plan lines.
        Category: [Your Category]
        Plan: [Your Plan]
        """

        llm_response = self.llm.invoke(planning_prompt).content
        logging.info(f"LLM Visualization Plan Response: {llm_response}")

        # Parse the LLM's response
        category_match = re.search(r"Category:\s*(.+?)(?:\n|$)", llm_response)
        plan_match = re.search(r"Plan:\s*(.+)", llm_response, re.DOTALL)

        if not category_match or not plan_match:
            return None, f"I had trouble understanding how to proceed with your request. Could you please rephrase?\nDebug Info:\n{llm_response}"

        category = category_match.group(1).strip()
        plan = plan_match.group(1).strip()
        logging.info(f"Parsed Category: {category}, Parsed Plan: {plan}")

        # --- Execute Plan ---
        if category == "Scatter Plot":
            x_col, y_col, hue_col = None, None, None
            col_specs = re.findall(r"(\w+)\s*=\s*([\w\s`'.]+?)(?:,|$)", plan)  # Allow quoted/backticked/spaced names
            for key, val in col_specs:
                key_lower = key.lower()
                # Remove potential quotes/backticks from column name
                val_stripped = val.strip().strip('`\'"')
                if key_lower == 'x':
                    x_col = val_stripped
                elif key_lower == 'y':
                    y_col = val_stripped
                elif key_lower == 'hue':
                    hue_col = val_stripped
            path, description = self.generate_scatter(data, x_col=x_col, y_col=y_col, hue_col=hue_col)
            return path, description
        elif category == "Distribution Plot":
            target_col = None
            col_specs = re.findall(r"(\w+)\s*=\s*([\w\s`'.]+?)(?:,|$)", plan)  # Allow quoted/backticked/spaced names
            for key, val in col_specs:
                key_lower = key.lower()
                # Remove potential quotes/backticks from column name
                val_stripped = val.strip().strip('`\'"')
                if key_lower == 'target':
                    target_col = val_stripped
            path, description = self.generate_distribution(data, hist_col=target_col)
            return path, description
        elif category == "Box Plot":
            target_col = None
            col_specs = re.findall(r"(\w+)\s*=\s*([\w\s`'.]+?)(?:,|$)", plan)  # Allow quoted/backticked/spaced names
            for key, val in col_specs:
                key_lower = key.lower()
                # Remove potential quotes/backticks from column name
                val_stripped = val.strip().strip('`\'"')
                if key_lower == 'target':
                    target_col = val_stripped
            path, description = self.generate_boxplot(data, box_col=target_col)
            return path, description
        elif category == "Bar Plot":
            target_col = None
            col_specs = re.findall(r"(\w+)\s*=\s*([\w\s`'.]+?)(?:,|$)", plan)  # Allow quoted/backticked/spaced names
            for key, val in col_specs:
                key_lower = key.lower()
                # Remove potential quotes/backticks from column name
                val_stripped = val.strip().strip('`\'"')
                if key_lower == 'target':
                    target_col = val_stripped
            path, description = self.generate_bar(data, cat_col=target_col)
            return path, description
        elif category == "Correlation Heatmap":
            path, description = self.generate_correlation_heatmap(data)
            return path, description
        elif category == "Unclear":
            return None, "I'm sorry, I couldn't understand what type of plot you're asking for. Please rephrase your request."
        else:
            return None, f"I recognized the category '{category}', but I don't know how to generate it yet."

    def generate_all(self, data: pd.DataFrame) -> dict:
        """Generates all standard plots, returning dict of {type: (path, cols)}."""
        results = {}
        if data is None or data.empty:
            empty_msg = "DataFrame is empty."
            return {
                'scatter': (empty_msg, {}),
                'distribution': (empty_msg, {}),
                'boxplot': (empty_msg, {}),
                'correlation_heatmap': (empty_msg, {}),
                'bar': (empty_msg, {})
            }

        plot_data = data.copy()  # Use copy for all plots
        results['scatter'] = self.generate_scatter(plot_data.copy())  # Pass copy to each
        results['distribution'] = self.generate_distribution(plot_data.copy())
        results['boxplot'] = self.generate_boxplot(plot_data.copy())
        results['correlation_heatmap'] = self.generate_correlation_heatmap(plot_data.copy())  # Pass copy
        results['bar'] = self.generate_bar(plot_data.copy())

        # Filter out None results if necessary, though errors are returned as strings
        return results
