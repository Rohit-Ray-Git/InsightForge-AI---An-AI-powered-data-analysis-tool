# agents/data_cleaning_agent.py
import pandas as pd
import numpy as np
import logging
import os
# No LLM needed for this version

# Configure logging for this module if needed, or rely on main.py's config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaningAgent:
    # No __init__ needed if not using LLM

    def clean_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame | None, str, str]:
        """
        Clean the provided DataFrame using standard techniques (median/mode imputation, IQR outlier capping).
        Generates both a technical log and a simplified explanation of the steps taken.

        Args:
            data (pd.DataFrame): The input DataFrame to clean.

        Returns:
            tuple[pd.DataFrame | None, str, str]: A tuple containing:
                - The cleaned DataFrame (or the original DataFrame if errors occurred).
                - A string summarizing the technical cleaning steps performed.
                - A string providing a simplified, layman's explanation of the steps.
                  Returns (None, "Error message", "Error explanation") if input data is invalid.
        """
        if data is None or data.empty:
            logging.warning("DataCleaningAgent: No data provided for cleaning.")
            return None, "No data provided for cleaning.", "Cannot clean data because no data was given."

        cleaned_data = data.copy()
        technical_log = []  # Log for technical details
        layman_log = []     # Log for simplified explanations

        logging.info("Starting data cleaning process...")

        try:
            # --- 1. Missing Value Handling ---
            technical_log.append("**1. Missing Value Handling:**")
            layman_log.append("**1. Filling Empty Spots:**")
            numeric_cols = cleaned_data.select_dtypes(include=np.number).columns
            categorical_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns
            any_missing_handled = False

            for col in cleaned_data.columns:
                missing_count = cleaned_data[col].isnull().sum()
                if missing_count > 0:
                    any_missing_handled = True
                    percentage_missing = (missing_count / len(cleaned_data)) * 100
                    tech_prefix = f"*   **`{col}`:** Found {missing_count} missing value(s) ({percentage_missing:.1f}%)."
                    layman_prefix = f"*   **'{col}' column:** Had {missing_count} empty spot(s)."

                    if col in numeric_cols:
                        if cleaned_data[col].isnull().all():
                            median_val = 0 # Default for all-NaN numeric
                            technical_log.append(f"{tech_prefix} All values were missing. Imputed with default ({median_val}).")
                            layman_log.append(f"{layman_prefix} Since the whole column was empty, we filled it with 0.")
                        else:
                            median_val = cleaned_data[col].median()
                            technical_log.append(f"{tech_prefix} Imputed with median ({median_val:.2f}).")
                            layman_log.append(f"{layman_prefix} We filled these with the typical middle value ({median_val:.2f}) found in that column to keep things consistent.")
                        cleaned_data[col].fillna(median_val, inplace=True)

                    elif col in categorical_cols:
                        if cleaned_data[col].isnull().all():
                            mode_val = "Unknown" # Default for all-NaN categorical
                            technical_log.append(f"{tech_prefix} All values were missing. Imputed with default ('{mode_val}').")
                            layman_log.append(f"{layman_prefix} Since the whole column was empty, we filled it with 'Unknown'.")
                        elif not cleaned_data[col].mode().empty:
                            mode_val = cleaned_data[col].mode()[0]
                            technical_log.append(f"{tech_prefix} Imputed with mode ('{mode_val}').")
                            layman_log.append(f"{layman_prefix} We filled these with the most common entry ('{mode_val}') found in that column.")
                            cleaned_data[col].fillna(mode_val, inplace=True)
                        else:
                            technical_log.append(f"{tech_prefix} Could not determine mode. No imputation applied.")
                            layman_log.append(f"{layman_prefix} We couldn't figure out the most common entry, so these spots were left empty.")

                    else:
                        technical_log.append(f"{tech_prefix} Data type not automatically handled for imputation. No imputation applied.")
                        layman_log.append(f"{layman_prefix} These weren't standard numbers or text, so we didn't automatically fill them.")

            if not any_missing_handled:
                 technical_log.append("   * No missing values found in any column.")
                 layman_log.append("   * Good news! No empty spots were found in the data.")


            # --- 2. Outlier Handling (IQR Method Capping for numeric columns) ---
            technical_log.append("\n**2. Outlier Handling (IQR Method - Capping):**")
            layman_log.append("\n**2. Adjusting Extreme Values:**")
            numeric_cols_after_imputation = cleaned_data.select_dtypes(include=np.number).columns
            any_outliers_handled = False

            for col in numeric_cols_after_imputation:
                # Skip if column has no variance
                if cleaned_data[col].isnull().all() or cleaned_data[col].nunique(dropna=False) <= 1:
                    technical_log.append(f"*   **`{col}`:** Skipped outlier detection (all NaN, single unique value, or no variance).")
                    # No layman log needed for skipped technical steps unless desired
                    continue

                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0:
                     technical_log.append(f"*   **`{col}`:** Skipped outlier detection (IQR is zero).")
                     continue

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                outlier_count = outliers_mask.sum()

                if outlier_count > 0:
                    any_outliers_handled = True
                    original_min = cleaned_data.loc[~outliers_mask, col].min() # Min of non-outliers
                    original_max = cleaned_data.loc[~outliers_mask, col].max() # Max of non-outliers
                    
                    # Cap outliers
                    cleaned_data[col] = np.where(cleaned_data[col] < lower_bound, lower_bound, cleaned_data[col])
                    cleaned_data[col] = np.where(cleaned_data[col] > upper_bound, upper_bound, cleaned_data[col])
                    capped_min = cleaned_data[col].min()
                    capped_max = cleaned_data[col].max()

                    technical_log.append(
                        f"*   **`{col}`:** Capped {outlier_count} outlier(s) outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]. "
                        f"Original non-outlier range approx: [{original_min:.2f}, {original_max:.2f}]. New range: [{capped_min:.2f}, {capped_max:.2f}]."
                    )
                    layman_log.append(
                        f"*   **'{col}' column:** Found {outlier_count} value(s) that seemed unusually high or low compared to the typical range (around {original_min:.2f} to {original_max:.2f}). "
                        f"We adjusted these extreme values to be no lower than {lower_bound:.2f} and no higher than {upper_bound:.2f} to prevent them from skewing results."
                    )
                else:
                    technical_log.append(f"*   **`{col}`:** No outliers detected by IQR method.")
                    # Only add layman log if you want to explicitly state nothing was done
                    # layman_log.append(f"*   **'{col}' column:** The numbers looked consistent; no extreme values needed adjustment.")

            if not any_outliers_handled:
                 technical_log.append("   * No outliers detected or capped in any numeric column.")
                 layman_log.append("   * The numbers in the data looked consistent; no extreme values needed adjustment.")


            # --- Combine log messages ---
            technical_summary = "\n".join(technical_log)
            layman_summary = "\n".join(layman_log)

            # Check if any actual cleaning steps were logged
            if not any_missing_handled and not any_outliers_handled:
                no_action_msg = "No specific cleaning actions were performed (data might already be clean based on implemented rules)."
                technical_summary = no_action_msg
                layman_summary = "The data looked pretty clean already, so no automatic adjustments were needed based on our standard checks."

            logging.info("Data cleaning process completed successfully.")
            return cleaned_data, technical_summary, layman_summary

        except Exception as e:
            logging.error(f"Error during data cleaning process: {e}", exc_info=True)
            # Return original data and error messages for both logs
            error_msg = f"An critical error occurred during cleaning: {e}. Returning original data."
            technical_error_summary = "\n".join(technical_log) + f"\n\n**ERROR:** {error_msg}"
            layman_error_summary = "\n".join(layman_log) + f"\n\n**Problem:** Something went wrong while trying to clean the data ({e}). We've stopped the cleaning process and are showing the original data."
            return data, technical_error_summary, layman_error_summary
