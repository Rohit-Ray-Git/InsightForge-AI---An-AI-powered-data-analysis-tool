# InsightForge AI - An AI-Powered Data Analysis Tool

InsightForge AI is a comprehensive data analysis platform that leverages advanced AI capabilities to provide actionable insights, visualizations, and reports. Designed for business stakeholders and data professionals, this tool simplifies the process of exploring, cleaning, analyzing, and visualizing data from various sources, including files and databases.

---

## Features

### 1. **Data Ingestion**
- Supports uploading CSV files or connecting to MySQL databases.
- Automatically detects and converts data types for seamless analysis.

### 2. **Exploratory Data Analysis (EDA)**
- Generates detailed reports with:
  - Data shape and column types.
  - Summary statistics for numeric columns.
  - Insights into categorical columns, including top values and unique counts.
  - Missing value analysis.

### 3. **Data Cleaning**
- Handles missing values and outliers using intelligent techniques.
- Provides both technical and layman summaries of cleaning operations.

### 4. **Data Visualization**
- Generates various visualizations, including:
  - Scatter plots.
  - Bar charts.
  - Correlation heatmaps.
  - Boxplots.
- Automatically determines the best visualization based on user prompts.

### 5. **AI-Powered Insights**
- Uses large language models (LLMs) to:
  - Summarize data findings.
  - Generate actionable insights.
  - Provide recommendations based on data trends.

### 6. **Customizable Reports**
- Creates professional, structured reports in HTML format.
- Includes sections for EDA, insights, visualizations, and conclusions.

### 7. **Web Research**
- Integrates web scraping capabilities to fetch external data for comparative analysis.

---

### Key Directories and Files

#### **`main.py`**
The entry point of the application. It initializes agents, handles user interactions via Streamlit, and orchestrates data analysis workflows.

#### **`agents/`**
Contains specialized agents for handling different tasks:
- **`analysis_agent.py`**: Performs exploratory data analysis (EDA) and generates insights.
- **`content_generation_agent.py`**: Synthesizes content using LLMs.
- **`data_cleaning_agent.py`**: Cleans data by handling missing values and outliers.
- **`data_ingestion_agent.py`**: Manages data ingestion from files or databases.
- **`database_agent.py`**: Interacts with MySQL databases to fetch schema and data.
- **`report_generation_agent.py`**: Generates professional reports based on analysis.
- **`storage_agent.py`**: Manages file storage for reports and visualizations.
- **`visualization_agent.py`**: Creates visualizations based on user prompts.

#### **`config/`**
Holds configuration files:
- **`logging_config.py`**: Configures logging for debugging and monitoring.
- **`settings.py`**: Stores application settings and constants.

#### **`data/`**
Stores uploaded files, generated reports, and other data artifacts.

#### **`templates/`**
Contains HTML templates for generating reports.

#### **`tests/`**
Includes unit tests to ensure the reliability of the application.

#### **`utils/`**
Provides utility functions for common operations.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/InsightForge-AI.git
   cd InsightForge-AI
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```bash
   export GOOGLE_API_KEY=your_google_api_key
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage
- **`Upload a File or Connect to a Database`**: Upload a CSV file or connect to a MySQL database to start analysis.
- **`Perform Operations`**: Choose from operations like data cleaning, visualization, or detailed analysis.
- **`Generate Reports`**: View and download professional reports summarizing the analysis.
- **`Interact with AI`**: Use the chat interface to ask questions or request specific actions.

---

## Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** Interactive user interface.
*   **Pandas:** Data manipulation and analysis.
*   **Matplotlib & Seaborn:** Data visualization.
*   **LangChain:** Integration with large language models.
*   **MySQL:** Database support.
*   **Google Generative AI:** For AI-powered insights and content generation.

---

## Contributing

Contributions to InsightForge AI are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear messages.
4.  Push your changes to your fork.
5.  Submit a pull request to the main repository.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Contact

For any questions or feedback, please contact me at rayrohit685@gmail.com.
