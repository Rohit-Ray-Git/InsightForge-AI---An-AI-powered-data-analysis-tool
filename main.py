# main.py
import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import markdown
from agents.report_generation_agent import ReportGenerationAgent
from models.llm_handler import LLMHandler  # Adjust based on your setup

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['REPORT_FOLDER'] = 'data/reports'  # Define report folder for serving files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

llm = LLMHandler()
report_agent = ReportGenerationAgent()
data = None  # Global variable to store uploaded data

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    global data
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        data = pd.read_excel(filepath)
    else:
        return jsonify({'message': 'Unsupported file format'}), 400

    return jsonify({'message': f'File {file.filename} uploaded successfully. Ask me about it!'})

@app.route('/report/<filename>')
def serve_report(filename):
    # Serve files from the report folder
    file_path = os.path.join(app.config['REPORT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    if filename.endswith('.html'):
        return send_file(file_path, mimetype='text/html')
    elif filename.endswith('.pdf'):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    if data is None:
        return jsonify({'response': 'Please upload a data file first.'})

    if "generate report" in message.lower():
        eda_results = {
            'stats': {'describe': data.describe(), 'missing_values': data.isnull().sum().to_dict()},
            'correlations': data.corr()
        }
        viz_paths = {
            'temperature_vs_has_children_scatter': 'temperature_vs_has_children_scatter.png',
            'temperature_distribution': 'temperature_distribution.png',
            'correlation_heatmap': 'correlation_heatmap.png'
        }
        insight = "Initial insight placeholder"
        # Get both HTML and PDF paths
        report_paths = report_agent.generate_report(data, eda_results, insight, viz_paths)
        html_url = f"/report/report.html"
        pdf_url = f"/report/analysis_report.pdf"
        response = (
            f'Report generated! '
            f'<a href="{html_url}" target="_blank">View HTML Report</a> | '
            f'<a href="{pdf_url}" download>Download PDF Report</a>'
        )
        return jsonify({'response': response})

    # Handle simple queries directly
    message_lower = message.lower()
    if "how many rows" in message_lower:
        row_count = len(data)
        return jsonify({'response': f'There are {row_count} rows in the dataset.'})
    elif "how many columns" in message_lower:
        col_count = len(data.columns)
        return jsonify({'response': f'There are {col_count} columns in the dataset.'})
    elif "what are the columns" in message_lower:
        columns = ", ".join(data.columns)
        return jsonify({'response': f'The columns are: {columns}'})

    # Fallback to LLM for other queries
    prompt = f"Analyze this data:\n{data.head().to_string()}\n\nUser question: {message}"
    print(f"Prompt sent to LLM: {prompt}")
    try:
        response = llm.get_completion(prompt, max_tokens=1000)
        print(f"Raw LLM response: {response}")
        response_html = markdown.markdown(response)
        print(f"Converted HTML response: {response_html}")
    except Exception as e:
        print(f"LLM error: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error processing your request.'})
    return jsonify({'response': response_html})

if __name__ == '__main__':
    app.run(debug=True)