# main.py
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from agents.report_generation_agent import ReportGenerationAgent
from models.llm_handler import LLMHandler  # Adjust based on your setup

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['REPORT_FOLDER'] = 'data/reports'
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
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'message': 'No file uploaded or selected'}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    global data
    try:
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            data = pd.read_excel(filepath)
        else:
            return jsonify({'message': 'Unsupported file format (use CSV or XLSX)'}), 400
    except Exception as e:
        return jsonify({'message': f'Error reading file: {str(e)}'}), 500

    return jsonify({'message': f'File "{file.filename}" uploaded successfully. Ask me about it or generate a report!'})

@app.route('/chat', methods=['POST'])
def chat():
    if data is None:
        return jsonify({'response': 'Please upload a data file first.'})

    message = request.json.get('message', '').strip()
    if not message:
        return jsonify({'response': 'Please enter a question.'})

    prompt = f"Analyze this data:\n{data.head().to_string()}\n\nUser question: {message}"
    try:
        response = llm.get_completion(prompt, max_tokens=1000)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error processing question: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if data is None:
        return jsonify({'response': 'Please upload a data file first.'})

    try:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        eda_results = {
            'stats': {
                'describe': data.describe(),
                'missing_values': data.isnull().sum().to_dict()
            },
            'correlations': numeric_data.corr() if not numeric_data.empty else "No numeric columns available"
        }
        viz_paths = {  # Placeholder; replace with actual visualization logic if available
            'temperature_vs_has_children_scatter': 'temperature_vs_has_children_scatter.png',
            'temperature_distribution': 'temperature_distribution.png',
            'correlation_heatmap': 'correlation_heatmap.png'
        }
        insight = "Initial insight placeholder"  # Overridden by LLM in report_agent
        
        report_paths = report_agent.generate_report(data, eda_results, insight, viz_paths)
        
        # Handle dict or string return from report_agent
        html_filename = os.path.basename(report_paths.get('html_path', 'report.html') if isinstance(report_paths, dict) else 'report.html')
        pdf_filename = os.path.basename(report_paths.get('pdf_path', 'analysis_report.pdf') if isinstance(report_paths, dict) else report_paths)
        
        response = (
            f"Report generated successfully! "
            f"<a href='/reports/{html_filename}' class='view-link' target='_blank'>View HTML Report</a> | "
            f"<a href='/download/{pdf_filename}' class='download-link' target='_blank'>Download PDF Report</a>"
        )
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error generating report: {str(e)}'}), 500

@app.route('/reports/<filename>', methods=['GET'])
def view_report(filename):
    """Serve HTML files for viewing in the browser."""
    return send_from_directory(app.config['REPORT_FOLDER'], filename, mimetype='text/html')

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Serve files (e.g., PDF) as downloads."""
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)