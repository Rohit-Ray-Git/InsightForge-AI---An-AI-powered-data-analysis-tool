# main.py
from config.settings import settings
from agents.data_ingestion_agent import ingest_data
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_generation_agent import ReportGenerationAgent

def main():
    """Main function to run the InsightForge AI pipeline."""
    try:
        # Ingest data from a sample file
        data = ingest_data(settings.INPUT_FILE_PATH, settings.INPUT_FILE_TYPE)
        print("Data ingested successfully!")
        print(data.head())

        # Analyze the data
        analyzer = AnalysisAgent(data)
        eda_results, insight = analyzer.analyze()
        
        print("\nEDA Results:")
        print("Statistics:")
        print(eda_results['stats']['describe'])
        print("\nMissing Values:")
        print(eda_results['stats']['missing_values'])
        print("\nCorrelations:")
        print(eda_results['correlations'])
        
        print("\nAI Insight:")
        print(insight)

        # Generate visualizations automatically
        visualizer = VisualizationAgent(data)
        viz_paths = visualizer.generate_visualizations()
        print("\nVisualization Files Generated:")
        for name, path in viz_paths.items():
            if path:
                print(f"{name}: {path}")

        # Generate report
        reporter = ReportGenerationAgent()
        report_path = reporter.generate_report(data, eda_results, insight, viz_paths)
        print(f"\nReport Generated: {report_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()