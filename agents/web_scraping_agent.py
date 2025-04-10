# Example within agents/web_scraping_agent.py
from serpapi import GoogleSearch
import os
import logging # Add logging

class WebScrapingAgent:
    def research(self, query: str, num_results: int = 5) -> str:
        logging.info(f"Web Agent: Starting research for query: '{query}'")
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            logging.error("Web Agent Error: SERPAPI_KEY environment variable not found.")
            return "Error: SerpAPI key is not configured. Cannot perform web search."

        try:
            params = {
                "api_key": api_key,
                "engine": "google",
                "q": query,
                "num": num_results
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            if 'error' in results:
                logging.error(f"Web Agent Error: SerpAPI returned an error: {results['error']}")
                return f"Error: Web search failed. (Details: {results['error']})"

            snippets = []
            if 'organic_results' in results:
                for result in results['organic_results']:
                    if 'snippet' in result:
                        snippets.append(result['snippet'])
                    # Optional: Also grab answer box snippets etc. if needed
                    # elif 'answer_box' in result and 'snippet' in result['answer_box']:
                    #    snippets.append(result['answer_box']['snippet'])

            if not snippets:
                logging.warning(f"Web Agent: No relevant snippets found for query '{query}'.")
                return "Web search completed, but no relevant text snippets were found."

            # Combine and provide raw snippets for the main LLM to synthesize
            combined_snippets = "\n\n".join(f"- {s}" for s in snippets)
            logging.info(f"Web Agent: Found {len(snippets)} snippets for query '{query}'.")
            # Return the raw findings for the main LLM to process
            return f"Web Search Findings for '{query}':\n{combined_snippets}"

        except Exception as e:
            logging.error(f"Web Agent Error: An unexpected error occurred during SerpAPI call: {e}", exc_info=True)
            return f"Error: An unexpected error occurred during web search: {e}"

