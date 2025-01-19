import functions_framework
import os
import json
import logging
from datetime import datetime
from scraper import scrape_all
from hdfs import ensure_hdfs_folder, write_to_hdfs_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hdfs_host = "http://34.67.32.69:50070/"
folder_path = "/user/adam_majczyk2001/nifi/bronze/news/"

@functions_framework.http
def main(request):
    try:
        logger.info("SCRAPING STARTED...")
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Datetime: {current_time}")
        
        request_json = request.get_json(silent=True)
        request_args = request.args

        logger.info("Loading site variables configuration.")
        with open("variables_dict.json", "r", encoding="utf-8") as f:
            site_variables_dict = json.load(f)

        logger.info("Scraping articles...")
        results = scrape_all(site_variables_dict, pages=1)

        # Logging results summary
        total_articles = len(results)
        logger.info(f"Scraped {total_articles} articles.")

        # Log article count per source
        source_counts = {}
        for result in results.values():
            source = result["source_site"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            logger.info(f"{count} articles for source {source}")

        # Save results to HDFS
        ensure_hdfs_folder(hdfs_host=hdfs_host, folder_path=folder_path)
        file_name = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        logger.info(f"Saving results to {folder_path}/{file_name}")

        write_to_hdfs_json(hdfs_host, folder_path, file_name, results)
        logger.info("Data saved successfully.")
        
        return "success"
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return f"Error: {e}", 500