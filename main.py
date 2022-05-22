import os
from datetime import datetime

from src.scraper import Scraper

if __name__ == '__main__':
    scraper = Scraper()
    start_time = datetime.now()
    print("Script started at:", start_time)
    dirname = os.path.dirname(__file__)

    input_dir = dirname + "/pdfs/raw/"
    html_dir = dirname + "/pdfs/html/"
    output_dir = dirname + "/pdfs/done/"

    #scraper.pdfs_to_html(input_dir, html_dir)
    data_frame = scraper.scrape_cover_html_files(html_dir, output_dir)

    print("Script took:", datetime.now() - start_time, "to run.")
