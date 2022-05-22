import csv
import glob
import json
import re

from PDFNetPython3 import *
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd


class Scraper:
    # HOW TO:
    # STEP 1: Add column names as constants here.
    STATUS = "autef_status"
    AUTEF_NUM = "autef_num"
    COR_GEO = "coordin_geo"

    def __init__(self):
        self.df = None
        self.columns = []
        self.rows = []

    # Convert the PDFs to HTML, which should be easier to crawl.
    @staticmethod
    def pdfs_to_html(input_dir, html_dir, error_dir):
        errors = []
        PDFNet.Initialize("demo:1653215173602:7b8663780300000000158a6730bd31c4ffd5b0091160797d38327d8f76")

        for file in sorted(glob.glob(os.path.join(input_dir, "*.pdf"))):
            print(file)
            filename = Path(os.path.basename(file)).stem.replace(' ', '_').lower()

            # Convert PDF document to HTML with fixed positioning option turned on (default)
            try:
                Convert.ToHtml(file, html_dir + filename)
            except:
                errors.append(filename)
                continue

        json_string = json.dumps(errors)
        json_file = open(error_dir + "last_run_errors.json", "w")
        json_file.write(json_string)
        json_file.close()

    def scrape_cover_html_files(self, html_dir, output_dir):
        i = 0
        for html_file in sorted(glob.glob(os.path.join(html_dir, "**/cover.xhtml"))):
            with open(html_file) as hf_buffer:
                self.rows.append(self.scrape_html_file(hf_buffer.read()))
            if i > 10:
                break
            i += 1

        df = pd.DataFrame(self.rows, columns=self.columns)

        return df.to_csv(output_dir + "final.csv", sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)

    def scrape_html_file(self, raw_html):
        values = []
        soup = BeautifulSoup(raw_html, 'html.parser')

        # STEP 2: Parse column-by-column.
        # Find the div or span or whatever that contains the text.
        # Repeat this stuff for every one of the fields:
        side_text_div = soup.find(id="TextContainer1").find('span')

        new_value = self.add_cols_and_values(self.STATUS, side_text_div)

        values.append(new_value)

        # <---- END REPEAT ---->

        # Autef num.

        autef_num_span = soup.findAll('span')[2]

        new_value = self.add_cols_and_values(self.AUTEF_NUM, autef_num_span)

        values.append(new_value)

        # Coord. geo.

        coor_geo_span_hr = soup.find('span', string=re.compile('^COORDENADAS'))
        if coor_geo_span_hr:
            coor_geo_span = coor_geo_span_hr.next_sibling

            new_value = self.add_cols_and_values(self.COR_GEO, coor_geo_span)

            values.append(new_value)

        return values

    def add_cols_and_values(self, col_name, div_containing_txt):
        values = []

        # Add a column to the list using the constant in STEP 1:
        if col_name not in self.columns:
            self.columns.append(col_name)
        # Get the text.
        txt = div_containing_txt.text

        if txt:  # If found.
            return txt.strip()
        else:  # If not found.
            return ""
