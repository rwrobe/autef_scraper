"""
Rotate PDF first page's files 90 degree clockwise
"""

from os import listdir
from datetime import datetime
import shutil

from tqdm import tqdm
import PyPDF2

input_dir = ".\\pdfs\\initial\\"
output_dir = ".\\pdfs\\rotated\\"
done_dir = ".\\pdfs\\scraped\\"
error_dir = ".\\pdfs\\error\\"


def main():
    start_time = datetime.now()
    print("Script started:", start_time)

    rotate_pdf(input_dir, output_dir, done_dir, error_dir)

    finished_time = datetime.now()
    print("Script took:", finished_time - start_time, "to run")


def rotate_pdf(i, o, d, e):
    for x in tqdm(listdir(i)):
        if not x.endswith(".pdf"):
            continue
        with open(i + x, "rb") as pdf_in:
            try:
                pdf_reader = PyPDF2.PdfFileReader(pdf_in)
            except EOFError:
                print("Could not open:", x)
                shutil.move(i + x, e + x)
                pass
            else:
                pdf_reader.decrypt("")
                page_one = pdf_reader.getPage(0)
                page_one.rotateClockwise(90)
                pdf_writer = PyPDF2.PdfFileWriter()
                pdf_writer.addPage(page_one)
                with open(o + x, 'wb') as pdf_out:
                    pdf_writer.write(pdf_out)
        shutil.move(i + x, d + x)


if __name__ == "__main__":
    main()
