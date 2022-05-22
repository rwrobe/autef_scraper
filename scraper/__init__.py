from datetime import datetime


def main():
    start_time = datetime.now()
    print("Script started at:", start_time)

    input_dir = f".\\pdfs\\raw\\"
    html_dir = f".\\pdfs\\html\\"
    output_dir = f".\\pdfs\\scraped\\"

    autef = Autef(input_dir, rot_dir, output_dir)
    autef.run()

    print("Script took:", datetime.now() - start_time, "to run.")