import yaml
import pdfplumber
from pathlib import Path 
import os 

ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"

with open(CONFIG_FILE, "r") as file:
    CONFIG = yaml.safe_load(file)

FILE_FOLDER = ROOT_DIR / CONFIG["data"]
list_of_files = os.listdir(FILE_FOLDER)

file = FILE_FOLDER / list_of_files[0]

with pdfplumber.open(file) as pdf:
    first_page = pdf.pages[7]
    text = first_page.extract_text(x_tolerance=2)

print(text)