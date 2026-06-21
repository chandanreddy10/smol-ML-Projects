import yaml
import pdfplumber
from pathlib import Path
import os
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
import json
from dotenv import load_dotenv

load_dotenv("../secrets.env")
groq_api_key = os.getenv("GROQ_API_KEY")

ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"
PROMPT_FILE = "prompts/extract_prompt.txt"
PROCESSED_DATA = "processed"
NUM_PAGE = 2

os.makedirs(PROCESSED_DATA, exist_ok=True)

with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)

with open(PROMPT_FILE, "r") as f:
    PROMPT = f.read()

FILE_FOLDER = ROOT_DIR / CONFIG["data"]

CLIENT = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1/",
)

class SectionHeaders(BaseModel):
    section_headers: list[str]
    model_config = ConfigDict(extra="forbid")


def extract_section_info(text: str, prompt: str) -> SectionHeaders:
    response = CLIENT.beta.chat.completions.parse(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\nPage:\n{text}"},
        ],
        response_format=SectionHeaders,
    )
    return response.choices[0].message.parsed


list_of_files = os.listdir(FILE_FOLDER)
year_section_details = {}

for filename in list_of_files:

    year = int(filename[:4])
    year_section_details.setdefault(year, [])

    print("Current Year:", year)

    pdf_path = FILE_FOLDER / filename

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        total_pages = len(pages)

        current_page = 0

        while current_page < total_pages:
            try:
                print(f"Current Page: {current_page}, Next: {current_page + NUM_PAGE}")

                current_pages = pages[current_page: current_page + NUM_PAGE]

                text = " ".join(
                    (page.extract_text(x_tolerance=2) or "") for page in current_pages
                )

                if not text.strip():
                    current_page += NUM_PAGE
                    continue

                sections = extract_section_info(text, PROMPT)

                year_section_details[year].extend(sections.section_headers)

                print("Sections:", sections.section_headers)

            except Exception as e:
                print("Error:", e)

            current_page += NUM_PAGE

with open(f"{PROCESSED_DATA}/sections.json", "w") as f:
    json.dump(year_section_details, f, indent=2)

print("Saved results at:", f"{PROCESSED_DATA}/sections.json")