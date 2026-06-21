import yaml
import pdfplumber
from pathlib import Path
import os
from openai import OpenAI
from pydantic import BaseModel
import json

ROOT_DIR = Path(__file__).parents[1]
CONFIG_FILE = ROOT_DIR / "config.yaml"
PROMPT_FILE = "prompts/extract_prompt.txt"
PROCESSED_DATA = "processed"
NUM_PAGE = 5

os.makedirs(PROCESSED_DATA, exist_ok=True)

with open(CONFIG_FILE, "r") as file:
    CONFIG = yaml.safe_load(file)

with open(PROMPT_FILE, "r") as file:
    PROMPT = file.read()

FILE_FOLDER = ROOT_DIR / CONFIG["data"]

CLIENT = OpenAI(base_url="http://127.0.0.1:8080", api_key="Not required")


class SectionHeaders(BaseModel):
    section_headers: list[str] | str | dict | None


def extract_section_info(
    text: str, prompt: str, CLIENT: OpenAI = CLIENT
) -> SectionHeaders:
    response = CLIENT.chat.completions.create(
        model="unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\nPage:\n{text}", "think": False},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SectionHeaders.model_json_schema(),
        },
    )
    content = response.choices[0].message.content
    content = content.replace("```json", "")
    content = content.replace("```", "")

    return SectionHeaders.model_validate_json(content)


list_of_files = os.listdir(FILE_FOLDER)

# file = FILE_FOLDER / list_of_files[0]
year_section_details = {}

for file in list_of_files:
    year = int(file[:4])
    year_section_details.update({year: []})

    print("Current Year", year)
    file = FILE_FOLDER / file
    with pdfplumber.open(file) as pdf:
        pages = pdf.pages

        total_pages = len(pages)

        current_page = 0

        while current_page < total_pages:
            print("Current Page :{}, Num Pages {}".format(current_page, current_page + NUM_PAGE))
            current_pages = pages[current_page : current_page + NUM_PAGE]

            text = " ".join(
                [page.extract_text(x_tolerance=2) for page in current_pages]
            )

            sections = extract_section_info(text, PROMPT)
            
            year_section_details[year].extend(sections)

            print("Sections", sections)
            current_page += NUM_PAGE
            
    print("Done : ", file)
    print("\n")

with open(f"{PROCESSED_DATA}/sections.json", "w") as file:
    json.dump(year_section_details, file)

print("Saved the results at ", f"{PROCESSED_DATA}/sections.json")