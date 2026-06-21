import requests
import os 

OUTPUT_DIR = "files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_url = "https://ir.aboutamazon.com/feed/FinancialReport.svc/GetFinancialReportList"

params = {
    "apiKey": "BF185719B0464B3CB809D23926182246",
    "LanguageId": 1,
    "reportTypes": "Annual Report",
    "reportSubType[]": "Annual Report",
    "reportSubTypeList[]": "Annual Report",
    "pageSize": -1,
    "pageNumber": 0,
    "tagList": "",
    "includeTags": True,
    "year": -1,
    "excludeSelection": 1
}

response = requests.get(base_url, params=params)
data = response.json()

for report_details in data["GetFinancialReportListResult"]:
    report_year = report_details["ReportYear"]
    annual_report  = report_details["Documents"][1]["DocumentPath"]

    if report_year == 2000:
        break
    #Download report and write to a file.
    response = requests.get(annual_report)

    with open(f"{OUTPUT_DIR}/{report_year}_annual_report.pdf", "wb") as f:
        f.write(response.content)

    print("Done with : {}, Year :{}".format(annual_report, report_year))



