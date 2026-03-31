import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag
import unicodedata

import os
from pathlib import Path
# Set working directory to project root (parent of eda folder)
os.chdir(Path(__file__).parent.parent)

BASE_URL = "https://www.cpc.ncep.noaa.gov/products/CDB/CDB_Archive_html/bulletin_{month_year}/Forecast/forecast.shtml"
YEARS = range(2018, 2026)
MONTH = "10"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def get_html(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def find_marker_tag(soup: BeautifulSoup, marker: str):
    for tag in soup.find_all(True):
        text = normalize_text(tag.get_text(" ", strip=True))
        if text.startswith(marker):
            return tag
    return None


def extract_outlook_section(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    outlook_tag = find_marker_tag(soup, "Outlook:")
    discussion_tag = find_marker_tag(soup, "Discussion:")

    if not outlook_tag:
        return ""

    collected = []
    started = False

    for tag in soup.find_all(True):
        if tag == outlook_tag:
            started = True

        if not started:
            continue

        if discussion_tag is not None and tag == discussion_tag:
            break

        text = normalize_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        collected.append(text)

    if not collected:
        return ""

    outlook_text = " ".join(collected)
    outlook_text = normalize_text(outlook_text)

    # Remove the Outlook header itself
    if outlook_text.startswith("Outlook:"):
        outlook_text = normalize_text(outlook_text[len("Outlook:"):])

    # Remove anything after Discussion if it slipped in
    if "Discussion:" in outlook_text:
        outlook_text = normalize_text(outlook_text.split("Discussion:")[0])

    return outlook_text


def classify_enso(outlook_text: str) -> str:
    text_ascii = strip_accents(outlook_text).lower()

    if "enso-neutral" in text_ascii or "enso neutral" in text_ascii:
        return "ENSO-Neutral"
    if "la nina" in text_ascii:
        return "La Niña"
    if "el nino" in text_ascii:
        return "El Niño"

    return "Unknown"


def scrape():
    rows = []

    for year in YEARS:
        month_year = f"{MONTH}{year}"
        url = BASE_URL.format(month_year=month_year)

        try:
            html = get_html(url)
            outlook_text = extract_outlook_section(html)
            classification = classify_enso(outlook_text)

            rows.append({
                "year": year,
                "enso_outlook": classification,
                "outlook_text": outlook_text,
            })

            print(f"{year}: {classification}")
            print(outlook_text)
            print("-" * 80)

        except Exception as e:
            rows.append({
                "year": year,
                "enso_outlook": "ERROR",
                "outlook_text": "",
            })
            print(f"{year}: ERROR - {e}")

    df = pd.DataFrame(rows)
    print(df[["year", "enso_outlook"]].to_string(index=False))
    df.to_csv("data/cleaned_data/el_nino_la_nina_outlook_october.csv", index=False)


if __name__ == "__main__":
    scrape()