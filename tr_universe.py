"""
Liest die offizielle Liste aller bei Trade Republic handelbaren Aktien.

Quelle: https://assets.traderepublic.com/assets/files/DE/Instrument_Universe_DE_de.pdf
(wird von Trade Republic täglich aktualisiert)

    .venv/bin/python -I tr_universe.py            # PDF neu laden und auswerten
    .venv/bin/python -I tr_universe.py --offline  # vorhandene PDF auswerten

Ergebnis: data/tr_aktien.csv  (isin, name)
"""
import csv
import os
import re
import sys
import urllib.request

URL = "https://assets.traderepublic.com/assets/files/DE/Instrument_Universe_DE_de.pdf"
HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "data", "tr", "Instrument_Universe_DE_de.pdf")
CSV = os.path.join(HERE, "data", "tr_aktien.csv")
ISIN = re.compile(r"^([A-Z]{2}[A-Z0-9]{9}\d)\s+(.*)$")


def download():
    os.makedirs(os.path.dirname(PDF), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(PDF, "wb") as f:
        f.write(r.read())


def parse():
    from pypdf import PdfReader

    section, rows, sections = None, {}, {}
    for page in PdfReader(PDF).pages:
        lines = [l.strip() for l in (page.extract_text() or "").splitlines()]
        for i, line in enumerate(lines):
            if line.startswith("HANDELSUNIVERSUM -") and i + 1 < len(lines):
                section = (line.replace("HANDELSUNIVERSUM -", "").strip() or lines[i + 1]).strip()
                continue
            m = ISIN.match(line)
            if m:
                sections[section] = sections.get(section, 0) + 1
                if section == "Aktien":
                    rows[m.group(1)] = m.group(2).strip()
    return rows, sections


def load():
    """→ {isin: name} aus der zuletzt erzeugten CSV."""
    with open(CSV, newline="", encoding="utf-8") as f:
        return {r["isin"]: r["name"] for r in csv.DictReader(f)}


def main():
    if "--offline" not in sys.argv:
        download()
    rows, sections = parse()
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["isin", "name"])
        w.writerows(sorted(rows.items()))
    print("Abschnitte:", sections)
    print(f"{len(rows)} Aktien → {os.path.relpath(CSV, HERE)}")
    top = {}
    for isin in rows:
        top[isin[:2]] = top.get(isin[:2], 0) + 1
    print("Nach Land:", sorted(top.items(), key=lambda x: -x[1])[:15])


if __name__ == "__main__":
    main()
