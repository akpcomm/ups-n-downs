"""
Ordnet den ISINs der Trade-Republic-Liste die Yahoo-Kürzel zu (über die Yahoo-Suche).

Das Ergebnis wird in data/isin_map.csv gespeichert und nur für neue ISINs ergänzt,
ein späterer Lauf ist deshalb schnell.

    .venv/bin/python isin_map.py            # fehlende ISINs nachschlagen
    .venv/bin/python isin_map.py US DE      # nur diese Länder
"""
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import yfinance as yf

import tr_universe

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, "data", "isin_map.csv")
# Länder, deren Aktien an US-Börsen oder auf Xetra gehandelt werden (in dieser Reihenfolge)
PRIORITY = ["US", "DE", "IE", "NL", "KY", "IL", "BM", "CH", "LU", "JE", "GG", "PA", "MH", "SG",
            "LR", "CW", "AN", "VG", "PR", "BS", "CY", "MU", "GI", "CA", "GB"]


def load():
    """→ {isin: [symbole]}"""
    if not os.path.exists(MAP):
        return {}
    with open(MAP, newline="", encoding="utf-8") as f:
        return {r["isin"]: [s for s in r["symbols"].split("|") if s] for r in csv.DictReader(f)}


def save(m):
    with open(MAP, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["isin", "symbols"])
        for isin in sorted(m):
            w.writerow([isin, "|".join(m[isin])])


def lookup(isin):
    for attempt in range(4):
        try:
            s = yf.Search(isin, max_results=8, news_count=0)
            return isin, [q["symbol"] for q in s.quotes if q.get("quoteType") == "EQUITY" and q.get("symbol")]
        except Exception:
            time.sleep(2 + 4 * attempt)
    return isin, None   # nicht gespeichert → wird beim nächsten Lauf erneut versucht


def main():
    tr = tr_universe.load()
    done = load()
    want = [c.upper() for c in sys.argv[1:]] or PRIORITY
    todo = sorted((i for i in tr if i[:2] in want and i not in done),
                  key=lambda i: (want.index(i[:2]), i))
    print(f"{len(tr)} TR-Aktien, {len(done)} schon zugeordnet, {len(todo)} offen", flush=True)
    fails, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(lookup, i) for i in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            isin, syms = fut.result()
            if syms is None:
                fails += 1
            else:
                done[isin] = syms
            if n % 500 == 0:
                save(done)
                print(f"  {n}/{len(todo)}  ({time.time() - t0:.0f} s, {fails} Fehler)", flush=True)
    save(done)
    hit = sum(1 for v in done.values() if v)
    print(f"Fertig: {len(done)} ISINs gespeichert, davon {hit} mit Yahoo-Kürzel, {fails} Fehler.")


if __name__ == "__main__":
    main()
