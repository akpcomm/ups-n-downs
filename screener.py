"""
Screener: Welche bei Trade Republic handelbaren Aktien schwanken am häufigsten und stärksten?

Ablauf
  1. Liquide Aktien von Yahoo holen (USA und Xetra, nach Börsenwert und Umsatz gefiltert)
  2. Nur behalten, was laut offizieller Trade-Republic-Liste handelbar ist (Abgleich über ISIN)
  3. Für jede Aktie die Swings der letzten 12 Monate zählen und ausmessen
  4. Rangliste nach data/screener.csv schreiben

    .venv/bin/python screener.py            # Rangliste erzeugen
    .venv/bin/python screener.py --check    # zusätzlich prüfen, ob Schwankungsfreude stabil ist

Vorher einmal:  .venv/bin/python -I tr_universe.py   und   .venv/bin/python isin_map.py
"""
import os
import sys
import time
from datetime import date
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as Q

import core
import isin_map
import research_4w
import research_window
import tr_universe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "screener.csv")

SWING = 0.10          # Mindestgröße eines Swings
YEAR = 252            # Handelstage
MIN_DAYS = 100        # Mindest-Historie (darunter keine Aussage)
MIN_PRICE = 5.0       # keine Pennystocks
LOOK = 120           # Rückblick fürs Ranking: 24 Wochen (bester Wert laut research_window.py)
MIN_TURNOVER = 10e6   # mittlerer Tagesumsatz (in Heimatwährung), damit der Spread klein bleibt


def yahoo_universe():
    """Liquide Aktien: USA ab 1 Mrd. $ Börsenwert, Xetra ab 300 Mio. €."""
    specs = [
        ("us", 1e9, 3e5, {"NMS", "NYQ", "NGM", "NCM", "ASE"}),
        ("de", 3e8, 1e4, {"GER"}),
    ]
    rows = {}
    for region, cap, vol, exchanges in specs:
        q = Q("and", [Q("eq", ["region", region]), Q("gt", ["intradaymarketcap", cap]),
                      Q("gt", ["avgdailyvol3m", vol])])
        offset = 0
        while True:
            r = yf.screen(q, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False)
            quotes = r.get("quotes", [])
            for x in quotes:
                if x.get("quoteType") == "EQUITY" and x.get("exchange") in exchanges:
                    rows[x["symbol"]] = {
                        "name_yahoo": x.get("longName") or x.get("shortName") or "",
                        "boersenwert_mrd": (x.get("marketCap") or 0) / 1e9,
                        "waehrung": x.get("currency", ""),
                    }
            offset += len(quotes)
            if not quotes or offset >= r.get("total", 0):
                break
    return rows


def tr_symbols():
    """→ {yahoo_symbol: (isin, tr_name)} für alle zugeordneten Trade-Republic-Aktien."""
    tr, m = tr_universe.load(), isin_map.load()
    out = {}
    for isin, syms in m.items():
        for s in syms:
            out.setdefault(s, (isin, tr.get(isin, "")))
    return out


def download(symbols, period="2y"):
    frames = {}
    symbols = list(symbols)
    for i in range(0, len(symbols), 150):
        chunk = symbols[i:i + 150]
        d = yf.download(chunk, period=period, auto_adjust=True, group_by="ticker",
                        threads=True, progress=False)
        for s in chunk:
            try:
                x = d[s][["Close", "Volume"]].dropna()
            except KeyError:
                continue
            if len(x):
                frames[s] = x
        print(f"  Kurse: {min(i + 150, len(symbols))}/{len(symbols)}", flush=True)
        time.sleep(2)   # Yahoo nicht überlasten
    return frames


def swing_metrics(close: pd.Series) -> dict:
    """Swings eines Zeitfensters: Anzahl (aufs Jahr gerechnet), Größe, Dauer, Pendel-Anteil."""
    piv = core.zigzag(close, SWING)
    pts = [close.iloc[i] for i, _ in piv]
    amps = [abs(b / a - 1) for a, b in zip(pts, pts[1:])]
    durs = [j - i for (i, _), (j, _) in zip(piv, piv[1:])]
    # Wegstrecke aller Schwünge im Verhältnis zur Netto-Bewegung:
    # 1,0 = reines Hin und Her, 0 = gerader Trend
    path = [close.iloc[0]] + pts + [close.iloc[-1]]
    travel = sum(abs(np.log(b / a)) for a, b in zip(path, path[1:]))
    net = abs(np.log(close.iloc[-1] / close.iloc[0]))
    return {
        "swings_pa": len(amps) * YEAR / len(close),
        "swing_groesse": float(np.mean(amps)) if amps else np.nan,
        "swing_dauer": float(np.mean(durs)) if durs else np.nan,
        "pendel_anteil": 1 - net / travel if travel > 0 else np.nan,
    }


def build():
    yu = yahoo_universe()
    trs = tr_symbols()
    tradable = {s: v for s, v in yu.items() if s in trs}
    print(f"Yahoo: {len(yu)} liquide Aktien, davon bei Trade Republic handelbar: {len(tradable)}", flush=True)
    # Kurse nur einmal am Tag laden (schont Yahoo); mit --fresh neu laden
    cache = os.path.join(HERE, "data", "prices_cache.pkl")
    fresh = os.path.exists(cache) and date.fromtimestamp(os.path.getmtime(cache)) == date.today()
    if fresh and "--fresh" not in sys.argv:
        frames = pd.read_pickle(cache)
    else:
        frames = download(tradable)
        pd.to_pickle(frames, cache)

    rows, prev = [], {}
    for s, x in frames.items():
        c = x["Close"]
        if len(c) < MIN_DAYS:
            continue
        last = c.iloc[-YEAR:]
        turnover = float((x["Close"] * x["Volume"]).iloc[-60:].median())
        if c.iloc[-1] < MIN_PRICE or turnover < MIN_TURNOVER:
            continue
        m = swing_metrics(last)
        ma, sd = c.rolling(20).mean().iloc[-1], c.rolling(20).std().iloc[-1]
        # Letzte vier Wochen: Schwünge ab 5 %, Hin-und-Her-Anteil, Spanne, Lage in der Spanne
        arr = c.values.astype(float)
        legs4, pendel4, pos4, lo4, hi4 = research_4w.window_metrics(arr[-research_4w.W:])
        # Ranking-Grundlage: Schwünge ab 5 % in den letzten 24 Wochen, umgerechnet auf vier Wochen
        look = min(len(arr), LOOK)
        legs24, _ = research_window.osc(arr[-look:])
        # Beständigkeit: in wie vielen der letzten sechs Vier-Wochen-Blöcke gab es ein klares Muster?
        B = research_window.BLOCK
        nb = min(len(arr) // B, research_window.NBLOCKS)
        blocks = [research_window.clear(*research_window.osc(arr[len(arr) - (k + 1) * B:len(arr) - k * B]))
                  for k in range(nb)]
        isin, tr_name = trs[s]
        rows.append({
            "symbol": s, "name": tr_name or tradable[s]["name_yahoo"], "isin": isin,
            "kurs": float(c.iloc[-1]), "waehrung": tradable[s]["waehrung"],
            "swings_pa": m["swings_pa"], "swing_groesse": m["swing_groesse"],
            "swing_dauer": m["swing_dauer"], "pendel_anteil": m["pendel_anteil"],
            "vola_pa": float(np.log(last).diff().std() * np.sqrt(YEAR)),
            "veraenderung_12m": float(last.iloc[-1] / last.iloc[0] - 1),
            "vom_60t_hoch": float(c.iloc[-1] / c.iloc[-60:].max() - 1),
            "z_score": float((c.iloc[-1] - ma) / sd) if sd else np.nan,
            "swings_4w": legs4, "pendel_4w": pendel4, "spanne_4w": hi4 / lo4 - 1, "lage_4w": pos4,
            "muster_4w": research_4w.is_clear(legs4, pendel4),
            "schwuenge_24w": legs24 * B / look, "muster_bloecke": int(sum(blocks)), "bloecke": nb,
            "tagesumsatz_mio": turnover / 1e6,
            "boersenwert_mrd": tradable[s]["boersenwert_mrd"],
            "handelstage": int(len(last)),
        })
        if len(c) >= 2 * YEAR - 5:
            prev[s] = swing_metrics(c.iloc[-2 * YEAR:-YEAR])
    df = pd.DataFrame(rows)
    # Punktzahl 0–100: Rang nach Schwung-Häufigkeit der letzten 24 Wochen. Dieser Wert sagte im Test
    # am besten vorher, welche Aktien danach wieder ein klares Auf und Ab zeigen (research_window.py).
    df["punktzahl"] = 100 * df["schwuenge_24w"].rank(pct=True)
    df = df.sort_values(["punktzahl", "muster_bloecke", "swings_pa"], ascending=False)
    df.to_csv(OUT, index=False)
    return df, prev


def check(df, prev):
    """Sind die Aktien, die im Vorjahr am meisten schwankten, auch dieses Jahr die schwankungsfreudigsten?"""
    p = pd.DataFrame(prev).T.astype(float).add_prefix("vor_")
    j = df.set_index("symbol").join(p, how="inner")
    print(f"\n=== Stabilität (Vorjahr → letzte 12 Monate, {len(j)} Aktien mit 2 Jahren Historie) ===")
    for col, label in [("swings_pa", "Anzahl Swings"), ("swing_groesse", "Größe der Swings"),
                       ("pendel_anteil", "Pendel-Anteil")]:
        r = j[col].rank().corr(j["vor_" + col].rank())   # Rang-Korrelation (Spearman)
        print(f"  {label:18} Rang-Korrelation {r:+.2f}")
    top = j.sort_values("vor_swings_pa", ascending=False)
    k = max(len(j) // 10, 1)
    print(f"  Die {k} Aktien mit den meisten Swings im Vorjahr ({top['vor_swings_pa'].head(k).mean():.1f} pro Jahr) "
          f"hatten danach {top['swings_pa'].head(k).mean():.1f}; alle Aktien im Schnitt {j['swings_pa'].mean():.1f}.")
    stay = len(set(top.index[:k]) & set(j.sort_values("swings_pa", ascending=False).index[:k]))
    print(f"  Von diesen {k} sind {stay} auch in den letzten 12 Monaten wieder unter den obersten {k}.")


def main():
    df, prev = build()
    print(f"\n{len(df)} Aktien nach Filtern (Kurs ≥ {MIN_PRICE:.0f}, Tagesumsatz ≥ {MIN_TURNOVER / 1e6:.0f} Mio.) "
          f"→ {os.path.relpath(OUT, HERE)}\n")
    show = df.head(40)[["symbol", "name", "kurs", "punktzahl", "schwuenge_24w", "muster_bloecke", "muster_4w", "swings_pa", "swing_groesse", "swing_dauer",
                        "pendel_anteil", "vola_pa", "veraenderung_12m", "tagesumsatz_mio"]]
    print(show.to_string(index=False, formatters={
        "name": lambda s: str(s)[:26], "punktzahl": "{:.0f}".format, "schwuenge_24w": "{:.1f}".format, "muster_4w": lambda b: "klar" if b else "-", "kurs": "{:.2f}".format, "swings_pa": "{:.1f}".format,
        "swing_groesse": "{:.0%}".format, "swing_dauer": "{:.0f}".format,
        "pendel_anteil": "{:.2f}".format, "vola_pa": "{:.0%}".format,
        "veraenderung_12m": "{:+.0%}".format, "tagesumsatz_mio": "{:.0f}".format}))
    if "--check" in sys.argv:
        check(df, prev)


if __name__ == "__main__":
    main()
