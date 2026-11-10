"""
Gegenprobe für die Swing-Scanner-Idee.

Fragen:
 1. Hat das Signal (Kurs weit unter/über dem gleitenden Durchschnitt) über viele Aktien
    und viele Jahre einen Vorsprung gegenüber "einfach irgendwann kaufen"?
 2. Bleibt davon nach Kosten etwas übrig?
 3. Sind Aktien, bei denen es früher funktioniert hat, auch später die, bei denen es funktioniert?

Start:  .venv/bin/python backtest.py
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = """
TSLA NVDA PLTR RKLB AMD COIN MSTR SMCI SHOP ROKU SNAP UBER NET CRWD DDOG ZS MDB SNOW U RBLX
HOOD SOFI AFRM UPST RIVN LCID NIO PLUG ENPH FSLR MRNA BNTX ZM DOCU PINS ETSY TWLO OKTA TTD
MELI SE BABA JD PDD META NFLX AMZN GOOGL AAPL MSFT INTC MU AVGO QCOM BA DIS PYPL
SAP.DE RHM.DE SIE.DE IFX.DE VOW3.DE BAS.DE BAYN.DE DBK.DE CBK.DE ADS.DE ZAL.DE DHER.DE
HFG.DE TKA.DE ENR.DE AIXA.DE
""".split()

COST = 0.004  # Hin und zurück: 2 × 1 € Gebühr auf ~1.000 € + Spread/Slippage

print(f"Lade {len(UNIVERSE)} Aktien, 10 Jahre …")
DATA = yf.download(UNIVERSE, period="10y", auto_adjust=True, group_by="ticker",
                   threads=True, progress=False)


def events(win: int, zt: float, h: int) -> pd.DataFrame:
    rows = []
    for t in UNIVERSE:
        try:
            d = DATA[t][["Open", "Close"]].dropna()
        except KeyError:
            continue
        if len(d) < 300:
            continue
        c, o = d["Close"], d["Open"]
        z = (c - c.rolling(win).mean()) / c.rolling(win).std()
        buy = (z < -zt) & (z.shift() >= -zt)
        sell = (z > zt) & (z.shift() <= zt)
        fwd = c.shift(-h) / c - 1                      # Schlusskurs → Schlusskurs
        trade = c.shift(-h) / o.shift(-1) - 1 - COST   # Kauf zur Eröffnung am Folgetag, nach Kosten
        any_day = c.shift(-h) / o.shift(-1) - 1 - COST
        half = d.index[len(d) // 2]
        for name, m in [("buy", buy), ("sell", sell)]:
            rows.append(pd.DataFrame({
                "date": d.index[m], "ticker": t, "sig": name,
                "fwd": fwd[m].values, "trade": trade[m].values,
                "base_up": float((fwd > 0).mean()), "base_mean": float(fwd.mean()),
                "base_trade": float(any_day.mean()),
                "late": d.index[m] >= half,
            }))
    return pd.concat(rows).dropna()


def summary(E: pd.DataFrame, sig: str) -> dict:
    e = E[E.sig == sig]
    d = 1 if sig == "buy" else -1
    exc = e.fwd - e.base_mean
    return {
        "n": len(e),
        "Treffer": ((e.fwd * d) > 0).mean(),
        "Basis": e.base_up.mean() if d == 1 else 1 - e.base_up.mean(),
        "Ø danach": e.fwd.mean(),
        "Ø normal": e.base_mean.mean(),
        "Vorsprung": exc.mean(),
    }


pd.set_option("display.width", 200)
pct = lambda v: f"{v:+.2%}"

print("\n=== 1) Alle Parameter-Kombinationen, KAUFSIGNAL (alle Aktien gepoolt) ===")
out = []
for win in (20,):
    for zt in (1.5, 2.0, 2.5):
        for h in (5, 10, 20):
            s = summary(events(win, zt, h), "buy")
            out.append({"Schwelle": zt, "Horizont": h, **s})
T = pd.DataFrame(out)
print(T.to_string(index=False, formatters={
    "Treffer": "{:.1%}".format, "Basis": "{:.1%}".format, "Ø danach": pct,
    "Ø normal": pct, "Vorsprung": pct}))

print("\n=== 2) Standard-Einstellung (20 Tage, 2.0 SD, 10 Tage Horizont) ===")
E = events(20, 2.0, 10)
for sig, label in [("buy", "KAUFSIGNAL"), ("sell", "VERKAUFSSIGNAL")]:
    for part, m in [("gesamt", slice(None)), ("frühe Hälfte", ~E.late), ("späte Hälfte", E.late)]:
        s = summary(E[m] if not isinstance(m, slice) else E, sig)
        print(f"{label:15} {part:13} n={s['n']:5}  Treffer {s['Treffer']:.1%} (Basis {s['Basis']:.1%})  "
              f"Ø danach {s['Ø danach']:+.2%} vs. normal {s['Ø normal']:+.2%}  "
              f"Vorsprung {s['Vorsprung']:+.2%}")

print("\n=== 3) Als Handelsregel: Kauf am Folgetag zur Eröffnung, Verkauf nach 10 Tagen, nach Kosten ===")
b = E[E.sig == "buy"]
print(f"Trades: {len(b)}   Gewinner: {(b.trade > 0).mean():.1%}   "
      f"Ø Ergebnis je Trade: {b.trade.mean():+.2%}   Median: {b.trade.median():+.2%}")
print(f"Zum Vergleich – an einem beliebigen Tag gekauft, gleiche Aktien, gleiche Kosten: {b.base_trade.mean():+.2%}")
print(f"Schlechtestes Zehntel der Trades: {b.trade.quantile(0.10):+.1%}   "
      f"bestes Zehntel: {b.trade.quantile(0.90):+.1%}")

print("\n=== 4) Bleibt der Vorsprung bei denselben Aktien bestehen? ===")
b = b.assign(exc=b.fwd - b.base_mean)
per = b.groupby(["ticker", "late"]).agg(n=("exc", "size"), exc=("exc", "mean")).unstack()
ok = per[(per[("n", False)] >= 8) & (per[("n", True)] >= 8)]
r = ok[("exc", False)].corr(ok[("exc", True)])
print(f"{len(ok)} Aktien mit ≥ 8 Signalen in beiden Hälften.")
print(f"Korrelation Vorsprung frühe Hälfte ↔ späte Hälfte: {r:+.2f}   "
      "(nahe 0 = 'gute' Aktien von früher sind später nicht besser)")
top = ok.sort_values(("exc", False), ascending=False)
k = max(len(ok) // 4, 1)
print(f"Beste {k} Aktien der frühen Hälfte: Vorsprung früh {top[('exc', False)].head(k).mean():+.2%} "
      f"→ spät {top[('exc', True)].head(k).mean():+.2%}")
print(f"Schlechteste {k}: früh {top[('exc', False)].tail(k).mean():+.2%} "
      f"→ spät {top[('exc', True)].tail(k).mean():+.2%}")
