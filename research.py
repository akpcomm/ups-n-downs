"""
Signal-Suche mit Gegenprobe.

Jede Regel besteht aus
  - Zustand  (z.B. "Aktie im Aufwärtstrend")  → definiert die faire Vergleichsgruppe
  - Auslöser (z.B. "Kurs fällt 2 SD unter den 20-Tage-Schnitt")
  - Haltedauer in Handelstagen

Gemessen wird: Kauf zur Eröffnung am Folgetag, Verkauf zum Schlusskurs nach h Tagen,
nach Kosten – minus dem, was ein Kauf an einem beliebigen Tag im selben Zustand gebracht hätte.

Entdeckung: 2010–2018.  Bestätigung: 2019–heute.  Nur was in BEIDEN hält, zählt.

Start:  .venv/bin/python research.py
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = sorted(set("""
TSLA NVDA PLTR RKLB AMD COIN MSTR SMCI SHOP ROKU SNAP UBER NET CRWD DDOG ZS MDB SNOW U RBLX
HOOD SOFI AFRM UPST RIVN LCID NIO PLUG ENPH FSLR MRNA BNTX ZM DOCU PINS ETSY TWLO OKTA TTD
MELI SE BABA JD PDD META NFLX AMZN GOOGL AAPL MSFT INTC MU AVGO QCOM BA DIS PYPL
JPM BAC WFC GS MS C V MA AXP BRK-B JNJ PFE MRK ABBV LLY UNH TMO ABT AMGN GILD BMY CVS
XOM CVX COP SLB PG KO PEP WMT COST HD LOW MCD SBUX NKE TGT CAT DE HON GE MMM UPS FDX
LMT RTX F GM T VZ CMCSA ORCL CRM ADBE CSCO IBM TXN AMAT LRCX NOW INTU
SAP.DE RHM.DE SIE.DE IFX.DE VOW3.DE BAS.DE BAYN.DE DBK.DE CBK.DE ADS.DE ZAL.DE DHER.DE
HFG.DE TKA.DE ENR.DE AIXA.DE ALV.DE MUV2.DE DTE.DE BMW.DE MBG.DE EOAN.DE RWE.DE HEI.DE
MRK.DE FRE.DE DB1.DE HEN3.DE CON.DE PUM.DE LHA.DE
""".split()))
COST = 0.004
SPLIT = pd.Timestamp("2019-01-01")


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def cross(cond):
    return cond & ~cond.shift(fill_value=False)


def rules(c, mkt_z):
    """→ {Name: (Zustand, Auslöser, Haltedauer)}"""
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    z = (c - ma20) / sd20
    up = c > c.rolling(200).mean()
    always = pd.Series(True, index=c.index)
    dd60 = c / c.rolling(60).max() - 1
    return {
        "A  Dip (Z<-2), 10 T  [bisherige App]":        (always, cross(z < -2), 10),
        "B  Dip im Aufwärtstrend, 10 T":               (up, cross(z < -2), 10),
        "C  Dip im Abwärtstrend, 10 T":                (~up, cross(z < -2), 10),
        "D  RSI(2)<10 im Aufwärtstrend, 5 T":          (up, cross(rsi(c, 2) < 10), 5),
        "E  -20 % vom 60-Tage-Hoch, 20 T":             (always, cross(dd60 < -0.20), 20),
        "F  Starker Dip (Z<-2.5), 20 T":               (always, cross(z < -2.5), 20),
        "G  Dip bei Markt-Ausverkauf, 10 T":           (always, cross((z < -2) & (mkt_z < -1.5)), 10),
        "H  Neues 100-Tage-Hoch (Stärke kaufen), 20 T": (always, cross(c >= c.rolling(100).max()), 20),
    }


def main():
    print(f"Lade {len(UNIVERSE)} Aktien ab 2010 …")
    data = yf.download(UNIVERSE + ["SPY"], start="2010-01-01", auto_adjust=True,
                       group_by="ticker", threads=True, progress=False)
    spy = data["SPY"]["Close"].dropna()
    spy_z = (spy - spy.rolling(20).mean()) / spy.rolling(20).std()

    rows = []
    for t in UNIVERSE:
        try:
            d = data[t][["Open", "Close"]].dropna()
        except KeyError:
            continue
        if len(d) < 400:
            continue
        d.index = pd.to_datetime(d.index)
        c, o = d["Close"], d["Open"]
        mz = spy_z.reindex(c.index, method="ffill")
        late = pd.Series(c.index >= SPLIT, index=c.index)
        for name, (state, trig, h) in rules(c, mz).items():
            ret = c.shift(-h) / o.shift(-1) - 1 - COST
            for is_late in (False, True):
                per = late == is_late
                bench = ret[state & per].mean()
                m = trig & state & per & ret.notna()
                if m.sum() == 0 or np.isnan(bench):
                    continue
                rows.append(pd.DataFrame({
                    "rule": name, "late": is_late, "date": c.index[m], "ticker": t,
                    "ret": ret[m].values, "exc": ret[m].values - bench,
                }))
    E = pd.concat(rows)
    E["month"] = E["date"].dt.to_period("M")

    def stats(e):
        n = len(e)
        if n < 30:
            return None
        g = e.groupby("month")["exc"].agg(["sum", "size"])
        mean = g["sum"].sum() / n
        M = len(g)
        se = np.sqrt(((g["sum"] - mean * g["size"]) ** 2).sum()) / n * np.sqrt(M / (M - 1))
        return n, (e.ret > 0).mean(), e.ret.mean(), mean, mean / se

    print(f"\nKosten je Trade (hin und zurück): {COST:.1%}.  "
          "Vorsprung = Ergebnis minus 'beliebiger Tag im selben Zustand'.")
    print("t > 2 = statistisch auffällig (nach Monat gebündelt).\n")
    hdr = f"{'Regel':46} | {'ENTDECKUNG 2010–2018':^38} | {'BESTÄTIGUNG 2019–heute':^38}"
    print(hdr)
    print(f"{'':46} | {'n':>5} {'Gewinner':>8} {'Ø netto':>8} {'Vorspr.':>8} {'t':>5} "
          f"| {'n':>5} {'Gewinner':>8} {'Ø netto':>8} {'Vorspr.':>8} {'t':>5}")
    print("-" * len(hdr))
    for name in E["rule"].unique():
        line = f"{name:46}"
        for is_late in (False, True):
            s = stats(E[(E.rule == name) & (E.late == is_late)])
            line += (" | " + (f"{s[0]:5d} {s[1]:8.1%} {s[2]:+8.2%} {s[3]:+8.2%} {s[4]:+5.1f}"
                              if s else f"{'zu wenige Fälle':^38}"))
        print(line)

    # Jahr für Jahr für die Regeln, damit man sieht, ob ein Vorsprung an einzelnen Jahren hängt
    print("\nVorsprung je Kalenderjahr (Ø je Trade):")
    E["year"] = E["date"].dt.year
    piv = E.pivot_table(index="rule", columns="year", values="exc", aggfunc="mean")
    print((piv * 100).round(1).to_string())


if __name__ == "__main__":
    main()
