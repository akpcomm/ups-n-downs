"""Gemeinsame Rechenlogik für die App (app.py) und die Alerts (alerts.py)."""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

MKT_PANIC = -1.5  # Markt gilt als "im Ausverkauf", wenn der S&P 500 so viele SD unter seinem 20-Tage-Schnitt liegt

# Ergebnis von research.py (149 Aktien, 2010–heute, nach 0,4 % Kosten je Trade, Stand 10.11.2026).
# Vorsprung = Ergebnis je Trade minus "Kauf an einem beliebigen Tag".
EVIDENCE_MD = """
**Was der breite Test sagt** (149 Aktien, 2010 bis heute, nach 0,4 % Kosten je Trade;
Vorsprung gegenüber einem Kauf an einem beliebigen Tag, erst 2010–2018, dann 2019–heute):

| Signal | 2010–2018 | 2019–heute | Einordnung |
|---|---|---|---|
| Kaufzone allein | +0,49 % | −0,13 % | kein belastbarer Vorsprung |
| **Kaufzone + Markt-Ausverkauf** | +1,10 % | +0,57 % | in 13 von 17 Jahren positiv, statistisch aber nicht gesichert; 2020: −1,8 % |
| Verkaufszone / neues Hoch | – | – | kein Vorsprung; Stärke lief im Schnitt eher weiter |

Die Quote einer einzelnen Aktie trägt nicht in die Zukunft: Aktien, bei denen das Signal früher
gut lief, waren später nicht besser als die übrigen.
"""


SCREENER_MD = """
**So ist die Rangliste zu lesen:** Oben stehen die Aktien, die in den letzten 24 Wochen am häufigsten
um mindestens 5 % hin- und hergelaufen sind. Das sagt, **wo** viel Bewegung ist, aber nicht,
**in welche Richtung** die nächste Bewegung geht. Aktien mit vielen Swings sind zugleich die mit
den größten Verlustphasen.

**Wie stabil ist das?** (Vorjahr gegen die letzten 12 Monate, 1.605 Aktien, Stand 10.11.2026)

| Eigenschaft | Rang-Korrelation | Einordnung |
|---|---|---|
| Anzahl der Swings | +0,70 | stabil: wer viel schwankte, schwankt meist weiter viel |
| Größe der Swings | +0,28 | nur schwach stabil |
| Pendel-Anteil | −0,01 | nicht stabil: ob eine Aktie hin- und herläuft oder trendet, lässt sich aus dem Vorjahr nicht ablesen |

Die 160 Aktien mit den meisten Swings im Vorjahr (Ø 22 pro Jahr) hatten danach Ø 21, alle Aktien
im Schnitt 9. Von diesen 160 standen 90 wieder im obersten Zehntel.

**Punktzahl: wonach die Liste sortiert ist** (58.352 Stichtage aus zwei Jahren, Stand 10.11.2026)

Die Punktzahl (0–100) ist der Rang nach der Häufigkeit von Schwüngen ab 5 % in den letzten
24 Wochen. Getestet wurden Rückblicke von 2 bis 50 Wochen: Ab etwa 24 Wochen wird es nicht mehr besser.

| Rückblick | Trefferquote der besten 10 % |
|---|---|
| 2 Wochen | 17 % |
| 4 Wochen | 20 % |
| 8 Wochen | 22 % |
| 24 Wochen | 24 % |
| Grundquote aller Aktien | 7 % |

Trefferquote heißt: Die Aktie zeigt in den folgenden vier Wochen ein klares Auf und Ab (mindestens
3 Schwünge von je 5 % bei wenig Netto-Bewegung). Von den **30 bestplatzierten Aktien** war das bei
etwa jeder dritten der Fall; im Schnitt machten sie danach 3,7 Schwünge in vier Wochen, alle Aktien 1,1.

„Klare Monate“ zeigt, in wie vielen der letzten sechs Vier-Wochen-Blöcke die Aktie ein klares
Muster hatte. Bei 3 von 6 folgte in 24 % der Fälle wieder eines, bei 0 von 6 in 4 %.
"""


def fetch(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def market_z(period: str, win: int = 20) -> pd.Series:
    """Z-Score des S&P 500 (SPY) zu seinem 20-Tage-Schnitt."""
    spy = fetch("SPY", period)["Close"]
    return (spy - spy.rolling(win).mean()) / spy.rolling(win).std()


def zigzag(close: pd.Series, thr: float):
    """Bestätigte Wendepunkte: ein Hoch/Tief zählt erst, wenn der Kurs danach
    um mindestens `thr` (z.B. 0.10 = 10 %) in die Gegenrichtung gelaufen ist."""
    v = close.values
    pivots = []  # (position, "Hoch"/"Tief")
    trend, hi, lo = 0, 0, 0
    for i in range(1, len(v)):
        p = v[i]
        if trend == 0:
            if p > v[hi]:
                hi = i
            if p < v[lo]:
                lo = i
            if v[hi] / v[lo] - 1 >= thr:
                if hi > lo:
                    pivots.append((lo, "Tief"))
                    trend = 1
                else:
                    pivots.append((hi, "Hoch"))
                    trend = -1
        elif trend == 1:
            if p > v[hi]:
                hi = i
            elif p <= v[hi] * (1 - thr):
                pivots.append((hi, "Hoch"))
                trend, lo = -1, i
        else:
            if p < v[lo]:
                lo = i
            elif p >= v[lo] * (1 + thr):
                pivots.append((lo, "Tief"))
                trend, hi = 1, i
    return pivots


def swing_stats(close: pd.Series, thr: float) -> dict:
    piv = zigzag(close, thr)
    amps, durs = [], []
    for (i1, _), (i2, _) in zip(piv, piv[1:]):
        amps.append(abs(close.iloc[i2] / close.iloc[i1] - 1))
        durs.append(i2 - i1)
    return {
        "pivots": piv,
        "n_swings": len(amps),
        "avg_amp": float(np.mean(amps)) if amps else np.nan,
        "avg_dur": float(np.mean(durs)) if durs else np.nan,
    }


def variance_ratio(close: pd.Series, k: int = 10) -> float:
    """< 1: Kurs nimmt Bewegungen eher zurück. ≈ 1: Zufallspfad. > 1: Trends laufen weiter."""
    r = np.log(close).diff().dropna()
    if len(r) < k * 5:
        return np.nan
    rk = np.log(close).diff(k).dropna()
    return float(rk.var() / (k * r.var()))


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def add_indicators(df: pd.DataFrame, win: int, z_thr: float, mkt_z=None) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["MA"] = c.rolling(win).mean()
    df["SD"] = c.rolling(win).std()
    df["Z"] = (c - df["MA"]) / df["SD"]
    df["RSI"] = rsi(c)
    df["SMA200"] = c.rolling(200).mean()
    df["DD60"] = c / c.rolling(60, min_periods=20).max() - 1
    df["MKT_Z"] = mkt_z.reindex(df.index, method="ffill") if mkt_z is not None else np.nan
    z, zp = df["Z"], df["Z"].shift()
    # Signal nur am Tag, an dem die Schwelle überschritten wird (nicht jeden Tag in der Zone)
    df["BUY"] = (z < -z_thr) & (zp >= -z_thr)
    df["SELL"] = (z > z_thr) & (zp <= z_thr)
    panic = (z < -z_thr) & (df["MKT_Z"] < MKT_PANIC)
    df["PANIC"] = panic & ~panic.shift(fill_value=False)
    return df


def wilson(k: int, n: int, zc: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + zc**2 / n
    mid = (p + zc**2 / (2 * n)) / den
    half = zc * np.sqrt(p * (1 - p) / n + zc**2 / (4 * n * n)) / den
    return (mid - half, mid + half)


def evaluate(df: pd.DataFrame, col: str, direction: int, h: int, mask=None) -> dict:
    """Wie oft lag der Kurs h Tage nach dem Signal in der vorhergesagten Richtung?"""
    c = df["Close"]
    fwd = c.shift(-h) / c - 1
    m = df[col] if mask is None else (df[col] & mask)
    r = fwd[m].dropna()
    n = len(r)
    k = int(((r * direction) > 0).sum())
    allfwd = fwd.dropna() if mask is None else fwd[mask].dropna()
    base = float(((allfwd * direction) > 0).mean()) if len(allfwd) else np.nan
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "hit": k / n if n else np.nan,
        "lo": lo,
        "hi": hi,
        "base": base,
        "avg_move": float((r * direction).mean()) if n else np.nan,
    }


def signal_state(last: pd.Series, z_thr: float) -> str:
    z, mz = last["Z"], last["MKT_Z"]
    if z < -z_thr:
        return "KAUFZONE + Markt-Ausverkauf" if mz < MKT_PANIC else "KAUFZONE"
    if z > z_thr:
        return "VERKAUFSZONE"
    if z < -0.75 * z_thr:
        return "nahe Kaufzone"
    if z > 0.75 * z_thr:
        return "nahe Verkaufszone"
    return "neutral"


def trend_label(last: pd.Series) -> str:
    if pd.isna(last["SMA200"]):
        return "–"
    return "↑ aufwärts" if last["Close"] > last["SMA200"] else "↓ abwärts"


def analyse(ticker: str, df: pd.DataFrame, mkt_z, swing_thr, win, z_thr, h):
    if df is None or df.empty or len(df) < win + h + 10:
        return None
    df = add_indicators(df, win, z_thr, mkt_z)
    c = df["Close"]
    sw = swing_stats(c, swing_thr)
    last = df.iloc[-1]
    return {
        "df": df,
        "swing": sw,
        "buy": evaluate(df, "BUY", +1, h),
        "sell": evaluate(df, "SELL", -1, h),
        "row": {
            "Ticker": ticker,
            "Kurs": float(last["Close"]),
            "Signal": signal_state(last, z_thr),
            "Z-Score": float(last["Z"]),
            "Markt-Z": float(last["MKT_Z"]),
            "Trend (200 T)": trend_label(last),
            "vom 60-T-Hoch": float(last["DD60"]),
            "Vola p.a.": float(np.log(c).diff().std() * np.sqrt(252)),
            "Swings": sw["n_swings"],
            "Ø Swing": sw["avg_amp"],
            "Ø Dauer (Tage)": sw["avg_dur"],
            "Pendel-Index (VR)": variance_ratio(c),
            "RSI": float(last["RSI"]),
            "Handelstage": len(df),
        },
    }
