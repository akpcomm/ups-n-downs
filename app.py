"""
ups'n'downs
=============
Zeigt, wie stark Aktien schwanken, markiert überdehnte Kurse (Kauf-/Verkaufszone)
und ordnet ein, wie belastbar diese Signale im breiten Test waren.

Keine Anlageberatung. Historische Statistik ist keine Garantie.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
from core import evaluate

st.set_page_config(page_title="ups'n'downs", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    return core.fetch(ticker, period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_z(period: str) -> pd.Series:
    return core.market_z(period)


def render_screener():
    st.subheader("Screener: die schwankungsfreudigsten Aktien bei Trade Republic")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "screener.csv")
    if not os.path.exists(path):
        st.info("Noch keine Rangliste vorhanden. Im Terminal erzeugen: `.venv/bin/python screener.py`")
        return
    df = pd.read_csv(path)
    stand = datetime.fromtimestamp(os.path.getmtime(path))
    st.caption(
        f"Stand: {stand:%d.%m.%Y %H:%M}. Grundlage: liquide Aktien aus den USA und von Xetra, die laut "
        "offizieller Trade-Republic-Liste handelbar sind. Sortiert nach der Häufigkeit von Auf- und "
        "Ab-Bewegungen ab 5 % in den letzten 24 Wochen. Aktualisieren: `.venv/bin/python screener.py`"
    )
    c1, c2, c3 = st.columns(3)
    min_sw = c1.slider("Mindestens so viele 10-%-Swings pro Jahr", 0, 40, 0)
    min_pendel = c2.slider(
        "Mindest-Pendel-Anteil (nur Rückblick)", 0.0, 1.0, 0.0, 0.05,
        help="1,0 = reines Hin und Her ohne Netto-Bewegung. 0 = gerader Trend in eine Richtung. "
        "Achtung: Dieser Wert war im Test von Jahr zu Jahr nicht stabil, er beschreibt nur die Vergangenheit.",
    )
    cur = c3.multiselect("Währung", sorted(df["waehrung"].dropna().unique()),
                         default=sorted(df["waehrung"].dropna().unique()))
    only_clear = st.checkbox(
        "Nur Aktien mit klarem Auf und Ab in den letzten vier Wochen",
        help="Mindestens 3 Schwünge von je 5 % in 20 Handelstagen und wenig Netto-Bewegung. "
        "Im Test wiederholte sich so ein Muster in etwa jedem fünften Fall.",
    )
    f = df[(df["swings_pa"] >= min_sw) & (df["pendel_anteil"] >= min_pendel) & df["waehrung"].isin(cur)]
    if only_clear:
        f = f[f["muster_4w"]]
    f = f.assign(
        muster_4w=f["muster_4w"].map({True: "klar", False: "–"}),
        klare_bloecke=f["muster_bloecke"].astype(str) + " von " + f["bloecke"].astype(str),
    )
    st.markdown(f"**{len(f)} von {len(df)} Aktien** erfüllen die Filter.")
    cols = {
        "symbol": "Kürzel", "name": "Name", "isin": "ISIN", "kurs": "Kurs", "waehrung": "Währung",
        "punktzahl": "Punktzahl", "schwuenge_24w": "Schwünge je 4 Wo.", "klare_bloecke": "Klare Monate",
        "muster_4w": "Muster aktuell",
        "swings_pa": "Swings/Jahr", "swing_groesse": "Ø Swing", "swing_dauer": "Ø Dauer (Tage)",
        "pendel_anteil": "Pendel-Anteil", "vola_pa": "Vola p.a.", "veraenderung_12m": "12 Monate",
        "vom_60t_hoch": "vom 60-T-Hoch", "z_score": "Z-Score", "tagesumsatz_mio": "Tagesumsatz (Mio.)",
        "boersenwert_mrd": "Börsenwert (Mrd.)", "handelstage": "Handelstage",
    }
    st.dataframe(
        f[list(cols)].rename(columns=cols).set_index("Kürzel").style.format(
            {"Kurs": "{:.2f}", "Punktzahl": "{:.0f}", "Schwünge je 4 Wo.": "{:.1f}", "Swings/Jahr": "{:.1f}", "Ø Swing": "{:.0%}", "Ø Dauer (Tage)": "{:.0f}",
             "Pendel-Anteil": "{:.2f}", "Vola p.a.": "{:.0%}", "12 Monate": "{:+.0%}",
             "vom 60-T-Hoch": "{:+.0%}", "Z-Score": "{:+.2f}", "Tagesumsatz (Mio.)": "{:.0f}",
             "Börsenwert (Mrd.)": "{:.1f}"}, na_rep="–"),
        use_container_width=True, height=560,
    )
    st.markdown(core.SCREENER_MD)
    st.text_area("Kürzel der gefilterten Liste (für die Watchlist)", ", ".join(f["symbol"].head(30)), height=80)


# --------------------------------------------------------------------------- #
# Oberfläche
# --------------------------------------------------------------------------- #
st.title("📈 ups'n'downs")
st.caption(
    "Zeigt, wie stark Aktien schwanken, und markiert überdehnte Kurse. "
    "**Keine Anlageberatung** – Statistik aus der Vergangenheit, keine Vorhersage."
)

view = st.radio("Ansicht", ["Watchlist", "Screener"], horizontal=True, label_visibility="collapsed")
if view == "Screener":
    render_screener()
    st.stop()

with st.sidebar:
    st.header("Einstellungen")
    try:
        default_tickers = ", ".join(
            l.split("#")[0].strip() for l in open("watchlist.txt") if l.split("#")[0].strip()
        )
    except OSError:
        default_tickers = "SPCX, TSLA, NVDA, PLTR, RKLB, AMD, COIN, MSTR, SAP.DE, RHM.DE"
    tickers_raw = st.text_area(
        "Ticker (Komma-getrennt)",
        default_tickers,
        help="Yahoo-Finance-Symbole. Deutsche Börse: Endung .DE (z.B. SAP.DE). "
        "Die Vorgabe kommt aus watchlist.txt – dieselbe Liste nutzen die Alerts.",
    )
    period = st.selectbox("Historie", ["1y", "2y", "5y", "10y", "max"], index=2)
    swing_thr = st.slider("Mindestgröße eines Swings", 3, 30, 10, format="%d %%") / 100
    st.divider()
    st.subheader("Signal")
    win = st.slider("Gleitender Durchschnitt (Tage)", 10, 60, 20)
    z_thr = st.slider(
        "Überdehnung (Standardabweichungen)", 1.0, 3.0, 2.0, 0.1,
        help="Kaufzone: Kurs liegt mehr als X Standardabweichungen unter dem Durchschnitt. "
        "Verkaufszone: entsprechend darüber.",
    )
    h = st.slider("Auswertungs-Horizont (Handelstage)", 3, 40, 10)

tickers = [t.strip().upper() for t in tickers_raw.replace(";", ",").split(",") if t.strip()]

results, failed = {}, []
with st.spinner("Lade Kurse …"):
    try:
        mkt_z = load_market_z(period)
    except Exception:
        mkt_z = None
    for t in tickers:
        try:
            res = core.analyse(t, load_prices(t, period), mkt_z, swing_thr, win, z_thr, h)
        except Exception:
            res = None
        if res is None:
            failed.append(t)
        else:
            results[t] = res

if failed:
    st.warning("Keine oder zu wenige Daten für: " + ", ".join(failed))
if not results:
    st.stop()

# ---------- Marktlage ----------
if mkt_z is not None and len(mkt_z.dropna()):
    mz = float(mkt_z.dropna().iloc[-1])
    if mz < core.MKT_PANIC:
        st.error(f"**Markt-Ausverkauf:** Der S&P 500 liegt {mz:+.1f} Standardabweichungen unter seinem "
                 "20-Tage-Schnitt. Kaufzonen in dieser Lage waren im breiten Test die mit dem besten Bild.")
    else:
        st.info(f"**Marktlage normal** (S&P 500: {mz:+.1f} SD zum 20-Tage-Schnitt). "
                f"Als Ausverkauf gilt ein Wert unter {core.MKT_PANIC}.")

# ---------- Übersicht ----------
st.subheader("Übersicht")
table = pd.DataFrame([r["row"] for r in results.values()]).set_index("Ticker")
order = {"KAUFZONE + Markt-Ausverkauf": 0, "KAUFZONE": 1, "VERKAUFSZONE": 2,
         "nahe Kaufzone": 3, "nahe Verkaufszone": 4, "neutral": 5}
table = table.assign(_o=table["Signal"].map(order)).sort_values(["_o", "Z-Score"]).drop(columns="_o")


def color_signal(v):
    if "AUSVERKAUF" in v.upper():
        return "background-color: rgba(34,197,94,.55); font-weight: 600"
    if "KAUF" in v.upper() and "VER" not in v.upper():
        return "background-color: rgba(34,197,94,.22)"
    if "VERKAUF" in v.upper():
        return "background-color: rgba(239,68,68,.22)"
    return ""


st.dataframe(
    table.style.format(
        {
            "Kurs": "{:.2f}",
            "Z-Score": "{:+.2f}",
            "Markt-Z": "{:+.2f}",
            "vom 60-T-Hoch": "{:+.0%}",
            "Vola p.a.": "{:.0%}",
            "Ø Swing": "{:.1%}",
            "Ø Dauer (Tage)": "{:.0f}",
            "Pendel-Index (VR)": "{:.2f}",
            "RSI": "{:.0f}",
        },
        na_rep="–",
    ).map(color_signal, subset=["Signal"]),
    use_container_width=True,
)

st.markdown(core.EVIDENCE_MD)

with st.expander("Was bedeuten die Spalten?"):
    st.markdown(
        f"""
- **Signal** – Kaufzone bei Z < −{z_thr}, Verkaufszone bei Z > +{z_thr}. „+ Markt-Ausverkauf“ heißt:
  gleichzeitig liegt der S&P 500 mehr als {abs(core.MKT_PANIC)} SD unter seinem 20-Tage-Schnitt.
- **Z-Score** – wie viele Standardabweichungen der Kurs vom {win}-Tage-Durchschnitt entfernt ist.
- **Markt-Z** – derselbe Wert für den S&P 500.
- **Trend (200 T)** – Kurs über oder unter dem 200-Tage-Durchschnitt.
- **vom 60-T-Hoch** – Abstand zum höchsten Schlusskurs der letzten 60 Handelstage.
- **Swings / Ø Swing / Ø Dauer** – Anzahl bestätigter Auf- und Ab-Bewegungen von mindestens {swing_thr:.0%},
  deren durchschnittliche Größe und Länge. Hohe Werte = die Aktie schwankt stark.
- **Pendel-Index (VR)** – Variance Ratio über 10 Tage. Unter 1: Bewegungen wurden eher
  zurückgenommen. Um 1: Zufallspfad. Über 1: Trends liefen weiter.
"""
    )

# ---------- Detail ----------
st.subheader("Detailansicht")
pick = st.selectbox("Aktie", list(table.index))
R = results[pick]
df = R["df"]
row = R["row"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kurs", f"{row['Kurs']:.2f}")
c2.metric("Aktuelles Signal", row["Signal"])
c3.metric("Z-Score", f"{row['Z-Score']:+.2f}")
c4.metric("Ø Swing", f"{row['Ø Swing']:.1%}" if pd.notna(row["Ø Swing"]) else "–",
          f"{row['Swings']} Swings")

# Chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                    vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=df.index, y=df["MA"] + z_thr * df["SD"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"), 1, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["MA"] - z_thr * df["SD"], line=dict(width=0),
                         fill="tonexty", fillcolor="rgba(120,120,160,.15)",
                         name=f"Normalbereich (±{z_thr} SD)", hoverinfo="skip"), 1, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["MA"], line=dict(color="#888", width=1, dash="dot"),
                         name=f"{win}-Tage-Schnitt"), 1, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="#3b82f6", width=1.6),
                         name="Kurs"), 1, 1)

piv = R["swing"]["pivots"]
if piv:
    px = [df.index[i] for i, _ in piv]
    py = [df["Close"].iloc[i] for i, _ in piv]
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="#f59e0b", width=1),
                             name=f"Swings ≥ {swing_thr:.0%}", opacity=0.7), 1, 1)

b, s = df[df["BUY"]], df[df["SELL"]]
fig.add_trace(go.Scatter(x=b.index, y=b["Close"], mode="markers", name="Kaufsignal",
                         marker=dict(symbol="triangle-up", size=11, color="#22c55e",
                                     line=dict(width=1, color="#14532d"))), 1, 1)
pk = df[df["PANIC"]]
fig.add_trace(go.Scatter(x=pk.index, y=pk["Close"], mode="markers", name="Kauf bei Markt-Ausverkauf",
                         marker=dict(symbol="star", size=14, color="#16a34a",
                                     line=dict(width=1, color="#052e16"))), 1, 1)
fig.add_trace(go.Scatter(x=s.index, y=s["Close"], mode="markers", name="Verkaufssignal",
                         marker=dict(symbol="triangle-down", size=11, color="#ef4444",
                                     line=dict(width=1, color="#7f1d1d"))), 1, 1)

fig.add_trace(go.Scatter(x=df.index, y=df["Z"], line=dict(color="#a855f7", width=1.2),
                         name="Z-Score"), 2, 1)
for lvl, col in [(z_thr, "#ef4444"), (-z_thr, "#22c55e")]:
    fig.add_hline(y=lvl, line=dict(color=col, width=1, dash="dash"), row=2, col=1)
fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified")
fig.update_yaxes(title_text="Kurs", row=1, col=1)
fig.update_yaxes(title_text="Z", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# Trefferquoten mit Test auf "frische" Daten
st.markdown(f"#### Signale dieser Aktie im Rückblick (Horizont: {h} Handelstage)")
st.warning("Diese Zahlen beschreiben nur die Vergangenheit **dieser einen Aktie**. Im breiten Test hat sich "
           "gezeigt, dass eine gute Quote einzelner Aktien später nicht bestehen bleibt. Für die "
           "Einordnung zählt die Tabelle „Was der breite Test sagt“ oben.")
split = df.index[int(len(df) * 0.7)]
early = pd.Series(df.index < split, index=df.index)
late = ~early


def stats_rows(col, direction):
    out = []
    for label, mask in [("Gesamter Zeitraum", None),
                        (f"Erste 70 % (bis {split:%d.%m.%Y})", early),
                        ("Letzte 30 % (Gegenprobe)", late)]:
        e = evaluate(df, col, direction, h, mask)
        out.append({
            "Zeitraum": label,
            "n Signale": e["n"],
            "Trefferquote": e["hit"],
            "95 %-Bereich": (f"{e['lo']:.0%} – {e['hi']:.0%}" if e["n"] else "–"),
            "Basisrate": e["base"],
            "Vorsprung": e["hit"] - e["base"] if e["n"] else np.nan,
            "Ø Bewegung in Signalrichtung": e["avg_move"],
        })
    return pd.DataFrame(out).set_index("Zeitraum")


fmt = {"Trefferquote": "{:.0%}", "Basisrate": "{:.0%}", "Vorsprung": "{:+.0%}",
       "Ø Bewegung in Signalrichtung": "{:+.1%}"}
st.markdown("**🟢 Kaufsignal** → Kurs danach höher?")
st.dataframe(stats_rows("BUY", +1).style.format(fmt, na_rep="–"), use_container_width=True)
st.markdown("**🔴 Verkaufssignal** → Kurs danach tiefer?")
st.dataframe(stats_rows("SELL", -1).style.format(fmt, na_rep="–"), use_container_width=True)

nb, ns = R["buy"]["n"], R["sell"]["n"]
if min(nb, ns) < 20:
    st.info(
        f"Nur {nb} Kauf- und {ns} Verkaufssignale in der Historie – die Quoten sind bei so wenigen "
        "Fällen sehr unsicher (siehe 95 %-Bereich). Längere Historie wählen oder Schwelle senken."
    )
st.caption(
    "Lesehilfe: Ein Signal ist nur dann interessant, wenn die Trefferquote **deutlich über der "
    "Basisrate** liegt **und** das auch in der Gegenprobe (letzte 30 %) noch so ist. "
    "Wenn die Quote nur in den ersten 70 % gut war, ist das meist Zufall."
)
