"""
Bereitet die Daten für die iPhone-Oberfläche auf: site/data.json

Enthält
  - die Rangliste (beste 300 Aktien plus Watchlist) mit Kursverlauf, Schwüngen und aktuellem Schwung
  - den laufenden Ranking-Check: Für jeden zurückliegenden Stichtag wird die Rangliste so berechnet,
    wie sie damals ausgesehen hätte, und geprüft, wie viele der besten 30 danach wirklich ein
    klares Auf und Ab gezeigt haben. Jeder neue Lauf verlängert diese Reihe um den jüngsten
    Stichtag, dessen vier Folgewochen inzwischen vorbei sind.

Vorher:  .venv/bin/python screener.py
Start:   .venv/bin/python build_site.py
"""
import json
import os
import sys
import warnings
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    BERLIN = ZoneInfo("Europe/Berlin")
except Exception:
    BERLIN = None

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import core
import research_window as rw
from research_4w import THR, pivots
from research_hot import state as swing_state

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "site", "data.json")
TOP, LOOK, B = 300, 120, rw.BLOCK
CHECK_TOP = 30


def read_watchlist():
    with open(os.path.join(HERE, "watchlist.txt")) as f:
        return [l.split("#")[0].strip().upper() for l in f if l.split("#")[0].strip()]


def ranking_check(frames, symbols):
    """Rangliste rückwirkend für jeden Stichtag nachstellen und gegen die Folgewochen prüfen."""
    ref = max((frames[s] for s in symbols), key=len).index
    series, top5 = [], []
    for back in range(250, B - 1, -5):
        scored = []
        for s in symbols:
            a = frames[s]["Close"].values.astype(float)
            t = len(a) - back
            if t < LOOK:
                continue
            legs, _ = rw.osc(a[t - LOOK:t])
            scored.append((legs, rw.clear(*rw.osc(a[t:t + B]))))
        if len(scored) < 200:
            continue
        scored.sort(key=lambda x: -x[0])
        top = scored[:CHECK_TOP]
        top5.append(np.mean([y for _, y in scored[:5]]))
        series.append({
            "date": ref[-(back + 1)].strftime("%Y-%m-%d"),
            "hits": int(sum(y for _, y in top)),
            "n": CHECK_TOP,
            "base": round(float(np.mean([y for _, y in scored])), 4),
        })
    last = series[-8:]
    rate = np.mean([x["hits"] / x["n"] for x in last])
    base = np.mean([x["base"] for x in last])
    lift = rate / base if base > 0 else 0
    state = "wie erwartet" if lift >= 2.5 else "schwächer als üblich" if lift >= 1.5 else "derzeit ohne Aussagekraft"
    return {
        "series": series,
        "avg_hits": round(float(np.mean([x["hits"] for x in series])), 1),
        "recent_hits": round(float(np.mean([x["hits"] for x in last])), 1),
        "recent_base_hits": round(float(base * CHECK_TOP), 1),
        "lift": round(float(lift), 1),
        "top5_rate": round(float(np.mean(top5)), 3),
        "base_rate": round(float(np.mean([x["base"] for x in series])), 3),
        "state": state,
    }


def hot_stats(frames, symbols):
    """Was folgte, wenn eine Aktie aus dem Raster tief stand? Wird bei jedem Lauf neu ausgezählt.

    Raster: am Stichtag unter den 10 % mit den häufigsten Schwüngen der letzten 24 Wochen.
    Tief:   Abschwung seit dem letzten Hoch mindestens so groß wie ein typischer Abschwung der Aktie.
    Danach: Was kam in den nächsten 20 Handelstagen zuerst – ein typischer Aufschwung oder ein
            weiterer typischer Abschwung?"""
    rows = []
    for s in symbols:
        a = frames[s]["Close"].values.astype(float)
        n = len(a)
        for back in range(20, n - LOOK, 5):
            t = n - back
            st = swing_state(a[t - LOOK:t])
            if st is None:
                continue
            fut = a[t:t + 20] / a[t - 1] - 1
            up = int(np.argmax(fut >= st["typ_up"])) if (fut >= st["typ_up"]).any() else 99
            dn = int(np.argmax(fut <= -st["typ_down"])) if (fut <= -st["typ_down"]).any() else 99
            rows.append((back, st["legs"], st["progress"], up, dn, fut[-1], fut[9], st["typ_up"]))
    D = pd.DataFrame(rows, columns=["back", "legs", "progress", "up", "dn", "f20", "f10", "typ_up"])
    grid = D.groupby("back")["legs"].rank(pct=True) >= 0.9

    def summ(g):
        won = g[g.up < g.dn]
        return {
            "n": int(len(g)),
            "p_up": round(float((g.up < g.dn).mean()), 3),
            "p_dn": round(float((g.dn < g.up).mean()), 3),
            "p_none": round(float((g.up == g.dn).mean()), 3),
            "days_up": int(won.up.median() + 1) if len(won) else None,
            "mean20": round(float(g.f20.mean()), 4), "med20": round(float(g.f20.median()), 4),
            "p_up10": round(float((g.f10 > 0).mean()), 3),
            "r20": round(float((g.f20 / g.typ_up).mean()), 3),
        }
    return {"hot": summ(D[grid & (D.progress >= 1.0)]), "any": summ(D[grid])}


def stock_record(sym, row, frame, rank):
    a = frame["Close"].values.astype(float)
    look = min(len(a), LOOK)
    w = a[-look:]
    piv = pivots(w, THR)
    legs = [{"pct": round(w[j] / w[i] - 1, 4), "days": int(j - i)} for i, j in zip(piv, piv[1:])]
    sw = None
    if piv:
        i0 = piv[-1]
        cur = w[-1] / w[i0] - 1
        same = [abs(l["pct"]) for l in legs if (l["pct"] > 0) == (cur > 0)] or [abs(l["pct"]) for l in legs]
        days = [l["days"] for l in legs]
        sw = {
            "from": int(i0), "pct": round(float(cur), 4), "days": int(len(w) - 1 - i0),
            "typ_pct": round(float(np.median(same)), 4) if same else None,
            "typ_days": int(np.median(days)) if days else None,
        }
    ma, sd = pd.Series(a).rolling(20).mean().iloc[-1], pd.Series(a).rolling(20).std().iloc[-1]
    nb = min(len(a) // B, rw.NBLOCKS)
    kb = sum(rw.clear(*rw.osc(a[len(a) - (k + 1) * B:len(a) - k * B])) for k in range(nb))
    status = "Zu neu" if nb < rw.NBLOCKS else "Beständig" if kb >= 3 else "Wechselhaft" if kb >= 1 else "Ruhig"
    legs24, _ = rw.osc(w)
    m4 = rw.clear(*rw.osc(a[-B:]))
    st = swing_state(w)
    hs = None if st is None else {
        "up": round(st["typ_up"], 4), "up_days": int(round(st["typ_up_days"])),
        "dn": round(st["typ_down"], 4), "prog": round(st["progress"], 2),
    }
    rec = {
        "id": sym.replace(".", "-"), "s": sym, "p": round(float(a[-1]), 2),
        "d": round(float(a[-1] / a[-2] - 1), 4), "z": round(float((a[-1] - ma) / sd), 2) if sd else None,
        "f": round(legs24 * B / look, 1), "kb": int(kb), "nb": int(nb), "m4": bool(m4), "st": status,
        "typ": round(float(np.median([abs(l["pct"]) for l in legs])), 4) if legs else None,
        "c": [round(float(v), 2) for v in w], "piv": [int(i) for i in piv], "legs": legs[-6:], "sw": sw,
        "d0": frame.index[-look].strftime("%Y-%m-%d"), "d1": frame.index[-1].strftime("%Y-%m-%d"),
        "r": rank, "hs": hs,
    }
    if row is not None:
        rec.update({"n": str(row["name"]), "isin": str(row["isin"]), "cur": str(row["waehrung"]),
                    "sc": int(round(row["punktzahl"]))})
    else:
        rec.update({"n": sym, "isin": "", "cur": "", "sc": None})
    return rec


def main():
    frames = pd.read_pickle(os.path.join(HERE, "data", "prices_cache.pkl"))
    df = pd.read_csv(os.path.join(HERE, "data", "screener.csv")).reset_index(drop=True)
    df["rank"] = df.index + 1
    by_sym = df.set_index("symbol")
    symbols = [s for s in df["symbol"] if s in frames and len(frames[s]) >= 300]

    print("Ranking-Check …", flush=True)
    check = ranking_check(frames, symbols)
    print("Auszählung der Tiefs …", flush=True)
    hot = hot_stats(frames, symbols)

    watch = read_watchlist()
    wanted = list(df["symbol"].head(TOP)) + [s for s in watch if s not in set(df["symbol"].head(TOP))]
    stocks = []
    for s in wanted:
        frame = frames.get(s)
        if frame is None:
            try:
                frame = core.fetch(s, "2y")[["Close", "Volume"]]
            except Exception:
                continue
        if frame is None or len(frame) < 40:
            continue
        row = by_sym.loc[s] if s in by_sym.index else None
        stocks.append(stock_record(s, row, frame, int(row["rank"]) if row is not None else None))

    try:
        mz = float(core.market_z("6mo").dropna().iloc[-1])
    except Exception:
        mz = None

    try:
        fx = round(float(core.fetch("EURUSD=X", "5d")["Close"].iloc[-1]), 4)   # US-Dollar je Euro
    except Exception:
        fx = None

    data = {
        "fx": fx,
        "generated": datetime.now(BERLIN).strftime("%Y-%m-%d %H:%M"),   # deutsche Zeit, auch auf GitHubs Servern
        "universe": int(len(df)),
        "market_z": None if mz is None else round(mz, 2),
        "market_panic": core.MKT_PANIC,
        "watchlist": [s for s in watch if any(x["s"] == s for x in stocks)],
        "check": check,
        "hot": hot,
        "stocks": stocks,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(stocks)} Aktien → site/data.json ({os.path.getsize(OUT) / 1024:.0f} KB)")
    c = check
    print(f"Ranking-Check: {len(c['series'])} Stichtage, im Schnitt {c['avg_hits']} von {CHECK_TOP} Treffern; "
          f"zuletzt {c['recent_hits']} (Zufall wären {c['recent_base_hits']}) → {c['state']}")

    h = hot["hot"]
    print(f"Nach einem Tief ({h['n']} Fälle): zuerst Aufschwung {h['p_up']:.0%}, zuerst weiterer Abschwung "
          f"{h['p_dn']:.0%}, keins {h['p_none']:.0%}; Ø nach 20 Tagen {h['mean20']:+.1%}, Mitte {h['med20']:+.1%}")

    # Warnung aufs iPhone, wenn das Ranking nachlässt (höchstens einmal pro Woche, montags)
    topic = os.environ.get("NTFY_TOPIC")
    if "--notify" in sys.argv and topic and c["state"] != "wie erwartet" and datetime.now(BERLIN).weekday() == 0:
        import alerts
        alerts.send({
            "title": "ups'n'downs: Ranking " + c["state"],
            "message": f"Von den 30 bestplatzierten Aktien zeigten zuletzt im Schnitt {c['recent_hits']} danach ein "
                       f"klares Auf und Ab. Zufall wären {c['recent_base_hits']}, im Jahresschnitt waren es {c['avg_hits']}.",
            "priority": 3, "tags": ["warning"],
        }, topic)
        print("Warnung gesendet.")


if __name__ == "__main__":
    main()
