"""
Was folgte, wenn eine schwungstarke Aktie "tief" stand?

"Im Raster":  gehört am Stichtag zu den 10 % mit den häufigsten Schwüngen der letzten 24 Wochen.
"Tief" (HOT): Der Kurs fällt seit dem letzten bestätigten Hoch und hat dabei mindestens so viel
              verloren wie ein typischer Abschwung dieser Aktie (Median ihrer Abschwünge).
Beides ist am Stichtag selbst feststellbar, ohne Blick in die Zukunft.

Gemessen wird der Kurs 5, 10 und 20 Handelstage später, verglichen mit denselben Aktien an
beliebigen Tagen.

    .venv/bin/python research_hot.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from research_4w import THR, pivots

HERE = os.path.dirname(os.path.abspath(__file__))
LOOK = 120


def state(w):
    """Lage am Ende des Fensters → dict oder None, wenn es zu wenige Schwünge gibt."""
    piv = pivots(w, THR)
    if len(piv) < 4:
        return None
    legs = [w[j] / w[i] - 1 for i, j in zip(piv, piv[1:])]
    ups = [l for l in legs if l > 0]
    downs = [-l for l in legs if l < 0]
    if len(ups) < 2 or len(downs) < 2:
        return None
    i0 = piv[-1]
    cur = w[-1] / w[i0] - 1
    typ_up, typ_down = float(np.median(ups)), float(np.median(downs))
    up_days = [j - i for i, j in zip(piv, piv[1:]) if w[j] > w[i]]
    return {
        "legs": len(legs), "down": cur < 0, "drop": -cur, "days": len(w) - 1 - i0,
        "typ_up": typ_up, "typ_down": typ_down, "typ_up_days": float(np.median(up_days)),
        "progress": (-cur / typ_down) if cur < 0 else 0.0,
        "turning": w[-1] > w[-2],
    }


def main():
    frames = pd.read_pickle(os.path.join(HERE, "data", "prices_cache.pkl"))
    keep = set(pd.read_csv(os.path.join(HERE, "data", "screener.csv"))["symbol"])
    rows = []
    for sym, x in frames.items():
        if sym not in keep:
            continue
        a = x["Close"].values.astype(float)
        n = len(a)
        if n < 300:
            continue
        for back in range(20, n - LOOK, 5):
            t = n - back
            s = state(a[t - LOOK:t])
            if s is None:
                continue
            p = a[t - 1]
            fut = a[t:t + 20]
            s.update({"sym": sym, "back": back, "late": back < (n - LOOK) / 2,
                      "f5": a[t + 4] / p - 1, "f10": a[t + 9] / p - 1, "f20": a[t + 19] / p - 1,
                      "peak": fut.max() / p - 1, "peak_day": int(fut.argmax()) + 1,
                      "trough": fut.min() / p - 1})
            rows.append(s)
    D = pd.DataFrame(rows)
    D["grid"] = D.groupby("back")["legs"].rank(pct=True) >= 0.9
    G = D[D.grid]
    print(f"{len(D)} Stichtage x Aktien, davon im Raster: {len(G)}\n")

    def line(label, g):
        if len(g) < 30:
            print(f"  {label:44} zu wenige Fälle ({len(g)})")
            return
        print(f"  {label:44} n={len(g):5}  nach 5 T: {g.f5.mean():+.1%} ({(g.f5 > 0).mean():.0%} im Plus)  "
              f"nach 10 T: {g.f10.mean():+.1%} ({(g.f10 > 0).mean():.0%})  "
              f"nach 20 T: {g.f20.mean():+.1%} ({(g.f20 > 0).mean():.0%})")

    for part, g in [("GESAMT", G), ("ERSTES JAHR", G[~G.late]), ("ZWEITES JAHR", G[G.late])]:
        print(f"=== {part} ===")
        line("Raster, beliebiger Tag", g)
        line("Tief: Abschwung ≥ 80 % des typischen", g[g.progress >= 0.8])
        line("Tief: Abschwung ≥ 100 %", g[g.progress >= 1.0])
        line("Tief: Abschwung ≥ 150 %", g[g.progress >= 1.5])
        line("Tief ≥ 100 % und heute im Plus (dreht)", g[(g.progress >= 1.0) & g.turning])
        print()

    H = G[G.progress >= 1.0]
    print("=== Nach 'Tief ≥ 100 %': Verteilung der nächsten 10 Handelstage ===")
    q = H.f10.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("  Kursänderung:  schlechtestes Zehntel {:+.0%} | unteres Viertel {:+.0%} | Mitte {:+.0%} | "
          "oberes Viertel {:+.0%} | bestes Zehntel {:+.0%}".format(*q))
    r = (H.f10 / H.typ_up).quantile([0.25, 0.5, 0.75])
    print("  im Verhältnis zum typischen Aufschwung der Aktie:  {:+.2f} | {:+.2f} | {:+.2f}".format(*r))
    print(f"  Höchster Stand in den nächsten 20 Tagen: Mitte {H.peak.median():+.0%}, erreicht nach "
          f"{H.peak_day.median():.0f} Tagen; tiefster Stand: Mitte {H.trough.median():+.0%}")
    print(f"  Typischer Aufschwung dieser Aktien zuvor: {H.typ_up.median():.0%} in {H.typ_up_days.median():.0f} Tagen")
    Gq = G.f10.quantile([0.25, 0.5, 0.75])
    print("  Zum Vergleich Raster, beliebiger Tag: {:+.0%} | {:+.0%} | {:+.0%}".format(*Gq))


if __name__ == "__main__":
    main()
