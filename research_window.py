"""
Welcher Rückblick findet Aktien mit wiederkehrendem Auf und Ab am besten?

Ziel: Aktien vorhersagen, die in den NÄCHSTEN vier Wochen ein klares Muster zeigen
(mindestens 3 Schwünge von je 5 %, Pendel-Anteil ≥ 0,8).

Verglichen werden
  A) Rückblicke von 2, 4, 8, 12 und 24 Wochen (Muster-Wert = Schwünge je 4 Wochen × Pendel-Anteil)
  B) Beständigkeit: in wie vielen der letzten sechs Vier-Wochen-Blöcke gab es ein klares Muster?

    .venv/bin/python research_window.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from research_4w import MIN_LEGS, MIN_PENDEL, THR, pivots

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKS = [10, 20, 40, 60, 120]   # Handelstage
BLOCK, NBLOCKS = 20, 6


def osc(w):
    """→ (Schwünge, Pendel-Anteil) eines Kursfensters."""
    pts = [w[i] for i in pivots(w, THR)]
    legs = max(len(pts) - 1, 0)
    path = [w[0]] + pts + [w[-1]]
    travel = sum(abs(np.log(b / a)) for a, b in zip(path, path[1:]))
    return legs, (1 - abs(np.log(w[-1] / w[0])) / travel) if travel > 0 else 0.0


def clear(legs, pendel, days=BLOCK):
    return legs >= MIN_LEGS * days / BLOCK and pendel >= MIN_PENDEL


def main():
    frames = pd.read_pickle(os.path.join(HERE, "data", "prices_cache.pkl"))
    keep = set(pd.read_csv(os.path.join(HERE, "data", "screener.csv"))["symbol"])
    rows = []
    for sym, x in frames.items():
        if sym not in keep:
            continue
        c = x["Close"].values.astype(float)
        n = len(c)
        if n < 300:
            continue
        for t in range(BLOCK * NBLOCKS, n - BLOCK, 10):
            r = {"sym": sym, "t": t}
            for L in LOOKS:
                legs, pen = osc(c[t - L:t])
                r[f"s{L}"] = legs * BLOCK / L * pen
            blocks = [clear(*osc(c[t - (k + 1) * BLOCK:t - k * BLOCK])) for k in range(NBLOCKS)]
            r["last"] = blocks[0]
            r["count"] = sum(blocks)
            # Beständigkeits-Wert: jüngere Blöcke zählen mehr
            r["bestand"] = sum(b * w for b, w in zip(blocks, [6, 5, 4, 3, 2, 1])) / 21
            legs, pen = osc(c[t:t + BLOCK])
            r["y"] = clear(legs, pen)
            rows.append(r)
    D = pd.DataFrame(rows)
    base = D["y"].mean()
    print(f"{D['sym'].nunique()} Aktien, {len(D)} Stichtage. Grundquote: {base:.1%} zeigen in den "
          "folgenden vier Wochen ein klares Muster.\n")

    def top_decile(col):
        """Trefferquote der 10 % besten Aktien je Stichtag."""
        hit = D.groupby("t").apply(lambda g: g.nlargest(max(len(g) // 10, 1), col)["y"].mean())
        return hit.mean()

    print("=== A) Länge des Rückblicks ===")
    print("  Von den 10 % Aktien mit dem höchsten Muster-Wert zeigen danach ein klares Muster:")
    for L in LOOKS:
        p = top_decile(f"s{L}")
        print(f"    Rückblick {L // 5:2d} Wochen: {p:.1%}   (das {p / base:.1f}-Fache der Grundquote)")

    print("\n=== B) Beständigkeit über sechs Vier-Wochen-Blöcke (24 Wochen) ===")
    print("  Blöcke mit klarem Muster → danach wieder klares Muster")
    for k, g in D.groupby("count"):
        print(f"    {k} von 6: {g['y'].mean():6.1%}   (n = {len(g)})")
    p = top_decile("bestand")
    print(f"  Die 10 % mit dem höchsten Beständigkeits-Wert: {p:.1%}   (das {p / base:.1f}-Fache)")

    print("\n=== C) Kombination: Beständigkeit + Muster-Wert der letzten 8 Wochen ===")
    for a in (0.0, 0.3, 0.5, 0.7, 1.0):
        D["combo"] = a * D.groupby("t")["bestand"].rank(pct=True) + (1 - a) * D.groupby("t")["s40"].rank(pct=True)
        p = top_decile("combo")
        print(f"    {a:.0%} Beständigkeit / {1 - a:.0%} letzte 8 Wochen: {p:.1%}   (das {p / base:.1f}-Fache)")

    print("\n=== D) Hält das in beiden Hälften des Zeitraums? (beste Einzelwerte) ===")
    mid = D["t"].median()
    for label, g in [("frühe Hälfte", D[D.t <= mid]), ("späte Hälfte", D[D.t > mid])]:
        out = []
        for col in ("s20", "s40", "s120", "bestand"):
            hit = g.groupby("t").apply(lambda q: q.nlargest(max(len(q) // 10, 1), col)["y"].mean()).mean()
            out.append(f"{col}: {hit:.1%}")
        print(f"    {label}: Grundquote {g['y'].mean():.1%} | " + " | ".join(out))


if __name__ == "__main__":
    main()
