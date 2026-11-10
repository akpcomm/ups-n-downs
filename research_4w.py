"""
Prüft die Idee: "Wenn eine Aktie in den letzten vier Wochen ein klares Auf-und-Ab-Muster zeigt,
sollte das ihr Ranking verstärken."

Dazu wird jedes Vier-Wochen-Fenster (20 Handelstage) der letzten zwei Jahre vermessen und
geschaut, was in den Wochen DANACH passiert ist.

Klares Muster (vorab festgelegt):
  mindestens 3 abgeschlossene Schwünge von je mindestens 5 % in 20 Handelstagen
  und Pendel-Anteil ≥ 0,8 (viel Hin und Her, wenig Netto-Bewegung)

Nutzt die Kurse, die screener.py heute schon geladen hat (data/prices_cache.pkl).

    .venv/bin/python research_4w.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W, STEP, H = 20, 10, 10      # Fensterlänge, Abstand der Stichtage, Blick nach vorn (Handelstage)
THR = 0.05                   # Mindestgröße eines Schwungs im Vier-Wochen-Fenster
MIN_LEGS, MIN_PENDEL = 3, 0.8


def pivots(v, thr):
    """Wendepunkte wie core.zigzag, aber auf einem nackten Array (schnell)."""
    out, trend, hi, lo = [], 0, 0, 0
    for i in range(1, len(v)):
        p = v[i]
        if trend == 0:
            if p > v[hi]:
                hi = i
            if p < v[lo]:
                lo = i
            if v[hi] / v[lo] - 1 >= thr:
                if hi > lo:
                    out.append(lo); trend = 1
                else:
                    out.append(hi); trend = -1
        elif trend == 1:
            if p > v[hi]:
                hi = i
            elif p <= v[hi] * (1 - thr):
                out.append(hi); trend, lo = -1, i
        else:
            if p < v[lo]:
                lo = i
            elif p >= v[lo] * (1 + thr):
                out.append(lo); trend, hi = 1, i
    return out


def window_metrics(w):
    piv = pivots(w, THR)
    pts = [w[i] for i in piv]
    legs = max(len(pts) - 1, 0)
    path = [w[0]] + pts + [w[-1]]
    travel = sum(abs(np.log(b / a)) for a, b in zip(path, path[1:]))
    net = abs(np.log(w[-1] / w[0]))
    pendel = 1 - net / travel if travel > 0 else 0.0
    lo, hi = w.min(), w.max()
    pos = (w[-1] - lo) / (hi - lo) if hi > lo else 0.5
    return legs, pendel, pos, lo, hi


def is_clear(legs, pendel):
    return legs >= MIN_LEGS and pendel >= MIN_PENDEL


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
        for end in range(W, n - W, STEP):          # Fenster = c[end-W:end], danach folgt c[end:]
            legs, pendel, pos, lo, hi = window_metrics(c[end - W:end])
            legs2, pendel2, _, _, _ = window_metrics(c[end:end + W])
            nxt = c[end:end + H]
            rows.append({
                "sym": sym, "t": end, "late": end > n // 2,
                "legs": legs, "pendel": pendel, "pos": pos, "clear": is_clear(legs, pendel),
                "legs2": legs2, "pendel2": pendel2, "clear2": is_clear(legs2, pendel2),
                "fwd": c[end - 1 + H] / c[end - 1] - 1,
                "stay": float(((nxt >= lo) & (nxt <= hi)).mean()),   # Anteil der Folgetage in der alten Spanne
            })
    D = pd.DataFrame(rows)
    print(f"{D['sym'].nunique()} Aktien, {len(D)} Vier-Wochen-Fenster\n")

    print("=== 1) Wie häufig ist ein klares Muster? ===")
    print(f"  {D['clear'].mean():.1%} aller Fenster\n")

    print("=== 2) Setzt sich das Muster fort? (nächste vier Wochen) ===")
    print(f"  Klares Muster jetzt  → danach wieder klares Muster: {D[D.clear]['clear2'].mean():.1%}")
    print(f"  Kein klares Muster   → danach klares Muster:        {D[~D.clear]['clear2'].mean():.1%}")
    for a, b, label in [("legs", "legs2", "Anzahl Schwünge"), ("pendel", "pendel2", "Pendel-Anteil")]:
        r = D.groupby("t").apply(lambda g: g[a].rank().corr(g[b].rank())).mean()
        print(f"  Rang-Korrelation {label:16} jetzt ↔ danach: {r:+.2f}")
    print(f"  Folgetage innerhalb der alten Spanne: klares Muster {D[D.clear]['stay'].mean():.0%}, "
          f"sonst {D[~D.clear]['stay'].mean():.0%}\n")

    print("=== 3) Lohnt 'unten in der Spanne kaufen'? (Kurs 10 Handelstage später) ===")

    def low_high(g):
        lo, hi = g[g.pos < 1 / 3], g[g.pos > 2 / 3]
        return lo, hi

    def tstat(g):
        per = g.groupby("t").apply(lambda q: q[q.pos < 1 / 3]["fwd"].mean() - q[q.pos > 2 / 3]["fwd"].mean()).dropna()
        return per.mean() / per.std() * np.sqrt(len(per)) if len(per) > 2 else np.nan

    for label, g in [("Klares Muster, gesamt", D[D.clear]),
                     ("Klares Muster, 1. Jahr", D[D.clear & ~D.late]),
                     ("Klares Muster, 2. Jahr", D[D.clear & D.late]),
                     ("Kein klares Muster", D[~D.clear])]:
        lo, hi = low_high(g)
        print(f"  {label:24} unten: n={len(lo):6} Ø {lo.fwd.mean():+.2%} ({(lo.fwd > 0).mean():.0%} im Plus)   "
              f"oben: n={len(hi):6} Ø {hi.fwd.mean():+.2%} ({(hi.fwd > 0).mean():.0%} im Plus)   "
              f"Unterschied {lo.fwd.mean() - hi.fwd.mean():+.2%}  t={tstat(g):+.1f}")

    print("\n=== 4) Stand heute: letzte vier Wochen ===")
    today = []
    for sym, x in frames.items():
        if sym not in keep:
            continue
        c = x["Close"].values.astype(float)
        if len(c) < W:
            continue
        legs, pendel, pos, lo, hi = window_metrics(c[-W:])
        today.append({"sym": sym, "legs": legs, "pendel": pendel, "pos": pos,
                      "spanne": hi / lo - 1, "clear": is_clear(legs, pendel)})
    T = pd.DataFrame(today).set_index("sym")
    print(f"  {int(T.clear.sum())} von {len(T)} Aktien zeigen aktuell ein klares Muster.")
    for s in ["SPCX", "RKLB", "TSLA", "NVDA", "ASTS"]:
        if s in T.index:
            r = T.loc[s]
            print(f"  {s:5} Schwünge {int(r.legs)}, Pendel-Anteil {r.pendel:.2f}, Spanne {r.spanne:.0%}, "
                  f"Lage in der Spanne {r.pos:.0%} → {'klares Muster' if r.clear else 'kein klares Muster'}")


if __name__ == "__main__":
    main()
