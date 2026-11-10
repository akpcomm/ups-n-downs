"""
Scan der Watchlist → Push-Nachricht aufs iPhone (über die App "ntfy").

Läuft während der Handelszeiten alle 15 Minuten (siehe .github/workflows/alerts.yml).

    .venv/bin/python alerts.py                    # Probelauf: zeigt die Nachricht nur an
    .venv/bin/python alerts.py --date 2025-04-04  # so tun, als wäre dieser Tag heute (zum Testen)
    .venv/bin/python alerts.py --always           # auch melden, wenn es nichts Neues gibt
    NTFY_TOPIC=… .venv/bin/python alerts.py       # wirklich senden

Ablauf je Aktie und Handelstag
  1. Tritt die Aktie im Tagesverlauf in eine Zone ein, kommt EINMAL eine Meldung "vorläufig".
  2. Nach Börsenschluss kommt EINMAL das Ergebnis: "bestätigt" oder "nicht bestätigt".
     Die getesteten Signale beruhen auf Schlusskursen – erst dann gilt ein Signal.
Was schon gemeldet wurde, steht in state/sent.json.

Keine Anlageberatung; die Order gibst du selbst auf.
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, time, timedelta

import pandas as pd

import core

try:
    from zoneinfo import ZoneInfo
    BERLIN = ZoneInfo("Europe/Berlin")
except Exception:  # pragma: no cover
    BERLIN = None

WIN, Z_THR = 20, 2.0
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh")
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state", "sent.json")
# Ab wann der Tageskurs als Schlusskurs gilt (deutsche Zeit, mit etwas Puffer)
CLOSE_DE = time(17, 50)   # Xetra schließt 17:30
CLOSE_US = time(22, 15)   # US-Börsen schließen 22:00


def read_watchlist(path="watchlist.txt"):
    with open(os.path.join(HERE, path)) as f:
        return [l.split("#")[0].strip().upper() for l in f if l.split("#")[0].strip()]


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state, today):
    keep = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    state = {k: v for k, v in state.items() if k.split("|")[1] >= keep}
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f, indent=0, sort_keys=True)


def is_final(ticker, bar_date, now):
    """Ist der letzte Tageskurs schon ein Schlusskurs?"""
    if bar_date.date() < now.date():
        return True
    return now.time() >= (CLOSE_DE if ticker.endswith(".DE") else CLOSE_US)


def scan(now, as_of=None):
    """→ Liste aller Aktien mit ihrem aktuellen Stand, Fehler, Markt-Z."""
    mkt_z = core.market_z("2y")
    if as_of is not None:
        mkt_z = mkt_z[:as_of]
    rows, errors = [], []
    for t in read_watchlist():
        try:
            df = core.fetch(t, "2y")
            if as_of is not None:
                df = df[:as_of]
            if len(df) < WIN + 5:
                errors.append(t)
                continue
            df = core.add_indicators(df, WIN, Z_THR, mkt_z)
        except Exception:
            errors.append(t)
            continue
        last, bar = df.iloc[-1], df.index[-1]
        if pd.Timestamp(now.date()) - bar > timedelta(days=4):   # veraltete Daten nicht melden
            errors.append(t)
            continue
        kind = "panic" if last["PANIC"] else "buy" if last["BUY"] else "sell" if last["SELL"] else None
        rows.append({
            "ticker": t, "kind": kind, "bar": bar.strftime("%Y-%m-%d"),
            "final": as_of is not None or is_final(t, bar, now),
            "close": float(last["Close"]), "z": float(last["Z"]), "dd60": float(last["DD60"]),
            "trend": core.trend_label(last),
            "day": float(last["Close"] / df["Close"].iloc[-2] - 1),
        })
    mz = float(mkt_z.dropna().iloc[-1])
    return rows, errors, mz


def decide(rows, state):
    """Was ist neu zu melden? → (events, neue Einträge für den Zustand)"""
    events, new = [], {}
    for r in rows:
        base = f"{r['ticker']}|{r['bar']}"
        prov = state.get(base + "|vorlaeufig")       # im Tagesverlauf gemeldete Signalart
        done = base + "|schluss" in state
        if r["kind"] and not r["final"]:
            if prov != r["kind"]:
                events.append({**r, "status": "vorläufig"})
                new[base + "|vorlaeufig"] = r["kind"]
        elif r["final"] and not done:
            if r["kind"]:
                events.append({**r, "status": "bestätigt" if prov else "Schlusskurs"})
                new[base + "|schluss"] = r["kind"]
            elif prov:
                events.append({**r, "kind": prov, "status": "nicht bestätigt"})
                new[base + "|schluss"] = "keins"
    return events, new


LABEL = {
    "panic": "🟢🟢 KAUFZONE + Markt-Ausverkauf",
    "buy": "🟢 Kaufzone",
    "sell": "🔴 Verkaufszone",
}
NOTE = {
    "panic": "Im breiten Test das Signal mit dem besten Bild (+0,6 bis +1,1 % Vorsprung je Trade, "
             "13 von 17 Jahren positiv), statistisch aber nicht gesichert.",
    "buy": "Im breiten Test ohne belastbaren Vorsprung.",
    "sell": "Im breiten Test ohne Vorsprung; Stärke lief im Schnitt eher weiter.",
}


def build_message(events, errors, mz, now, always=False):
    if not events and not always:
        return None
    lines = []
    live = [e for e in events if e["status"] != "nicht bestätigt"]
    for kind in ("panic", "buy", "sell"):
        group = sorted((e for e in live if e["kind"] == kind), key=lambda e: e["z"])
        if not group:
            continue
        lines.append(LABEL[kind])
        for e in group:
            lines.append(f"  {e['ticker']}  {e['close']:.2f}  [{e['status']}]  ({e['day']:+.1%} heute, "
                         f"{e['dd60']:+.0%} vom 60-T-Hoch, Trend {e['trend']})")
        lines.append(f"  → {NOTE[kind]}")
        lines.append("")
    gone = [e for e in events if e["status"] == "nicht bestätigt"]
    if gone:
        lines.append("⚪ Nicht bestätigt (zum Schlusskurs wieder im Normalbereich)")
        for e in gone:
            lines.append(f"  {e['ticker']}  {e['close']:.2f}  (war: {LABEL[e['kind']].split(' ', 1)[1]})")
        lines.append("")
    if any(e["status"] == "vorläufig" for e in events):
        lines.append("Vorläufig = Zwischenstand im Handelstag. Ob das Signal gilt, entscheidet der "
                     "Schlusskurs; dazu kommt eine zweite Meldung.\n")
    if not events:
        lines.append("Nichts Neues in deiner Watchlist.\n")
    state_txt = "AUSVERKAUF" if mz < core.MKT_PANIC else "normal"
    lines.append(f"Markt (S&P 500): {mz:+.1f} SD → {state_txt}")
    if errors:
        lines.append("Ohne aktuelle Daten: " + ", ".join(errors))
    lines.append(f"Stand {now:%H:%M} Uhr, Kurse der Heimatbörse (teils 15 Min. verzögert). Keine Anlageberatung.")
    n = len(events)
    strong = any(e["kind"] == "panic" and e["status"] != "nicht bestätigt" for e in events)
    title = f"ups'n'downs {now:%d.%m. %H:%M}: " + (f"{n} Meldung{'en' if n != 1 else ''}" if n else "nichts Neues")
    return {"title": title, "message": "\n".join(lines), "priority": 4 if strong else 3,
            "tags": ["chart_with_upwards_trend"]}


def send(msg, topic):
    body = json.dumps({"topic": topic, **msg}).encode("utf-8")
    req = urllib.request.Request(NTFY_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Stichtag JJJJ-MM-TT (zum Testen, ändert den Zustand nicht)")
    ap.add_argument("--always", action="store_true", help="auch melden, wenn es nichts Neues gibt")
    a = ap.parse_args()
    as_of = pd.Timestamp(a.date) if a.date else None
    now = datetime.now(BERLIN).replace(tzinfo=None) if BERLIN else datetime.now()
    if as_of is not None:
        now = as_of.to_pydatetime().replace(hour=23)

    rows, errors, mz = scan(now, as_of)
    if not rows:
        print("Keine Kursdaten erhalten: " + ", ".join(errors))
        return 1
    state = {} if as_of is not None else load_state()
    events, new = decide(rows, state)

    msg = build_message(events, errors, mz, now, always=a.always)
    if msg is None:
        print(f"{now:%H:%M} Uhr: nichts Neues – keine Nachricht. ({len(rows)} Aktien geprüft)")
        return 0
    print(msg["title"] + "\n" + "-" * len(msg["title"]) + "\n" + msg["message"])
    topic = os.environ.get("NTFY_TOPIC")
    if topic and as_of is None:
        print(f"\nGesendet (HTTP {send(msg, topic)}).")
        state.update(new)
        save_state(state, now)      # erst nach erfolgreichem Senden merken
    else:
        print("\n[Probelauf – nichts gesendet, Zustand unverändert]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
