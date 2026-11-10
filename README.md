# ups'n'downs

Zeigt, welche bei Trade Republic handelbaren Aktien am häufigsten auf- und abschwingen, verfolgt eine
Watchlist und schickt Alerts aufs iPhone. Die Order gibst du selbst bei Trade Republic auf.
Keine Anlageberatung.

## Was automatisch läuft (GitHub Actions)

| Ablauf | Wann | Was |
|---|---|---|
| `Alerts` | Mo–Fr alle 15 Minuten während der Handelszeiten, dazu ein Abschlusslauf | Prüft die Watchlist und meldet über ntfy: einmal „vorläufig“, nach Börsenschluss „bestätigt“ oder „nicht bestätigt“ |
| `Rangliste und App` | Mo–Fr 15:45 und 22:40 UTC | Lädt die Trade-Republic-Liste, berechnet die Rangliste neu, prüft das Ranking gegen die Vergangenheit, zählt aus, was nach einem Tief folgte (Ansicht „Hot“), und veröffentlicht die App über GitHub Pages |

Lässt das Ranking nach, kommt montags eine Warnung aufs iPhone.

## Dateien

| Datei | Zweck |
|---|---|
| `watchlist.txt` | Deine Aktien (Yahoo-Symbole), eine pro Zeile. Gilt für Alerts und App. |
| `site/index.html` | Die iPhone-Oberfläche. `site/_head.html` wird beim Veröffentlichen davorgesetzt. |
| `build_site.py` | Bereitet die Daten der App auf (`site/data.json`) und führt den Ranking-Check durch |
| `screener.py` | Rangliste: liquide Yahoo-Aktien, die laut Trade-Republic-Liste handelbar sind |
| `tr_universe.py` | Liest die offizielle Trade-Republic-Liste (PDF) |
| `isin_map.py` | Ordnet ISINs den Yahoo-Kürzeln zu (`data/isin_map.csv`) |
| `alerts.py` | Scan der Watchlist und Push über ntfy. Ohne `NTFY_TOPIC` nur Probelauf. |
| `core.py` | Gemeinsame Rechenlogik |
| `app.py` | Älteres Dashboard für den Mac: `.venv/bin/streamlit run app.py` |
| `research*.py`, `backtest.py` | Die Tests, auf denen Ranking und Einordnung beruhen |

## Stand der Erkenntnis (10.11.2026)

- Verlässlich ist, wie oft eine Aktie schwingt. Der beste Rückblick dafür sind 24 Wochen.
- Von den 30 bestplatzierten Aktien zeigte im Jahresschnitt etwa jede dritte danach wieder ein klares Auf und Ab.
- Wann der nächste Schwung dreht, war in keinem Test vorhersagbar. Kauf- und Verkaufszonen haben keinen
  gesicherten Vorsprung.

## Bedienung

- Watchlist ändern: `watchlist.txt` anpassen, committen, pushen.
- Lauf von Hand: GitHub → Actions → Ablauf wählen → „Run workflow“. Ein Handstart der Alerts meldet immer.
- Der Kanalname für ntfy steht bei GitHub als Secret `NTFY_TOPIC`, nicht im Code.
