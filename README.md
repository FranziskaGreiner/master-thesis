Masterarbeit Franziska Greiner
## Optimierung der Nachhaltigkeit von Softwarearchitekturen durch KI-basierte Prognose der Kohlenstoffintensität

### Repo Aufbau
- thesis: Schriftliche Ausarbeitung
  - thesis.pdf: Finale PDF-Datei
  - thesis.bib: Literaturquellen
- time-series-forecasting: Zeitreihenvorhersage
  - notebooks: Jupyter Notebooks
  - reports: Weights & Biases Reports
  - src: Modelle und Konfiguration
  - app.py: API zum lokalen Abrufen der Modellvorhersagen

*Wegen Lizenzvereinbarung mit WattTime können die MOER-Daten nicht zur Verfügung gestellt werden*

### Abstract

### Gliederung
1. Einleitung
   - Motivation
   - Forschungsziele
   - Struktur der Arbeit
   - Themenabgrenzung
   - Verwandte Arbeiten
2. Grundlagen nachhaltiger Software
   - Definition Nachhaltigkeit
   - Nachhaltige Software
   - Kohlenstoffintensität als Maß für Nachhaltigkeit
3. Strategien zur Nachhaltigkeitsoptimierung
   - Scheduling
   - Scaling
   - Demand Shifting
   - Demand Shaping
   - Voraussetzungen zur Anwendung der Strategien
4. Grundlagen zur Zeitreihenprognose
   - Einführung in die Zeitreihenprognose
   - SARIMAX als statistisches Basismodell
   - Temporal Fusion Transformer als KI-basiertes Modell
5. Prognose der Kohlenstoffintensität
   - Analyse und Vorverarbeitung der Daten
   - Auswahl der korrelierten Eingabedaten
   - Stromnetzdaten
   - Modellentwicklung und Training mit Temporal Fusion Transformer
   - Evaluierung und Vergleich der Modelle
6. Anwendungsmöglichkeiten der Strategien auf Basis der Prognose
   - Anwendungsszenarien
   - Verfügbare Tools
   - API zur Abfrage der Prognose
7. Diskussion
   - Datenmodell und Prognose
   - Anwendung der Vorhersagen
   - Nachhaltige Softwareanwendungen
8. Fazit
