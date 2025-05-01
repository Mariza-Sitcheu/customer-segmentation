# Customer Segmentation mit KMeans

Dieses Projekt segmentiert Kunden anhand von RFM-Features (Recency, Frequency, Monetary) mithilfe von Clustering-Algorithmen wie **KMeans**. Ziel ist es, Kundenverhalten besser zu verstehen und sinnvolle Gruppen zu identifizieren.

## 🔍 Funktionen

- **RFM-Analyse** aus Transaktionsdaten
- **Standardisierung** der Daten
- **Clustering** mit KMeans (inkl. Elbow & Silhouette Score)
- **Visualisierung** mit PCA (2D)
- **Modell speichern** (KMeans)
- **Evaluierung** der Cluster

---

## 🗂️ Projektstruktur

```
project/
│
├── data/                    # Rohdaten (z. B. Online Retail Dataset)
│   └── Online Retail.xlsx
│
├── models/                  # Gespeicherte Modelle
│   └── kmeans_model.pkl
│
├── src/                     # Source-Code
│   ├── preprocess.py        # Laden & RFM-Erstellung
│   ├── segment.py           # Clustering-Funktionen
│   ├── visualize.py         # Visualisierungen (PCA-Plot etc.)
│   └── evaluate.py          # Elbow, Silhouette Score, Bewertung
│
├── main.py                  # Hauptskript zum Ausführen
├── README.md                # Projektdokumentation
└── requirements.txt         # Python-Abhängigkeiten
```

---

## ⚙️ Installation

```bash
# Repository klonen
git clone https://github.com/dein-nutzername/customer-segmentation.git
cd customer-segmentation

# Virtuelle Umgebung erstellen (optional)
python -m venv .venv
source .venv/bin/activate  # oder .venv\Scripts\activate unter Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

---

## 🚀 Ausführen

```bash
python main.py
```

---

## 🧪 Beispiel: RFM-Kennzahlen

| Kundennr | Recency | Frequency | Monetary |
|----------|---------|-----------|----------|
| 12345    | 15      | 12        | 358.0    |
| 54321    | 200     | 1         | 40.0     |

---

## 📈 Visualisierung

Das Projekt verwendet **PCA**, um die hochdimensionalen RFM-Daten auf 2D zu reduzieren und die Cluster sichtbar zu machen.

---

## 🛠 Verwendete Tools

- Python 3.10+
- pandas, scikit-learn, matplotlib
- KMeans, PCA, StandardScaler

---

## 📚 Datenquelle

Das Projekt verwendet den **[Online Retail Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/online+retail)** mit Bestelldaten aus einem britischen Onlinehandel.

---

