import pandas as pd
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    # Lade Excel-Datei
    df = pd.read_excel(path)

    # Konvertiere InvoiceDate in datetime-Format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d.%m.%Y %H:%M')

    # Konvertiere UnitPrice von Komma zu Punkt
    df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(',', '.').astype(float)

    # Entferne Zeilen ohne CustomerID
    df = df.dropna(subset=['CustomerID'])

    return df

def create_rfm(df: pd.DataFrame, ref_date: datetime) -> pd.DataFrame:
    # Berechne TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Erzeuge RFM-Features
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                              # Frequency
        'TotalPrice': 'sum'                                  # Monetary
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

