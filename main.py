from datetime import datetime
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_data, create_rfm
from src.segment import segment_customers, save_model
from src.visualize import plot_clusters
from src.evaluate import evaluate_elbow, evaluate_silhouette, evaluate_clustering
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # vermeidet eine Fehlermeldung

def main():
    # 1. Daten laden und RFM-Matrix erstellen
    df = load_data("data/Online Retail.xlsx")
    rfm = create_rfm(df, ref_date=datetime(2011, 1, 1))

    # 2. Daten skalieren für Clustering
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # 3. Clusteranzahl evaluieren
    evaluate_elbow(rfm_scaled)
    evaluate_silhouette(rfm_scaled)

    # 4. Clustering mit KMeans (Clusteranzahl manuell wählen)
    rfm, kmeans = segment_customers(rfm, n_clusters=6)

    # 5. Modell speichern
    save_model(kmeans)

    # 6. Cluster visualisieren
    plot_clusters(rfm)

    # 7. Evaluation richtig machen
    evaluate_clustering(kmeans, rfm_scaled)

if __name__ == "__main__":
    main()




