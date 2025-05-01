import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle

def segment_customers(rfm: pd.DataFrame, n_clusters: int = 4):
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    pca = PCA(n_components=2)
    rfm[['PCA1', 'PCA2']] = pca.fit_transform(rfm_scaled)

    return rfm, kmeans

def save_model(model, path='models/kmeans_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


