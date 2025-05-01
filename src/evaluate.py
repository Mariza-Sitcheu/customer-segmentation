import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_elbow(data, max_k=10):
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title("Elbow-Methode: Optimale Clusteranzahl")
    plt.xlabel("Anzahl der Cluster")
    plt.ylabel("Inertia (SSE)")
    plt.grid(True)
    plt.show()

def evaluate_silhouette(data, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), scores, marker='o')
    plt.title("Silhouette Score vs Clusteranzahl")
    plt.xlabel("Anzahl der Cluster")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()
# Bewertet das Clustering mit Silhouette-Score und Inertia (SSE)
def evaluate_clustering(model, X):
    labels = model.labels_
    silhouette = silhouette_score(X, labels)
    inertia = model.inertia_

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Inertia (SSE): {inertia:.4f}")