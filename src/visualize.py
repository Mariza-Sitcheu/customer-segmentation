import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_clusters(rfm: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=100)
    plt.title('Kundensegmente (PCA-Reduktion)')
    plt.show()

