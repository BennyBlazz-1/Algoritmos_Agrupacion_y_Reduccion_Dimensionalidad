"""
DBSCAN aplicado a eventos sísmicos sintéticos.
Genera datos, aplica DBSCAN, muestra: 
 - scatter lat/lon con clusters y ruido
 - bar plot de #clusters vs eps (barrido rápido)
 - silhouette score (excluyendo ruido)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
import random

random.seed(42)
np.random.seed(42)

# 1) Generar dataset sintético de eventos sísmicos
# Creamos varias "zonas de falla" con clusters espaciales y eventos aislados (ruido)
def generate_seismic_events():
    clusters = [
        {"center": (35.5, -117.7), "n": 180, "scale": 0.02},
        {"center": (36.1, -118.2), "n": 120, "scale": 0.015},
        {"center": (35.9, -117.3), "n": 80,  "scale": 0.01},
        {"center": (36.4, -117.9), "n": 60,  "scale": 0.025}
    ]
    points = []
    for c in clusters:
        lat0, lon0 = c["center"]
        lat = np.random.normal(lat0, c["scale"], size=c["n"])
        lon = np.random.normal(lon0, c["scale"], size=c["n"])
        depth = np.abs(np.random.normal(8.0, 3.0, size=c["n"]))  # km
        mag = np.clip(np.random.normal(3.2, 0.6, size=c["n"]), 1.5, 6.0)
        t = np.random.randint(0, 1000, size=c["n"])
        points.append(np.column_stack([lat, lon, depth, mag, t]))
    # ruido: eventos aislados alrededor
    noise_n = 70
    noise_lat = np.random.uniform(35.2, 36.6, size=noise_n)
    noise_lon = np.random.uniform(-118.4, -117.0, size=noise_n)
    noise_depth = np.abs(np.random.normal(10.0, 6.0, size=noise_n))
    noise_mag = np.clip(np.random.normal(2.5, 1.0, size=noise_n), 1.0, 5.5)
    noise_t = np.random.randint(0, 1000, size=noise_n)
    points.append(np.column_stack([noise_lat, noise_lon, noise_depth, noise_mag, noise_t]))
    allp = np.vstack(points)
    df = pd.DataFrame(allp, columns=["lat", "lon", "depth_km", "magnitude", "time"])
    return df

df = generate_seismic_events()
print(f"Eventos generados: {len(df)}")

# 2) Selección de características para clustering
# Clustering espacial (lat, lon) — si quisieras incluir profundidad o tiempo, agregarlos.
X = df[["lat", "lon"]].values

# Estándar o escala: para lat/lon en grados, escalamos para que eps sea interpretable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Función para ejecutar DBSCAN y mostrar resultados
def run_dbscan_and_plot(X_scaled, X_original, eps=0.3, min_samples=8, ax=None):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap('tab20', max(1, len(unique_labels)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X_original[class_member_mask]
        if k == -1:
            ax.scatter(xy[:,1], xy[:,0], s=18, c='k', marker='x', label='ruido')
        else:
            ax.scatter(xy[:,1], xy[:,0], s=30, color=colors(k), label=f'cluster {k}')
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"DBSCAN eps={eps}, min_samples={min_samples} -> clusters={n_clusters}, ruido={n_noise}")
    ax.legend(loc='upper right', fontsize='small')
    # Silhouette (solo si hay al menos 2 clusters no-ruido)
    if n_clusters >= 2:
        mask = labels != -1
        sil = silhouette_score(X_scaled[mask], labels[mask])
        print(f"Silhouette (excluyendo ruido): {sil:.3f}")
    else:
        print("No hay al menos 2 clusters para calcular silhouette (excluyendo ruido).")
    return labels, n_clusters, n_noise

# 4) Ejecutar con parámetros iniciales y mostrar plot
fig, ax = plt.subplots(1,2, figsize=(14,6))
labels_base, ncl, nnoise = run_dbscan_and_plot(X_scaled, X, eps=0.35, min_samples=10, ax=ax[0])

# 5) Barrido rápido de eps para ver #clusters vs eps (min_samples fijo)
eps_vals = np.linspace(0.15, 0.6, 10)
min_samples = 8
clusters_list = []
noise_list = []
for e in eps_vals:
    db = DBSCAN(eps=e, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    clusters_list.append(n_clusters)
    noise_list.append(n_noise)

ax[1].plot(eps_vals, clusters_list, marker='o', label='#clusters')
ax[1].plot(eps_vals, noise_list, marker='x', label='#ruido')
ax[1].set_xlabel("eps (en espacio estandarizado)")
ax[1].set_ylabel("Cantidad")
ax[1].set_title("Sensibilidad a eps (min_samples=8)")
ax[1].legend()

plt.tight_layout()
plt.show()

# 6) Ejemplo de evaluación/optimización: buscar combinación que maximice silhouette
eps_grid = np.linspace(0.18, 0.5, 15)
min_grid = [5, 8, 10, 12]
best = {"sil": -1, "eps": None, "min_samples": None, "labels": None}
for eps, mn in itertools.product(eps_grid, min_grid):
    db = DBSCAN(eps=eps, min_samples=mn).fit(X_scaled)
    labels = db.labels_
    # Only compute silhouette when >=2 clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters >= 2:
        mask = labels != -1
        try:
            sil = silhouette_score(X_scaled[mask], labels[mask])
        except Exception:
            sil = -1
        if sil > best["sil"]:
            best.update({"sil": sil, "eps": eps, "min_samples": mn, "labels": labels})
print("\nMejor combinación según silhouette (excluyendo ruido):")
print(best)

# 7) Plot del mejor resultado encontrado
fig, ax = plt.subplots(figsize=(8,6))
if best["labels"] is not None:
    run_dbscan_and_plot(X_scaled, X, eps=best["eps"], min_samples=best["min_samples"], ax=ax)
else:
    ax.text(0.5, 0.5, "No se encontró configuración con >=2 clusters", ha='center')
plt.show()
