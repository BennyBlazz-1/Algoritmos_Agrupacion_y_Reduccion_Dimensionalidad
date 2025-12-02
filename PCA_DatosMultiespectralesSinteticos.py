"""
PCA aplicado a datos multiespectrales sintéticos de parcelas agrícolas.
Genera datos para 4 tipos de cultivo en 8 bandas, aplica PCA, muestra:
 - scatter PC1 vs PC2 coloreado por clase
 - varianza explicada y varianza acumulada
 - curva MSE de reconstrucción vs número de componentes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
np.random.seed(123)

# 1) Generar dataset sintético multiespectral (8 bandas) para 4 clases de vegetación
n_classes = 4
bands = 8
samples_per_class = 300
means = [
    [0.2, 0.3, 0.25, 0.5, 0.6, 0.55, 0.4, 0.35],  # cultivo A (bajas en visible, altas en NIR)
    [0.4, 0.45, 0.5, 0.35, 0.3, 0.25, 0.4, 0.45], # cultivo B (más visible)
    [0.15, 0.2, 0.18, 0.42, 0.7, 0.6, 0.45, 0.3], # cultivo C (alto NIR y SWIR)
    [0.55, 0.5, 0.45, 0.25, 0.2, 0.15, 0.22, 0.3]  # cultivo D (muy distinto)
]
data = []
labels = []
for i in range(n_classes):
    cov = np.diag([0.01]*bands)  # pequeña varianza por banda
    samples = np.random.multivariate_normal(means[i], cov, size=samples_per_class)
    # simular saturación en [0,1]
    samples = np.clip(samples, 0.0, 1.0)
    data.append(samples)
    labels += [f"class_{i}"] * samples_per_class

X = np.vstack(data)
y = np.array(labels)
print("Muestras:", X.shape)

# 2) Escalar (PCA sensible a escala)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) PCA - calcular componentes
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

# 4) Scatter PC1 vs PC2
plt.figure(figsize=(8,6))
classes = np.unique(y)
colors = plt.cm.get_cmap('tab10', len(classes))
for i, cl in enumerate(classes):
    mask = y == cl
    plt.scatter(X_pca[mask,0], X_pca[mask,1], s=18, alpha=0.7, label=cl, color=colors(i))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: PC1 vs PC2 (datos multiespectrales)")
plt.legend()
plt.grid(True)
plt.show()

# 5) Varianza explicada
plt.figure(figsize=(8,4))
plt.bar(range(1, len(explained)+1), explained*100, alpha=0.7, label='Explained per PC')
plt.plot(range(1, len(cum_explained)+1), cum_explained*100, marker='o', color='k', label='Cumulative (%)')
plt.xlabel("Número de componente principal")
plt.ylabel("Varianza explicada (%)")
plt.title("Varianza explicada por componente")
plt.legend()
plt.show()

# 6) Evaluación de reconstrucción: MSE vs número de componentes
mse_list = []
components_range = list(range(1, bands+1))
for k in components_range:
    pca_k = PCA(n_components=k)
    Xk = pca_k.fit_transform(X_scaled)
    X_rec = pca_k.inverse_transform(Xk)
    mse = mean_squared_error(X_scaled, X_rec)
    mse_list.append(mse)

plt.figure(figsize=(8,5))
plt.plot(components_range, mse_list, marker='o')
plt.xlabel("Número de componentes")
plt.ylabel("MSE reconstrucción (en datos escalados)")
plt.title("Error de reconstrucción vs # componentes")
plt.grid(True)
plt.show()

# 7) Elegir k que explique >= 90% varianza
k90 = np.searchsorted(cum_explained, 0.90) + 1
print(f"Número de componentes para explicar >=90% varianza: {k90}")
# Proyección usando k90 y scatter coloreado
pca_k90 = PCA(n_components=k90)
Xk90 = pca_k90.fit_transform(X_scaled)
if k90 >= 2:
    plt.figure(figsize=(8,6))
    for i, cl in enumerate(classes):
        mask = y == cl
        plt.scatter(Xk90[mask,0], Xk90[mask,1] if k90>1 else np.zeros(mask.sum()), 
                    s=18, alpha=0.7, label=cl, color=colors(i))
    plt.xlabel("PC1")
    plt.ylabel("PC2" if k90>1 else "")
    plt.title(f"PCA proyectado a {k90} componentes (>=90% varianza)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("k90 < 2, no es posible visualizar en 2D.")

# 8) Observaciones finales impresas
print("\nVarianza explicada por componente (primeras 6):")
for i, v in enumerate(explained[:6], start=1):
    print(f"PC{i}: {v*100:.2f}%")
print(f"Varianza acumulada (primeras {k90}): {cum_explained[k90-1]*100:.2f}%")
