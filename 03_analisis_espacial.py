# -*- coding: utf-8 -*-
"""
================================================================================
ANÁLISIS ESPACIAL AVANZADO (sección 4.5.4.6 de la tesis)
================================================================================
A partir de muestras_132.csv (generado por 01_exportar_muestras.py):
  1. Índice I de Moran con correlograma (bandas de distancia) para cada una de
     las 8 variables predictoras, con p-valor por permutaciones (999).
     Implementado con numpy puro: no requiere librerías adicionales.
  2. Validación cruzada espacial por bloques (block sizes de 10, 20 y 30 km),
     comparada con la validación cruzada aleatoria estratificada 10-fold,
     para RF, SVM y ANN con los hiperparámetros finales.

Salidas: moran_correlograma.csv/.png y cv_espacial_vs_aleatoria.csv/.png
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False


class Config:
    # === AJUSTA ESTAS RUTAS / VALORES ===
    CSV_MUESTRAS = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\muestras_132.csv"
    OUTPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\Analisis_Espacial"

    # <<< REEMPLAZAR por los valores finales de tu SVM (model_summary_svm.csv) >>>
    SVM_C = 1
    SVM_GAMMA = 0.1

    # Bandas de distancia para el correlograma de Moran (metros)
    BANDAS_MORAN = [5000, 10000, 20000, 30000, 50000]
    N_PERMUTACIONES = 999

    # Tamaños de bloque para la validación cruzada espacial (metros)
    BLOQUES = [10000, 20000, 30000]
    SEED = 42


# ----------------------------------------------------------------------------
# I de Moran (implementación propia, pesos binarios por banda de distancia)
# ----------------------------------------------------------------------------
def morans_i(valores, coords, dmax, n_perm=999, seed=42):
    """I de Moran global con pesos binarios w_ij = 1 si 0 < d_ij <= dmax.
    Devuelve (I, I_esperado, p_valor_permutacional)."""
    z = valores - valores.mean()
    n = len(z)
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    d = np.sqrt(dx ** 2 + dy ** 2)
    W = ((d > 0) & (d <= dmax)).astype(float)
    s0 = W.sum()
    if s0 == 0:
        return np.nan, -1.0 / (n - 1), np.nan
    denom = (z ** 2).sum()
    I_obs = (n / s0) * (z @ W @ z) / denom
    # p-valor por permutaciones
    rng = np.random.default_rng(seed)
    cuenta = 0
    for _ in range(n_perm):
        zp = rng.permutation(z)
        I_p = (n / s0) * (zp @ W @ zp) / denom
        if abs(I_p) >= abs(I_obs):
            cuenta += 1
    p = (cuenta + 1) / (n_perm + 1)
    return I_obs, -1.0 / (n - 1), p


def construir_modelos():
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, max_features='log2',
                                random_state=Config.SEED, n_jobs=-1)
    svm = Pipeline([('scaler', StandardScaler()),
                    ('svc', SVC(kernel='rbf', C=Config.SVM_C, gamma=Config.SVM_GAMMA,
                                probability=True, class_weight='balanced',
                                random_state=Config.SEED))])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                        alpha=0.001, learning_rate_init=0.005, max_iter=500,
                        early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=30, random_state=Config.SEED)
    if SMOTE_OK:
        ann = ImbPipeline([('scaler', StandardScaler()),
                           ('smote', SMOTE(sampling_strategy=1.0, random_state=Config.SEED,
                                           k_neighbors=5)),
                           ('mlp', mlp)])
    else:
        ann = Pipeline([('scaler', StandardScaler()), ('mlp', mlp)])
    return {'RF': rf, 'SVM': svm, 'ANN': ann}


def cv_por_grupos(modelo_fn, X, y, grupos, min_clase=3):
    """CV dejando fuera un bloque (o grupo de bloques) a la vez.
    Omite los folds cuyo conjunto de validación no tenga ambas clases."""
    aucs = []
    for g in np.unique(grupos):
        test = grupos == g
        train = ~test
        if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
            continue
        if min(np.bincount(y[train])) < min_clase:
            continue
        m = modelo_fn()
        m.fit(X[train], y[train])
        aucs.append(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1]))
    return np.array(aucs)


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(Config.CSV_MUESTRAS)
    no_pred = ['clase', 'X', 'Y', 'fila', 'columna']
    variables = [c for c in df.columns if c not in no_pred]
    X = df[variables].values
    y = df['clase'].values.astype(int)
    coords = df[['X', 'Y']].values
    print(f"Muestras: {len(y)} | Variables: {len(variables)}")

    # ---------- 1. Correlograma de Moran ----------
    print("\n[1/2] I de Moran por variable y banda de distancia...")
    filas = []
    for var in variables + ['clase']:
        vals = df[var].values.astype(float)
        for dmax in Config.BANDAS_MORAN:
            I, Ie, p = morans_i(vals, coords, dmax, Config.N_PERMUTACIONES, Config.SEED)
            filas.append({'Variable': var, 'Distancia_m': dmax, 'Moran_I': round(I, 4),
                          'I_esperado': round(Ie, 4), 'p_valor': round(p, 4),
                          'Significativo_0.05': 'Sí' if (not np.isnan(p) and p < 0.05) else 'No'})
        print(f"  {var}: listo")

    moran_df = pd.DataFrame(filas)
    moran_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'moran_correlograma.csv'),
                    index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(11, 6))
    for var in variables:
        sub = moran_df[moran_df['Variable'] == var]
        ax.plot(sub['Distancia_m'] / 1000, sub['Moran_I'], 'o-', label=var, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Distancia máxima de vecindad (km)')
    ax.set_ylabel("I de Moran")
    ax.set_title('Correlograma de autocorrelación espacial de las variables predictoras')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'moran_correlograma.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Guardado: moran_correlograma.csv / .png")

    # ---------- 2. Validación cruzada espacial por bloques ----------
    print("\n[2/2] Validación cruzada espacial por bloques...")
    resultados = []
    modelos = construir_modelos()

    # Referencia: CV aleatoria estratificada 10-fold
    cv_rand = StratifiedKFold(n_splits=10, shuffle=True, random_state=Config.SEED)
    for nombre in modelos:
        aucs = []
        for tr, te in cv_rand.split(X, y):
            m = construir_modelos()[nombre]
            m.fit(X[tr], y[tr])
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        aucs = np.array(aucs)
        resultados.append({'Modelo': nombre, 'Esquema': 'Aleatoria 10-fold',
                           'AUC_media': round(aucs.mean(), 4), 'AUC_DE': round(aucs.std(), 4),
                           'Folds_validos': len(aucs)})
        print(f"  {nombre} | aleatoria: AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")

    # Espacial: bloques cuadrados de tamaño creciente
    for bloque in Config.BLOQUES:
        gx = np.floor(coords[:, 0] / bloque).astype(int)
        gy = np.floor(coords[:, 1] / bloque).astype(int)
        grupos = gx * 100000 + gy
        n_bloques = len(np.unique(grupos))
        for nombre in modelos:
            aucs = cv_por_grupos(lambda n=nombre: construir_modelos()[n], X, y, grupos)
            if len(aucs) == 0:
                print(f"  {nombre} | bloques {bloque/1000:.0f} km: sin folds válidos")
                continue
            resultados.append({'Modelo': nombre, 'Esquema': f'Espacial {bloque/1000:.0f} km '
                                                            f'({n_bloques} bloques)',
                               'AUC_media': round(aucs.mean(), 4),
                               'AUC_DE': round(aucs.std(), 4), 'Folds_validos': len(aucs)})
            print(f"  {nombre} | bloques {bloque/1000:.0f} km: AUC = "
                  f"{aucs.mean():.4f} ± {aucs.std():.4f} ({len(aucs)} folds válidos)")

    res_df = pd.DataFrame(resultados)
    res_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'cv_espacial_vs_aleatoria.csv'),
                  index=False, encoding='utf-8-sig')

    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(11, 6))
    esquemas = res_df['Esquema'].unique()
    ancho = 0.8 / len(esquemas)
    xpos = np.arange(3)
    colores = plt.cm.viridis(np.linspace(0.15, 0.85, len(esquemas)))
    for i, esq in enumerate(esquemas):
        sub = res_df[res_df['Esquema'] == esq].set_index('Modelo').reindex(['RF', 'SVM', 'ANN'])
        ax.bar(xpos + i * ancho, sub['AUC_media'], ancho, yerr=sub['AUC_DE'],
               label=esq, color=colores[i], edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xticks(xpos + ancho * (len(esquemas) - 1) / 2)
    ax.set_xticklabels(['RF', 'SVM', 'ANN'])
    ax.set_ylabel('AUC-ROC (media ± DE)')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Clasificador aleatorio')
    ax.set_title('Validación cruzada aleatoria vs espacial por bloques')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cv_espacial_vs_aleatoria.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGuardado: cv_espacial_vs_aleatoria.csv / .png")
    print("Nota: es esperable que el AUC espacial sea menor que el aleatorio; esa "
          "diferencia es el sesgo optimista que la sección 4.5.4.6 debe reportar.")


if __name__ == "__main__":
    main()
