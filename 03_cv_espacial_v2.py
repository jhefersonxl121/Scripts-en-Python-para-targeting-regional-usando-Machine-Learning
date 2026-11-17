# -*- coding: utf-8 -*-
"""
================================================================================
VALIDACIÓN CRUZADA ESPACIAL — VERSIÓN 2 (fiel a la metodología de la tesis)
================================================================================
Cambios respecto a la v1:
  - La ANN usa sus 7 variables (excluye Densidad_Fallas).
  - SMOTE eliminado (conjunto balanceado por diseño; esto elimina el ValueError).
  - Ya no recalcula el I de Moran: moran_correlograma.csv de la v1 es definitivo.
RF y SVM no cambian respecto a la v1.

Salidas: cv_espacial_vs_aleatoria_v2.csv / .png
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


class Config:
    # === AJUSTA ESTAS RUTAS ===
    CSV_MUESTRAS = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\muestras_132.csv"
    OUTPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\Analisis_Espacial"

    SVM_C = 1
    SVM_GAMMA = 0.1
    VAR_EXCLUIDA_ANN = "Densidad_Fallas"

    BLOQUES = [10000, 20000, 30000]  # metros
    SEED = 42


def construir_modelos():
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, max_features='log2',
                                random_state=Config.SEED, n_jobs=-1)
    svm = Pipeline([('scaler', StandardScaler()),
                    ('svc', SVC(kernel='rbf', C=Config.SVM_C, gamma=Config.SVM_GAMMA,
                                probability=True, class_weight='balanced',
                                random_state=Config.SEED))])
    ann = Pipeline([('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                                          solver='adam', alpha=0.001,
                                          learning_rate_init=0.005, max_iter=500,
                                          early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=30,
                                          random_state=Config.SEED))])
    return {'RF': rf, 'SVM': svm, 'ANN': ann}


def cv_por_grupos(nombre, X, y, grupos, min_clase=3):
    """CV dejando fuera un bloque a la vez; omite folds sin ambas clases."""
    aucs = []
    for g in np.unique(grupos):
        test = grupos == g
        train = ~test
        if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
            continue
        if min(np.bincount(y[train])) < min_clase:
            continue
        m = construir_modelos()[nombre]
        m.fit(X[train], y[train])
        aucs.append(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1]))
    return np.array(aucs)


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(Config.CSV_MUESTRAS)
    no_pred = ['clase', 'X', 'Y', 'fila', 'columna']
    variables = [c for c in df.columns if c not in no_pred]
    vars_ann = [v for v in variables if v != Config.VAR_EXCLUIDA_ANN]

    y = df['clase'].values.astype(int)
    coords = df[['X', 'Y']].values
    X_por_modelo = {'RF': df[variables].values,
                    'SVM': df[variables].values,
                    'ANN': df[vars_ann].values}
    print(f"Muestras: {len(y)} | RF/SVM: {len(variables)} variables | "
          f"ANN: {len(vars_ann)} variables (sin {Config.VAR_EXCLUIDA_ANN})")

    resultados = []

    # Referencia: CV aleatoria estratificada 10-fold
    cv_rand = StratifiedKFold(n_splits=10, shuffle=True, random_state=Config.SEED)
    for nombre in ['RF', 'SVM', 'ANN']:
        X = X_por_modelo[nombre]
        aucs = []
        for tr, te in cv_rand.split(X, y):
            m = construir_modelos()[nombre]
            m.fit(X[tr], y[tr])
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        aucs = np.array(aucs)
        resultados.append({'Modelo': nombre, 'Esquema': 'Aleatoria 10-fold',
                           'AUC_media': round(aucs.mean(), 4),
                           'AUC_DE': round(aucs.std(), 4), 'Folds_validos': len(aucs)})
        print(f"  {nombre} | aleatoria: AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")

    # Espacial por bloques
    for bloque in Config.BLOQUES:
        gx = np.floor(coords[:, 0] / bloque).astype(int)
        gy = np.floor(coords[:, 1] / bloque).astype(int)
        grupos = gx * 100000 + gy
        n_bloques = len(np.unique(grupos))
        for nombre in ['RF', 'SVM', 'ANN']:
            aucs = cv_por_grupos(nombre, X_por_modelo[nombre], y, grupos)
            if len(aucs) == 0:
                print(f"  {nombre} | bloques {bloque/1000:.0f} km: sin folds válidos")
                continue
            resultados.append({'Modelo': nombre,
                               'Esquema': f'Espacial {bloque/1000:.0f} km ({n_bloques} bloques)',
                               'AUC_media': round(aucs.mean(), 4),
                               'AUC_DE': round(aucs.std(), 4),
                               'Folds_validos': len(aucs)})
            print(f"  {nombre} | bloques {bloque/1000:.0f} km: AUC = "
                  f"{aucs.mean():.4f} ± {aucs.std():.4f} ({len(aucs)} folds válidos)")

    res_df = pd.DataFrame(resultados)
    ruta_csv = os.path.join(Config.OUTPUT_DIR, 'cv_espacial_vs_aleatoria_v2.csv')
    res_df.to_csv(ruta_csv, index=False, encoding='utf-8-sig')

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
    ax.set_xticklabels(['RF', 'SVM', 'ANN (7 var.)'])
    ax.set_ylabel('AUC-ROC (media ± DE)')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Clasificador aleatorio')
    ax.set_title('Validación cruzada aleatoria vs espacial por bloques (v2)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cv_espacial_vs_aleatoria_v2.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGuardado: {ruta_csv} y su figura.")
    print("Solo las filas de ANN deberían diferir de la v1; RF y SVM deben coincidir.")


if __name__ == "__main__":
    main()
