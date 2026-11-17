# -*- coding: utf-8 -*-
"""
================================================================================
DIAGNÓSTICO DE SOBREAJUSTE (sección 4.2.7.4 de la tesis)
================================================================================
A partir de muestras_132.csv (generado por 01_exportar_muestras.py):
  1. Curvas de aprendizaje (AUC-ROC en entrenamiento vs validación cruzada)
     para RF, SVM y ANN, con los hiperparámetros finales de la tesis.
  2. Tabla de brecha train-test (AUC en entrenamiento vs conjunto de prueba).

Salidas: learning_curves.png y brecha_train_test.csv
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
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False
    print("Advertencia: imbalanced-learn no disponible; la ANN se evaluará sin SMOTE.")


class Config:
    # === AJUSTA ESTAS RUTAS / VALORES ===
    CSV_MUESTRAS = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\muestras_132.csv"
    OUTPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\Sobreajuste"

    # <<< REEMPLAZAR por los valores finales de tu SVM (model_summary_svm.csv) >>>
    SVM_C = 1
    SVM_GAMMA = 0.1

    TEST_SIZE = 0.33
    CV_FOLDS = 10
    SEED = 42


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


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(Config.CSV_MUESTRAS)
    no_pred = ['clase', 'X', 'Y', 'fila', 'columna']
    variables = [c for c in df.columns if c not in no_pred]
    X = df[variables].values
    y = df['clase'].values
    print(f"Muestras: {len(y)} | Variables: {len(variables)}")

    modelos = construir_modelos()
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.SEED)
    colores = {'RF': '#2E7D32', 'SVM': '#1565C0', 'ANN': '#C62828'}

    # ---------- 1. Curvas de aprendizaje ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    train_sizes = np.linspace(0.2, 1.0, 8)

    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        print(f"\nCurva de aprendizaje: {nombre} ...")
        sizes, tr_scores, cv_scores = learning_curve(
            modelo, X, y, cv=cv, scoring='roc_auc',
            train_sizes=train_sizes, n_jobs=-1, random_state=Config.SEED)
        tr_m, tr_s = tr_scores.mean(axis=1), tr_scores.std(axis=1)
        cv_m, cv_s = cv_scores.mean(axis=1), cv_scores.std(axis=1)

        ax.plot(sizes, tr_m, 'o-', color=colores[nombre], label='Entrenamiento')
        ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color=colores[nombre])
        ax.plot(sizes, cv_m, 's--', color='gray', label='Validación cruzada')
        ax.fill_between(sizes, cv_m - cv_s, cv_m + cv_s, alpha=0.15, color='gray')
        ax.set_title(nombre, fontsize=13, fontweight='bold')
        ax.set_xlabel('Tamaño del conjunto de entrenamiento')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        print(f"  Brecha final train-CV (AUC): {tr_m[-1] - cv_m[-1]:.3f}")

    axes[0].set_ylabel('AUC-ROC')
    plt.suptitle('Curvas de aprendizaje (AUC-ROC, validación cruzada estratificada 10-fold)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    ruta_fig = os.path.join(Config.OUTPUT_DIR, 'learning_curves.png')
    plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGuardado: {ruta_fig}")

    # ---------- 2. Brecha train-test ----------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                              random_state=Config.SEED, stratify=y)
    filas = []
    for nombre, modelo in construir_modelos().items():
        modelo.fit(X_tr, y_tr)
        auc_tr = roc_auc_score(y_tr, modelo.predict_proba(X_tr)[:, 1])
        auc_te = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])
        filas.append({'Modelo': nombre,
                      'AUC entrenamiento': round(auc_tr, 4),
                      'AUC prueba': round(auc_te, 4),
                      'Brecha (train - test)': round(auc_tr - auc_te, 4)})
        print(f"{nombre}: AUC train={auc_tr:.4f} | AUC test={auc_te:.4f} | "
              f"brecha={auc_tr - auc_te:.4f}")

    tabla = pd.DataFrame(filas)
    ruta_csv = os.path.join(Config.OUTPUT_DIR, 'brecha_train_test.csv')
    tabla.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
    print(f"Guardado: {ruta_csv}")
    print("\nInterpretación rápida: brechas < 0.05 sugieren sobreajuste limitado; "
          "una curva de entrenamiento muy por encima de la de CV indica varianza alta.")


if __name__ == "__main__":
    main()
