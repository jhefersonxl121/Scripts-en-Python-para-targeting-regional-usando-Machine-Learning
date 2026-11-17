# -*- coding: utf-8 -*-
"""
================================================================================
DIAGNÓSTICO DE SOBREAJUSTE — VERSIÓN 2 (fiel a la metodología de la tesis)
================================================================================
Cambios respecto a la v1:
  - La ANN se evalúa con sus 7 variables (excluye Densidad_Fallas), igual que
    el modelo oficial de la tesis.
  - SMOTE eliminado: el conjunto de entrenamiento está balanceado por diseño.
RF y SVM no cambian respecto a la v1.

Salidas: learning_curves_v2.png y brecha_train_test_v2.csv
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


class Config:
    # === AJUSTA ESTAS RUTAS ===
    CSV_MUESTRAS = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\muestras_132.csv"
    OUTPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\Sobreajuste"

    # Hiperparámetros finales de la tesis
    SVM_C = 1
    SVM_GAMMA = 0.1

    # Variable excluida por la ANN (decisión metodológica de la tesis)
    VAR_EXCLUIDA_ANN = "Densidad_Fallas"

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
    ann = Pipeline([('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                                          solver='adam', alpha=0.001,
                                          learning_rate_init=0.005, max_iter=500,
                                          early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=30,
                                          random_state=Config.SEED))])
    return {'RF': rf, 'SVM': svm, 'ANN': ann}


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(Config.CSV_MUESTRAS)
    no_pred = ['clase', 'X', 'Y', 'fila', 'columna']
    variables = [c for c in df.columns if c not in no_pred]
    vars_ann = [v for v in variables if v != Config.VAR_EXCLUIDA_ANN]

    y = df['clase'].values
    X_por_modelo = {'RF': df[variables].values,
                    'SVM': df[variables].values,
                    'ANN': df[vars_ann].values}
    print(f"Muestras: {len(y)} | RF/SVM: {len(variables)} variables | "
          f"ANN: {len(vars_ann)} variables (sin {Config.VAR_EXCLUIDA_ANN})")

    modelos = construir_modelos()
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.SEED)
    colores = {'RF': '#2E7D32', 'SVM': '#1565C0', 'ANN': '#C62828'}

    # ---------- 1. Curvas de aprendizaje ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    train_sizes = np.linspace(0.2, 1.0, 8)

    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        print(f"\nCurva de aprendizaje: {nombre} ...")
        X = X_por_modelo[nombre]
        sizes, tr_scores, cv_scores = learning_curve(
            modelo, X, y, cv=cv, scoring='roc_auc',
            train_sizes=train_sizes, n_jobs=-1, random_state=Config.SEED)
        tr_m, tr_s = tr_scores.mean(axis=1), tr_scores.std(axis=1)
        cv_m, cv_s = cv_scores.mean(axis=1), cv_scores.std(axis=1)

        ax.plot(sizes, tr_m, 'o-', color=colores[nombre], label='Entrenamiento')
        ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color=colores[nombre])
        ax.plot(sizes, cv_m, 's--', color='gray', label='Validación cruzada')
        ax.fill_between(sizes, cv_m - cv_s, cv_m + cv_s, alpha=0.15, color='gray')
        titulo = nombre if nombre != 'ANN' else 'ANN (7 variables)'
        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel('Tamaño del conjunto de entrenamiento')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        print(f"  Brecha final train-CV (AUC): {tr_m[-1] - cv_m[-1]:.3f}")

    axes[0].set_ylabel('AUC-ROC')
    plt.suptitle('Curvas de aprendizaje (AUC-ROC, validación cruzada estratificada 10-fold)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    ruta_fig = os.path.join(Config.OUTPUT_DIR, 'learning_curves_v2.png')
    plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGuardado: {ruta_fig}")

    # ---------- 2. Brecha train-test ----------
    filas = []
    for nombre, modelo in construir_modelos().items():
        X = X_por_modelo[nombre]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                  random_state=Config.SEED, stratify=y)
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
    ruta_csv = os.path.join(Config.OUTPUT_DIR, 'brecha_train_test_v2.csv')
    tabla.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
    print(f"Guardado: {ruta_csv}")
    print("\nVerificación esperada: el AUC de prueba de la ANN debería acercarse "
          "al 0.932 reportado en la tesis.")


if __name__ == "__main__":
    main()
