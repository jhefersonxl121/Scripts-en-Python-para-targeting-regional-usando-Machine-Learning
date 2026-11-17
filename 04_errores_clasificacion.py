# -*- coding: utf-8 -*-
"""
================================================================================
IDENTIFICACIÓN DE FALSOS NEGATIVOS Y FALSOS POSITIVOS (sección 6.6.3)
================================================================================
Reproduce la partición entrenamiento/prueba de la tesis (test_size=0.33,
random_state=42, estratificada) sobre muestras_132.csv y los tres modelos
finales (RF y SVM con 8 variables; ANN con 7, sin Densidad_Fallas; sin SMOTE).

Exporta errores_clasificacion.csv con, para cada error del conjunto de prueba:
  - tipo de error (FN / FP) y qué modelos lo cometen
  - coordenadas X, Y (UTM 18S) y probabilidad de cada modelo
  - valores de las 8 variables predictoras
  - atributos del depósito más cercano (de yacimientos.shp) y su distancia
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class Config:
    # === AJUSTA ESTAS RUTAS ===
    BASE = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning"
    CSV_MUESTRAS = os.path.join(BASE, "Results", "muestras_132.csv")
    DEPOSITS_SHP = os.path.join(BASE, "Shapefiles", "yacimientos.shp")
    OUTPUT_CSV = os.path.join(BASE, "Results", "errores_clasificacion.csv")

    OUTPUT_CRS = "EPSG:32718"
    VAR_EXCLUIDA_ANN = "Densidad_Fallas"
    TEST_SIZE = 0.33
    SEED = 42


def modelos():
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, max_features='log2',
                                random_state=Config.SEED, n_jobs=-1)
    svm = Pipeline([('scaler', StandardScaler()),
                    ('svc', SVC(kernel='rbf', C=1, gamma=0.1, probability=True,
                                class_weight='balanced', random_state=Config.SEED))])
    ann = Pipeline([('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                                          solver='adam', alpha=0.001,
                                          learning_rate_init=0.005, max_iter=500,
                                          early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=30, random_state=Config.SEED))])
    return {'RF': rf, 'SVM': svm, 'ANN': ann}


def main():
    df = pd.read_csv(Config.CSV_MUESTRAS)
    no_pred = ['clase', 'X', 'Y', 'fila', 'columna']
    variables = [c for c in df.columns if c not in no_pred]
    vars_ann = [v for v in variables if v != Config.VAR_EXCLUIDA_ANN]
    y = df['clase'].values.astype(int)

    # Misma partición que los scripts de la tesis (mismo orden de filas y semilla)
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=Config.TEST_SIZE,
                                      random_state=Config.SEED, stratify=y)
    print(f"Entrenamiento: {len(idx_tr)} | Prueba: {len(idx_te)} "
          f"({y[idx_te].sum()} depósitos, {(y[idx_te] == 0).sum()} no-depósitos)")

    test = df.iloc[idx_te].copy()
    for nombre, modelo in modelos().items():
        cols = vars_ann if nombre == 'ANN' else variables
        modelo.fit(df.iloc[idx_tr][cols].values, y[idx_tr])
        prob = modelo.predict_proba(test[cols].values)[:, 1]
        test[f'P_{nombre}'] = np.round(prob, 4)
        test[f'Pred_{nombre}'] = (prob >= 0.5).astype(int)

    # Clasificar errores
    filas = []
    for _, r in test.iterrows():
        fallan = [m for m in ['RF', 'SVM', 'ANN'] if r[f'Pred_{m}'] != r['clase']]
        if fallan:
            tipo = 'FN (depósito omitido)' if r['clase'] == 1 else 'FP (falsa alarma)'
            filas.append({**r.to_dict(), 'Tipo_error': tipo,
                          'Modelos_que_fallan': ', '.join(fallan)})
    err = pd.DataFrame(filas)
    if err.empty:
        print("No se encontraron errores en el conjunto de prueba.")
        return

    # Depósito más cercano (nombre y atributos del shapefile)
    dep = gpd.read_file(Config.DEPOSITS_SHP).to_crs(Config.OUTPUT_CRS)
    pts = gpd.GeoDataFrame(err, geometry=gpd.points_from_xy(err['X'], err['Y']),
                           crs=Config.OUTPUT_CRS)
    unido = gpd.sjoin_nearest(pts, dep, how='left', distance_col='Dist_deposito_m')
    unido = unido.drop(columns=['geometry', 'index_right'], errors='ignore')
    unido['Dist_deposito_m'] = unido['Dist_deposito_m'].round(0)

    unido = unido.sort_values(['Tipo_error', 'Modelos_que_fallan'])
    unido.to_csv(Config.OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\nFalsos negativos: {(unido['Tipo_error'].str.startswith('FN')).sum()}")
    print(f"Falsos positivos: {(unido['Tipo_error'].str.startswith('FP')).sum()}")
    print(f"Guardado: {Config.OUTPUT_CSV}")
    print("\nNota: para los FN, la distancia al depósito más cercano debería ser ~0 m "
          "(es el propio depósito). Para los FP, indica qué tan lejos está la falsa "
          "alarma del depósito conocido más próximo.")


if __name__ == "__main__":
    main()
