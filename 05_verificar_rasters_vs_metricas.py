# -*- coding: utf-8 -*-
"""
================================================================================
VERIFICACIÓN: ¿LOS MAPAS OFICIALES COINCIDEN CON LAS MÉTRICAS DE LA TESIS?
================================================================================
Lee los rásters de prospectividad oficiales (RF, SVM, ANN) en las 44 celdas del
conjunto de prueba (misma partición de la tesis: test_size=0.33, semilla 42) y
recalcula sus métricas. Como el mapa y las métricas salen del mismo modelo,
deben coincidir con las reportadas en la tesis (Tabla 20).

No entrena ningún modelo: solo lee los rásters.
================================================================================
"""

import os
import datetime
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix


class Config:
    # === AJUSTA ESTAS RUTAS: apunta a los rásters con que hiciste los MAPAS de la tesis ===
    BASE = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning"
    CSV_MUESTRAS = os.path.join(BASE, "Results", "muestras_132.csv")
    RASTERS = {
        'RF': os.path.join(BASE, "Rasters", "3. RF_Version 3.0", "prospectivity_rf.tif"),
        'SVM': os.path.join(BASE, "Rasters", "5. SVM_Version 2.0", "prospectivity_svm.tif"),
        'ANN': os.path.join(BASE, "Rasters", "7. ANN_V.2.0", "prospectivity_ann.tif"),
    }
    OUTPUT_CSV = os.path.join(BASE, "Results", "verificacion_rasters.csv")

    # Métricas de prueba reportadas en la tesis (Tabla 20)
    TESIS = {
        'RF':  {'Sensibilidad': 0.8182, 'Especificidad': 0.8636, 'AUC': 0.9360},
        'SVM': {'Sensibilidad': 0.8182, 'Especificidad': 0.9091, 'AUC': 0.9463},
        'ANN': {'Sensibilidad': 0.8182, 'Especificidad': 0.9545, 'AUC': 0.9318},
    }
    TEST_SIZE = 0.33
    SEED = 42


def main():
    df = pd.read_csv(Config.CSV_MUESTRAS)
    y = df['clase'].values.astype(int)
    idx = np.arange(len(df))
    _, idx_te = train_test_split(idx, test_size=Config.TEST_SIZE,
                                 random_state=Config.SEED, stratify=y)
    test = df.iloc[idx_te].reset_index(drop=True)
    y_te = test['clase'].values.astype(int)
    coords = list(zip(test['X'], test['Y']))
    print(f"Celdas de prueba: {len(test)} ({y_te.sum()} depósitos, {(y_te == 0).sum()} negativas)\n")

    resumen = []
    for nombre, ruta in Config.RASTERS.items():
        if not os.path.exists(ruta):
            print(f"[{nombre}] NO ENCONTRADO: {ruta}\n")
            continue
        fecha = datetime.datetime.fromtimestamp(os.path.getmtime(ruta)).strftime('%Y-%m-%d %H:%M')
        with rasterio.open(ruta) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
        test[f'P_{nombre}_raster'] = np.round(vals, 4)

        ok = ~np.isnan(vals)
        pred = (vals >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te[ok], pred[ok], labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        auc = roc_auc_score(y_te[ok], vals[ok])
        t = Config.TESIS[nombre]
        coincide = (abs(sens - t['Sensibilidad']) < 0.001 and
                    abs(spec - t['Especificidad']) < 0.001 and
                    abs(auc - t['AUC']) < 0.002)

        print(f"[{nombre}] {os.path.basename(ruta)}  (modificado: {fecha})")
        print(f"   Ráster -> Sensibilidad {sens:.4f} | Especificidad {spec:.4f} | AUC {auc:.4f} "
              f"| FN={fn} FP={fp}")
        print(f"   Tesis  -> Sensibilidad {t['Sensibilidad']:.4f} | Especificidad "
              f"{t['Especificidad']:.4f} | AUC {t['AUC']:.4f}")
        print(f"   ¿Coincide con la tesis? {'SÍ' if coincide else 'NO'}\n")
        resumen.append({'Modelo': nombre, 'Archivo': ruta, 'Modificado': fecha,
                        'Sens_raster': round(sens, 4), 'Espec_raster': round(spec, 4),
                        'AUC_raster': round(auc, 4), 'FN': fn, 'FP': fp,
                        'Coincide_tesis': 'SÍ' if coincide else 'NO'})

    os.makedirs(os.path.dirname(Config.OUTPUT_CSV), exist_ok=True)
    test.to_csv(Config.OUTPUT_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(resumen).to_csv(Config.OUTPUT_CSV.replace('.csv', '_resumen.csv'),
                                 index=False, encoding='utf-8-sig')
    print(f"Guardado: {Config.OUTPUT_CSV} y su _resumen.csv")
    print("Si un modelo dice 'NO', ese ráster no es el que generó las métricas de la tesis.")


if __name__ == "__main__":
    main()
