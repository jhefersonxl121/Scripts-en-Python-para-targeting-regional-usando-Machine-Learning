# -*- coding: utf-8 -*-
"""
================================================================================
EXPORTACIÓN DE LAS 132 MUESTRAS DE ENTRENAMIENTO A CSV (clase, X, Y, 8 variables)
================================================================================
Reconstruye las 66 muestras positivas y las 66 negativas con EXACTAMENTE el
mismo protocolo y semilla (random_state=42) de los scripts RF/SVM/ANN de la
tesis, leyendo las 8 variables desde los GeoTIFF ya generados, y exporta todo
a muestras_132.csv con coordenadas UTM 18S (EPSG:32718).

Verificación: la consola debe imprimir 66 depósitos y 66 no-depósitos.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')


class Config:
    # === AJUSTA ESTAS RUTAS SI DIFIEREN ===
    INPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Shapefiles"
    RASTER_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Rasters"
    OUTPUT_CSV = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Results\muestras_132.csv"

    STUDY_AREA_SHP = os.path.join(INPUT_DIR, "area.shp")
    DEPOSITS_SHP = os.path.join(INPUT_DIR, "yacimientos.shp")

    OUTPUT_CRS = "EPSG:32718"
    CELL_SIZE = 500
    NON_DEPOSIT_BUFFER = 5000

    # Las 8 variables, en el MISMO orden que el script de RF (que usa las 8)
    PREDICTORES = [
        ("prox_favorable_lithology", "Prox_Lith_Favorable"),
        ("fault_density", "Densidad_Fallas"),
        ("prox_faults", "Prox_Fallas"),
        ("prox_fault_intersections", "Prox_Intersecciones"),
        ("lithological_contacts_density", "Densidad_Contactos"),
        ("geochemical_pc1", "PC1_Geoquimica"),
        ("geochemical_pc2", "PC2_Geoquimica"),
        ("geochemical_tungsten", "W_Geoquimica"),
    ]


def main():
    print("=" * 70)
    print("EXPORTACIÓN DE MUESTRAS DE ENTRENAMIENTO A CSV")
    print("=" * 70)

    # 1. Grid idéntico al de los scripts de modelamiento
    study_area_gdf = gpd.read_file(Config.STUDY_AREA_SHP).to_crs(Config.OUTPUT_CRS)
    minx, miny, maxx, maxy = study_area_gdf.total_bounds
    width = int(np.ceil((maxx - minx) / Config.CELL_SIZE))
    height = int(np.ceil((maxy - miny) / Config.CELL_SIZE))
    maxx_adj = minx + width * Config.CELL_SIZE
    maxy_adj = miny + height * Config.CELL_SIZE
    transform = from_bounds(minx, miny, maxx_adj, maxy_adj, width, height)

    shapes = [(g, 1) for g in study_area_gdf.geometry if g is not None]
    study_area_mask = features.rasterize(shapes=shapes, out_shape=(height, width),
                                         transform=transform, fill=0,
                                         dtype=np.uint8).astype(bool)
    print(f"  Grid: {width} x {height} | píxeles en el área: {study_area_mask.sum():,}")

    # 2. Cargar los 8 rasters predictores ya generados
    predictor_stack, nombres = [], []
    for archivo, nombre in Config.PREDICTORES:
        ruta = os.path.join(Config.RASTER_DIR, f"{archivo}.tif")
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró {ruta}. Verifica RASTER_DIR.")
        with rasterio.open(ruta) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            if data.shape != (height, width):
                raise ValueError(f"{archivo}.tif tiene forma {data.shape}, se esperaba "
                                 f"{(height, width)}. ¿Se generó con otro grid?")
        predictor_stack.append(data)
        nombres.append(nombre)
        print(f"  Cargado: {archivo}.tif")

    # 3. Muestras positivas (idéntico a generate_training_data de tus scripts)
    deposits_gdf = gpd.read_file(Config.DEPOSITS_SHP).to_crs(Config.OUTPUT_CRS)
    deposit_rows, deposit_cols = [], []
    for point in deposits_gdf.geometry:
        if point is not None:
            col = int((point.x - minx) / Config.CELL_SIZE)
            row = int((maxy_adj - point.y) / Config.CELL_SIZE)
            if 0 <= row < height and 0 <= col < width and study_area_mask[row, col]:
                deposit_rows.append(row)
                deposit_cols.append(col)
    n_dep = len(deposit_rows)
    print(f"\n  Depósitos dentro del área: {n_dep}  (esperado: 66)")

    # 4. Muestras negativas (misma semilla y protocolo)
    deposit_mask = np.zeros((height, width), dtype=bool)
    for r, c in zip(deposit_rows, deposit_cols):
        deposit_mask[r, c] = True
    dist = distance_transform_edt(~deposit_mask) * Config.CELL_SIZE

    valid_mask = study_area_mask.copy()
    valid_mask &= (dist > Config.NON_DEPOSIT_BUFFER)
    for p in predictor_stack:
        valid_mask &= ~np.isnan(p) & ~np.isinf(p)

    valid_rows, valid_cols = np.where(valid_mask)
    n_neg = min(n_dep, len(valid_rows))
    np.random.seed(42)
    idx = np.random.choice(len(valid_rows), size=n_neg, replace=False)
    neg_rows, neg_cols = valid_rows[idx], valid_cols[idx]
    print(f"  No-depósitos generados: {n_neg}  (esperado: 66)")

    # 5. Armar tabla: clase, X, Y (centro de celda) y las 8 variables
    def filas(rows, cols, clase):
        out = []
        for r, c in zip(rows, cols):
            x = minx + (c + 0.5) * Config.CELL_SIZE
            y = maxy_adj - (r + 0.5) * Config.CELL_SIZE
            vals = [p[r, c] for p in predictor_stack]
            out.append([clase, x, y, r, c] + vals)
        return out

    data = filas(deposit_rows, deposit_cols, 1) + filas(list(neg_rows), list(neg_cols), 0)
    df = pd.DataFrame(data, columns=["clase", "X", "Y", "fila", "columna"] + nombres)

    # Eliminar muestras con NaN (igual que en tus scripts)
    antes = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < antes:
        print(f"  Eliminadas {antes - len(df)} muestras con NaN")

    os.makedirs(os.path.dirname(Config.OUTPUT_CSV), exist_ok=True)
    df.to_csv(Config.OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Total exportado: {len(df)} muestras "
          f"({(df['clase'] == 1).sum()} positivas, {(df['clase'] == 0).sum()} negativas)")
    print(f"  Guardado: {Config.OUTPUT_CSV}")
    print("\n  Si los conteos no son 66/66, avísame antes de usar los resultados.")


if __name__ == "__main__":
    main()
