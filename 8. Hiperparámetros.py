# -*- coding: utf-8 -*-
"""
================================================================================
OPTIMIZACIÓN DE HIPERPARÁMETROS PARA MODELOS DE MACHINE LEARNING
================================================================================
Script para encontrar los mejores hiperparámetros de Random Forest, SVM y ANN
utilizando GridSearchCV con validación cruzada estratificada de 10 pliegues.

Objetivo: Asegurar que la comparación entre algoritmos refleje el mejor 
desempeño alcanzable por cada uno bajo configuración óptima.

Modelos optimizados:
1. Random Forest (RF)
2. Support Vector Machine (SVM)
3. Artificial Neural Network (ANN)

Autor: Adaptado para proyecto de tesis
================================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, convolve
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
import os
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Intentar importar SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Advertencia: imbalanced-learn no instalado. SMOTE no disponible.")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración de rutas y parámetros"""
    
    # Directorios
    INPUT_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Shapefiles"
    RASTER_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Rasters"
    OUTPUT_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Results\Optimizacion_Hiperparametros"
    
    # Archivos de entrada
    FAULTS_SHP = os.path.join(INPUT_DIR, "fallas.shp")
    LITHOLOGY_SHP = os.path.join(INPUT_DIR, "litologias.shp")
    DEPOSITS_SHP = os.path.join(INPUT_DIR, "yacimientos.shp")
    STUDY_AREA_SHP = os.path.join(INPUT_DIR, "area.shp")
    PC1_RASTER = os.path.join(INPUT_DIR, "PC1.tif")
    PC2_RASTER = os.path.join(INPUT_DIR, "PC2.tif")
    W_RASTER = os.path.join(INPUT_DIR, "W.tif")
    
    # Parámetros espaciales
    OUTPUT_CRS = "EPSG:32718"
    CELL_SIZE = 500
    FAVORABLE_COLUMN = "FAVORABLE"
    NON_DEPOSIT_BUFFER = 5000
    FAULT_SEARCH_RADIUS = 2500
    CONTACT_SEARCH_RADIUS = 500
    TEST_SIZE = 0.33
    
    # Validación cruzada
    CV_FOLDS = 10
    SCORING = 'roc_auc'  # Métrica principal para optimización
    
    # =========================================================================
    # GRIDS DE HIPERPARÁMETROS
    # =========================================================================
    
    # Random Forest - Grid completo
    RF_PARAM_GRID = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Random Forest - Grid simplificado (más rápido)
    RF_PARAM_GRID_SIMPLE = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 15, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # SVM - Grid completo
    SVM_PARAM_GRID = {
        'C': [0.1, 0.5, 1, 5, 10, 25, 50],
        'gamma': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    }
    
    # SVM - Grid simplificado
    SVM_PARAM_GRID_SIMPLE = {
        'C': [0.1, 1, 10, 50],
        'gamma': [0.1, 0.3, 0.5, 1.0]
    }
    
    # ANN - Grid completo
    ANN_PARAM_GRID = {
        'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32), (128, 64), (64, 32, 16)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.005, 0.01]
    }
    
    # ANN - Grid simplificado
    ANN_PARAM_GRID_SIMPLE = {
        'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)],
        'activation': ['relu'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.005]
    }
    
    # Usar grids simplificados para ejecución más rápida
    USE_SIMPLE_GRIDS = True
    
    # Usar SMOTE para ANN
    USE_SMOTE_FOR_ANN = True


def create_output_directory():
    """Crear directorio de salida"""
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    return Config.OUTPUT_DIR


# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

def load_study_area():
    """Cargar área de estudio"""
    print(f"Cargando área de estudio: {Config.STUDY_AREA_SHP}")
    study_area_gdf = gpd.read_file(Config.STUDY_AREA_SHP)
    study_area_gdf = study_area_gdf.to_crs(Config.OUTPUT_CRS)
    study_area_geom = unary_union(study_area_gdf.geometry)
    return study_area_gdf, study_area_geom


def get_study_area_bounds(study_area_gdf):
    return list(study_area_gdf.total_bounds)


def calculate_grid_dimensions(bounds, cell_size):
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    maxx_adj = minx + width * cell_size
    maxy_adj = miny + height * cell_size
    return width, height, (minx, miny, maxx_adj, maxy_adj)


def create_transform(bounds, width, height):
    minx, miny, maxx, maxy = bounds
    return from_bounds(minx, miny, maxx, maxy, width, height)


def create_study_area_mask(study_area_gdf, bounds, width, height, transform):
    shapes = [(geom, 1) for geom in study_area_gdf.geometry if geom is not None]
    mask = features.rasterize(shapes=shapes, out_shape=(height, width),
                              transform=transform, fill=0, dtype=np.uint8)
    return mask.astype(bool)


def apply_study_area_mask(raster, mask):
    result = raster.copy().astype(np.float32)
    result[~mask] = np.nan
    return result


def load_predictor_from_raster(raster_path, bounds, width, height, transform, mask):
    """Cargar y reproyectar un raster predictor"""
    with rasterio.open(raster_path) as src:
        dst_array = np.zeros((height, width), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dst_array,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=Config.OUTPUT_CRS,
                  resampling=Resampling.bilinear)
        if src.nodata is not None:
            dst_array[dst_array == src.nodata] = np.nan
        return apply_study_area_mask(dst_array, mask)


def normalize_raster(raster):
    """Normalización min-max"""
    valid_mask = ~np.isnan(raster) & ~np.isinf(raster)
    if not valid_mask.any():
        return raster
    result = raster.copy()
    min_val = np.nanmin(raster[valid_mask])
    max_val = np.nanmax(raster[valid_mask])
    if max_val > min_val:
        result[valid_mask] = (raster[valid_mask] - min_val) / (max_val - min_val)
    else:
        result[valid_mask] = 0
    return result


def rasterize_polygon_to_binary(gdf, bounds, width, height, transform, 
                                 attribute=None, value=1, mask=None):
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    if attribute is not None:
        gdf_proj = gdf_proj[gdf_proj[attribute] == value]
    if len(gdf_proj) == 0:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    raster = features.rasterize(shapes=shapes, out_shape=(height, width),
                                transform=transform, fill=0, dtype=np.float32)
    if mask is not None:
        raster = apply_study_area_mask(raster, mask)
    return raster


def calculate_euclidean_distance(binary_raster, cell_size):
    binary_clean = np.nan_to_num(binary_raster, nan=0)
    inverted = 1 - binary_clean
    return distance_transform_edt(inverted) * cell_size


def rasterize_lines_to_density(gdf, bounds, width, height, transform, 
                                search_radius=2500, mask=None):
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    if not shapes:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    line_raster = features.rasterize(shapes=shapes, out_shape=(height, width),
                                     transform=transform, fill=0, dtype=np.float32)
    kernel_size = int(search_radius / Config.CELL_SIZE)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = (x**2 + y**2 <= (kernel_size//2)**2).astype(float)
    kernel /= kernel.sum()
    density = convolve(line_raster.astype(float), kernel, mode='constant')
    if mask is not None:
        density = apply_study_area_mask(density, mask)
    return density


def calculate_proximity_to_features(gdf, bounds, width, height, transform, 
                                     cell_size, mask=None):
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    if not shapes:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    feature_raster = features.rasterize(shapes=shapes, out_shape=(height, width),
                                        transform=transform, fill=0, dtype=np.float32)
    distance = calculate_euclidean_distance(feature_raster, cell_size)
    if mask is not None:
        distance = apply_study_area_mask(distance, mask)
    return distance


def extract_fault_intersections(gdf, study_area_geom=None):
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    if study_area_geom is not None:
        gdf_proj = gdf_proj.copy()
        gdf_proj['geometry'] = gdf_proj.geometry.intersection(study_area_geom)
        gdf_proj = gdf_proj[~gdf_proj.geometry.is_empty]
    intersections = []
    geometries = list(gdf_proj.geometry)
    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            if geometries[i] is not None and geometries[j] is not None:
                if geometries[i].intersects(geometries[j]):
                    intersection = geometries[i].intersection(geometries[j])
                    if intersection.geom_type == 'Point':
                        if study_area_geom is None or study_area_geom.contains(intersection):
                            intersections.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            if study_area_geom is None or study_area_geom.contains(point):
                                intersections.append(point)
    if intersections:
        return gpd.GeoDataFrame(geometry=intersections, crs=Config.OUTPUT_CRS)
    return None


def extract_favorable_lithology_contacts(lithology_gdf, study_area_geom=None):
    gdf_proj = lithology_gdf.to_crs(Config.OUTPUT_CRS)
    favorable_gdf = gdf_proj[gdf_proj[Config.FAVORABLE_COLUMN] == 1].copy()
    favorable_gdf = favorable_gdf.reset_index(drop=True)
    if len(favorable_gdf) < 2:
        return None
    contact_lines = []
    for i in range(len(favorable_gdf)):
        geom_i = favorable_gdf.geometry.iloc[i]
        for j in range(i + 1, len(favorable_gdf)):
            geom_j = favorable_gdf.geometry.iloc[j]
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                intersection = geom_i.intersection(geom_j)
                if intersection.geom_type == 'LineString':
                    contact_lines.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    for line in intersection.geoms:
                        contact_lines.append(line)
                elif intersection.geom_type == 'GeometryCollection':
                    for geom in intersection.geoms:
                        if geom.geom_type in ['LineString', 'MultiLineString']:
                            if geom.geom_type == 'LineString':
                                contact_lines.append(geom)
                            else:
                                for line in geom.geoms:
                                    contact_lines.append(line)
    if not contact_lines:
        return None
    contacts_gdf = gpd.GeoDataFrame(geometry=contact_lines, crs=Config.OUTPUT_CRS)
    if study_area_geom is not None:
        contacts_gdf['geometry'] = contacts_gdf.geometry.intersection(study_area_geom)
        contacts_gdf = contacts_gdf[~contacts_gdf.geometry.is_empty]
    return contacts_gdf


def calculate_favorable_contacts_density(lithology_gdf, bounds, width, height, 
                                          transform, cell_size, study_area_geom=None, 
                                          mask=None):
    contacts_gdf = extract_favorable_lithology_contacts(lithology_gdf, study_area_geom)
    if contacts_gdf is None or len(contacts_gdf) == 0:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    shapes = [(geom, 1) for geom in contacts_gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    contact_raster = features.rasterize(shapes=shapes, out_shape=(height, width),
                                        transform=transform, fill=0, dtype=np.float32)
    kernel_size = int(Config.CONTACT_SEARCH_RADIUS / cell_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = (x**2 + y**2 <= (kernel_size//2)**2).astype(float)
    kernel /= kernel.sum()
    density = convolve(contact_raster.astype(float), kernel, mode='constant')
    if mask is not None:
        density = apply_study_area_mask(density, mask)
    return density


def generate_training_data(deposits_gdf, predictor_stack, transform, bounds, 
                           width, height, study_area_mask, buffer_distance=5000):
    """Generar datos de entrenamiento"""
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    deposit_rows, deposit_cols = [], []
    minx, miny, maxx, maxy = bounds
    
    for point in deposits_proj.geometry:
        if point is not None:
            col = int((point.x - minx) / Config.CELL_SIZE)
            row = int((maxy - point.y) / Config.CELL_SIZE)
            if 0 <= row < height and 0 <= col < width:
                if study_area_mask[row, col]:
                    deposit_rows.append(row)
                    deposit_cols.append(col)
    
    n_deposits = len(deposit_rows)
    
    deposit_mask = np.zeros((height, width), dtype=bool)
    for r, c in zip(deposit_rows, deposit_cols):
        deposit_mask[r, c] = True
    
    distance_from_deposits = distance_transform_edt(~deposit_mask) * Config.CELL_SIZE
    
    valid_mask = study_area_mask.copy()
    valid_mask &= (distance_from_deposits > buffer_distance)
    for predictor in predictor_stack:
        valid_mask &= ~np.isnan(predictor) & ~np.isinf(predictor)
    
    valid_rows, valid_cols = np.where(valid_mask)
    n_non_deposits = min(n_deposits, len(valid_rows))
    
    np.random.seed(42)
    indices = np.random.choice(len(valid_rows), size=n_non_deposits, replace=False)
    non_deposit_rows = valid_rows[indices]
    non_deposit_cols = valid_cols[indices]
    
    X_deposits = np.column_stack([p[deposit_rows, deposit_cols] for p in predictor_stack])
    X_non_deposits = np.column_stack([p[non_deposit_rows, non_deposit_cols] for p in predictor_stack])
    
    X = np.vstack([X_deposits, X_non_deposits])
    y = np.array([1] * n_deposits + [0] * n_non_deposits)
    
    return X, y


def load_and_prepare_data():
    """Cargar todos los datos y preparar conjunto de entrenamiento"""
    print("\n" + "="*70)
    print("CARGANDO Y PREPARANDO DATOS")
    print("="*70)
    
    # Cargar área de estudio
    study_area_gdf, study_area_geom = load_study_area()
    bounds = get_study_area_bounds(study_area_gdf)
    width, height, adjusted_bounds = calculate_grid_dimensions(bounds, Config.CELL_SIZE)
    transform = create_transform(adjusted_bounds, width, height)
    study_area_mask = create_study_area_mask(study_area_gdf, adjusted_bounds, width, height, transform)
    
    print(f"  Grid: {width}x{height} píxeles")
    print(f"  Área de estudio: {study_area_mask.sum():,} píxeles válidos")
    
    # Cargar shapefiles
    print("\n  Cargando datos vectoriales...")
    faults_gdf = gpd.read_file(Config.FAULTS_SHP)
    lithology_gdf = gpd.read_file(Config.LITHOLOGY_SHP)
    deposits_gdf = gpd.read_file(Config.DEPOSITS_SHP)
    
    # Generar predictores
    print("\n  Generando variables predictoras...")
    predictor_names = []
    predictor_stack = []
    
    # 1. Proximidad a litología favorable
    print("    - Proximidad a litología favorable")
    fav_lith = rasterize_polygon_to_binary(lithology_gdf, adjusted_bounds, width, height, transform,
                                           attribute=Config.FAVORABLE_COLUMN, value=1, mask=study_area_mask)
    prox_fav = calculate_euclidean_distance(fav_lith, Config.CELL_SIZE)
    prox_fav = apply_study_area_mask(prox_fav, study_area_mask)
    prox_fav_norm = 1 - normalize_raster(prox_fav)
    prox_fav_norm = apply_study_area_mask(prox_fav_norm, study_area_mask)
    predictor_names.append("Prox_Lith_Favorable")
    predictor_stack.append(prox_fav_norm)
    
    # 2. Densidad de fallas
    print("    - Densidad de fallas")
    fault_dens = rasterize_lines_to_density(faults_gdf, adjusted_bounds, width, height, transform,
                                            search_radius=Config.FAULT_SEARCH_RADIUS, mask=study_area_mask)
    fault_dens_norm = normalize_raster(fault_dens)
    predictor_names.append("Densidad_Fallas")
    predictor_stack.append(fault_dens_norm)
    
    # 3. Proximidad a fallas
    print("    - Proximidad a fallas")
    prox_faults = calculate_proximity_to_features(faults_gdf, adjusted_bounds, width, height, transform,
                                                   Config.CELL_SIZE, mask=study_area_mask)
    prox_faults_norm = 1 - normalize_raster(prox_faults)
    prox_faults_norm = apply_study_area_mask(prox_faults_norm, study_area_mask)
    predictor_names.append("Prox_Fallas")
    predictor_stack.append(prox_faults_norm)
    
    # 4. Proximidad a intersecciones de fallas
    print("    - Proximidad a intersecciones de fallas")
    fault_ints = extract_fault_intersections(faults_gdf, study_area_geom)
    if fault_ints is not None and len(fault_ints) > 0:
        prox_ints = calculate_proximity_to_features(fault_ints, adjusted_bounds, width, height, transform,
                                                    Config.CELL_SIZE, mask=study_area_mask)
        prox_ints_norm = 1 - normalize_raster(prox_ints)
        prox_ints_norm = apply_study_area_mask(prox_ints_norm, study_area_mask)
        predictor_names.append("Prox_Intersecciones")
        predictor_stack.append(prox_ints_norm)
    
    # 5. Densidad de contactos litológicos
    print("    - Densidad de contactos litológicos")
    contact_dens = calculate_favorable_contacts_density(lithology_gdf, adjusted_bounds, width, height, transform,
                                                        Config.CELL_SIZE, study_area_geom, mask=study_area_mask)
    contact_dens_norm = normalize_raster(contact_dens)
    predictor_names.append("Densidad_Contactos")
    predictor_stack.append(contact_dens_norm)
    
    # 6-8. Datos geoquímicos
    print("    - Componente Principal 1 (PC1)")
    pc1 = load_predictor_from_raster(Config.PC1_RASTER, adjusted_bounds, width, height, transform, study_area_mask)
    pc1_norm = normalize_raster(pc1)
    predictor_names.append("PC1_Geoquimica")
    predictor_stack.append(pc1_norm)
    
    print("    - Componente Principal 2 (PC2)")
    pc2 = load_predictor_from_raster(Config.PC2_RASTER, adjusted_bounds, width, height, transform, study_area_mask)
    pc2_norm = normalize_raster(pc2)
    predictor_names.append("PC2_Geoquimica")
    predictor_stack.append(pc2_norm)
    
    print("    - Anomalía de Tungsteno (W)")
    w_geo = load_predictor_from_raster(Config.W_RASTER, adjusted_bounds, width, height, transform, study_area_mask)
    w_geo_norm = normalize_raster(w_geo)
    predictor_names.append("W_Geoquimica")
    predictor_stack.append(w_geo_norm)
    
    print(f"\n  Total de predictores: {len(predictor_stack)}")
    
    # Generar datos de entrenamiento
    print("\n  Generando datos de entrenamiento...")
    X, y = generate_training_data(deposits_gdf, predictor_stack, transform, adjusted_bounds,
                                  width, height, study_area_mask, buffer_distance=Config.NON_DEPOSIT_BUFFER)
    
    # Eliminar muestras con NaN
    valid_samples = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X = X[valid_samples]
    y = y[valid_samples]
    
    print(f"  Muestras totales: {len(y)}")
    print(f"  Depósitos (clase 1): {sum(y == 1)}")
    print(f"  No-depósitos (clase 0): {sum(y == 0)}")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=42, stratify=y
    )
    
    print(f"\n  Conjunto de entrenamiento: {len(y_train)}")
    print(f"  Conjunto de prueba: {len(y_test)}")
    
    return X_train, X_test, y_train, y_test, predictor_names


# ============================================================================
# FUNCIONES DE OPTIMIZACIÓN
# ============================================================================

def optimize_random_forest(X_train, y_train):
    """Optimizar hiperparámetros de Random Forest"""
    print("\n" + "-"*70)
    print("OPTIMIZANDO RANDOM FOREST")
    print("-"*70)
    
    param_grid = Config.RF_PARAM_GRID_SIMPLE if Config.USE_SIMPLE_GRIDS else Config.RF_PARAM_GRID
    
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    
    print(f"  Grid de parámetros: {param_grid}")
    print(f"  Combinaciones a evaluar: {n_combinations}")
    print(f"  Total de ajustes: {n_combinations * Config.CV_FOLDS}")
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
    
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=cv,
        scoring=Config.SCORING,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n  Tiempo de optimización: {elapsed_time/60:.2f} minutos")
    print(f"\n  MEJORES PARÁMETROS ENCONTRADOS:")
    print(f"  {'-'*50}")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  {'-'*50}")
    print(f"  Mejor {Config.SCORING} (CV): {grid_search.best_score_:.4f}")
    
    return grid_search


def optimize_svm(X_train_scaled, y_train):
    """Optimizar hiperparámetros de SVM"""
    print("\n" + "-"*70)
    print("OPTIMIZANDO SUPPORT VECTOR MACHINE")
    print("-"*70)
    
    param_grid = Config.SVM_PARAM_GRID_SIMPLE if Config.USE_SIMPLE_GRIDS else Config.SVM_PARAM_GRID
    
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    
    print(f"  Grid de parámetros: {param_grid}")
    print(f"  Combinaciones a evaluar: {n_combinations}")
    print(f"  Total de ajustes: {n_combinations * Config.CV_FOLDS}")
    
    svm_base = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        svm_base,
        param_grid,
        cv=cv,
        scoring=Config.SCORING,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n  Tiempo de optimización: {elapsed_time/60:.2f} minutos")
    print(f"\n  MEJORES PARÁMETROS ENCONTRADOS:")
    print(f"  {'-'*50}")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  {'-'*50}")
    print(f"  Mejor {Config.SCORING} (CV): {grid_search.best_score_:.4f}")
    
    return grid_search


def optimize_ann(X_train_scaled, y_train):
    """Optimizar hiperparámetros de ANN"""
    print("\n" + "-"*70)
    print("OPTIMIZANDO ARTIFICIAL NEURAL NETWORK")
    print("-"*70)
    
    # Aplicar SMOTE si está disponible y configurado
    if Config.USE_SMOTE_FOR_ANN and SMOTE_AVAILABLE:
        print("  Aplicando SMOTE oversampling...")
        smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"  Después de SMOTE: Clase 0={sum(y_train_balanced==0)}, Clase 1={sum(y_train_balanced==1)}")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    param_grid = Config.ANN_PARAM_GRID_SIMPLE if Config.USE_SIMPLE_GRIDS else Config.ANN_PARAM_GRID
    
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    
    print(f"  Grid de parámetros: {param_grid}")
    print(f"  Combinaciones a evaluar: {n_combinations}")
    print(f"  Total de ajustes: {n_combinations * Config.CV_FOLDS}")
    
    ann_base = MLPClassifier(
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30
    )
    
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        ann_base,
        param_grid,
        cv=cv,
        scoring=Config.SCORING,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n  Tiempo de optimización: {elapsed_time/60:.2f} minutos")
    print(f"\n  MEJORES PARÁMETROS ENCONTRADOS:")
    print(f"  {'-'*50}")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  {'-'*50}")
    print(f"  Mejor {Config.SCORING} (CV): {grid_search.best_score_:.4f}")
    
    return grid_search


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN Y REPORTE
# ============================================================================

def plot_cv_results(grid_search, model_name, param_x, param_y, output_path):
    """Generar heatmap de resultados de validación cruzada"""
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Extraer valores únicos de los parámetros
    param_x_name = f'param_{param_x}'
    param_y_name = f'param_{param_y}'
    
    if param_x_name not in results.columns or param_y_name not in results.columns:
        print(f"  Advertencia: Parámetros {param_x} o {param_y} no encontrados en resultados")
        return
    
    pivot_table = results.pivot_table(
        values='mean_test_score',
        index=param_y_name,
        columns=param_x_name,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': f'Mean {Config.SCORING}'})
    plt.title(f'{model_name}: Resultados de GridSearchCV\n({param_x} vs {param_y})', fontsize=12)
    plt.xlabel(param_x, fontsize=11)
    plt.ylabel(param_y, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Guardado: {output_path}")
    plt.close()


def generate_report(rf_results, svm_results, ann_results, output_dir):
    """Generar reporte de resultados de optimización"""
    
    # Resumen de mejores parámetros
    summary = []
    
    # Random Forest
    rf_best = rf_results.best_params_.copy()
    rf_best['Modelo'] = 'Random Forest'
    rf_best['Mejor_Score'] = rf_results.best_score_
    summary.append(rf_best)
    
    # SVM
    svm_best = svm_results.best_params_.copy()
    svm_best['Modelo'] = 'SVM'
    svm_best['Mejor_Score'] = svm_results.best_score_
    summary.append(svm_best)
    
    # ANN
    ann_best = ann_results.best_params_.copy()
    ann_best['Modelo'] = 'ANN'
    ann_best['Mejor_Score'] = ann_results.best_score_
    summary.append(ann_best)
    
    summary_df = pd.DataFrame(summary)
    
    # Guardar resumen
    summary_path = os.path.join(output_dir, "resumen_mejores_parametros.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n  Guardado: {summary_path}")
    
    # Guardar resultados detallados de cada modelo
    pd.DataFrame(rf_results.cv_results_).to_csv(
        os.path.join(output_dir, "gridsearch_rf_detalle.csv"), index=False)
    pd.DataFrame(svm_results.cv_results_).to_csv(
        os.path.join(output_dir, "gridsearch_svm_detalle.csv"), index=False)
    pd.DataFrame(ann_results.cv_results_).to_csv(
        os.path.join(output_dir, "gridsearch_ann_detalle.csv"), index=False)
    
    # Generar reporte de texto
    report_path = os.path.join(output_dir, "reporte_optimizacion_hiperparametros.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE OPTIMIZACIÓN DE HIPERPARÁMETROS\n")
        f.write("Modelos de Machine Learning para Prospectividad Mineral\n")
        f.write("="*80 + "\n\n")
        f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Validación cruzada: {Config.CV_FOLDS}-fold estratificada\n")
        f.write(f"Métrica de optimización: {Config.SCORING}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("1. RANDOM FOREST\n")
        f.write("-"*80 + "\n")
        f.write(f"Mejor score ({Config.SCORING}): {rf_results.best_score_:.4f}\n")
        f.write("Mejores parámetros:\n")
        for param, value in rf_results.best_params_.items():
            f.write(f"  - {param}: {value}\n")
        f.write("\nPara usar en tu script, copia esta configuración:\n")
        f.write("RF_PARAMS = {\n")
        for param, value in rf_results.best_params_.items():
            if isinstance(value, str):
                f.write(f"    '{param}': '{value}',\n")
            else:
                f.write(f"    '{param}': {value},\n")
        f.write("    'oob_score': True,\n")
        f.write("    'random_state': 42,\n")
        f.write("    'n_jobs': -1\n")
        f.write("}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("2. SUPPORT VECTOR MACHINE\n")
        f.write("-"*80 + "\n")
        f.write(f"Mejor score ({Config.SCORING}): {svm_results.best_score_:.4f}\n")
        f.write("Mejores parámetros:\n")
        for param, value in svm_results.best_params_.items():
            f.write(f"  - {param}: {value}\n")
        f.write("\nPara usar en tu script, copia esta configuración:\n")
        f.write("SVM_PARAMS = {\n")
        f.write("    'kernel': 'rbf',\n")
        for param, value in svm_results.best_params_.items():
            f.write(f"    '{param}': {value},\n")
        f.write("    'probability': True,\n")
        f.write("    'random_state': 42,\n")
        f.write("    'class_weight': 'balanced'\n")
        f.write("}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("3. ARTIFICIAL NEURAL NETWORK\n")
        f.write("-"*80 + "\n")
        f.write(f"Mejor score ({Config.SCORING}): {ann_results.best_score_:.4f}\n")
        f.write("Mejores parámetros:\n")
        for param, value in ann_results.best_params_.items():
            f.write(f"  - {param}: {value}\n")
        f.write("\nPara usar en tu script, copia esta configuración:\n")
        f.write("ANN_PARAMS = {\n")
        for param, value in ann_results.best_params_.items():
            if isinstance(value, str):
                f.write(f"    '{param}': '{value}',\n")
            elif isinstance(value, tuple):
                f.write(f"    '{param}': {value},\n")
            else:
                f.write(f"    '{param}': {value},\n")
        f.write("    'solver': 'adam',\n")
        f.write("    'max_iter': 500,\n")
        f.write("    'random_state': 42,\n")
        f.write("    'early_stopping': True,\n")
        f.write("    'validation_fraction': 0.15,\n")
        f.write("    'n_iter_no_change': 30\n")
        f.write("}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("4. COMPARACIÓN DE MODELOS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Modelo':<25} {'Mejor Score':>15}\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Random Forest':<25} {rf_results.best_score_:>15.4f}\n")
        f.write(f"{'SVM':<25} {svm_results.best_score_:>15.4f}\n")
        f.write(f"{'ANN':<25} {ann_results.best_score_:>15.4f}\n")
        
        # Determinar el mejor modelo
        scores = {
            'Random Forest': rf_results.best_score_,
            'SVM': svm_results.best_score_,
            'ANN': ann_results.best_score_
        }
        best_model = max(scores, key=scores.get)
        f.write(f"\n  → Mejor modelo: {best_model} (score = {scores[best_model]:.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
    
    print(f"  Guardado: {report_path}")
    
    return summary_df


def plot_model_comparison(rf_results, svm_results, ann_results, output_path):
    """Generar gráfico comparativo de los mejores scores"""
    models = ['Random Forest', 'SVM', 'ANN']
    scores = [rf_results.best_score_, svm_results.best_score_, ann_results.best_score_]
    
    colors = ['#2E7D32', '#1565C0', '#C62828']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Agregar valores sobre las barras
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(f'Mejor {Config.SCORING} (CV)', fontsize=12)
    ax.set_title('Comparación de Modelos - Mejores Scores de GridSearchCV', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) * 1.15)
    
    # Línea de referencia
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Clasificador aleatorio')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Guardado: {output_path}")
    plt.close()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    print("\n" + "="*80)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS PARA MODELOS DE MACHINE LEARNING")
    print("Random Forest | Support Vector Machine | Artificial Neural Network")
    print("="*80)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Validación cruzada: {Config.CV_FOLDS}-fold estratificada")
    print(f"Métrica de optimización: {Config.SCORING}")
    print(f"Grids simplificados: {Config.USE_SIMPLE_GRIDS}")
    
    # Crear directorio de salida
    output_dir = create_output_directory()
    print(f"Directorio de salida: {output_dir}")
    
    # Cargar y preparar datos
    X_train, X_test, y_train, y_test, predictor_names = load_and_prepare_data()
    
    # Escalar datos para SVM y ANN
    print("\n  Escalando características (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # =========================================================================
    # OPTIMIZACIÓN DE MODELOS
    # =========================================================================
    
    total_start = time.time()
    
    # 1. Random Forest
    rf_results = optimize_random_forest(X_train, y_train)
    
    # 2. SVM
    svm_results = optimize_svm(X_train_scaled, y_train)
    
    # 3. ANN
    ann_results = optimize_ann(X_train_scaled, y_train)
    
    total_time = time.time() - total_start
    
    # =========================================================================
    # GENERAR REPORTES Y VISUALIZACIONES
    # =========================================================================
    
    print("\n" + "="*70)
    print("GENERANDO REPORTES Y VISUALIZACIONES")
    print("="*70)
    
    # Generar reporte
    summary_df = generate_report(rf_results, svm_results, ann_results, output_dir)
    
    # Visualizaciones
    plot_model_comparison(rf_results, svm_results, ann_results,
                         os.path.join(output_dir, "comparacion_modelos.png"))
    
    # Heatmaps de GridSearch (si hay 2 parámetros principales)
    try:
        plot_cv_results(svm_results, 'SVM', 'C', 'gamma',
                       os.path.join(output_dir, "heatmap_svm_C_gamma.png"))
    except:
        pass
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    
    print("\n" + "="*80)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*80)
    print(f"\nTiempo total de ejecución: {total_time/60:.2f} minutos")
    
    print("\n" + "-"*60)
    print("MEJORES PARÁMETROS ENCONTRADOS")
    print("-"*60)
    
    print("\n  RANDOM FOREST:")
    for param, value in rf_results.best_params_.items():
        print(f"    {param}: {value}")
    print(f"    → Mejor score: {rf_results.best_score_:.4f}")
    
    print("\n  SVM:")
    for param, value in svm_results.best_params_.items():
        print(f"    {param}: {value}")
    print(f"    → Mejor score: {svm_results.best_score_:.4f}")
    
    print("\n  ANN:")
    for param, value in ann_results.best_params_.items():
        print(f"    {param}: {value}")
    print(f"    → Mejor score: {ann_results.best_score_:.4f}")
    
    print(f"\nArchivos generados en: {output_dir}")
    print("\nArchivos principales:")
    print("  - resumen_mejores_parametros.csv")
    print("  - reporte_optimizacion_hiperparametros.txt")
    print("  - gridsearch_rf_detalle.csv")
    print("  - gridsearch_svm_detalle.csv")
    print("  - gridsearch_ann_detalle.csv")
    print("  - comparacion_modelos.png")
    
    print("\n" + "="*80)
    print("¡Copia los mejores parámetros al archivo de configuración de cada modelo!")
    print("="*80)
    
    return rf_results, svm_results, ann_results


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    try:
        rf, svm, ann = main()
    except FileNotFoundError as e:
        print(f"\nERROR: Archivo no encontrado - {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()