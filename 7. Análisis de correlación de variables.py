# -*- coding: utf-8 -*-
"""
================================================================================
ANÁLISIS DE CORRELACIÓN Y MULTICOLINEALIDAD DE VARIABLES PREDICTORAS
================================================================================
Análisis estadístico de las variables predictoras utilizadas en los modelos
de Machine Learning para mapeo de prospectividad mineral.

Incluye:
- Matriz de correlación de Pearson con significancia estadística
- Matriz de correlación de Spearman (relaciones no lineales)
- Factor de Inflación de la Varianza (VIF) para multicolinealidad
- Número de condición de la matriz de correlación
- Tolerancia (1/VIF)
- Visualizaciones: heatmaps, dendrogramas, scatter matrix

Autor: Adaptado para proyecto de tesis
================================================================================
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio import features
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración de rutas y parámetros"""
    
    # Directorio de datos
    INPUT_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Shapefiles"
    RASTER_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Rasters"
    OUTPUT_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Results\Analisis_Correlacion"
    
    # Archivos de entrada
    STUDY_AREA_SHP = os.path.join(INPUT_DIR, "area.shp")
    
    # Lista de predictores (en el orden que se generaron)
    PREDICTORS = {
        'prox_favorable_lithology': 'Proximidad a Litología Favorable',
        'fault_density': 'Densidad de Fallas',
        'prox_faults': 'Proximidad a Fallas',
        'prox_fault_intersections': 'Proximidad a Intersecciones de Fallas',
        'lithological_contacts_density': 'Densidad de Contactos Litológicos',
        'geochemical_pc1': 'Componente Principal 1 (Geoquímica)',
        'geochemical_pc2': 'Componente Principal 2 (Geoquímica)',
        'geochemical_tungsten': 'Anomalía de Tungsteno'
    }
    
    # Nombres cortos para visualizaciones
    PREDICTORS_SHORT = {
        'prox_favorable_lithology': 'Prox_Lith_Fav',
        'fault_density': 'Dens_Fallas',
        'prox_faults': 'Prox_Fallas',
        'prox_fault_intersections': 'Prox_Inters',
        'lithological_contacts_density': 'Dens_Contactos',
        'geochemical_pc1': 'PC1',
        'geochemical_pc2': 'PC2',
        'geochemical_tungsten': 'W_Anomaly'
    }
    
    # Sistema de coordenadas
    OUTPUT_CRS = "EPSG:32718"
    CELL_SIZE = 500
    
    # Nivel de significancia para pruebas estadísticas
    ALPHA = 0.05
    
    # Umbral de VIF para multicolinealidad problemática
    VIF_THRESHOLD = 5.0  # Valores > 5 indican multicolinealidad moderada
    VIF_SEVERE = 10.0    # Valores > 10 indican multicolinealidad severa


def create_output_directory():
    """Crear directorio de salida si no existe"""
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        print(f"Directorio creado: {Config.OUTPUT_DIR}")
    return Config.OUTPUT_DIR


def load_study_area():
    """Cargar y proyectar el área de estudio"""
    print(f"Cargando área de estudio: {Config.STUDY_AREA_SHP}")
    study_area_gdf = gpd.read_file(Config.STUDY_AREA_SHP)
    study_area_gdf = study_area_gdf.to_crs(Config.OUTPUT_CRS)
    study_area_geom = unary_union(study_area_gdf.geometry)
    return study_area_gdf, study_area_geom


def load_predictor_raster(predictor_name):
    """Cargar un raster predictor"""
    raster_path = os.path.join(Config.RASTER_DIR, f"{predictor_name}.tif")
    
    if not os.path.exists(raster_path):
        print(f"  ADVERTENCIA: No se encontró {raster_path}")
        return None, None, None
    
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        return data, src.transform, src.crs


def extract_valid_samples(predictor_dict, sample_fraction=0.1, max_samples=50000):
    """
    Extraer muestras válidas de todos los predictores.
    
    Parameters:
    -----------
    predictor_dict : dict
        Diccionario con nombre: array de cada predictor
    sample_fraction : float
        Fracción de píxeles válidos a muestrear (para eficiencia)
    max_samples : int
        Número máximo de muestras
    
    Returns:
    --------
    DataFrame con valores de cada predictor
    """
    # Crear máscara de píxeles válidos (válidos en TODOS los predictores)
    valid_mask = None
    
    for name, data in predictor_dict.items():
        if data is None:
            continue
        predictor_valid = ~np.isnan(data) & ~np.isinf(data)
        if valid_mask is None:
            valid_mask = predictor_valid
        else:
            valid_mask &= predictor_valid
    
    if valid_mask is None or not valid_mask.any():
        print("ERROR: No hay píxeles válidos en común entre todos los predictores")
        return None
    
    # Obtener índices de píxeles válidos
    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)
    print(f"  Píxeles válidos totales: {n_valid:,}")
    
    # Muestrear para eficiencia si hay muchos píxeles
    n_samples = min(int(n_valid * sample_fraction), max_samples, n_valid)
    
    if n_samples < n_valid:
        np.random.seed(42)
        indices = np.random.choice(n_valid, size=n_samples, replace=False)
        sample_rows = valid_rows[indices]
        sample_cols = valid_cols[indices]
        print(f"  Muestras seleccionadas: {n_samples:,} ({sample_fraction*100:.1f}%)")
    else:
        sample_rows = valid_rows
        sample_cols = valid_cols
        print(f"  Usando todos los píxeles válidos: {n_valid:,}")
    
    # Extraer valores
    data_dict = {}
    for name, data in predictor_dict.items():
        if data is not None:
            values = data[sample_rows, sample_cols]
            data_dict[name] = values
    
    df = pd.DataFrame(data_dict)
    
    return df


def calculate_pearson_correlation(df):
    """
    Calcular matriz de correlación de Pearson con valores p.
    
    Returns:
    --------
    corr_matrix : DataFrame - Matriz de correlación
    pval_matrix : DataFrame - Matriz de valores p
    """
    n_vars = len(df.columns)
    var_names = df.columns.tolist()
    
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                # Correlación de Pearson
                r, p = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p
    
    corr_df = pd.DataFrame(corr_matrix, index=var_names, columns=var_names)
    pval_df = pd.DataFrame(pval_matrix, index=var_names, columns=var_names)
    
    return corr_df, pval_df


def calculate_spearman_correlation(df):
    """
    Calcular matriz de correlación de Spearman con valores p.
    Útil para detectar relaciones monotónicas no lineales.
    
    Returns:
    --------
    corr_matrix : DataFrame - Matriz de correlación
    pval_matrix : DataFrame - Matriz de valores p
    """
    n_vars = len(df.columns)
    var_names = df.columns.tolist()
    
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                # Correlación de Spearman
                rho, p = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
                corr_matrix[i, j] = rho
                pval_matrix[i, j] = p
    
    corr_df = pd.DataFrame(corr_matrix, index=var_names, columns=var_names)
    pval_df = pd.DataFrame(pval_matrix, index=var_names, columns=var_names)
    
    return corr_df, pval_df


def calculate_vif(df):
    """
    Calcular el Factor de Inflación de la Varianza (VIF) para cada variable.
    
    VIF = 1 / (1 - R²)
    
    Donde R² es el coeficiente de determinación de la regresión de una variable
    contra todas las demás.
    
    Interpretación:
    - VIF = 1: No hay correlación con otras variables
    - VIF 1-5: Correlación moderada (generalmente aceptable)
    - VIF 5-10: Alta correlación (puede ser problemática)
    - VIF > 10: Multicolinealidad severa (problemática)
    
    Returns:
    --------
    DataFrame con VIF y Tolerancia para cada variable
    """
    # Agregar constante para el cálculo de VIF
    df_const = add_constant(df)
    
    vif_data = []
    
    for i, col in enumerate(df.columns):
        vif = variance_inflation_factor(df_const.values, i + 1)  # +1 por la constante
        tolerance = 1 / vif if vif != 0 else np.inf
        
        # Interpretación
        if vif < 5:
            interpretation = "Aceptable"
        elif vif < 10:
            interpretation = "Moderada"
        else:
            interpretation = "Severa"
        
        vif_data.append({
            'Variable': col,
            'VIF': vif,
            'Tolerancia': tolerance,
            'Interpretación': interpretation
        })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)
    
    return vif_df


def calculate_condition_number(df):
    """
    Calcular el número de condición de la matriz de correlación.
    
    El número de condición es la raíz cuadrada del cociente entre el mayor
    y el menor autovalor de la matriz de correlación.
    
    Interpretación:
    - < 10: No hay problema de multicolinealidad
    - 10-30: Multicolinealidad moderada
    - > 30: Multicolinealidad severa
    
    Returns:
    --------
    condition_number : float
    eigenvalues : array
    """
    # Calcular matriz de correlación
    corr_matrix = df.corr().values
    
    # Calcular autovalores
    eigenvalues = np.linalg.eigvals(corr_matrix)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Ordenar descendente
    
    # Número de condición
    condition_number = np.sqrt(eigenvalues[0] / eigenvalues[-1])
    
    return condition_number, eigenvalues


def calculate_determinant(df):
    """
    Calcular el determinante de la matriz de correlación.
    
    Un determinante cercano a 0 indica alta multicolinealidad.
    Un determinante cercano a 1 indica ortogonalidad.
    
    Returns:
    --------
    determinant : float
    """
    corr_matrix = df.corr().values
    det = np.linalg.det(corr_matrix)
    return det


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_correlation_heatmap(corr_df, pval_df, title, output_path, 
                             short_names=None, annotate_significance=True):
    """
    Generar heatmap de correlación con indicadores de significancia.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Usar nombres cortos si se proporcionan
    if short_names:
        labels = [short_names.get(col, col) for col in corr_df.columns]
    else:
        labels = corr_df.columns.tolist()
    
    # Crear matriz de anotaciones con significancia
    if annotate_significance:
        annot = corr_df.copy()
        for i in range(len(corr_df)):
            for j in range(len(corr_df.columns)):
                r = corr_df.iloc[i, j]
                p = pval_df.iloc[i, j]
                if i == j:
                    annot.iloc[i, j] = "1.00"
                elif p < 0.001:
                    annot.iloc[i, j] = f"{r:.2f}***"
                elif p < 0.01:
                    annot.iloc[i, j] = f"{r:.2f}**"
                elif p < 0.05:
                    annot.iloc[i, j] = f"{r:.2f}*"
                else:
                    annot.iloc[i, j] = f"{r:.2f}"
        annot_array = annot.values
    else:
        annot_array = np.round(corr_df.values, 2)
    
    # Crear heatmap
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)  # Triángulo superior
    
    sns.heatmap(corr_df, mask=mask, annot=annot_array, fmt='',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Coeficiente de Correlación', 'shrink': 0.8},
                annot_kws={'size': 9})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Agregar leyenda de significancia
    if annotate_significance:
        legend_text = "Significancia: *** p<0.001, ** p<0.01, * p<0.05"
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {output_path}")
    plt.close()


def plot_vif_barplot(vif_df, output_path, short_names=None):
    """
    Generar gráfico de barras para VIF.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Usar nombres cortos
    if short_names:
        labels = [short_names.get(v, v) for v in vif_df['Variable']]
    else:
        labels = vif_df['Variable'].tolist()
    
    # Colores según nivel de multicolinealidad
    colors = []
    for vif in vif_df['VIF']:
        if vif < Config.VIF_THRESHOLD:
            colors.append('#2E7D32')  # Verde - Aceptable
        elif vif < Config.VIF_SEVERE:
            colors.append('#FFA000')  # Naranja - Moderada
        else:
            colors.append('#C62828')  # Rojo - Severa
    
    bars = ax.barh(labels, vif_df['VIF'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Líneas de umbral
    ax.axvline(x=Config.VIF_THRESHOLD, color='orange', linestyle='--', 
               linewidth=2, label=f'Umbral moderado (VIF={Config.VIF_THRESHOLD})')
    ax.axvline(x=Config.VIF_SEVERE, color='red', linestyle='--', 
               linewidth=2, label=f'Umbral severo (VIF={Config.VIF_SEVERE})')
    
    # Etiquetas de valor en barras
    for bar, vif in zip(bars, vif_df['VIF']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{vif:.2f}', va='center', ha='left', fontsize=9)
    
    ax.set_xlabel('Factor de Inflación de la Varianza (VIF)', fontsize=12)
    ax.set_ylabel('Variable Predictora', fontsize=12)
    ax.set_title('Análisis de Multicolinealidad - VIF', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max(vif_df['VIF']) * 1.2)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {output_path}")
    plt.close()


def plot_correlation_dendrogram(df, output_path, short_names=None):
    """
    Generar dendrograma de clustering jerárquico basado en correlación.
    """
    # Calcular matriz de distancia (1 - |correlación|)
    corr_matrix = df.corr().abs()
    distance_matrix = 1 - corr_matrix
    
    # Linkage jerárquico
    linkage = hierarchy.linkage(
        hierarchy.distance.squareform(distance_matrix.values),
        method='average'
    )
    
    # Usar nombres cortos
    if short_names:
        labels = [short_names.get(col, col) for col in df.columns]
    else:
        labels = df.columns.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hierarchy.dendrogram(
        linkage,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax
    )
    
    ax.set_xlabel('Variables Predictoras', fontsize=12)
    ax.set_ylabel('Distancia (1 - |r|)', fontsize=12)
    ax.set_title('Dendrograma de Correlación entre Variables Predictoras', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {output_path}")
    plt.close()


def plot_eigenvalues(eigenvalues, condition_number, output_path):
    """
    Generar gráfico de autovalores de la matriz de correlación.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de barras de autovalores
    x = range(1, len(eigenvalues) + 1)
    ax1.bar(x, eigenvalues, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='λ = 1')
    ax1.set_xlabel('Componente', fontsize=12)
    ax1.set_ylabel('Autovalor (λ)', fontsize=12)
    ax1.set_title('Autovalores de la Matriz de Correlación', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(x)
    
    # Gráfico de varianza acumulada
    variance_explained = eigenvalues / np.sum(eigenvalues) * 100
    cumulative_variance = np.cumsum(variance_explained)
    
    ax2.plot(x, cumulative_variance, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='95%')
    ax2.fill_between(x, cumulative_variance, alpha=0.3)
    ax2.set_xlabel('Número de Componentes', fontsize=12)
    ax2.set_ylabel('Varianza Acumulada (%)', fontsize=12)
    ax2.set_title('Varianza Explicada Acumulada', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_ylim(0, 105)
    
    # Agregar número de condición como texto
    fig.suptitle(f'Número de Condición = {condition_number:.2f}', 
                 fontsize=11, style='italic', y=0.02)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {output_path}")
    plt.close()


def plot_scatter_matrix(df, output_path, short_names=None, max_vars=6):
    """
    Generar matriz de dispersión para las variables más importantes.
    """
    # Limitar número de variables para legibilidad
    if len(df.columns) > max_vars:
        print(f"  Nota: Mostrando solo las primeras {max_vars} variables en scatter matrix")
        df_plot = df.iloc[:, :max_vars]
    else:
        df_plot = df
    
    # Usar nombres cortos
    if short_names:
        df_plot.columns = [short_names.get(col, col) for col in df_plot.columns]
    
    # Submuestrear para eficiencia
    if len(df_plot) > 2000:
        df_plot = df_plot.sample(n=2000, random_state=42)
    
    fig = plt.figure(figsize=(14, 14))
    
    axes = pd.plotting.scatter_matrix(df_plot, alpha=0.3, figsize=(14, 14),
                                       diagonal='hist', marker='.', s=10)
    
    # Ajustar formato
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
        ax.tick_params(axis='both', labelsize=8)
    
    plt.suptitle('Matriz de Dispersión de Variables Predictoras', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Guardado: {output_path}")
    plt.close()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del análisis de correlación y multicolinealidad"""
    
    print("="*80)
    print("ANÁLISIS DE CORRELACIÓN Y MULTICOLINEALIDAD")
    print("Variables Predictoras para Modelos de Machine Learning")
    print("="*80)
    
    # Crear directorio de salida
    output_dir = create_output_directory()
    
    # -------------------------------------------------------------------------
    # PASO 1: Cargar predictores
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 1: Cargando rasters predictores...")
    print("-"*60)
    
    predictor_data = {}
    predictor_info = []
    
    for predictor_name, predictor_desc in Config.PREDICTORS.items():
        print(f"\n  Cargando: {predictor_name}")
        data, transform, crs = load_predictor_raster(predictor_name)
        
        if data is not None:
            predictor_data[predictor_name] = data
            
            # Estadísticas básicas
            valid_data = data[~np.isnan(data)]
            predictor_info.append({
                'Variable': predictor_name,
                'Descripción': predictor_desc,
                'Min': valid_data.min(),
                'Max': valid_data.max(),
                'Media': valid_data.mean(),
                'Desv. Est.': valid_data.std(),
                'N Válidos': len(valid_data)
            })
            print(f"    - Rango: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
            print(f"    - Media ± DE: {valid_data.mean():.4f} ± {valid_data.std():.4f}")
    
    print(f"\n  Total de predictores cargados: {len(predictor_data)}")
    
    # Guardar estadísticas descriptivas
    stats_df = pd.DataFrame(predictor_info)
    stats_path = os.path.join(output_dir, "estadisticas_descriptivas_predictores.csv")
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"\n  Guardado: {stats_path}")
    
    # -------------------------------------------------------------------------
    # PASO 2: Extraer muestras válidas
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 2: Extrayendo muestras para análisis...")
    print("-"*60)
    
    df = extract_valid_samples(predictor_data, sample_fraction=0.1, max_samples=50000)
    
    if df is None:
        print("ERROR: No se pudieron extraer muestras válidas")
        return
    
    print(f"\n  DataFrame resultante: {df.shape[0]:,} muestras × {df.shape[1]} variables")
    
    # -------------------------------------------------------------------------
    # PASO 3: Correlación de Pearson
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 3: Calculando correlación de Pearson...")
    print("-"*60)
    
    pearson_corr, pearson_pval = calculate_pearson_correlation(df)
    
    print("\n  Matriz de Correlación de Pearson:")
    print(pearson_corr.round(3).to_string())
    
    # Guardar resultados
    pearson_corr.to_csv(os.path.join(output_dir, "correlacion_pearson.csv"), 
                        encoding='utf-8-sig')
    pearson_pval.to_csv(os.path.join(output_dir, "correlacion_pearson_pvalores.csv"), 
                        encoding='utf-8-sig')
    
    # Identificar correlaciones altas
    print("\n  Correlaciones significativas (|r| > 0.5):")
    high_corr = []
    for i in range(len(pearson_corr)):
        for j in range(i+1, len(pearson_corr.columns)):
            r = pearson_corr.iloc[i, j]
            p = pearson_pval.iloc[i, j]
            if abs(r) > 0.5:
                var1 = pearson_corr.index[i]
                var2 = pearson_corr.columns[j]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {var1} ↔ {var2}: r = {r:.3f}{sig}")
                high_corr.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Pearson r': r,
                    'p-valor': p,
                    'Significancia': sig
                })
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        high_corr_df.to_csv(os.path.join(output_dir, "correlaciones_altas.csv"), 
                           index=False, encoding='utf-8-sig')
    
    # Visualización
    plot_correlation_heatmap(pearson_corr, pearson_pval, 
                            "Matriz de Correlación de Pearson",
                            os.path.join(output_dir, "heatmap_correlacion_pearson.png"),
                            Config.PREDICTORS_SHORT)
    
    # -------------------------------------------------------------------------
    # PASO 4: Correlación de Spearman
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 4: Calculando correlación de Spearman...")
    print("-"*60)
    
    spearman_corr, spearman_pval = calculate_spearman_correlation(df)
    
    print("\n  Matriz de Correlación de Spearman:")
    print(spearman_corr.round(3).to_string())
    
    # Guardar resultados
    spearman_corr.to_csv(os.path.join(output_dir, "correlacion_spearman.csv"), 
                         encoding='utf-8-sig')
    spearman_pval.to_csv(os.path.join(output_dir, "correlacion_spearman_pvalores.csv"), 
                         encoding='utf-8-sig')
    
    # Visualización
    plot_correlation_heatmap(spearman_corr, spearman_pval, 
                            "Matriz de Correlación de Spearman",
                            os.path.join(output_dir, "heatmap_correlacion_spearman.png"),
                            Config.PREDICTORS_SHORT)
    
    # -------------------------------------------------------------------------
    # PASO 5: Factor de Inflación de la Varianza (VIF)
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 5: Calculando Factor de Inflación de la Varianza (VIF)...")
    print("-"*60)
    
    vif_df = calculate_vif(df)
    
    print("\n  Resultados VIF:")
    print(vif_df.to_string(index=False))
    
    # Guardar resultados
    vif_df.to_csv(os.path.join(output_dir, "vif_multicolinealidad.csv"), 
                  index=False, encoding='utf-8-sig')
    
    # Diagnóstico
    n_problematic = sum(vif_df['VIF'] >= Config.VIF_THRESHOLD)
    n_severe = sum(vif_df['VIF'] >= Config.VIF_SEVERE)
    
    print(f"\n  Diagnóstico de Multicolinealidad:")
    print(f"    - Variables con VIF ≥ {Config.VIF_THRESHOLD} (moderada): {n_problematic}")
    print(f"    - Variables con VIF ≥ {Config.VIF_SEVERE} (severa): {n_severe}")
    
    if n_severe > 0:
        print("    ⚠️ ADVERTENCIA: Se detectó multicolinealidad severa")
    elif n_problematic > 0:
        print("    ⚠️ NOTA: Se detectó multicolinealidad moderada")
    else:
        print("    ✓ No se detectó multicolinealidad problemática")
    
    # Visualización
    plot_vif_barplot(vif_df, os.path.join(output_dir, "barplot_vif.png"),
                    Config.PREDICTORS_SHORT)
    
    # -------------------------------------------------------------------------
    # PASO 6: Número de Condición y Autovalores
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 6: Calculando número de condición...")
    print("-"*60)
    
    condition_number, eigenvalues = calculate_condition_number(df)
    determinant = calculate_determinant(df)
    
    print(f"\n  Número de condición: {condition_number:.2f}")
    print(f"  Determinante de la matriz de correlación: {determinant:.6f}")
    print(f"\n  Autovalores: {np.round(eigenvalues, 4)}")
    
    # Interpretación del número de condición
    if condition_number < 10:
        cond_interp = "No hay problema de multicolinealidad"
    elif condition_number < 30:
        cond_interp = "Multicolinealidad moderada"
    else:
        cond_interp = "Multicolinealidad severa"
    
    print(f"\n  Interpretación: {cond_interp}")
    
    # Guardar autovalores
    eigen_df = pd.DataFrame({
        'Componente': range(1, len(eigenvalues) + 1),
        'Autovalor': eigenvalues,
        'Varianza Explicada (%)': eigenvalues / np.sum(eigenvalues) * 100,
        'Varianza Acumulada (%)': np.cumsum(eigenvalues / np.sum(eigenvalues) * 100)
    })
    eigen_df.to_csv(os.path.join(output_dir, "autovalores_matriz_correlacion.csv"), 
                    index=False, encoding='utf-8-sig')
    
    # Visualización
    plot_eigenvalues(eigenvalues, condition_number, 
                    os.path.join(output_dir, "autovalores_varianza.png"))
    
    # -------------------------------------------------------------------------
    # PASO 7: Dendrograma de correlación
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 7: Generando dendrograma de correlación...")
    print("-"*60)
    
    plot_correlation_dendrogram(df, os.path.join(output_dir, "dendrograma_correlacion.png"),
                               Config.PREDICTORS_SHORT)
    
    # -------------------------------------------------------------------------
    # PASO 8: Matriz de dispersión
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 8: Generando matriz de dispersión...")
    print("-"*60)
    
    plot_scatter_matrix(df, os.path.join(output_dir, "scatter_matrix.png"),
                       Config.PREDICTORS_SHORT)
    
    # -------------------------------------------------------------------------
    # PASO 9: Resumen ejecutivo
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("PASO 9: Generando resumen ejecutivo...")
    print("-"*60)
    
    resumen = {
        'Métrica': [
            'Número de variables predictoras',
            'Número de muestras analizadas',
            'Correlación máxima (Pearson)',
            'Par con mayor correlación',
            'Número de condición',
            'Determinante matriz correlación',
            'VIF máximo',
            'Variable con mayor VIF',
            'Variables con VIF > 5',
            'Variables con VIF > 10'
        ],
        'Valor': [
            len(df.columns),
            len(df),
            f"{pearson_corr.abs().values[np.triu_indices_from(pearson_corr.values, k=1)].max():.3f}",
            f"{high_corr[0]['Variable 1']} ↔ {high_corr[0]['Variable 2']}" if high_corr else "N/A",
            f"{condition_number:.2f}",
            f"{determinant:.6f}",
            f"{vif_df['VIF'].max():.2f}",
            vif_df.iloc[0]['Variable'],
            n_problematic,
            n_severe
        ]
    }
    
    resumen_df = pd.DataFrame(resumen)
    resumen_path = os.path.join(output_dir, "resumen_analisis_correlacion.csv")
    resumen_df.to_csv(resumen_path, index=False, encoding='utf-8-sig')
    print(f"\n  Guardado: {resumen_path}")
    
    # Generar reporte de texto
    report_path = os.path.join(output_dir, "reporte_correlacion_multicolinealidad.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE ANÁLISIS DE CORRELACIÓN Y MULTICOLINEALIDAD\n")
        f.write("Variables Predictoras para Modelos de Prospectividad Mineral\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. RESUMEN DE DATOS\n")
        f.write("-"*40 + "\n")
        f.write(f"   Número de variables: {len(df.columns)}\n")
        f.write(f"   Número de muestras: {len(df):,}\n\n")
        
        f.write("2. CORRELACIÓN DE PEARSON\n")
        f.write("-"*40 + "\n")
        f.write("   Matriz de correlación:\n")
        f.write(pearson_corr.round(3).to_string() + "\n\n")
        
        if high_corr:
            f.write("   Correlaciones altas (|r| > 0.5):\n")
            for hc in high_corr:
                f.write(f"   - {hc['Variable 1']} ↔ {hc['Variable 2']}: r = {hc['Pearson r']:.3f} (p = {hc['p-valor']:.2e})\n")
            f.write("\n")
        
        f.write("3. FACTOR DE INFLACIÓN DE LA VARIANZA (VIF)\n")
        f.write("-"*40 + "\n")
        f.write(vif_df.to_string(index=False) + "\n\n")
        
        f.write("   Interpretación VIF:\n")
        f.write("   - VIF < 5: Aceptable (no hay multicolinealidad problemática)\n")
        f.write("   - VIF 5-10: Multicolinealidad moderada\n")
        f.write("   - VIF > 10: Multicolinealidad severa\n\n")
        
        f.write("4. NÚMERO DE CONDICIÓN\n")
        f.write("-"*40 + "\n")
        f.write(f"   Número de condición: {condition_number:.2f}\n")
        f.write(f"   Determinante: {determinant:.6f}\n")
        f.write(f"   Interpretación: {cond_interp}\n\n")
        
        f.write("   Criterios de interpretación:\n")
        f.write("   - < 10: No hay problema de multicolinealidad\n")
        f.write("   - 10-30: Multicolinealidad moderada\n")
        f.write("   - > 30: Multicolinealidad severa\n\n")
        
        f.write("5. AUTOVALORES DE LA MATRIZ DE CORRELACIÓN\n")
        f.write("-"*40 + "\n")
        f.write(eigen_df.to_string(index=False) + "\n\n")
        
        f.write("6. CONCLUSIONES\n")
        f.write("-"*40 + "\n")
        if n_severe > 0:
            f.write("   ⚠️ Se detectó multicolinealidad SEVERA en las variables predictoras.\n")
            f.write("   Se recomienda considerar la eliminación o combinación de variables\n")
            f.write("   altamente correlacionadas antes del modelado.\n")
        elif n_problematic > 0:
            f.write("   ⚠️ Se detectó multicolinealidad MODERADA en algunas variables.\n")
            f.write("   Los modelos basados en árboles (Random Forest) son robustos a este problema.\n")
            f.write("   Para SVM y ANN, se recomienda monitorear el rendimiento.\n")
        else:
            f.write("   ✓ No se detectó multicolinealidad problemática.\n")
            f.write("   Las variables predictoras son suficientemente independientes.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Fin del reporte\n")
    
    print(f"  Guardado: {report_path}")
    
    # -------------------------------------------------------------------------
    # FIN
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("ANÁLISIS DE CORRELACIÓN Y MULTICOLINEALIDAD COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados en: {output_dir}")
    print("\nArchivos CSV:")
    print("  - estadisticas_descriptivas_predictores.csv")
    print("  - correlacion_pearson.csv")
    print("  - correlacion_pearson_pvalores.csv")
    print("  - correlacion_spearman.csv")
    print("  - correlacion_spearman_pvalores.csv")
    print("  - correlaciones_altas.csv")
    print("  - vif_multicolinealidad.csv")
    print("  - autovalores_matriz_correlacion.csv")
    print("  - resumen_analisis_correlacion.csv")
    print("\nFiguras:")
    print("  - heatmap_correlacion_pearson.png")
    print("  - heatmap_correlacion_spearman.png")
    print("  - barplot_vif.png")
    print("  - autovalores_varianza.png")
    print("  - dendrograma_correlacion.png")
    print("  - scatter_matrix.png")
    print("\nReporte:")
    print("  - reporte_correlacion_multicolinealidad.txt")
    
    return df, pearson_corr, vif_df


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    try:
        df, corr, vif = main()
    except FileNotFoundError as e:
        print(f"\nERROR: Archivo no encontrado - {e}")
        print("Verifique que los rasters predictores existan en el directorio especificado.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()