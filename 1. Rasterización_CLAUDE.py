# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:33:20 2026

@author: JHEF
"""

"""
SCRIPT DE RASTERIZACIÓN Y ESTANDARIZACIÓN DE DATOS
Mineral Prospectivity Mapping - Proyecto Tesis

Autor: JHEF
Fecha: 2026

Este script:
1. Define área de estudio y parámetros comunes
2. Reproyecta rasters geoquímicos a grid estándar
3. Rasteriza litologías favorables
4. Crea contactos litológicos con multi-buffer
5. Rasteriza fallas con buffer
6. Estandariza todo a mismo extent y cell size
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARÁMETROS GLOBALES
# ============================================================================

# Rutas
INPUT_DIR = r"C:\Users\JHEF\Documents\Proyecto Tesis\Información\Machine Learning\Shapefiles"
OUTPUT_DIR = r"C:\Users\JHEF\Documents\Proyecto Tesis\Información\Machine Learning\Rasters"

# Nombres de archivos
LITOLOGIAS_SHP = "litologias.shp"
FALLAS_SHP = "fallas.shp"
YACIMIENTOS_SHP = "yacimientos.shp"
PC1_RASTER = "PC1.tif"
PC2_RASTER = "PC2.tif"

# Parámetros de rasterización
TARGET_CRS = "EPSG:4326"  # GCS_WGS_1984
CELL_SIZE_DEGREES = 0.0045  # Aproximadamente 500m en el ecuador
# Nota: 500m ≈ 0.0045 grados en latitud (ajustar si estás muy lejos del ecuador)

# Parámetros de procesamiento
FAULT_BUFFER_METERS = 500  # Buffer para fallas
CONTACT_BUFFERS_METERS = [50, 100, 150, 200]  # Multi-buffer para contactos

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("RASTERIZACIÓN Y ESTANDARIZACIÓN DE DATOS PARA MPM")
print("="*80)
print(f"\nDirectorio de entrada: {INPUT_DIR}")
print(f"Directorio de salida: {OUTPUT_DIR}")
print(f"Sistema de coordenadas: {TARGET_CRS}")
print(f"Tamaño de celda: {CELL_SIZE_DEGREES} grados (~500m)")
print("="*80)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_common_extent_and_grid():
    """
    Calcula la extensión común y parámetros de grid basándose en todos los datos
    """
    print("\n[PASO 1] Calculando extensión común del área de estudio...")
    
    # Leer todos los shapefiles
    litologias = gpd.read_file(os.path.join(INPUT_DIR, LITOLOGIAS_SHP))
    fallas = gpd.read_file(os.path.join(INPUT_DIR, FALLAS_SHP))
    yacimientos = gpd.read_file(os.path.join(INPUT_DIR, YACIMIENTOS_SHP))
    
    # Reproyectar todos a WGS84
    litologias = litologias.to_crs(TARGET_CRS)
    fallas = fallas.to_crs(TARGET_CRS)
    yacimientos = yacimientos.to_crs(TARGET_CRS)
    
    # Leer rasters geoquímicos
    with rasterio.open(os.path.join(INPUT_DIR, PC1_RASTER)) as src:
        pc1_bounds = src.bounds
    
    with rasterio.open(os.path.join(INPUT_DIR, PC2_RASTER)) as src:
        pc2_bounds = src.bounds
    
    # Combinar todos los bounds
    all_bounds = [
        litologias.total_bounds,
        fallas.total_bounds,
        yacimientos.total_bounds,
        [pc1_bounds.left, pc1_bounds.bottom, pc1_bounds.right, pc1_bounds.top],
        [pc2_bounds.left, pc2_bounds.bottom, pc2_bounds.right, pc2_bounds.top]
    ]
    
    # Calcular extensión común (máxima)
    minx = min([b[0] for b in all_bounds])
    miny = min([b[1] for b in all_bounds])
    maxx = max([b[2] for b in all_bounds])
    maxy = max([b[3] for b in all_bounds])
    
    # Ajustar a múltiplos del tamaño de celda
    minx = np.floor(minx / CELL_SIZE_DEGREES) * CELL_SIZE_DEGREES
    miny = np.floor(miny / CELL_SIZE_DEGREES) * CELL_SIZE_DEGREES
    maxx = np.ceil(maxx / CELL_SIZE_DEGREES) * CELL_SIZE_DEGREES
    maxy = np.ceil(maxy / CELL_SIZE_DEGREES) * CELL_SIZE_DEGREES
    
    # Calcular dimensiones del grid
    width = int((maxx - minx) / CELL_SIZE_DEGREES)
    height = int((maxy - miny) / CELL_SIZE_DEGREES)
    
    # Crear transformación
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    print(f"  ✓ Extensión común calculada:")
    print(f"    Min X (Lon): {minx:.6f}")
    print(f"    Max X (Lon): {maxx:.6f}")
    print(f"    Min Y (Lat): {miny:.6f}")
    print(f"    Max Y (Lat): {maxy:.6f}")
    print(f"  ✓ Dimensiones del grid: {width} x {height} pixels")
    
    return (minx, miny, maxx, maxy), width, height, transform


def reproject_and_resample_raster(input_path, output_path, bounds, width, height, transform):
    """
    Reproyecta y remuestrea un raster al grid estándar
    """
    with rasterio.open(input_path) as src:
        # Leer datos
        data = src.read(1)
        
        # Crear raster de salida
        kwargs = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.float32,
            'crs': TARGET_CRS,
            'transform': transform,
            'nodata': np.nan,
            'compress': 'lzw'
        }
        
        # Crear array de salida
        output_array = np.full((height, width), np.nan, dtype=np.float32)
        
        # Reproyectar
        reproject(
            source=data,
            destination=output_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )
        
        # Guardar
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(output_array, 1)
    
    print(f"  ✓ Raster reproyectado: {os.path.basename(output_path)}")


def rasterize_favorable_lithology(bounds, width, height, transform):
    """
    Rasteriza litologías favorables (FAVORABLE = 1)
    """
    print("\n[PASO 3] Rasterizando litologías favorables...")
    
    # Leer shapefile
    gdf = gpd.read_file(os.path.join(INPUT_DIR, LITOLOGIAS_SHP))
    gdf = gdf.to_crs(TARGET_CRS)
    
    # Filtrar solo favorables
    gdf_favorable = gdf[gdf['FAVORABLE'] == 1].copy()
    
    print(f"  - Total de polígonos: {len(gdf)}")
    print(f"  - Polígonos favorables: {len(gdf_favorable)}")
    
    if len(gdf_favorable) == 0:
        print("  ⚠ ADVERTENCIA: No se encontraron litologías favorables!")
        return
    
    # Preparar geometrías para rasterización
    shapes = [(geom, 1) for geom in gdf_favorable.geometry]
    
    # Rasterizar
    output_path = os.path.join(OUTPUT_DIR, "lithology_favorable.tif")
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=TARGET_CRS,
        transform=transform,
        nodata=0,
        compress='lzw'
    ) as dst:
        burned = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )
        dst.write(burned, 1)
    
    print(f"  ✓ Litologías favorables rasterizadas: lithology_favorable.tif")


def create_lithology_contact_buffers(bounds, width, height, transform):
    """
    Crea buffers múltiples alrededor de contactos de litologías FAVORABLES
    """
    print("\n[PASO 4] Creando buffers de contactos litológicos...")
    
    # Leer shapefile
    gdf = gpd.read_file(os.path.join(INPUT_DIR, LITOLOGIAS_SHP))
    
    # Reproyectar a métrico para buffers (usamos UTM)
    # Detectar zona UTM automáticamente
    centroid = gdf.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    
    if hemisphere == 'north':
        utm_crs = f"EPSG:326{utm_zone:02d}"
    else:
        utm_crs = f"EPSG:327{utm_zone:02d}"
    
    print(f"  - Reproyectando a {utm_crs} para buffers precisos")
    
    gdf_utm = gdf.to_crs(utm_crs)
    
    # Filtrar solo favorables
    gdf_favorable = gdf_utm[gdf_utm['FAVORABLE'] == 1].copy()
    
    # Extraer bordes (contactos)
    boundaries = gdf_favorable.boundary
    
    # Crear multi-buffer
    for i, distance in enumerate(CONTACT_BUFFERS_METERS, 1):
        print(f"  - Creando buffer de {distance}m...")
        
        # Buffer
        buffered = boundaries.buffer(distance)
        
        # Convertir de vuelta a WGS84
        buffered_gdf = gpd.GeoDataFrame(
            {'geometry': buffered, 'value': i},
            crs=utm_crs
        ).to_crs(TARGET_CRS)
        
        # Rasterizar
        output_path = os.path.join(OUTPUT_DIR, f"contacts_buffer_{distance}m.tif")
        
        shapes = [(geom, val) for geom, val in zip(buffered_gdf.geometry, buffered_gdf['value'])]
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint8,
            crs=TARGET_CRS,
            transform=transform,
            nodata=0,
            compress='lzw'
        ) as dst:
            burned = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=rasterio.uint8
            )
            dst.write(burned, 1)
        
        print(f"    ✓ Buffer {distance}m creado")


def rasterize_faults_with_buffer(bounds, width, height, transform):
    """
    Rasteriza fallas con buffer de 500m (todas las orientaciones)
    """
    print("\n[PASO 5] Rasterizando fallas con buffer...")
    
    # Leer shapefile
    gdf = gpd.read_file(os.path.join(INPUT_DIR, FALLAS_SHP))
    
    print(f"  - Total de fallas: {len(gdf)}")
    
    # Reproyectar a métrico para buffer
    centroid = gdf.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    
    if hemisphere == 'north':
        utm_crs = f"EPSG:326{utm_zone:02d}"
    else:
        utm_crs = f"EPSG:327{utm_zone:02d}"
    
    gdf_utm = gdf.to_crs(utm_crs)
    
    # Aplicar buffer
    print(f"  - Aplicando buffer de {FAULT_BUFFER_METERS}m...")
    gdf_buffered = gdf_utm.copy()
    gdf_buffered['geometry'] = gdf_buffered.buffer(FAULT_BUFFER_METERS)
    
    # Convertir de vuelta a WGS84
    gdf_buffered = gdf_buffered.to_crs(TARGET_CRS)
    gdf_buffered['value'] = 1
    
    # Rasterizar
    output_path = os.path.join(OUTPUT_DIR, "faults_buffer_500m.tif")
    
    shapes = [(geom, 1) for geom in gdf_buffered.geometry]
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=TARGET_CRS,
        transform=transform,
        nodata=0,
        compress='lzw'
    ) as dst:
        burned = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )
        dst.write(burned, 1)
    
    print(f"  ✓ Fallas rasterizadas: faults_buffer_500m.tif")


def verify_rasters():
    """
    Verifica que todos los rasters tengan el mismo extent y resolución
    """
    print("\n[VERIFICACIÓN] Comprobando consistencia de rasters...")
    
    raster_files = [
        "PC1_resampled.tif",
        "PC2_resampled.tif",
        "lithology_favorable.tif",
        "faults_buffer_500m.tif"
    ] + [f"contacts_buffer_{d}m.tif" for d in CONTACT_BUFFERS_METERS]
    
    reference = None
    all_ok = True
    
    for raster_file in raster_files:
        path = os.path.join(OUTPUT_DIR, raster_file)
        
        if not os.path.exists(path):
            print(f"  ✗ {raster_file}: NO ENCONTRADO")
            all_ok = False
            continue
        
        with rasterio.open(path) as src:
            if reference is None:
                reference = {
                    'width': src.width,
                    'height': src.height,
                    'bounds': src.bounds,
                    'crs': src.crs,
                    'transform': src.transform
                }
                print(f"  ✓ {raster_file}: REFERENCIA")
            else:
                # Comparar
                if (src.width == reference['width'] and
                    src.height == reference['height'] and
                    src.bounds == reference['bounds'] and
                    src.crs == reference['crs']):
                    print(f"  ✓ {raster_file}: OK")
                else:
                    print(f"  ✗ {raster_file}: INCONSISTENTE")
                    all_ok = False
    
    if all_ok:
        print("\n  ✓✓✓ TODOS LOS RASTERS SON CONSISTENTES ✓✓✓")
    else:
        print("\n  ✗✗✗ HAY INCONSISTENCIAS - REVISAR ✗✗✗")
    
    return all_ok


def create_summary_report():
    """
    Crea un reporte resumen de los rasters generados
    """
    print("\n" + "="*80)
    print("RESUMEN DE RASTERS GENERADOS")
    print("="*80)
    
    raster_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.tif')]
    
    print(f"\nTotal de rasters generados: {len(raster_files)}\n")
    
    for raster_file in sorted(raster_files):
        path = os.path.join(OUTPUT_DIR, raster_file)
        
        with rasterio.open(path) as src:
            data = src.read(1)
            
            # Estadísticas
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) > 0:
                print(f"📄 {raster_file}")
                print(f"   Dimensiones: {src.width} x {src.height}")
                print(f"   CRS: {src.crs}")
                print(f"   Valores únicos: {len(np.unique(valid_data))}")
                print(f"   Rango: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
                print(f"   Pixels válidos: {len(valid_data):,} ({len(valid_data)/(src.width*src.height)*100:.1f}%)")
                print()


# ============================================================================
# WORKFLOW PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta el workflow completo de rasterización
    """
    print("\nIniciando procesamiento...\n")
    
    try:
        # Paso 1: Calcular extensión común y grid
        bounds, width, height, transform = get_common_extent_and_grid()
        
        # Paso 2: Reproyectar rasters geoquímicos
        print("\n[PASO 2] Reproyectando y remuestreando rasters geoquímicos...")
        
        reproject_and_resample_raster(
            os.path.join(INPUT_DIR, PC1_RASTER),
            os.path.join(OUTPUT_DIR, "PC1_resampled.tif"),
            bounds, width, height, transform
        )
        
        reproject_and_resample_raster(
            os.path.join(INPUT_DIR, PC2_RASTER),
            os.path.join(OUTPUT_DIR, "PC2_resampled.tif"),
            bounds, width, height, transform
        )
        
        # Paso 3: Rasterizar litologías favorables
        rasterize_favorable_lithology(bounds, width, height, transform)
        
        # Paso 4: Crear buffers de contactos
        create_lithology_contact_buffers(bounds, width, height, transform)
        
        # Paso 5: Rasterizar fallas
        rasterize_faults_with_buffer(bounds, width, height, transform)
        
        # Verificación
        verification_ok = verify_rasters()
        
        # Reporte final
        create_summary_report()
        
        print("\n" + "="*80)
        if verification_ok:
            print("✓✓✓ RASTERIZACIÓN COMPLETADA EXITOSAMENTE ✓✓✓")
        else:
            print("⚠⚠⚠ RASTERIZACIÓN COMPLETADA CON ADVERTENCIAS ⚠⚠⚠")
        print("="*80)
        
        print(f"\nTodos los rasters están en: {OUTPUT_DIR}")
        print("\n✓ Puedes proceder con el análisis de Machine Learning")
        
    except Exception as e:
        print(f"\n✗✗✗ ERROR EN EL PROCESAMIENTO ✗✗✗")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()