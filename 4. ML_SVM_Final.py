# -*- coding: utf-8 -*-
"""
================================================================================
MINERAL PROSPECTIVITY MAPPING USING SUPPORT VECTOR MACHINE (SVM)
================================================================================
Based on: Sun et al. (2019) - "GIS-based mineral prospectivity mapping using 
machine learning methods: A case study from Tongling ore district, eastern China"
Ore Geology Reviews 109 (2019) 26-49

Author: Adapted for thesis project
Purpose: Generate mineral prospectivity maps for Au-Ag exploration using SVM

Changes in this version:
- All processing limited to study area polygon
- Lithological contacts only between favorable lithologies (FAVORABLE=1)
- Excludes contacts between favorable and non-favorable lithologies
- Excludes contacts with study area boundary
================================================================================
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import convolve
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from shapely.ops import unary_union, linemerge
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, StratifiedKFold, 
                                      train_test_split, GridSearchCV)
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, accuracy_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS AND PARAMETERS
# ============================================================================

class Config:
    """Configuration class for all parameters"""
    
    # Input paths
    INPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\Shapefiles"
    OUTPUT_DIR = r"C:\Users\jhefe\Desktop\Jheff\Estudios\San Marcos\Tesis\Proyecto Tesis_V3\Información\Machine Learning\borrar"
    
    # Input files
    FAULTS_SHP = os.path.join(INPUT_DIR, "fallas.shp")
    LITHOLOGY_SHP = os.path.join(INPUT_DIR, "litologias.shp")
    DEPOSITS_SHP = os.path.join(INPUT_DIR, "yacimientos.shp")
    STUDY_AREA_SHP = os.path.join(INPUT_DIR, "area.shp")  # Study area shapefile
    PC1_RASTER = os.path.join(INPUT_DIR, "PC1.tif")
    PC2_RASTER = os.path.join(INPUT_DIR, "PC2.tif")
    W_RASTER = os.path.join(INPUT_DIR, "W.tif")
    
    # Output CRS and resolution
    # UTM Zone 18S (EPSG:32718) for Peru/Arequipa region
    OUTPUT_CRS = "EPSG:32718"
    CELL_SIZE = 500  # meters - Adjusted for 30,000 km² study area
    
    # Column names in shapefiles
    FAVORABLE_COLUMN = "FAVORABLE"
    
    # SVM parameters based on Sun et al. (2019)
    # Paper tested: gamma 0.1-1, cost 0.1-50
    # RBF kernel was chosen for its low errors and simplicity
    SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1,
    'gamma': 0.1,
    'probability': True,
    'random_state': 42,
    'class_weight': 'balanced'
    }
    
    # Parameter grid for hyperparameter tuning (based on paper Table 3)
    SVM_PARAM_GRID = {
        'C': [0.1, 0.5, 1, 5, 10, 25, 50],
        'gamma': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    }
    
    # Cross-validation folds (paper used 10-fold)
    CV_FOLDS = 10
    
    # Buffer distance for non-deposit selection (meters)
    # Adjusted for 10km typical distance between deposits
    NON_DEPOSIT_BUFFER = 5000
    
    # Validation split
    TEST_SIZE = 0.33  # 1/3 for testing as in the paper
    
    # Search radius for fault density (meters)
    # Based on 2500m mineralization distance to faults
    FAULT_SEARCH_RADIUS = 2500
    
    # Search radius for lithological contacts (meters)
    # Contacts less important, 500m radius
    CONTACT_SEARCH_RADIUS = 500


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        print(f"Created output directory: {Config.OUTPUT_DIR}")
    return Config.OUTPUT_DIR


def load_study_area():
    """Load and project study area shapefile"""
    print(f"Loading study area: {Config.STUDY_AREA_SHP}")
    study_area_gdf = gpd.read_file(Config.STUDY_AREA_SHP)
    study_area_gdf = study_area_gdf.to_crs(Config.OUTPUT_CRS)
    
    # Get the unified geometry (in case there are multiple polygons)
    study_area_geom = unary_union(study_area_gdf.geometry)
    
    print(f"  - Study area loaded successfully")
    print(f"  - Area: {study_area_geom.area / 1e6:.2f} km²")
    
    return study_area_gdf, study_area_geom


def get_study_area_bounds(study_area_gdf):
    """Get bounds from study area shapefile"""
    bounds = study_area_gdf.total_bounds
    return list(bounds)


def calculate_grid_dimensions(bounds, cell_size):
    """Calculate grid dimensions from bounds and cell size"""
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    
    maxx_adj = minx + width * cell_size
    maxy_adj = miny + height * cell_size
    
    return width, height, (minx, miny, maxx_adj, maxy_adj)


def create_transform(bounds, width, height):
    """Create affine transform for rasterization"""
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return transform


def create_study_area_mask(study_area_gdf, bounds, width, height, transform):
    """
    Create binary mask of the study area
    Pixels inside study area = 1, outside = 0
    """
    shapes = [(geom, 1) for geom in study_area_gdf.geometry if geom is not None]
    
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    return mask.astype(bool)


def apply_study_area_mask(raster, mask):
    """Apply study area mask to raster - set values outside to NaN"""
    result = raster.copy().astype(np.float32)
    result[~mask] = np.nan
    return result


# ============================================================================
# RASTERIZATION FUNCTIONS
# ============================================================================

def rasterize_polygon_to_binary(gdf, bounds, width, height, transform, 
                                 attribute=None, value=1, mask=None):
    """Rasterize polygon shapefile to binary raster"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    if attribute is not None:
        gdf_proj = gdf_proj[gdf_proj[attribute] == value]
    
    if len(gdf_proj) == 0:
        print(f"Warning: No features found with {attribute}={value}")
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    
    raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    # Apply study area mask
    if mask is not None:
        raster = apply_study_area_mask(raster, mask)
    
    return raster


def rasterize_lines_to_density(gdf, bounds, width, height, transform, 
                                search_radius=2500, mask=None):
    """Calculate line density using kernel density estimation"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    
    if not shapes:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    
    line_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    kernel_size = int(search_radius / Config.CELL_SIZE)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3
    
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = (x**2 + y**2 <= (kernel_size//2)**2).astype(float)
    kernel /= kernel.sum()
    
    density = convolve(line_raster.astype(float), kernel, mode='constant')
    
    # Apply study area mask
    if mask is not None:
        density = apply_study_area_mask(density, mask)
    
    return density


def calculate_euclidean_distance(binary_raster, cell_size):
    """Calculate Euclidean distance from features in meters"""
    # Handle NaN values for distance calculation
    binary_clean = np.nan_to_num(binary_raster, nan=0)
    inverted = 1 - binary_clean
    distance = distance_transform_edt(inverted) * cell_size
    return distance


def calculate_proximity_to_features(gdf, bounds, width, height, transform, 
                                     cell_size, mask=None):
    """Calculate proximity (distance) raster from vector features"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    
    if not shapes:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    
    feature_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    distance = calculate_euclidean_distance(feature_raster, cell_size)
    
    # Apply study area mask
    if mask is not None:
        distance = apply_study_area_mask(distance, mask)
    
    return distance


def extract_fault_intersections(gdf, study_area_geom=None):
    """Extract fault intersection points from line geometries within study area"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    # Clip faults to study area if provided
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
                        # Check if inside study area
                        if study_area_geom is None or study_area_geom.contains(intersection):
                            intersections.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            if study_area_geom is None or study_area_geom.contains(point):
                                intersections.append(point)
    
    if intersections:
        intersection_gdf = gpd.GeoDataFrame(geometry=intersections, crs=Config.OUTPUT_CRS)
        return intersection_gdf
    else:
        return None


def extract_favorable_lithology_contacts(lithology_gdf, study_area_geom=None):
    """
    Extract contacts ONLY between favorable lithologies (FAVORABLE=1)
    
    This function identifies shared boundaries between polygons where
    BOTH polygons have FAVORABLE=1. It excludes:
    - Contacts between favorable and non-favorable lithologies
    - Contacts with the study area boundary
    - External boundaries of favorable polygons
    
    Returns:
        GeoDataFrame with contact lines between favorable lithologies
    """
    gdf_proj = lithology_gdf.to_crs(Config.OUTPUT_CRS)
    
    # Filter only favorable lithologies
    favorable_gdf = gdf_proj[gdf_proj[Config.FAVORABLE_COLUMN] == 1].copy()
    favorable_gdf = favorable_gdf.reset_index(drop=True)
    
    if len(favorable_gdf) < 2:
        print("    Warning: Less than 2 favorable lithology polygons found")
        return None
    
    print(f"    - Number of favorable lithology polygons: {len(favorable_gdf)}")
    
    # Find shared boundaries between favorable polygons
    contact_lines = []
    
    for i in range(len(favorable_gdf)):
        geom_i = favorable_gdf.geometry.iloc[i]
        
        for j in range(i + 1, len(favorable_gdf)):
            geom_j = favorable_gdf.geometry.iloc[j]
            
            # Check if polygons touch or overlap
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                # Get the shared boundary
                intersection = geom_i.intersection(geom_j)
                
                # We only want linear intersections (shared edges), not points or polygons
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
        print("    Warning: No contacts found between favorable lithologies")
        return None
    
    # Create GeoDataFrame
    contacts_gdf = gpd.GeoDataFrame(geometry=contact_lines, crs=Config.OUTPUT_CRS)
    
    # Clip to study area if provided
    if study_area_geom is not None:
        contacts_gdf['geometry'] = contacts_gdf.geometry.intersection(study_area_geom)
        contacts_gdf = contacts_gdf[~contacts_gdf.geometry.is_empty]
    
    total_length = contacts_gdf.geometry.length.sum()
    print(f"    - Number of contact segments: {len(contacts_gdf)}")
    print(f"    - Total contact length: {total_length/1000:.2f} km")
    
    return contacts_gdf


def calculate_favorable_contacts_density(lithology_gdf, bounds, width, height, 
                                          transform, cell_size, study_area_geom=None, 
                                          mask=None):
    """
    Calculate density of contacts between favorable lithologies only
    """
    # Extract contacts between favorable lithologies
    contacts_gdf = extract_favorable_lithology_contacts(lithology_gdf, study_area_geom)
    
    if contacts_gdf is None or len(contacts_gdf) == 0:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    
    # Rasterize contact lines
    shapes = [(geom, 1) for geom in contacts_gdf.geometry if geom is not None and not geom.is_empty]
    
    if not shapes:
        result = np.zeros((height, width), dtype=np.float32)
        if mask is not None:
            result[~mask] = np.nan
        return result
    
    contact_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    # Apply kernel density
    kernel_size = int(Config.CONTACT_SEARCH_RADIUS / cell_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3
    
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = (x**2 + y**2 <= (kernel_size//2)**2).astype(float)
    kernel /= kernel.sum()
    
    density = convolve(contact_raster.astype(float), kernel, mode='constant')
    
    # Apply study area mask
    if mask is not None:
        density = apply_study_area_mask(density, mask)
    
    return density


# ============================================================================
# RASTER PROCESSING FUNCTIONS
# ============================================================================

def resample_raster_to_grid(input_path, bounds, width, height, transform, 
                             output_crs, mask=None):
    """Resample existing raster to match target grid"""
    with rasterio.open(input_path) as src:
        dst_array = np.zeros((height, width), dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=output_crs,
            resampling=Resampling.bilinear
        )
        
        if src.nodata is not None:
            dst_array[dst_array == src.nodata] = np.nan
        
        # Apply study area mask
        if mask is not None:
            dst_array = apply_study_area_mask(dst_array, mask)
        
        return dst_array


def normalize_raster(raster, method='minmax'):
    """Normalize raster values (only within valid/masked area)"""
    valid_mask = ~np.isnan(raster) & ~np.isinf(raster)
    
    if not valid_mask.any():
        return raster
    
    result = raster.copy()
    
    if method == 'minmax':
        min_val = np.nanmin(raster[valid_mask])
        max_val = np.nanmax(raster[valid_mask])
        if max_val > min_val:
            result[valid_mask] = (raster[valid_mask] - min_val) / (max_val - min_val)
        else:
            result[valid_mask] = 0
    
    elif method == 'zscore':
        mean_val = np.nanmean(raster[valid_mask])
        std_val = np.nanstd(raster[valid_mask])
        if std_val > 0:
            result[valid_mask] = (raster[valid_mask] - mean_val) / std_val
        else:
            result[valid_mask] = 0
    
    return result


def save_raster(array, output_path, transform, crs, nodata=-9999):
    """Save numpy array as GeoTIFF"""
    array_out = array.copy()
    array_out[np.isnan(array_out)] = nodata
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array_out.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(array_out, 1)
    
    print(f"Saved: {output_path}")


def load_raster(input_path):
    """Load raster from GeoTIFF"""
    with rasterio.open(input_path) as src:
        array = src.read(1).astype(np.float32)
        if src.nodata is not None:
            array[array == src.nodata] = np.nan
        return array, src.transform, src.crs


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def generate_training_data(deposits_gdf, predictor_stack, transform, bounds, 
                           width, height, study_area_mask, n_non_deposits=None, 
                           buffer_distance=5000):
    """
    Generate training dataset following Sun et al. (2019) methodology
    All samples are restricted to the study area mask
    """
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    
    deposit_rows = []
    deposit_cols = []
    
    minx, miny, maxx, maxy = bounds
    
    for point in deposits_proj.geometry:
        if point is not None:
            col = int((point.x - minx) / Config.CELL_SIZE)
            row = int((maxy - point.y) / Config.CELL_SIZE)
            
            # Only include deposits within study area
            if 0 <= row < height and 0 <= col < width:
                if study_area_mask[row, col]:  # Check if inside study area
                    deposit_rows.append(row)
                    deposit_cols.append(col)
    
    n_deposits = len(deposit_rows)
    print(f"Number of deposit locations within study area: {n_deposits}")
    
    if n_non_deposits is None:
        n_non_deposits = n_deposits
    
    # Create deposit mask
    deposit_mask = np.zeros((height, width), dtype=bool)
    for r, c in zip(deposit_rows, deposit_cols):
        deposit_mask[r, c] = True
    
    # Calculate distance from deposits
    distance_from_deposits = distance_transform_edt(~deposit_mask) * Config.CELL_SIZE
    
    # Valid mask: inside study area AND beyond buffer distance AND all predictors valid
    valid_mask = study_area_mask.copy()
    valid_mask &= (distance_from_deposits > buffer_distance)
    
    for predictor in predictor_stack:
        valid_mask &= ~np.isnan(predictor) & ~np.isinf(predictor)
    
    valid_rows, valid_cols = np.where(valid_mask)
    
    if len(valid_rows) < n_non_deposits:
        print(f"Warning: Only {len(valid_rows)} valid non-deposit locations available")
        n_non_deposits = len(valid_rows)
    
    np.random.seed(42)
    indices = np.random.choice(len(valid_rows), size=n_non_deposits, replace=False)
    non_deposit_rows = valid_rows[indices]
    non_deposit_cols = valid_cols[indices]
    
    print(f"Number of non-deposit locations: {n_non_deposits}")
    
    # Extract predictor values
    X_deposits = np.column_stack([
        predictor[deposit_rows, deposit_cols] for predictor in predictor_stack
    ])
    
    X_non_deposits = np.column_stack([
        predictor[non_deposit_rows, non_deposit_cols] for predictor in predictor_stack
    ])
    
    X = np.vstack([X_deposits, X_non_deposits])
    y = np.array([1] * n_deposits + [0] * n_non_deposits)
    
    locations = {
        'deposit_rows': deposit_rows,
        'deposit_cols': deposit_cols,
        'non_deposit_rows': non_deposit_rows.tolist(),
        'non_deposit_cols': non_deposit_cols.tolist()
    }
    
    return X, y, locations


# ============================================================================
# SVM MODEL FUNCTIONS
# ============================================================================

def optimize_svm_hyperparameters(X, y, param_grid, cv_folds=10):
    """
    Optimize SVM hyperparameters using GridSearchCV
    Based on Sun et al. (2019) methodology - Figure 12c shows MSE variation with gamma and cost
    """
    print("\n  Performing hyperparameter optimization...")
    print(f"  Testing C values: {param_grid['C']}")
    print(f"  Testing gamma values: {param_grid['gamma']}")
    
    svm_base = SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    
    grid_search = GridSearchCV(
        svm_base,
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search


def train_svm(X, y, params=None):
    """Train SVM classifier with specified parameters"""
    if params is None:
        params = Config.SVM_PARAMS.copy()
    
    # Merge with base params
    svm_params = {
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42,
        'class_weight': 'balanced'
    }
    svm_params.update(params)
    
    svm_model = SVC(**svm_params)
    svm_model.fit(X, y)
    
    return svm_model


def perform_cross_validation(X, y, params=None, n_folds=10):
    """Perform k-fold cross-validation following Sun et al. (2019)"""
    if params is None:
        params = {}
    
    svm_params = {
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42,
        'class_weight': 'balanced'
    }
    svm_params.update(params)
    
    svm_model = SVC(**svm_params)
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm_model, X, y, cv=cv, scoring='accuracy')
    cv_scores_auc = cross_val_score(svm_model, X, y, cv=cv, scoring='roc_auc')
    
    mse_scores = 1 - cv_scores
    
    results = {
        'cv_scores': cv_scores,
        'cv_scores_auc': cv_scores_auc,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'mean_auc': cv_scores_auc.mean(),
        'std_auc': cv_scores_auc.std(),
        'mean_mse': mse_scores.mean(),
        'std_mse': mse_scores.std()
    }
    
    return results


def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation following Sun et al. (2019)"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    kappa = cohen_kappa_score(y_test, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results = {
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'positive_predictive_value': ppv,
        'negative_predictive_value': npv,
        'accuracy': accuracy,
        'kappa': kappa,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return results


def get_feature_importance_svm(model, X, y, feature_names):
    """
    Estimate feature importance for SVM using permutation importance
    Since SVM doesn't have built-in feature importance like RF
    """
    from sklearn.inspection import permutation_importance
    
    perm_importance = permutation_importance(
        model, X, y, n_repeats=30, random_state=7, n_jobs=-1
    )
    
    importances = perm_importance.importances_mean
    indices = np.argsort(importances)[::-1]
    
    # Normalize to percentages
    importances_normalized = importances / importances.sum() * 100
    
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Importance_Percent': importances_normalized[indices]
    })
    
    return importance_df


# ============================================================================
# PROSPECTIVITY MAP GENERATION
# ============================================================================

def generate_prospectivity_map(model, predictor_stack, width, height, 
                                study_area_mask, scaler=None):
    """Generate prospectivity map by applying trained model to entire study area"""
    n_predictors = len(predictor_stack)
    
    # Valid mask: inside study area and all predictors valid
    valid_mask = study_area_mask.copy()
    for predictor in predictor_stack:
        valid_mask &= ~np.isnan(predictor) & ~np.isinf(predictor)
    
    valid_rows, valid_cols = np.where(valid_mask)
    
    X_predict = np.column_stack([
        predictor[valid_rows, valid_cols] for predictor in predictor_stack
    ])
    
    # Apply scaling if provided
    if scaler is not None:
        X_predict = scaler.transform(X_predict)
    
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    prospectivity = np.full((height, width), np.nan, dtype=np.float32)
    prospectivity[valid_rows, valid_cols] = probabilities
    
    return prospectivity


def classify_prospectivity(prospectivity, success_rate_thresholds=None):
    """Classify prospectivity map into potential zones"""
    if success_rate_thresholds is None:
        success_rate_thresholds = {
            'very_high': 0.75,
            'high': 0.50,
            'moderate': 0.25
        }
    
    classified = np.full_like(prospectivity, np.nan)
    valid_mask = ~np.isnan(prospectivity)
    
    classified[valid_mask & (prospectivity >= success_rate_thresholds['very_high'])] = 4
    classified[valid_mask & (prospectivity >= success_rate_thresholds['high']) & 
               (prospectivity < success_rate_thresholds['very_high'])] = 3
    classified[valid_mask & (prospectivity >= success_rate_thresholds['moderate']) & 
               (prospectivity < success_rate_thresholds['high'])] = 2
    classified[valid_mask & (prospectivity < success_rate_thresholds['moderate'])] = 1
    
    return classified


def calculate_success_rate_curve(prospectivity, deposit_rows, deposit_cols):
    """Calculate success-rate curve as in Sun et al. (2019)"""
    valid_mask = ~np.isnan(prospectivity)
    
    thresholds = np.linspace(1.0, 0.0, 100)
    
    total_area = valid_mask.sum()
    total_deposits = len(deposit_rows)
    
    area_percentages = []
    deposit_percentages = []
    
    for threshold in thresholds:
        prospective_mask = valid_mask & (prospectivity >= threshold)
        area_count = prospective_mask.sum()
        
        deposit_count = sum(
            1 for r, c in zip(deposit_rows, deposit_cols)
            if 0 <= r < prospectivity.shape[0] and 0 <= c < prospectivity.shape[1]
            and prospective_mask[r, c]
        )
        
        area_percentages.append(area_count / total_area * 100)
        deposit_percentages.append(deposit_count / total_deposits * 100 if total_deposits > 0 else 0)
    
    return np.array(area_percentages), np.array(deposit_percentages), thresholds


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_roc_curve(results, output_path=None):
    """Plot ROC curve with AUC value"""
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr'], results['tpr'], 'g-', linewidth=2, 
             label=f'SVM (AUC = {results["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve: {output_path}")
    plt.close()


def plot_success_rate_curve(area_pct, deposit_pct, thresholds, output_path=None):
    """Plot success-rate curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(area_pct, deposit_pct, 'g-', linewidth=2, label='SVM')
    plt.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Random Model')
    
    plt.xlabel('Percentage of Cumulative Area (%)', fontsize=12)
    plt.ylabel('Percentage of Captured Deposits (%)', fontsize=12)
    plt.title('Success-Rate Curve - SVM', fontsize=14)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved success-rate curve: {output_path}")
    plt.close()


def plot_feature_importance(importance_df, output_path=None):
    """Plot feature importance bar chart"""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(importance_df)))[::-1]
    
    bars = plt.barh(importance_df['Feature'], importance_df['Importance_Percent'], 
                    color=colors, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Contribution (%)', fontsize=12)
    plt.ylabel('Predictor Variable', fontsize=12)
    plt.title('Feature Importance - SVM Model (Permutation Importance)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot: {output_path}")
    plt.close()


def plot_confusion_matrix(cm, output_path=None):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    
    labels = ['Non-Deposit', 'Deposit']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14})
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - SVM', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {output_path}")
    plt.close()


def plot_prospectivity_map(prospectivity, deposits_gdf, bounds, study_area_gdf,
                           output_path=None):
    """Plot prospectivity map with deposits overlay"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('prospectivity', colors, N=256)
    
    minx, miny, maxx, maxy = bounds
    extent = [minx, maxx, miny, maxy]
    
    im = ax.imshow(prospectivity, extent=extent, cmap=cmap, 
                   vmin=0, vmax=1, origin='upper')
    
    # Plot study area boundary
    study_area_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, 
                                  linestyle='--', label='Study Area')
    
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    ax.scatter(deposits_proj.geometry.x, deposits_proj.geometry.y,
               c='black', s=50, marker='^', edgecolors='white', 
               linewidths=0.5, label='Known Deposits', zorder=5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Prospectivity (Probability)', fontsize=12)
    
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Mineral Prospectivity Map - SVM Model', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved prospectivity map: {output_path}")
    plt.close()


def plot_classified_map(classified, deposits_gdf, bounds, study_area_gdf,
                        output_path=None):
    """Plot classified prospectivity map"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('classified', colors, N=4)
    
    minx, miny, maxx, maxy = bounds
    extent = [minx, maxx, miny, maxy]
    
    im = ax.imshow(classified, extent=extent, cmap=cmap, 
                   vmin=1, vmax=4, origin='upper')
    
    # Plot study area boundary
    study_area_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, 
                                  linestyle='--', label='Study Area')
    
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    ax.scatter(deposits_proj.geometry.x, deposits_proj.geometry.y,
               c='black', s=50, marker='^', edgecolors='white', 
               linewidths=0.5, label='Known Deposits', zorder=5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[1.375, 2.125, 2.875, 3.625])
    cbar.ax.set_yticklabels(['Low', 'Moderate', 'High', 'Very High'])
    cbar.set_label('Potential Classification', fontsize=12)
    
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Mineral Prospectivity Classification - SVM Model', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved classified map: {output_path}")
    plt.close()


def plot_hyperparameter_heatmap(grid_search, output_path=None):
    """
    Plot heatmap of hyperparameter optimization results
    Similar to Figure 12c in Sun et al. (2019)
    """
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Extract C and gamma values
    c_values = Config.SVM_PARAM_GRID['C']
    gamma_values = Config.SVM_PARAM_GRID['gamma']
    
    # Create MSE matrix (1 - accuracy)
    mse_matrix = np.zeros((len(gamma_values), len(c_values)))
    
    for idx, row in results.iterrows():
        c_idx = c_values.index(row['param_C'])
        gamma_idx = gamma_values.index(row['param_gamma'])
        mse_matrix[gamma_idx, c_idx] = 1 - row['mean_test_score']
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(mse_matrix, 
                xticklabels=c_values, 
                yticklabels=gamma_values,
                cmap='RdYlGn_r',
                annot=True, 
                fmt='.3f',
                cbar_kws={'label': 'MSE (1 - Accuracy)'})
    
    plt.xlabel('Cost (C)', fontsize=12)
    plt.ylabel('Gamma (γ)', fontsize=12)
    plt.title('SVM Hyperparameter Optimization - MSE Heatmap', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved hyperparameter heatmap: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for SVM mineral prospectivity mapping"""
    print("="*80)
    print("MINERAL PROSPECTIVITY MAPPING USING SUPPORT VECTOR MACHINE (SVM)")
    print("Based on Sun et al. (2019) Methodology")
    print("="*80)
    
    output_dir = create_output_directory()
    
    # -------------------------------------------------------------------------
    # STEP 1: Load study area and input data
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 1: Loading input data...")
    print("-"*60)
    
    # Load study area first
    study_area_gdf, study_area_geom = load_study_area()
    
    print(f"\nLoading faults: {Config.FAULTS_SHP}")
    faults_gdf = gpd.read_file(Config.FAULTS_SHP)
    print(f"  - {len(faults_gdf)} fault features loaded")
    
    print(f"Loading lithologies: {Config.LITHOLOGY_SHP}")
    lithology_gdf = gpd.read_file(Config.LITHOLOGY_SHP)
    print(f"  - {len(lithology_gdf)} lithology features loaded")
    n_favorable = len(lithology_gdf[lithology_gdf[Config.FAVORABLE_COLUMN] == 1])
    print(f"  - {n_favorable} favorable lithology polygons (FAVORABLE=1)")
    
    print(f"Loading deposits: {Config.DEPOSITS_SHP}")
    deposits_gdf = gpd.read_file(Config.DEPOSITS_SHP)
    print(f"  - {len(deposits_gdf)} deposit locations loaded")
    
    # -------------------------------------------------------------------------
    # STEP 2: Define study area grid
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 2: Defining study area grid...")
    print("-"*60)
    
    bounds = get_study_area_bounds(study_area_gdf)
    width, height, adjusted_bounds = calculate_grid_dimensions(bounds, Config.CELL_SIZE)
    transform = create_transform(adjusted_bounds, width, height)
    
    # Create study area mask
    study_area_mask = create_study_area_mask(study_area_gdf, adjusted_bounds, 
                                              width, height, transform)
    
    print(f"  - CRS: {Config.OUTPUT_CRS}")
    print(f"  - Cell size: {Config.CELL_SIZE} meters")
    print(f"  - Grid dimensions: {width} x {height} pixels")
    print(f"  - Bounds: {adjusted_bounds}")
    print(f"  - Pixels inside study area: {study_area_mask.sum()} ({study_area_mask.sum()/(width*height)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 3: Generate predictor maps (within study area only)
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 3: Generating predictor maps (within study area)...")
    print("-"*60)
    
    predictor_names = []
    predictor_stack = []
    
    # 3.1 Proximity to favorable lithologies
    print("\n  3.1 Processing favorable lithologies...")
    favorable_lith = rasterize_polygon_to_binary(
        lithology_gdf, adjusted_bounds, width, height, transform,
        attribute=Config.FAVORABLE_COLUMN, value=1, mask=study_area_mask
    )
    prox_favorable = calculate_euclidean_distance(favorable_lith, Config.CELL_SIZE)
    prox_favorable = apply_study_area_mask(prox_favorable, study_area_mask)
    prox_favorable_norm = normalize_raster(prox_favorable, method='minmax')
    prox_favorable_norm = 1 - prox_favorable_norm  # Invert: closer = higher value
    prox_favorable_norm = apply_study_area_mask(prox_favorable_norm, study_area_mask)
    
    save_raster(prox_favorable_norm, 
                os.path.join(output_dir, "prox_favorable_lithology.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Proximity to Favorable Lithology")
    predictor_stack.append(prox_favorable_norm)
    
    # 3.2 Fault density (using configured search radius)
    print("\n  3.2 Processing fault density...")
    print(f"      Search radius: {Config.FAULT_SEARCH_RADIUS} m")
    fault_density = rasterize_lines_to_density(
        faults_gdf, adjusted_bounds, width, height, transform, 
        search_radius=Config.FAULT_SEARCH_RADIUS, mask=study_area_mask
    )
    fault_density_norm = normalize_raster(fault_density, method='minmax')
    
    save_raster(fault_density_norm,
                os.path.join(output_dir, "fault_density.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Fault Density")
    predictor_stack.append(fault_density_norm)
    
    # 3.3 Proximity to faults
    print("\n  3.3 Processing proximity to faults...")
    prox_faults = calculate_proximity_to_features(
        faults_gdf, adjusted_bounds, width, height, transform, 
        Config.CELL_SIZE, mask=study_area_mask
    )
    prox_faults_norm = normalize_raster(prox_faults, method='minmax')
    prox_faults_norm = 1 - prox_faults_norm  # Invert
    prox_faults_norm = apply_study_area_mask(prox_faults_norm, study_area_mask)
    
    save_raster(prox_faults_norm,
                os.path.join(output_dir, "prox_faults.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Proximity to Faults")
    predictor_stack.append(prox_faults_norm)
    
    # 3.4 Proximity to fault intersections
    print("\n  3.4 Processing fault intersections...")
    fault_intersections = extract_fault_intersections(faults_gdf, study_area_geom)
    if fault_intersections is not None and len(fault_intersections) > 0:
        print(f"      Number of fault intersections: {len(fault_intersections)}")
        prox_intersections = calculate_proximity_to_features(
            fault_intersections, adjusted_bounds, width, height, transform, 
            Config.CELL_SIZE, mask=study_area_mask
        )
        prox_intersections_norm = normalize_raster(prox_intersections, method='minmax')
        prox_intersections_norm = 1 - prox_intersections_norm  # Invert
        prox_intersections_norm = apply_study_area_mask(prox_intersections_norm, study_area_mask)
        
        save_raster(prox_intersections_norm,
                    os.path.join(output_dir, "prox_fault_intersections.tif"),
                    transform, Config.OUTPUT_CRS)
        predictor_names.append("Proximity to Fault Intersections")
        predictor_stack.append(prox_intersections_norm)
    else:
        print("      Warning: No fault intersections found within study area")
    
    # 3.5 Lithological contacts density (CORRECTED - only between favorable lithologies)
    print("\n  3.5 Processing lithological contacts (favorable-favorable only)...")
    print(f"      Search radius: {Config.CONTACT_SEARCH_RADIUS} m")
    contact_density = calculate_favorable_contacts_density(
        lithology_gdf, adjusted_bounds, width, height, transform, 
        Config.CELL_SIZE, study_area_geom, mask=study_area_mask
    )
    contact_density_norm = normalize_raster(contact_density, method='minmax')
    
    save_raster(contact_density_norm,
                os.path.join(output_dir, "lithological_contacts_density.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Lithological Contacts Density")
    predictor_stack.append(contact_density_norm)
    
    # 3.6 Geochemical PC1
    print("\n  3.6 Processing geochemical PC1...")
    pc1 = resample_raster_to_grid(
        Config.PC1_RASTER, adjusted_bounds, width, height, transform, 
        Config.OUTPUT_CRS, mask=study_area_mask
    )
    pc1_norm = normalize_raster(pc1, method='minmax')
    
    save_raster(pc1_norm,
                os.path.join(output_dir, "geochemical_pc1.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Geochemical PC1")
    predictor_stack.append(pc1_norm)
    
    # 3.7 Geochemical PC2
    print("\n  3.7 Processing geochemical PC2...")
    pc2 = resample_raster_to_grid(
        Config.PC2_RASTER, adjusted_bounds, width, height, transform, 
        Config.OUTPUT_CRS, mask=study_area_mask
    )
    pc2_norm = normalize_raster(pc2, method='minmax')
    
    save_raster(pc2_norm,
                os.path.join(output_dir, "geochemical_pc2.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Geochemical PC2")
    predictor_stack.append(pc2_norm)
    
    # 3.8 Tungsten geochemistry
    print("\n  3.8 Processing Tungsten geochemistry...")
    w_geochem = resample_raster_to_grid(
        Config.W_RASTER, adjusted_bounds, width, height, transform, 
        Config.OUTPUT_CRS, mask=study_area_mask
    )
    w_geochem_norm = normalize_raster(w_geochem, method='minmax')
    
    save_raster(w_geochem_norm,
                os.path.join(output_dir, "geochemical_tungsten.tif"),
                transform, Config.OUTPUT_CRS)
    predictor_names.append("Tungsten Geochemistry")
    predictor_stack.append(w_geochem_norm)
    
    print(f"\n  Total predictor maps: {len(predictor_stack)}")
    for i, name in enumerate(predictor_names):
        print(f"    {i+1}. {name}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Generate training data (within study area only)
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 4: Generating training dataset (within study area)...")
    print("-"*60)
    
    X, y, locations = generate_training_data(
        deposits_gdf, predictor_stack, transform, adjusted_bounds,
        width, height, study_area_mask, buffer_distance=Config.NON_DEPOSIT_BUFFER
    )
    
    print(f"  - Total samples: {len(y)}")
    print(f"  - Deposit samples (class 1): {sum(y == 1)}")
    print(f"  - Non-deposit samples (class 0): {sum(y == 0)}")
    
    # Remove samples with NaN values
    valid_samples = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X = X[valid_samples]
    y = y[valid_samples]
    print(f"  - Valid samples after removing NaN: {len(y)}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=42, stratify=y
    )
    
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Testing samples: {len(y_test)}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Feature scaling (important for SVM)
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 5: Scaling features (StandardScaler)...")
    print("-"*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  - Features standardized (mean=0, std=1)")
    
    # -------------------------------------------------------------------------
    # STEP 6: Hyperparameter optimization
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 6: Optimizing SVM hyperparameters...")
    print("-"*60)
    
    best_params, grid_search = optimize_svm_hyperparameters(
        X_train_scaled, y_train, Config.SVM_PARAM_GRID, cv_folds=Config.CV_FOLDS
    )
    
    # Plot hyperparameter heatmap
    plot_hyperparameter_heatmap(grid_search, 
                                os.path.join(output_dir, "svm_hyperparameter_heatmap.png"))
    
    # -------------------------------------------------------------------------
    # STEP 7: Cross-validation with best parameters
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 7: Performing 10-fold cross-validation with best parameters...")
    print("-"*60)
    
    cv_results = perform_cross_validation(X_train_scaled, y_train, best_params, 
                                          n_folds=Config.CV_FOLDS)
    
    print(f"  - Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  - Mean MSE: {cv_results['mean_mse']:.4f} ± {cv_results['std_mse']:.4f}")
    print(f"  - Individual fold scores: {[f'{s:.3f}' for s in cv_results['cv_scores']]}")
    
    # -------------------------------------------------------------------------
    # STEP 8: Train final SVM model
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 8: Training final SVM model...")
    print("-"*60)
    
    svm_model = train_svm(X_train_scaled, y_train, best_params)
    
    print(f"  - Kernel: {svm_model.kernel}")
    print(f"  - C (Cost): {svm_model.C}")
    print(f"  - Gamma: {svm_model.gamma}")
    print(f"  - Number of support vectors: {svm_model.n_support_.sum()}")
    print(f"  - Support vectors per class: {svm_model.n_support_}")
    
    # -------------------------------------------------------------------------
    # STEP 9: Model evaluation
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 9: Evaluating model performance...")
    print("-"*60)
    
    # Evaluate on training data
    train_results = evaluate_model(svm_model, X_train_scaled, y_train)
    print("\n  Training Dataset Performance:")
    print(f"    - Accuracy: {train_results['accuracy']:.4f}")
    print(f"    - Sensitivity: {train_results['sensitivity']:.4f}")
    print(f"    - Specificity: {train_results['specificity']:.4f}")
    
    # Evaluate on test data
    test_results = evaluate_model(svm_model, X_test_scaled, y_test)
    print("\n  Test Dataset Performance:")
    print(f"    - Accuracy: {test_results['accuracy']:.4f}")
    print(f"    - Sensitivity: {test_results['sensitivity']:.4f}")
    print(f"    - Specificity: {test_results['specificity']:.4f}")
    print(f"    - Positive Predictive Value: {test_results['positive_predictive_value']:.4f}")
    print(f"    - Negative Predictive Value: {test_results['negative_predictive_value']:.4f}")
    print(f"    - Kappa Index: {test_results['kappa']:.4f}")
    print(f"    - AUC: {test_results['auc']:.4f}")
    
    print("\n  Confusion Matrix (Test Data):")
    print(test_results['confusion_matrix'])
    
    # Feature importance (permutation-based for SVM)
    print("\n  Calculating feature importance (permutation-based)...")
    importance_df = get_feature_importance_svm(svm_model, X_test_scaled, y_test, predictor_names)
    print("\n  Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"    - {row['Feature']}: {row['Importance_Percent']:.2f}%")
    
    # -------------------------------------------------------------------------
    # STEP 10: Generate prospectivity map
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 10: Generating prospectivity map...")
    print("-"*60)
    
    prospectivity = generate_prospectivity_map(svm_model, predictor_stack, 
                                                width, height, study_area_mask, scaler)
    
    # Save prospectivity raster
    save_raster(prospectivity,
                os.path.join(output_dir, "prospectivity_svm.tif"),
                transform, Config.OUTPUT_CRS)
    
    valid_prosp = prospectivity[~np.isnan(prospectivity)]
    print(f"  - Prospectivity range: {valid_prosp.min():.4f} to {valid_prosp.max():.4f}")
    print(f"  - Mean prospectivity: {valid_prosp.mean():.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 11: Calculate success-rate curve and classify
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 11: Calculating success-rate curve...")
    print("-"*60)
    
    area_pct, deposit_pct, thresholds = calculate_success_rate_curve(
        prospectivity, locations['deposit_rows'], locations['deposit_cols']
    )
    
    classification_thresholds = {
        'very_high': 0.75,
        'high': 0.50,
        'moderate': 0.25
    }
    
    classified = classify_prospectivity(prospectivity, classification_thresholds)
    
    save_raster(classified,
                os.path.join(output_dir, "prospectivity_svm_classified.tif"),
                transform, Config.OUTPUT_CRS)
    
    # Calculate statistics
    valid_class = classified[~np.isnan(classified)]
    total_pixels = len(valid_class)
    
    total_deposits = len(locations['deposit_rows'])
    category_stats = []
    for class_value, class_name in [(4, 'Very High'), (3, 'High'), (2, 'Moderate'), (1, 'Low')]:
        class_pixels = (valid_class == class_value).sum()
        area_percent = class_pixels / total_pixels * 100
        class_mask = ~np.isnan(classified) & (classified == class_value)
        deps_in_class = sum(
            1 for r, c in zip(locations['deposit_rows'], locations['deposit_cols'])
            if 0 <= r < height and 0 <= c < width and class_mask[r, c]
        )
        dep_pct = deps_in_class / total_deposits * 100 if total_deposits > 0 else 0
        category_stats.append({'Category': class_name, 'Area %': round(area_percent, 2),
                               'Deposits in Category': deps_in_class, 'Deposits Captured %': round(dep_pct, 2)})
        print(f"  - {class_name}: {area_percent:.2f}% area, {deps_in_class} deposits ({dep_pct:.2f}%)")
    
    prospective_mask = ~np.isnan(classified) & (classified >= 3)
    prospective_area_pct = prospective_mask.sum() / (~np.isnan(classified)).sum() * 100
    
    deposits_in_prospective = sum(
        1 for r, c in zip(locations['deposit_rows'], locations['deposit_cols'])
        if 0 <= r < height and 0 <= c < width and prospective_mask[r, c]
    )
    deposits_captured_pct = deposits_in_prospective / len(locations['deposit_rows']) * 100
    
    print(f"\n  Prospective areas (Very High + High):")
    print(f"    - Occupy {prospective_area_pct:.2f}% of study area")
    print(f"    - Capture {deposits_captured_pct:.2f}% of known deposits")
    
    # -------------------------------------------------------------------------
    # STEP 12: Generate visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 12: Generating visualizations...")
    print("-"*60)
    
    plot_roc_curve(test_results, 
                   os.path.join(output_dir, "roc_curve_svm.png"))
    
    plot_success_rate_curve(area_pct, deposit_pct, thresholds,
                           os.path.join(output_dir, "success_rate_curve_svm.png"))
    
    plot_feature_importance(importance_df,
                           os.path.join(output_dir, "feature_importance_svm.png"))
    
    plot_confusion_matrix(test_results['confusion_matrix'],
                         os.path.join(output_dir, "confusion_matrix_svm.png"))
    
    plot_prospectivity_map(prospectivity, deposits_gdf, adjusted_bounds,
                          study_area_gdf,
                          os.path.join(output_dir, "prospectivity_map_svm.png"))
    
    plot_classified_map(classified, deposits_gdf, adjusted_bounds,
                       study_area_gdf,
                       os.path.join(output_dir, "prospectivity_classified_svm.png"))
    
    # -------------------------------------------------------------------------
    # STEP 13: Save results summary
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 13: Saving results summary...")
    print("-"*60)
    
    summary = {
        'Model': 'Support Vector Machine (SVM)',
        'Kernel': svm_model.kernel,
        'C (Cost)': svm_model.C,
        'Gamma': svm_model.gamma,
        'Number of Support Vectors': svm_model.n_support_.sum(),
        'Number of Predictors': len(predictor_names),
        'Predictor Names': str(predictor_names),
        'Cell Size (m)': Config.CELL_SIZE,
        'Fault Search Radius (m)': Config.FAULT_SEARCH_RADIUS,
        'Contact Search Radius (m)': Config.CONTACT_SEARCH_RADIUS,
        'Non-Deposit Buffer (m)': Config.NON_DEPOSIT_BUFFER,
        'Study Area Pixels': int(study_area_mask.sum()),
        'Total Samples': len(y),
        'Training Samples': len(y_train),
        'Test Samples': len(y_test),
        'CV Mean Accuracy': cv_results['mean_accuracy'],
        'CV Std Accuracy': cv_results['std_accuracy'],
        'CV Mean AUC': cv_results['mean_auc'],
        'CV Std AUC': cv_results['std_auc'],
        'Test Accuracy': test_results['accuracy'],
        'Test Sensitivity': test_results['sensitivity'],
        'Test Specificity': test_results['specificity'],
        'Test PPV': test_results['positive_predictive_value'],
        'Test NPV': test_results['negative_predictive_value'],
        'Test Kappa': test_results['kappa'],
        'Test AUC': test_results['auc'],
        'Prospective Area %': prospective_area_pct,
        'Deposits Captured %': deposits_captured_pct
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, "model_summary_svm.csv"), index=False)
    
    # Save category breakdown
    category_df = pd.DataFrame(category_stats)
    category_df.to_csv(os.path.join(output_dir, "category_breakdown_svm.csv"), index=False)
    
    importance_df.to_csv(os.path.join(output_dir, "feature_importance_svm.csv"), index=False)
    
    # Save hyperparameter search results
    hp_results = pd.DataFrame(grid_search.cv_results_)
    hp_results.to_csv(os.path.join(output_dir, "hyperparameter_search_svm.csv"), index=False)
    
    print("\n" + "="*80)
    print("SVM MINERAL PROSPECTIVITY MAPPING COMPLETED!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    
    return svm_model, prospectivity, test_results, scaler


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        model, prospectivity_map, results, scaler = main()
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Please verify that all input files exist in the specified paths.")
        print(f"  - Study area: {Config.STUDY_AREA_SHP}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()