# -*- coding: utf-8 -*-
"""
================================================================================
MINERAL PROSPECTIVITY MAPPING USING ARTIFICIAL NEURAL NETWORKS (ANN)
================================================================================
Based on: Sun et al. (2019) - "GIS-based mineral prospectivity mapping using 
machine learning methods: A case study from Tongling ore district, eastern China"
Ore Geology Reviews 109 (2019) 26-49

Author: Adapted for thesis project
Purpose: Generate mineral prospectivity maps for Au-Ag exploration using ANN
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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.neural_network import MLPClassifier
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
    INPUT_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Shapefiles"
    OUTPUT_DIR = r"D:\Jheff\Proyecto Tesis\Información\Machine Learning\Rasters"
    
    # Input files
    FAULTS_SHP = os.path.join(INPUT_DIR, "fallas.shp")
    LITHOLOGY_SHP = os.path.join(INPUT_DIR, "litologias.shp")
    DEPOSITS_SHP = os.path.join(INPUT_DIR, "yacimientos.shp")
    PC1_RASTER = os.path.join(INPUT_DIR, "PC1.tif")
    PC2_RASTER = os.path.join(INPUT_DIR, "PC2.tif")
    W_RASTER = os.path.join(INPUT_DIR, "W.tif")
    
    # Output CRS and resolution
    # UTM Zone 18S (EPSG:32718) for Peru/Arequipa region
    OUTPUT_CRS = "EPSG:32718"
    CELL_SIZE = 500  # meters - Adjusted for 30,000 km² study area
    
    # Column names in shapefiles
    FAVORABLE_COLUMN = "FAVORABLE"
    
    # ANN parameters based on Sun et al. (2019) - Table 3, page 40
    # Paper tested: neurons 2-10, epochs 10-500, learning rate 0.1-0.5, momentum 0.05-0.5
    ANN_PARAMS = {
        'hidden_layer_sizes': (10,),    # Number of neurons in hidden layer
        'activation': 'logistic',        # Sigmoid activation function (as in paper)
        'solver': 'sgd',                 # Stochastic gradient descent with backpropagation
        'learning_rate_init': 0.1,       # Initial learning rate
        'momentum': 0.2,                 # Momentum for SGD
        'max_iter': 300,                 # Maximum epochs
        'random_state': 77,
        'early_stopping': True,          # Stop when validation score doesn't improve
        'validation_fraction': 0.1,      # Fraction for early stopping validation
        'n_iter_no_change': 20           # Epochs with no improvement before stopping
    }
    
    # Parameter grid for hyperparameter tuning (based on paper Table 3 and Figure 12a,b)
    ANN_PARAM_GRID = {
        'hidden_layer_sizes': [(5,), (7,), (10,), (5, 3), (7, 5), (10, 5)],
        'max_iter': [100, 200, 300, 400, 500],
        'learning_rate_init': [0.1, 0.2, 0.3],
        'momentum': [0.1, 0.2, 0.3, 0.4]
    }
    
    # Simplified parameter grid for faster optimization
    ANN_PARAM_GRID_SIMPLE = {
        'hidden_layer_sizes': [(5,), (10,), (10, 5)],
        'max_iter': [200, 300, 400],
        'learning_rate_init': [0.1, 0.2],
        'momentum': [0.2, 0.3]
    }
    
    # Cross-validation folds (paper used 10-fold)
    CV_FOLDS = 10
    
    # Buffer distance for non-deposit selection (meters)
    NON_DEPOSIT_BUFFER = 5000  # Adjusted for 10km typical distance between deposits
    
    # Validation split
    TEST_SIZE = 0.33  # 1/3 for testing as in the paper
    
    # Search radius for fault density (meters)
    FAULT_SEARCH_RADIUS = 2500  # Based on 2500m mineralization distance to faults
    
    # Search radius for lithological contacts (meters)
    CONTACT_SEARCH_RADIUS = 100  # Contacts less important
    
    # Use simplified grid for faster execution (set to False for full optimization)
    USE_SIMPLE_GRID = True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        print(f"Created output directory: {Config.OUTPUT_DIR}")
    return Config.OUTPUT_DIR


def get_study_area_bounds(shapefiles_list):
    """Get unified bounds from all input shapefiles in target CRS"""
    all_bounds = []
    
    for shp_path in shapefiles_list:
        if os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path)
            gdf_projected = gdf.to_crs(Config.OUTPUT_CRS)
            bounds = gdf_projected.total_bounds
            all_bounds.append(bounds)
    
    if not all_bounds:
        raise ValueError("No valid shapefiles found to determine study area bounds")
    
    all_bounds = np.array(all_bounds)
    unified_bounds = [
        all_bounds[:, 0].min(),
        all_bounds[:, 1].min(),
        all_bounds[:, 2].max(),
        all_bounds[:, 3].max()
    ]
    
    return unified_bounds


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


# ============================================================================
# RASTERIZATION FUNCTIONS
# ============================================================================

def rasterize_polygon_to_binary(gdf, bounds, width, height, transform, attribute=None, value=1):
    """Rasterize polygon shapefile to binary raster"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    if attribute is not None:
        gdf_proj = gdf_proj[gdf_proj[attribute] == value]
    
    if len(gdf_proj) == 0:
        print(f"Warning: No features found with {attribute}={value}")
        return np.zeros((height, width), dtype=np.float32)
    
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    
    raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    return raster


def rasterize_lines_to_density(gdf, bounds, width, height, transform, search_radius=2500):
    """Calculate line density using kernel density estimation"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    
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
    
    return density


def calculate_euclidean_distance(binary_raster, cell_size):
    """Calculate Euclidean distance from features in meters"""
    inverted = 1 - binary_raster
    distance = distance_transform_edt(inverted) * cell_size
    return distance


def calculate_proximity_to_features(gdf, bounds, width, height, transform, cell_size):
    """Calculate proximity (distance) raster from vector features"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    shapes = [(geom, 1) for geom in gdf_proj.geometry if geom is not None]
    
    feature_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    distance = calculate_euclidean_distance(feature_raster, cell_size)
    
    return distance


def extract_fault_intersections(gdf):
    """Extract fault intersection points from line geometries"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    intersections = []
    geometries = list(gdf_proj.geometry)
    
    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            if geometries[i] is not None and geometries[j] is not None:
                if geometries[i].intersects(geometries[j]):
                    intersection = geometries[i].intersection(geometries[j])
                    if intersection.geom_type == 'Point':
                        intersections.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            intersections.append(point)
    
    if intersections:
        intersection_gdf = gpd.GeoDataFrame(geometry=intersections, crs=Config.OUTPUT_CRS)
        return intersection_gdf
    else:
        return None


def extract_lithological_contacts(gdf, bounds, width, height, transform, cell_size):
    """Extract lithological contacts and calculate their density"""
    gdf_proj = gdf.to_crs(Config.OUTPUT_CRS)
    
    boundaries = gdf_proj.boundary
    
    shapes = [(geom, 1) for geom in boundaries if geom is not None]
    
    if not shapes:
        return np.zeros((height, width), dtype=np.float32)
    
    contact_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    kernel_size = int(Config.CONTACT_SEARCH_RADIUS / cell_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3
    
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = (x**2 + y**2 <= (kernel_size//2)**2).astype(float)
    kernel /= kernel.sum()
    
    density = convolve(contact_raster.astype(float), kernel, mode='constant')
    
    return density


# ============================================================================
# RASTER PROCESSING FUNCTIONS
# ============================================================================

def resample_raster_to_grid(input_path, bounds, width, height, transform, output_crs):
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
        
        return dst_array


def normalize_raster(raster, method='minmax'):
    """Normalize raster values"""
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
                           width, height, n_non_deposits=None, buffer_distance=5000):
    """Generate training dataset following Sun et al. (2019) methodology"""
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    
    deposit_rows = []
    deposit_cols = []
    
    minx, miny, maxx, maxy = bounds
    
    for point in deposits_proj.geometry:
        if point is not None:
            col = int((point.x - minx) / Config.CELL_SIZE)
            row = int((maxy - point.y) / Config.CELL_SIZE)
            
            if 0 <= row < height and 0 <= col < width:
                deposit_rows.append(row)
                deposit_cols.append(col)
    
    n_deposits = len(deposit_rows)
    print(f"Number of deposit locations: {n_deposits}")
    
    if n_non_deposits is None:
        n_non_deposits = n_deposits
    
    deposit_mask = np.zeros((height, width), dtype=bool)
    for r, c in zip(deposit_rows, deposit_cols):
        deposit_mask[r, c] = True
    
    distance_from_deposits = distance_transform_edt(~deposit_mask) * Config.CELL_SIZE
    
    valid_mask = distance_from_deposits > buffer_distance
    
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
# ANN MODEL FUNCTIONS
# ============================================================================

def optimize_ann_hyperparameters(X, y, param_grid, cv_folds=10):
    """
    Optimize ANN hyperparameters using GridSearchCV
    Based on Sun et al. (2019) methodology - Figure 12a,b shows MSE variation
    """
    print("\n  Performing hyperparameter optimization...")
    print(f"  Testing configurations: {len(param_grid.get('hidden_layer_sizes', [1]))} architectures")
    print(f"  Testing max_iter values: {param_grid.get('max_iter', [])}")
    
    ann_base = MLPClassifier(
        activation='logistic',
        solver='sgd',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    grid_search = GridSearchCV(
        ann_base,
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


def train_ann(X, y, params=None):
    """Train ANN classifier with specified parameters"""
    if params is None:
        params = Config.ANN_PARAMS.copy()
    
    # Base parameters
    ann_params = {
        'activation': 'logistic',
        'solver': 'sgd',
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20
    }
    ann_params.update(params)
    
    ann_model = MLPClassifier(**ann_params)
    ann_model.fit(X, y)
    
    return ann_model


def perform_cross_validation(X, y, params=None, n_folds=10):
    """Perform k-fold cross-validation following Sun et al. (2019)"""
    if params is None:
        params = {}
    
    ann_params = {
        'activation': 'logistic',
        'solver': 'sgd',
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20
    }
    ann_params.update(params)
    
    ann_model = MLPClassifier(**ann_params)
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ann_model, X, y, cv=cv, scoring='accuracy')
    
    mse_scores = 1 - cv_scores
    
    results = {
        'cv_scores': cv_scores,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
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


def get_feature_importance_ann(model, X, y, feature_names):
    """
    Estimate feature importance for ANN using permutation importance
    Since ANN doesn't have built-in feature importance
    """
    from sklearn.inspection import permutation_importance
    
    perm_importance = permutation_importance(
        model, X, y, n_repeats=30, random_state=42, n_jobs=-1
    )
    
    importances = perm_importance.importances_mean
    indices = np.argsort(importances)[::-1]
    
    # Normalize to percentages
    total = importances.sum()
    if total > 0:
        importances_normalized = importances / total * 100
    else:
        importances_normalized = np.zeros_like(importances)
    
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Importance_Percent': importances_normalized[indices]
    })
    
    return importance_df


# ============================================================================
# PROSPECTIVITY MAP GENERATION
# ============================================================================

def generate_prospectivity_map(model, predictor_stack, width, height, scaler=None):
    """Generate prospectivity map by applying trained model to entire study area"""
    n_predictors = len(predictor_stack)
    
    valid_mask = np.ones((height, width), dtype=bool)
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
    plt.plot(results['fpr'], results['tpr'], 'r-', linewidth=2, 
             label=f'ANN (AUC = {results["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - ANN', fontsize=14)
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
    plt.plot(area_pct, deposit_pct, 'r-', linewidth=2, label='ANN')
    plt.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Random Model')
    
    plt.xlabel('Percentage of Cumulative Area (%)', fontsize=12)
    plt.ylabel('Percentage of Captured Deposits (%)', fontsize=12)
    plt.title('Success-Rate Curve - ANN', fontsize=14)
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
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(importance_df)))[::-1]
    
    bars = plt.barh(importance_df['Feature'], importance_df['Importance_Percent'], 
                    color=colors, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Contribution (%)', fontsize=12)
    plt.ylabel('Predictor Variable', fontsize=12)
    plt.title('Feature Importance - ANN Model (Permutation Importance)', fontsize=14)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14})
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - ANN', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {output_path}")
    plt.close()


def plot_prospectivity_map(prospectivity, deposits_gdf, bounds, output_path=None):
    """Plot prospectivity map with deposits overlay"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('prospectivity', colors, N=256)
    
    minx, miny, maxx, maxy = bounds
    extent = [minx, maxx, miny, maxy]
    
    im = ax.imshow(prospectivity, extent=extent, cmap=cmap, 
                   vmin=0, vmax=1, origin='upper')
    
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    ax.scatter(deposits_proj.geometry.x, deposits_proj.geometry.y,
               c='black', s=50, marker='^', edgecolors='white', 
               linewidths=0.5, label='Known Deposits', zorder=5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Prospectivity (Probability)', fontsize=12)
    
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Mineral Prospectivity Map - ANN Model', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved prospectivity map: {output_path}")
    plt.close()


def plot_classified_map(classified, deposits_gdf, bounds, output_path=None):
    """Plot classified prospectivity map"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('classified', colors, N=4)
    
    minx, miny, maxx, maxy = bounds
    extent = [minx, maxx, miny, maxy]
    
    im = ax.imshow(classified, extent=extent, cmap=cmap, 
                   vmin=1, vmax=4, origin='upper')
    
    deposits_proj = deposits_gdf.to_crs(Config.OUTPUT_CRS)
    ax.scatter(deposits_proj.geometry.x, deposits_proj.geometry.y,
               c='black', s=50, marker='^', edgecolors='white', 
               linewidths=0.5, label='Known Deposits', zorder=5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[1.375, 2.125, 2.875, 3.625])
    cbar.ax.set_yticklabels(['Low', 'Moderate', 'High', 'Very High'])
    cbar.set_label('Potential Classification', fontsize=12)
    
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title('Mineral Prospectivity Classification - ANN Model', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved classified map: {output_path}")
    plt.close()


def plot_learning_curve(model, output_path=None):
    """
    Plot the learning curve (loss vs epochs) from the trained ANN
    Similar to analysis in Sun et al. (2019) Figure 12a
    """
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, 'b-', linewidth=2, label='Training Loss')
        
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('ANN Learning Curve', fontsize=14)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning curve: {output_path}")
        plt.close()
    else:
        print("  Warning: Loss curve not available (model may not have been trained with SGD)")


def plot_hyperparameter_analysis(grid_search, output_path=None):
    """
    Plot hyperparameter analysis similar to Figure 12a,b in Sun et al. (2019)
    Shows MSE variation with epochs and neurons
    """
    results = pd.DataFrame(grid_search.cv_results_)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MSE vs max_iter (epochs)
    ax1 = axes[0]
    max_iters = Config.ANN_PARAM_GRID_SIMPLE['max_iter'] if Config.USE_SIMPLE_GRID else Config.ANN_PARAM_GRID['max_iter']
    
    for hidden_size in results['param_hidden_layer_sizes'].unique():
        mask = results['param_hidden_layer_sizes'] == hidden_size
        subset = results[mask].groupby('param_max_iter')['mean_test_score'].mean()
        mse = 1 - subset
        ax1.plot(subset.index, mse, 'o-', label=f'Neurons: {hidden_size}', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Maximum Epochs', fontsize=12)
    ax1.set_ylabel('MSE (1 - Accuracy)', fontsize=12)
    ax1.set_title('MSE vs Training Epochs', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MSE vs learning rate and momentum
    ax2 = axes[1]
    
    for lr in results['param_learning_rate_init'].unique():
        mask = results['param_learning_rate_init'] == lr
        subset = results[mask].groupby('param_momentum')['mean_test_score'].mean()
        mse = 1 - subset
        ax2.plot(subset.index, mse, 'o-', label=f'LR: {lr}', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Momentum', fontsize=12)
    ax2.set_ylabel('MSE (1 - Accuracy)', fontsize=12)
    ax2.set_title('MSE vs Momentum (by Learning Rate)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved hyperparameter analysis: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for ANN mineral prospectivity mapping"""
    print("="*80)
    print("MINERAL PROSPECTIVITY MAPPING USING ARTIFICIAL NEURAL NETWORKS (ANN)")
    print("Based on Sun et al. (2019) Methodology")
    print("="*80)
    
    output_dir = create_output_directory()
    
    # -------------------------------------------------------------------------
    # STEP 1: Load input data
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 1: Loading input data...")
    print("-"*60)
    
    print(f"Loading faults: {Config.FAULTS_SHP}")
    faults_gdf = gpd.read_file(Config.FAULTS_SHP)
    print(f"  - {len(faults_gdf)} fault features loaded")
    
    print(f"Loading lithologies: {Config.LITHOLOGY_SHP}")
    lithology_gdf = gpd.read_file(Config.LITHOLOGY_SHP)
    print(f"  - {len(lithology_gdf)} lithology features loaded")
    
    print(f"Loading deposits: {Config.DEPOSITS_SHP}")
    deposits_gdf = gpd.read_file(Config.DEPOSITS_SHP)
    print(f"  - {len(deposits_gdf)} deposit locations loaded")
    
    # -------------------------------------------------------------------------
    # STEP 2: Define study area grid
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 2: Defining study area grid...")
    print("-"*60)
    
    shapefiles = [Config.FAULTS_SHP, Config.LITHOLOGY_SHP, Config.DEPOSITS_SHP]
    bounds = get_study_area_bounds(shapefiles)
    
    width, height, adjusted_bounds = calculate_grid_dimensions(bounds, Config.CELL_SIZE)
    transform = create_transform(adjusted_bounds, width, height)
    
    print(f"  - CRS: {Config.OUTPUT_CRS}")
    print(f"  - Cell size: {Config.CELL_SIZE} meters")
    print(f"  - Grid dimensions: {width} x {height} pixels")
    print(f"  - Bounds: {adjusted_bounds}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Load or generate predictor maps
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 3: Loading/generating predictor maps...")
    print("-"*60)
    
    predictor_names = []
    predictor_stack = []
    
    # Check if predictor rasters already exist (from RF or SVM script)
    predictor_files = {
        'prox_favorable_lithology.tif': 'Proximity to Favorable Lithology',
        'fault_density.tif': 'Fault Density',
        'prox_faults.tif': 'Proximity to Faults',
        'prox_fault_intersections.tif': 'Proximity to Fault Intersections',
        'lithological_contacts_density.tif': 'Lithological Contacts Density',
        'geochemical_pc1.tif': 'Geochemical PC1',
        'geochemical_pc2.tif': 'Geochemical PC2',
        'geochemical_tungsten.tif': 'Tungsten Geochemistry'
    }
    
    all_exist = all(os.path.exists(os.path.join(output_dir, f)) for f in predictor_files.keys())
    
    if all_exist:
        print("  Found existing predictor rasters, loading...")
        for filename, name in predictor_files.items():
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                raster, _, _ = load_raster(filepath)
                predictor_stack.append(raster)
                predictor_names.append(name)
                print(f"    - Loaded: {name}")
    else:
        print("  Generating predictor maps from scratch...")
        
        # 3.1 Proximity to favorable lithologies
        print("\n  3.1 Processing favorable lithologies...")
        favorable_lith = rasterize_polygon_to_binary(
            lithology_gdf, adjusted_bounds, width, height, transform,
            attribute=Config.FAVORABLE_COLUMN, value=1
        )
        prox_favorable = calculate_euclidean_distance(favorable_lith, Config.CELL_SIZE)
        prox_favorable_norm = normalize_raster(prox_favorable, method='minmax')
        prox_favorable_norm = 1 - prox_favorable_norm
        
        save_raster(prox_favorable_norm, 
                    os.path.join(output_dir, "prox_favorable_lithology.tif"),
                    transform, Config.OUTPUT_CRS)
        predictor_names.append("Proximity to Favorable Lithology")
        predictor_stack.append(prox_favorable_norm)
        
        # 3.2 Fault density
        print("\n  3.2 Processing fault density...")
        fault_density = rasterize_lines_to_density(
            faults_gdf, adjusted_bounds, width, height, transform, 
            search_radius=Config.FAULT_SEARCH_RADIUS
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
            faults_gdf, adjusted_bounds, width, height, transform, Config.CELL_SIZE
        )
        prox_faults_norm = normalize_raster(prox_faults, method='minmax')
        prox_faults_norm = 1 - prox_faults_norm
        
        save_raster(prox_faults_norm,
                    os.path.join(output_dir, "prox_faults.tif"),
                    transform, Config.OUTPUT_CRS)
        predictor_names.append("Proximity to Faults")
        predictor_stack.append(prox_faults_norm)
        
        # 3.4 Proximity to fault intersections
        print("\n  3.4 Processing fault intersections...")
        fault_intersections = extract_fault_intersections(faults_gdf)
        if fault_intersections is not None and len(fault_intersections) > 0:
            prox_intersections = calculate_proximity_to_features(
                fault_intersections, adjusted_bounds, width, height, transform, Config.CELL_SIZE
            )
            prox_intersections_norm = normalize_raster(prox_intersections, method='minmax')
            prox_intersections_norm = 1 - prox_intersections_norm
            
            save_raster(prox_intersections_norm,
                        os.path.join(output_dir, "prox_fault_intersections.tif"),
                        transform, Config.OUTPUT_CRS)
            predictor_names.append("Proximity to Fault Intersections")
            predictor_stack.append(prox_intersections_norm)
            print(f"    - Number of fault intersections: {len(fault_intersections)}")
        
        # 3.5 Lithological contacts density
        print("\n  3.5 Processing lithological contacts...")
        contact_density = extract_lithological_contacts(
            lithology_gdf, adjusted_bounds, width, height, transform, Config.CELL_SIZE
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
            Config.PC1_RASTER, adjusted_bounds, width, height, transform, Config.OUTPUT_CRS
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
            Config.PC2_RASTER, adjusted_bounds, width, height, transform, Config.OUTPUT_CRS
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
            Config.W_RASTER, adjusted_bounds, width, height, transform, Config.OUTPUT_CRS
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
    # STEP 4: Generate training data
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 4: Generating training dataset...")
    print("-"*60)
    
    X, y, locations = generate_training_data(
        deposits_gdf, predictor_stack, transform, adjusted_bounds,
        width, height, buffer_distance=Config.NON_DEPOSIT_BUFFER
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
    # STEP 5: Feature scaling (important for ANN)
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 5: Scaling features (StandardScaler)...")
    print("-"*60)
    
    # ANN is very sensitive to feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  - Features standardized (mean=0, std=1)")
    
    # -------------------------------------------------------------------------
    # STEP 6: Hyperparameter optimization
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 6: Optimizing ANN hyperparameters...")
    print("-"*60)
    
    param_grid = Config.ANN_PARAM_GRID_SIMPLE if Config.USE_SIMPLE_GRID else Config.ANN_PARAM_GRID
    
    if Config.USE_SIMPLE_GRID:
        print("  Using simplified parameter grid for faster execution")
    
    best_params, grid_search = optimize_ann_hyperparameters(
        X_train_scaled, y_train, param_grid, cv_folds=Config.CV_FOLDS
    )
    
    # Plot hyperparameter analysis
    plot_hyperparameter_analysis(grid_search, 
                                 os.path.join(output_dir, "ann_hyperparameter_analysis.png"))
    
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
    # STEP 8: Train final ANN model
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 8: Training final ANN model...")
    print("-"*60)
    
    ann_model = train_ann(X_train_scaled, y_train, best_params)
    
    print(f"  - Architecture: {ann_model.hidden_layer_sizes}")
    print(f"  - Activation: {ann_model.activation}")
    print(f"  - Solver: {ann_model.solver}")
    print(f"  - Learning rate: {ann_model.learning_rate_init}")
    print(f"  - Momentum: {ann_model.momentum}")
    print(f"  - Iterations completed: {ann_model.n_iter_}")
    print(f"  - Final loss: {ann_model.loss_:.6f}")
    
    # Plot learning curve
    plot_learning_curve(ann_model, os.path.join(output_dir, "ann_learning_curve.png"))
    
    # -------------------------------------------------------------------------
    # STEP 9: Model evaluation
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 9: Evaluating model performance...")
    print("-"*60)
    
    # Evaluate on training data
    train_results = evaluate_model(ann_model, X_train_scaled, y_train)
    print("\n  Training Dataset Performance:")
    print(f"    - Accuracy: {train_results['accuracy']:.4f}")
    print(f"    - Sensitivity: {train_results['sensitivity']:.4f}")
    print(f"    - Specificity: {train_results['specificity']:.4f}")
    
    # Evaluate on test data
    test_results = evaluate_model(ann_model, X_test_scaled, y_test)
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
    
    # Feature importance (permutation-based for ANN)
    print("\n  Calculating feature importance (permutation-based)...")
    importance_df = get_feature_importance_ann(ann_model, X_test_scaled, y_test, predictor_names)
    print("\n  Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"    - {row['Feature']}: {row['Importance_Percent']:.2f}%")
    
    # -------------------------------------------------------------------------
    # STEP 10: Generate prospectivity map
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 10: Generating prospectivity map...")
    print("-"*60)
    
    prospectivity = generate_prospectivity_map(ann_model, predictor_stack, width, height, scaler)
    
    # Save prospectivity raster
    save_raster(prospectivity,
                os.path.join(output_dir, "prospectivity_ann.tif"),
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
                os.path.join(output_dir, "prospectivity_ann_classified.tif"),
                transform, Config.OUTPUT_CRS)
    
    # Calculate statistics
    valid_class = classified[~np.isnan(classified)]
    total_pixels = len(valid_class)
    
    for class_value, class_name in [(4, 'Very High'), (3, 'High'), (2, 'Moderate'), (1, 'Low')]:
        class_pixels = (valid_class == class_value).sum()
        area_percent = class_pixels / total_pixels * 100
        print(f"  - {class_name} potential: {area_percent:.2f}% of area")
    
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
                   os.path.join(output_dir, "roc_curve_ann.png"))
    
    plot_success_rate_curve(area_pct, deposit_pct, thresholds,
                           os.path.join(output_dir, "success_rate_curve_ann.png"))
    
    plot_feature_importance(importance_df,
                           os.path.join(output_dir, "feature_importance_ann.png"))
    
    plot_confusion_matrix(test_results['confusion_matrix'],
                         os.path.join(output_dir, "confusion_matrix_ann.png"))
    
    plot_prospectivity_map(prospectivity, deposits_gdf, adjusted_bounds,
                          os.path.join(output_dir, "prospectivity_map_ann.png"))
    
    plot_classified_map(classified, deposits_gdf, adjusted_bounds,
                       os.path.join(output_dir, "prospectivity_classified_ann.png"))
    
    # -------------------------------------------------------------------------
    # STEP 13: Save results summary
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 13: Saving results summary...")
    print("-"*60)
    
    summary = {
        'Model': 'Artificial Neural Network (ANN)',
        'Architecture': str(ann_model.hidden_layer_sizes),
        'Activation': ann_model.activation,
        'Solver': ann_model.solver,
        'Learning Rate': ann_model.learning_rate_init,
        'Momentum': ann_model.momentum,
        'Iterations': ann_model.n_iter_,
        'Final Loss': ann_model.loss_,
        'Number of Predictors': len(predictor_names),
        'Predictor Names': str(predictor_names),
        'Total Samples': len(y),
        'Training Samples': len(y_train),
        'Test Samples': len(y_test),
        'CV Mean Accuracy': cv_results['mean_accuracy'],
        'CV Std Accuracy': cv_results['std_accuracy'],
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
    summary_df.to_csv(os.path.join(output_dir, "model_summary_ann.csv"), index=False)
    
    importance_df.to_csv(os.path.join(output_dir, "feature_importance_ann.csv"), index=False)
    
    # Save hyperparameter search results
    hp_results = pd.DataFrame(grid_search.cv_results_)
    hp_results.to_csv(os.path.join(output_dir, "hyperparameter_search_ann.csv"), index=False)
    
    print("\n" + "="*80)
    print("ANN MINERAL PROSPECTIVITY MAPPING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nOutput files:")
    print("  - Rasters: prospectivity_ann.tif, prospectivity_ann_classified.tif")
    print("  - Figures: roc_curve_ann.png, success_rate_curve_ann.png, etc.")
    print("  - Reports: model_summary_ann.csv, feature_importance_ann.csv")
    print("  - Learning: ann_learning_curve.png, ann_hyperparameter_analysis.png")
    
    return ann_model, prospectivity, test_results, scaler


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        model, prospectivity_map, results, scaler = main()
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Please verify that all input files exist in the specified paths.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
