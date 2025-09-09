# -*- coding: utf-8 -*-
"""
ENHANCED MACHINE LEARNING ANALYSIS WITH OPTIMIZATION
- Hyperparameter optimization using GridSearchCV
- Model comparison visualization
- Network importance analysis
- Complete results summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

DATA_PATH = r"E:\Gisung\2025년연구\carotidplaque연구\250828\250828_data.csv"
OUT_DIR = r"E:\Gisung\2025년연구\carotidplaque연구\ML_Enhanced"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS = 5

# ==============================================================================
# VARIABLES DEFINITION (from original code)
# ==============================================================================

COVARIATES = [
    'AG1_AGE', 'AG1_SEX', 'AG1_EDU', 'AG1_BMI',
    'ag1_smoke', 'ag1_drink', 'AG1_HTN', 'AG1_DM', 'AG1_CVD', 'AG1_BDISUM'
]

COGNITIVE_TESTS = {
    'LMI': {'baseline': 'LMIB', 'followup': 'LMIF2', 'name': 'Logical Memory Immediate'},
    'LMD': {'baseline': 'LMDB', 'followup': 'LMDF2', 'name': 'Logical Memory Delayed'},
    'VRD': {'baseline': 'VRDB', 'followup': 'VRDF2', 'name': 'Visual Reproduction Delayed'},
    'STR1': {'baseline': 'STR1B', 'followup': 'STR1F2', 'name': 'Stroop Word'},
    'STR2': {'baseline': 'STR2B', 'followup': 'STR2F2', 'name': 'Stroop Color'}
}

# FIXED: Each ROI should be a separate string in the list
SIGNIFICANT_ROIS_51 = [
    'AG1_rVisCent_ExStr_6',
    'AG1_lDefaultA_PFCm_2',
    'AG1_lLimbicB_OFC_8',
    'AG1_rVisCent_ExStr_8',
    'AG1_lDorsAttnA_SPL_4',
    'AG1_rDorsAttnA_TempOcc_3',
    'AG1_rLimbicB_OFC_10',
    'AG1_rSalVentAttnA_FrMed_3',
    'AG1_rVisCent_Striate_1',
    'AG1_lSomMotA_20',
    'AG1_rContB_PFCld_7',
    'AG1_lSalVentAttnA_FrOper_1',
    'AG1_lSalVentAttnA_FrMed_3',
    'AG1_lDefaultB_PFCl_1',
    'AG1_rVisCent_ExStr_3',
    'AG1_lSalVentAttnA_Ins_6',
    'AG1_lContB_PFCl_2',
    'AG1_lSalVentAttnA_Ins_3',
    'AG1_rVisCent_ExStr_5',
    'AG1_lDorsAttnB_FEF_2',
    'AG1_lDefaultB_Temp_2',
    'AG1_rVisCent_ExStr_7',
    'AG1_lDefaultA_PFCm_6',
    'AG1_rVisPeri_ExStrSup_1',
    'AG1_lSalVentAttnB_Ins_4',
    'AG1_rVisPeri_ExStrInf_1',
    'AG1_lSalVentAttnA_FrOper_2',
    'AG1_lDefaultC_Rsp_2',
    'AG1_lContA_PFCl_3',
    'AG1_rSalVentAttnA_PrC_1',
    'AG1_lContC_pCun_2',
    'AG1_rVisCent_ExStr_2',
    'AG1_lDefaultB_Temp_3',
    'AG1_rSomMotB_Cent_3',
    'AG1_rLimbicB_OFC_3',
    'AG1_lVisPeri_ExStrSup_3',
    'AG1_lSomMotB_Cent_8',
    'AG1_rVisCent_ExStr_4',
    'AG1_rSomMotB_Cent_6',
    'AG1_rVisCent_ExStr_13',
    'AG1_lSalVentAttnA_ParOper_2',
    'AG1_lDorsAttnB_FEF_3',
    'AG1_lDorsAttnA_SPL_10',
    'AG1_lVisCent_ExStr_3',
    'AG1_rSalVentAttnB_Ins_1',
    'AG1_lDefaultC_PHC_2',
    'AG1_rDefaultA_PFCm_3',
    'AG1_lDefaultB_PFCv_4',
    'AG1_lDefaultA_IPL_1',
    'AG1_lDefaultB_PFCv_6',
    'AG1_rDefaultB_AntTemp_3'
]

# Network mapping
NETWORK_GROUPS = {
    'Visual': [],
    'Somatomotor': [],
    'DorsalAttention': [],
    'VentralAttention': [],
    'Limbic': [],
    'Control': [],
    'Default': []
}

for roi in SIGNIFICANT_ROIS_51:
    if 'Vis' in roi:
        NETWORK_GROUPS['Visual'].append(roi)
    elif 'SomMot' in roi:
        NETWORK_GROUPS['Somatomotor'].append(roi)
    elif 'DorsAttn' in roi:
        NETWORK_GROUPS['DorsalAttention'].append(roi)
    elif 'SalVentAttn' in roi:
        NETWORK_GROUPS['VentralAttention'].append(roi)
    elif 'Limbic' in roi:
        NETWORK_GROUPS['Limbic'].append(roi)
    elif 'Cont' in roi:
        NETWORK_GROUPS['Control'].append(roi)
    elif 'Default' in roi:
        NETWORK_GROUPS['Default'].append(roi)

# ==============================================================================
# HYPERPARAMETER GRIDS FOR OPTIMIZATION
# ==============================================================================

PARAM_GRIDS = {
    'Ridge': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    },
    'LASSO': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000, 2000]
    },
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.2, 0.5, 0.8]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
}

# ==============================================================================
# LOAD AND PREPROCESS DATA
# ==============================================================================

print("="*70)
print("ENHANCED MACHINE LEARNING ANALYSIS")
print("="*70)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"\nData loaded: {df.shape[0]} subjects")

# Check which ROIs are actually in the dataset
missing_rois = [roi for roi in SIGNIFICANT_ROIS_51 if roi not in df.columns]
available_rois = [roi for roi in SIGNIFICANT_ROIS_51 if roi in df.columns]

if missing_rois:
    print(f"\nWarning: {len(missing_rois)} ROIs not found in dataset:")
    for roi in missing_rois[:5]:  # Show first 5 missing
        print(f"  - {roi}")
    if len(missing_rois) > 5:
        print(f"  ... and {len(missing_rois)-5} more")
    print(f"\nUsing {len(available_rois)} available ROIs")
    SIGNIFICANT_ROIS_51 = available_rois

# TIV normalization
if 'AG1_TIV' in df.columns and available_rois:
    print("Performing TIV normalization...")
    tiv = df['AG1_TIV'].replace(0, np.nan)
    roi_normalized = df[available_rois].div(tiv, axis=0)
elif available_rois:
    roi_normalized = df[available_rois].copy()
else:
    print("ERROR: No ROIs found in dataset!")
    roi_normalized = pd.DataFrame()

# Calculate network means
print("\nCalculating network means...")
network_means = pd.DataFrame(index=df.index)

# Re-map networks with available ROIs only
NETWORK_GROUPS = {
    'Visual': [],
    'Somatomotor': [],
    'DorsalAttention': [],
    'VentralAttention': [],
    'Limbic': [],
    'Control': [],
    'Default': []
}

for roi in available_rois:
    if 'Vis' in roi:
        NETWORK_GROUPS['Visual'].append(roi)
    elif 'SomMot' in roi:
        NETWORK_GROUPS['Somatomotor'].append(roi)
    elif 'DorsAttn' in roi:
        NETWORK_GROUPS['DorsalAttention'].append(roi)
    elif 'SalVentAttn' in roi:
        NETWORK_GROUPS['VentralAttention'].append(roi)
    elif 'Limbic' in roi:
        NETWORK_GROUPS['Limbic'].append(roi)
    elif 'Cont' in roi:
        NETWORK_GROUPS['Control'].append(roi)
    elif 'Default' in roi:
        NETWORK_GROUPS['Default'].append(roi)

for net_name, roi_list in NETWORK_GROUPS.items():
    if roi_list:
        network_means[net_name] = roi_normalized[roi_list].mean(axis=1)
        print(f"  {net_name}: {len(roi_list)} ROIs")

print(f"\nTotal features: {len(network_means.columns)} networks + 1 baseline + {len(COVARIATES)} covariates")

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_cognitive_optimized(df_all, follow_col, base_col, covariates, network_df, optimize=True):
    """
    Enhanced analysis pipeline with hyperparameter optimization
    """
    print(f"\n{'='*60}")
    print(f"{follow_col} Analysis {'(with optimization)' if optimize else ''}")
    print(f"{'='*60}")
    
    # Prepare features
    X = pd.concat([network_df, df_all[[base_col] + covariates]], axis=1)
    y = df_all[follow_col].copy()
    
    network_names = list(network_df.columns)
    print(f"Features: {len(network_names)} networks + 1 baseline + {len(covariates)} covariates = {X.shape[1]} total")
    
    # Remove missing values
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Handle outliers for STR tests
    if 'STR' in follow_col:
        z_scores = np.abs((y - y.mean()) / y.std())
        outlier_mask = z_scores <= 3
        X = X[outlier_mask]
        y = y[outlier_mask]
        print(f"Removed {(~outlier_mask).sum()} outliers")
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Samples: Train={len(X_train)}, Test={len(X_test)}")
    
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    scaler = StandardScaler()
    X_train_final = pd.DataFrame(
        scaler.fit_transform(X_train_imp),
        columns=X_train_imp.columns,
        index=X_train_imp.index
    )
    X_test_final = pd.DataFrame(
        scaler.transform(X_test_imp),
        columns=X_test_imp.columns,
        index=X_test_imp.index
    )
    
    # Model training
    models = {
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'LASSO': Lasso(random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    }
    
    model_results = {}
    best_params = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        if optimize and name in PARAM_GRIDS:
            # GridSearch for optimization
            grid_search = GridSearchCV(
                model, 
                PARAM_GRIDS[name], 
                cv=N_CV_FOLDS, 
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_final, y_train)
            best_model = grid_search.best_estimator_
            best_params[name] = grid_search.best_params_
            cv_score = grid_search.best_score_
            print(f"    Best params: {grid_search.best_params_}")
        else:
            # Default parameters
            best_model = model
            best_model.fit(X_train_final, y_train)
            cv_scores = cross_val_score(best_model, X_train_final, y_train, cv=N_CV_FOLDS, scoring='r2')
            cv_score = cv_scores.mean()
        
        # Evaluate
        train_pred = best_model.predict(X_train_final)
        test_pred = best_model.predict(X_test_final)
        
        model_results[name] = {
            'model': best_model,
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'cv_r2': cv_score,
            'params': best_params.get(name, 'default')
        }
        
        print(f"    CV R²: {cv_score:.3f}, Test R²: {model_results[name]['test_r2']:.3f}")
    
    # Find best model
    best_model_name = max(model_results.items(), key=lambda x: x[1]['test_r2'])[0]
    best_model = model_results[best_model_name]['model']
    print(f"\n  🏆 Best model: {best_model_name} (Test R² = {model_results[best_model_name]['test_r2']:.3f})")
    
    # SHAP analysis
    print(f"  Computing SHAP values...")
    n_shap_samples = min(200, len(X_test_final))
    X_test_shap = X_test_final.sample(n_shap_samples, random_state=RANDOM_STATE)
    
    try:
        if best_model_name in ['XGBoost', 'RandomForest']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_shap)
        elif best_model_name in ['Ridge', 'LASSO', 'ElasticNet']:
            explainer = shap.LinearExplainer(best_model, X_train_final)
            shap_values = explainer.shap_values(X_test_shap)
        else:
            # KernelExplainer as fallback
            sample_data = shap.sample(X_train_final, 100)
            explainer = shap.KernelExplainer(best_model.predict, sample_data)
            shap_values = explainer.shap_values(X_test_shap)
        print(f"    ✓ SHAP computation successful")
    except Exception as e:
        shap_values = None
        print(f"    ⚠️ SHAP computation failed: {e}")
    
    return {
        'target': follow_col,
        'best_model': best_model_name,
        'model_results': model_results,
        'best_params': best_params,
        'shap_values': shap_values,
        'X_test': X_test_shap,
        'feature_names': list(X_test_shap.columns),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

print("\n" + "="*70)
print("Running Analysis for Each Cognitive Test")
print("="*70)

all_results_optimized = {}
all_results_default = {}

# Check if we have valid network means
if network_means.empty:
    print("\nERROR: No network means calculated. Check ROI availability in dataset.")
else:
    for key, meta in COGNITIVE_TESTS.items():
        follow = meta['followup']
        base = meta['baseline']
        
        if follow not in df.columns or base not in df.columns:
            print(f"\n[SKIP] {follow}: Missing columns")
            continue
        
        # Check if we have required covariates
        missing_covariates = [cov for cov in COVARIATES if cov not in df.columns]
        if missing_covariates:
            print(f"\n[SKIP] {follow}: Missing covariates: {missing_covariates}")
            continue
        
        # Run optimized analysis
        results = analyze_cognitive_optimized(df, follow, base, COVARIATES, network_means, optimize=True)
        all_results_optimized[follow] = results
        
        # Also run default for comparison
        results_default = analyze_cognitive_optimized(df, follow, base, COVARIATES, network_means, optimize=False)
        all_results_default[follow] = results_default

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_model_comparison(all_results, save_path):
    """Create model comparison visualization"""
    n_tests = len(all_results)
    if n_tests == 0:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (test_name, results) in enumerate(all_results.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        # Prepare data
        model_names = list(results['model_results'].keys())
        train_scores = [results['model_results'][m]['train_r2'] for m in model_names]
        test_scores = [results['model_results'][m]['test_r2'] for m in model_names]
        cv_scores = [results['model_results'][m]['cv_r2'] for m in model_names]
        
        # Create grouped bar plot
        x = np.arange(len(model_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, train_scores, width, label='Train', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, cv_scores, width, label='CV', alpha=0.8, color='lightgreen')
        bars3 = ax.bar(x + width, test_scores, width, label='Test', alpha=0.8, color='salmon')
        
        # Highlight best model
        best_idx = model_names.index(results['best_model'])
        bars3[best_idx].set_color('red')
        bars3[best_idx].set_alpha(1.0)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{test_name} - Best: {results["best_model"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Hide unused subplots
    for idx in range(n_tests, 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_network_importance_detailed(all_results, network_names, save_path):
    """Create detailed network importance analysis"""
    n_tests = len(all_results)
    if n_tests == 0:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (test_name, results) in enumerate(all_results.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        
        if results['shap_values'] is None:
            ax.text(0.5, 0.5, 'SHAP not available', ha='center', va='center')
            ax.set_title(test_name)
            continue
            
        # Calculate network importance
        importance = np.abs(results['shap_values']).mean(axis=0)
        feature_names = results['feature_names']
        
        network_importance = {}
        for network in network_names:
            if network in feature_names:
                idx_net = feature_names.index(network)
                network_importance[network] = importance[idx_net]
            else:
                network_importance[network] = 0
        
        # Plot
        networks = list(network_importance.keys())
        values = list(network_importance.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(networks)))
        bars = ax.bar(networks, values, color=colors)
        ax.set_xlabel('Brain Networks')
        ax.set_ylabel('SHAP Importance')
        ax.set_title(f'{test_name} (R² = {results["model_results"][results["best_model"]]["test_r2"]:.3f})')
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_tests, 6):
        axes[idx].set_visible(False)
    
    plt.suptitle('Network Importance by Cognitive Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_results_summary_table(all_results, save_path):
    """Create comprehensive results summary table"""
    if not all_results:
        print("No results to summarize")
        return None
    
    summary_data = []
    
    for test_name, results in all_results.items():
        best_model = results['best_model']
        model_stats = results['model_results'][best_model]
        
        # Get top 3 features
        if results['shap_values'] is not None:
            importance = np.abs(results['shap_values']).mean(axis=0)
            feature_names = results['feature_names']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            top_features = importance_df.head(3)['feature'].tolist()
            top_features_str = ', '.join(top_features)
        else:
            top_features_str = 'N/A'
        
        summary_data.append({
            'Cognitive Test': test_name,
            'Best Model': best_model,
            'Train R²': f"{model_stats['train_r2']:.3f}",
            'CV R²': f"{model_stats['cv_r2']:.3f}",
            'Test R²': f"{model_stats['test_r2']:.3f}",
            'Test RMSE': f"{model_stats['test_rmse']:.2f}",
            'Test MAE': f"{model_stats['test_mae']:.2f}",
            'Top 3 Features': top_features_str,
            'N_train': results['n_train'],
            'N_test': results['n_test']
        })
    
    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Save to Excel
    df_summary.to_excel(save_path, index=False, engine='openpyxl')
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with color coding
    table = ax.table(cellText=df_summary.values,
                     colLabels=df_summary.columns,
                     cellLoc='center',
                     loc='center')
    
    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    
    # Color header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code performance
    for i in range(len(df_summary)):
        test_r2 = float(df_summary.iloc[i]['Test R²'])
        if test_r2 > 0.5:
            color = '#C6EFCE'  # Light green
        elif test_r2 > 0.4:
            color = '#FFEB9C'  # Light yellow
        else:
            color = '#FFC7CE'  # Light red
        
        for j in range(len(df_summary.columns)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    plt.title('Machine Learning Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path.replace('.xlsx', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_summary

def plot_shap_beeswarm_all_features(all_results, save_dir):
    """Create SHAP beeswarm plots showing all features for each cognitive test"""
    print("\nCreating SHAP beeswarm plots for all features...")
    
    for test_name, results in all_results.items():
        if results['shap_values'] is None:
            print(f"  Skipping {test_name}: No SHAP values available")
            continue
            
        print(f"  Creating beeswarm plot for {test_name}...")
        
        # Create figure for this test
        fig = plt.figure(figsize=(12, 8))
        
        # Create SHAP beeswarm plot
        shap.summary_plot(
            results['shap_values'],
            results['X_test'],
            plot_type="dot",
            show=False,
            max_display=20  # Show top 20 features
        )
        
        # Customize the plot
        plt.title(f'SHAP Feature Importance - {test_name}\n'
                  f'Best Model: {results["best_model"]} (Test R² = {results["model_results"][results["best_model"]]["test_r2"]:.3f})',
                  fontsize=12, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save individual plot
        save_path = os.path.join(save_dir, f'shap_beeswarm_{test_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print("  Beeswarm plots created successfully!")

def plot_shap_combined_beeswarm(all_results, save_path):
    """Create combined SHAP beeswarm plots for all cognitive tests in one figure"""
    print("\nCreating combined SHAP beeswarm plot...")
    
    # Filter tests with SHAP values
    valid_tests = {k: v for k, v in all_results.items() if v['shap_values'] is not None}
    
    if not valid_tests:
        print("No tests with SHAP values available")
        return
    
    n_tests = len(valid_tests)
    n_cols = 2
    n_rows = (n_tests + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_tests == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_tests > 1 else [axes]
    
    for idx, (test_name, results) in enumerate(valid_tests.items()):
        ax = axes[idx]
        
        # Calculate mean absolute SHAP values for ordering
        mean_abs_shap = np.abs(results['shap_values']).mean(axis=0)
        feature_names = results['feature_names']
        
        # Create DataFrame for easier plotting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Get top 15 features
        top_features = shap_df.head(15)
        
        # Create custom beeswarm-style plot
        top_indices = [feature_names.index(f) for f in top_features['feature']]
        
        # Plot using matplotlib scatter
        y_pos = np.arange(len(top_features))
        
        for i, feature_idx in enumerate(top_indices):
            shap_vals = results['shap_values'][:, feature_idx]
            feature_vals = results['X_test'].iloc[:, feature_idx]
            
            # Normalize feature values for coloring
            if feature_vals.std() > 0:
                colors = (feature_vals - feature_vals.mean()) / feature_vals.std()
            else:
                colors = np.zeros_like(feature_vals)
            
            # Add jitter for visibility
            jitter = np.random.normal(0, 0.1, len(shap_vals))
            
            scatter = ax.scatter(shap_vals, np.full_like(shap_vals, i) + jitter,
                               c=colors, cmap='coolwarm', alpha=0.6, s=20,
                               vmin=-2, vmax=2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].tolist())
        ax.set_xlabel('SHAP value (impact on model output)')
        ax.set_title(f'{test_name} (R² = {results["model_results"][results["best_model"]]["test_r2"]:.3f})')
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add colorbar for the first plot
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
            cbar.set_label('Feature value\n(normalized)', rotation=270, labelpad=20)
    
    # Hide unused subplots if odd number of tests
    if n_tests % 2 == 1:
        axes[-1].set_visible(False)
    
    plt.suptitle('SHAP Feature Importance - All Features\n(7 Networks + Baseline Cognition + 10 Covariates)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  Combined beeswarm plot created successfully!")

def plot_optimization_impact(all_results_default, all_results_optimized, save_path):
    """Compare default vs optimized model performance"""
    if not all_results_default or not all_results_optimized:
        print("Need both default and optimized results for comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    tests = list(all_results_optimized.keys())
    
    # Prepare data
    default_scores = []
    optimized_scores = []
    
    for test in tests:
        if test in all_results_default and test in all_results_optimized:
            default_best = all_results_default[test]['best_model']
            optimized_best = all_results_optimized[test]['best_model']
            
            default_scores.append(all_results_default[test]['model_results'][default_best]['test_r2'])
            optimized_scores.append(all_results_optimized[test]['model_results'][optimized_best]['test_r2'])
    
    if not default_scores:
        print("No common tests between default and optimized results")
        return
        
    # Plot 1: Bar comparison
    ax1 = axes[0]
    x = np.arange(len(tests))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, default_scores, width, label='Default', alpha=0.8, color='lightblue')
    bars2 = ax1.bar(x + width/2, optimized_scores, width, label='Optimized', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Cognitive Tests')
    ax1.set_ylabel('Test R² Score')
    ax1.set_title('Impact of Hyperparameter Optimization')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tests, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add improvement percentage
    for i, (d, o) in enumerate(zip(default_scores, optimized_scores)):
        improvement = ((o - d) / d) * 100 if d > 0 else 0
        if improvement > 0:
            ax1.text(i, max(d, o) + 0.01, f'+{improvement:.1f}%', 
                    ha='center', va='bottom', fontsize=8, color='green')
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(default_scores, optimized_scores, s=100, alpha=0.6, color='purple')
    
    # Add diagonal line
    min_val = min(min(default_scores), min(optimized_scores)) - 0.05
    max_val = max(max(default_scores), max(optimized_scores)) + 0.05
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No improvement')
    
    # Add labels
    for i, test in enumerate(tests):
        ax2.annotate(test, (default_scores[i], optimized_scores[i]), 
                    fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Default R² Score')
    ax2.set_ylabel('Optimized R² Score')
    ax2.set_title('Optimization Improvement')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Hyperparameter Optimization Impact', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_beeswarm_all_features(all_results, save_dir):
    """Create SHAP beeswarm plots showing all features for each cognitive test"""
    print("\nCreating SHAP beeswarm plots for all features...")
    
    for test_name, results in all_results.items():
        if results['shap_values'] is None:
            print(f"  Skipping {test_name}: No SHAP values available")
            continue
            
        print(f"  Creating beeswarm plot for {test_name}...")
        
        # Create figure for this test
        fig = plt.figure(figsize=(12, 8))
        
        # Create SHAP beeswarm plot
        shap.summary_plot(
            results['shap_values'],
            results['X_test'],
            plot_type="dot",
            show=False,
            max_display=20  # Show top 20 features
        )
        
        # Customize the plot
        plt.title(f'SHAP Feature Importance - {test_name}\n'
                  f'Best Model: {results["best_model"]} (Test R² = {results["model_results"][results["best_model"]]["test_r2"]:.3f})',
                  fontsize=12, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save individual plot
        save_path = os.path.join(save_dir, f'shap_beeswarm_{test_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print("  Beeswarm plots created successfully!")

def plot_shap_combined_beeswarm(all_results, save_path):
    """Create combined SHAP beeswarm plots for all cognitive tests in one figure"""
    print("\nCreating combined SHAP beeswarm plot...")
    
    # Filter tests with SHAP values
    valid_tests = {k: v for k, v in all_results.items() if v['shap_values'] is not None}
    
    if not valid_tests:
        print("No tests with SHAP values available")
        return
    
    n_tests = len(valid_tests)
    n_cols = 2
    n_rows = (n_tests + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_tests == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_tests > 1 else [axes]
    
    for idx, (test_name, results) in enumerate(valid_tests.items()):
        ax = axes[idx]
        
        # Calculate mean absolute SHAP values for ordering
        mean_abs_shap = np.abs(results['shap_values']).mean(axis=0)
        feature_names = results['feature_names']
        
        # Create DataFrame for easier plotting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Get top 15 features
        top_features = shap_df.head(15)
        
        # Create custom beeswarm-style plot
        top_indices = [feature_names.index(f) for f in top_features['feature']]
        
        # Plot using matplotlib scatter
        y_pos = np.arange(len(top_features))
        
        for i, feature_idx in enumerate(top_indices):
            shap_vals = results['shap_values'][:, feature_idx]
            feature_vals = results['X_test'].iloc[:, feature_idx]
            
            # Normalize feature values for coloring
            if feature_vals.std() > 0:
                colors = (feature_vals - feature_vals.mean()) / feature_vals.std()
            else:
                colors = np.zeros_like(feature_vals)
            
            # Add jitter for visibility
            jitter = np.random.normal(0, 0.1, len(shap_vals))
            
            scatter = ax.scatter(shap_vals, np.full_like(shap_vals, i) + jitter,
                               c=colors, cmap='coolwarm', alpha=0.6, s=20,
                               vmin=-2, vmax=2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].tolist())
        ax.set_xlabel('SHAP value (impact on model output)')
        ax.set_title(f'{test_name} (R² = {results["model_results"][results["best_model"]]["test_r2"]:.3f})')
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add colorbar for the first plot
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
            cbar.set_label('Feature value\n(normalized)', rotation=270, labelpad=20)
    
    # Hide unused subplots if odd number of tests
    if n_tests % 2 == 1:
        axes[-1].set_visible(False)
    
    plt.suptitle('SHAP Feature Importance - All Features\n(7 Networks + Baseline Cognition + 10 Covariates)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  Combined beeswarm plot created successfully!")

# ==============================================================================
# CREATE VISUALIZATIONS
# ==============================================================================

print("\n" + "="*70)
print("Creating Comprehensive Visualizations")
print("="*70)

# 1. Model Comparison
if all_results_optimized:
    plot_model_comparison(all_results_optimized, 
                         os.path.join(OUT_DIR, 'model_comparison.png'))

# 2. Network Importance Detailed (bar plot)
network_names_list = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 
                       'Limbic', 'Control', 'Default']
if all_results_optimized:
    plot_network_importance_detailed(all_results_optimized, network_names_list,
                                    os.path.join(OUT_DIR, 'network_importance_detailed.png'))

# 3. SHAP Beeswarm Plots - Individual files for each test
if all_results_optimized:
    plot_shap_beeswarm_all_features(all_results_optimized, OUT_DIR)

# 4. SHAP Beeswarm Plots - Combined view
if all_results_optimized:
    plot_shap_combined_beeswarm(all_results_optimized, 
                               os.path.join(OUT_DIR, 'shap_beeswarm_combined.png'))

# 5. Results Summary Table
if all_results_optimized:
    df_summary = create_results_summary_table(all_results_optimized,
                                             os.path.join(OUT_DIR, 'results_summary.xlsx'))

# 6. Optimization Impact
if all_results_default and all_results_optimized:
    plot_optimization_impact(all_results_default, all_results_optimized,
                           os.path.join(OUT_DIR, 'optimization_impact.png'))

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

if all_results_optimized:
    for follow, results in all_results_optimized.items():
        print(f"\n{follow}:")
        print(f"  Best Model: {results['best_model']}")
        print(f"  Best Parameters: {results['best_params'].get(results['best_model'], 'default')}")
        print(f"  CV R²: {results['model_results'][results['best_model']]['cv_r2']:.3f}")
        print(f"  Test R²: {results['model_results'][results['best_model']]['test_r2']:.3f}")
        print(f"  Test RMSE: {results['model_results'][results['best_model']]['test_rmse']:.2f}")
        
        # Top features
        if results['shap_values'] is not None:
            importance = np.abs(results['shap_values']).mean(axis=0)
            feature_names = results['feature_names']
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            top_3 = importance_df.head(3)
            print("  Top 3 Features:")
            for _, row in top_3.iterrows():
                print(f"    - {row['feature']}: {row['importance']:.3f}")

    print(f"\n✅ All analyses complete!")
    print(f"📁 Results saved to: {OUT_DIR}")
else:
    print("\nNo results generated. Please check:")
    print("  1. ROI column names match those in your dataset")
    print("  2. Cognitive test columns exist in the dataset")
    print("  3. Covariate columns exist in the dataset")