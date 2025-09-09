# -*- coding: utf-8 -*-
"""
Publication figure: Multicollinearity check with 51 ROIs (Adjusted)
- Panel A: 51x51 correlation matrix (TIV-normalized + covariate-adjusted residuals)
- Panel B: VIF distribution for the same 51 ROIs
- Times New Roman, font size 10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# =========================
# Paths & basic settings
# =========================
DATA_PATH = r"E:\Gisung\2025년연구\carotidplaque연구\250828\250828_data.csv"

# -------------------------
# Fonts: Times New Roman (fallback: serif)
# -------------------------
available_fonts = [f.name for f in fm.fontManager.ttflist]
for candidate in ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'serif']:
    if candidate in available_fonts or candidate == 'serif':
        plt.rcParams['font.family'] = candidate
        break
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.unicode_minus'] = False

# =========================
# ROI list (51개)
# =========================
ROI_NAMES_RAW = """
AG1_rVisCent_ExStr_6
AG1_lDefaultA_PFCm_2
AG1_lLimbicB_OFC_8
AG1_rVisCent_ExStr_8
AG1_lDorsAttnA_SPL_4
AG1_rDorsAttnA_TempOcc_3
AG1_rLimbicB_OFC_10
AG1_rSalVentAttnA_FrMed_3
AG1_rVisCent_Striate_1
AG1_lSomMotA_20
AG1_rContB_PFCld_7
AG1_lSalVentAttnA_FrOper_1
AG1_lSalVentAttnA_FrMed_3
AG1_lDefaultB_PFCl_1
AG1_rVisCent_ExStr_3
AG1_lSalVentAttnA_Ins_6
AG1_lContB_PFCl_2
AG1_lSalVentAttnA_Ins_3
AG1_rVisCent_ExStr_5
AG1_lDorsAttnB_FEF_2
AG1_lDefaultB_Temp_2
AG1_rVisCent_ExStr_7
AG1_lDefaultA_PFCm_6
AG1_rVisPeri_ExStrSup_1
AG1_lSalVentAttnB_Ins_4
AG1_rVisPeri_ExStrInf_1
AG1_lSalVentAttnA_FrOper_2
AG1_lDefaultC_Rsp_2
AG1_lContA_PFCl_3
AG1_rSalVentAttnA_PrC_1
AG1_lContC_pCun_2
AG1_rVisCent_ExStr_2
AG1_lDefaultB_Temp_3
AG1_rSomMotB_Cent_3
AG1_rLimbicB_OFC_3
AG1_lVisPeri_ExStrSup_3
AG1_lSomMotB_Cent_8
AG1_rVisCent_ExStr_4
AG1_rSomMotB_Cent_6
AG1_rVisCent_ExStr_13
AG1_lSalVentAttnA_ParOper_2
AG1_lDorsAttnB_FEF_3
AG1_lDorsAttnA_SPL_10
AG1_lVisCent_ExStr_3
AG1_rSalVentAttnB_Ins_1
AG1_lDefaultC_PHC_2
AG1_rDefaultA_PFCm_3
AG1_lDefaultB_PFCv_4
AG1_lDefaultA_IPL_1
AG1_lDefaultB_PFCv_6
AG1_rDefaultB_AntTemp_3
""".strip().splitlines()

SIGNIFICANT_ROIS = [x.strip() for x in ROI_NAMES_RAW if x.strip()]
M = len(SIGNIFICANT_ROIS)
print(f"[INFO] ROI count = {M}")

# =========================
# Covariates (for adjustment)
# =========================
COVARIATES = ['AG1_AGE','AG1_SEX','AG1_EDU','AG1_BMI',
              'ag1_smoke','ag1_drink','AG1_HTN','AG1_DM','AG1_CVD','AG1_BDISUM']

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

# Subset & drop NAs needed for adjustment
need_cols = ['AG1_TIV'] + SIGNIFICANT_ROIS + COVARIATES
df_sub = df[need_cols].apply(pd.to_numeric, errors='coerce').dropna().copy()
n_subjects = len(df_sub)
print(f"[INFO] Subjects used = {n_subjects}")

# =========================
# TIV normalization
# =========================
roi_tiv = pd.DataFrame(index=df_sub.index)
for c in SIGNIFICANT_ROIS:
    roi_tiv[c] = df_sub[c] / df_sub['AG1_TIV']

# =========================
# Covariate adjustment (residualization)
# =========================
X = df_sub[COVARIATES].to_numpy(dtype=float)
X = np.column_stack([np.ones(n_subjects), X])  # add intercept

def residualize_matrix(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta

Y = roi_tiv.to_numpy(dtype=float)  # shape (n, M)
roi_adj = pd.DataFrame(residualize_matrix(Y, X),
                       index=roi_tiv.index, columns=roi_tiv.columns)

# =========================
# Figure: 1x2 panels (A & B)
# =========================
fig = plt.figure(figsize=(12, 5.8))
gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.96, top=0.92, bottom=0.15)

# -------- Panel A: Correlation matrix --------
axA = fig.add_subplot(gs[0, 0])
corr_mat = roi_adj.corr()
im = axA.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
axA.set_title(f'A. ROI Correlations ({M}×{M}, adjusted)', fontsize=11, fontweight='bold', pad=8)
axA.set_xlabel(f'{M} ROIs', fontsize=10)
axA.set_ylabel(f'{M} ROIs', fontsize=10)
axA.set_xticks([]); axA.set_yticks([])

vals = corr_mat.values[np.tril_indices(M, k=-1)]
txt = (f'n = {len(vals):,} pairs\n'
       f'Mean |r| = {np.mean(np.abs(vals)):.3f}\n'
       f'|r| > 0.7: {np.sum(np.abs(vals) > 0.7)} pairs')
axA.text(0.02, 0.98, txt, transform=axA.transAxes,
         ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
         fontsize=8)
cbar = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)
cbar.set_label('Correlation coefficient', fontsize=10)

# -------- Panel B: VIF distribution --------
axB = fig.add_subplot(gs[0, 1])
scaler = StandardScaler()
X_roi = scaler.fit_transform(roi_adj.to_numpy(dtype=float))

vif_vals = []
for i in range(M):
    try:
        v = variance_inflation_factor(X_roi, i)
        if np.isnan(v) or np.isinf(v): v = 100.0
    except Exception:
        v = 1.0
    vif_vals.append(min(v, 100.0))

max_vif_vis = min(max(vif_vals) * 1.1, 20)
bins = np.linspace(0, max_vif_vis, 20)
counts, edges, patches = axB.hist(vif_vals, bins=bins, color='#95A5A6',
                                  edgecolor='black', linewidth=1.2, alpha=0.85)

for i, patch in enumerate(patches):
    if edges[i] >= 10:
        patch.set_facecolor('#E74C3C')   # red > 10
    elif edges[i] >= 5:
        patch.set_facecolor('#F39C12')   # orange 5-10
    else:
        patch.set_facecolor('#2ECC71')   # green < 5

axB.axvline(5, color='#F39C12', linestyle='--', linewidth=2, alpha=0.7, label='VIF = 5')
axB.axvline(10, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, label='VIF = 10')
axB.set_title(f'B. VIF Distribution – {M} ROIs (adjusted)', fontsize=11, fontweight='bold', pad=8)
axB.set_xlabel('Variance Inflation Factor', fontsize=10)
axB.set_ylabel('Number of ROIs', fontsize=10)
axB.legend(loc='upper right', fontsize=9, framealpha=0.9)
axB.grid(True, axis='y', alpha=0.3)

txtB = (f'Mean = {np.mean(vif_vals):.2f}\n'
        f'Max = {np.max(vif_vals):.2f}\n'
        f'VIF > 5: {np.sum(np.array(vif_vals) > 5)} ROIs\n'
        f'VIF > 10: {np.sum(np.array(vif_vals) > 10)} ROIs')
axB.text(0.98, 0.98, txtB, transform=axB.transAxes,
         ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
         fontsize=8)

# footer: sample size
fig.text(0.98, 0.06, f'n = {n_subjects} subjects', ha='right', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(f'Figure_Multicollinearity_{M}ROI.png', dpi=300, bbox_inches='tight')
plt.savefig(f'Figure_Multicollinearity_{M}ROI.pdf', bbox_inches='tight')
plt.show()

# =========================
# Console summary
# =========================
print("\n" + "="*80)
print(f"MULTICOLLINEARITY ({M} ROI, adjusted) — SUMMARY")
print("="*80)
print(f"Subjects used: {n_subjects}")
print(f"Correlation pairs: {len(vals):,}")
print(f"Mean |r|: {np.mean(np.abs(vals)):.3f}")
print(f"|r|>0.7 pairs: {np.sum(np.abs(vals)>0.7)}")
print(f"VIF mean: {np.mean(vif_vals):.2f} | VIF max: {np.max(vif_vals):.2f}")
print(f"VIF>5: {np.sum(np.array(vif_vals)>5)} | VIF>10: {np.sum(np.array(vif_vals)>10)}")
print(f"Saved: Figure_Multicollinearity_{M}ROI.(png|pdf)")
