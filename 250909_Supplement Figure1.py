# -*- coding: utf-8 -*-
"""
Cross-sectional ROI–Cognition Association Analysis (Manuscript-ready)
- Partial correlation with covariate adjustment
- FDR correction
- CI + n reporting
- Heatmap + Barplot outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr
from statsmodels.stats.multitest import multipletests

# =============================
# 1) 데이터 로드
# =============================
df = pd.read_csv(r'E:\Gisung\2025년연구\carotidplaque연구\250826\250826_data1.csv')

significant_rois = [
    'AG1_rVisCent_ExStr_6',
    'AG1_rVisCent_ExStr_8',
    'AG1_rDorsAttnA_TempOcc_3',
    'AG1_lDefaultA_PFCm_2',
    'AG1_lLimbicB_OFC_8',
    'AG1_lDorsAttnA_SPL_4'
]
roi_display = ['rVisCent_6','rVisCent_8','rDorsAttn_3',
               'lDefault_2','lLimbic_8','lDorsAttn_4']
baseline_cognitive = ['LMIB', 'LMDB', 'VRDB', 'STR1B', 'STR2B']
full_covariates = ['AG1_AGE','AG1_SEX','AG1_EDU','AG1_BMI',
                   'ag1_smoke','ag1_drink','AG1_HTN','AG1_DM',
                   'AG1_CVD','AG1_BDISUM']

# =============================
# 2) 전처리
# =============================
all_cols = ['AG1_TIV'] + significant_rois + full_covariates + baseline_cognitive
df_clean = df[all_cols].dropna().copy()

# TIV 정규화
for roi in significant_rois:
    df_clean[f'{roi}_norm'] = df_clean[roi] / df_clean['AG1_TIV']
roi_normalized = [f'{roi}_norm' for roi in significant_rois]

# =============================
# 3) Partial correlation
# =============================

# CI 계산 함수
def fisher_ci(r, n, alpha=0.05):
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z_crit = 1.959964  # 95%
    lo = np.tanh(z - z_crit*se)
    hi = np.tanh(z + z_crit*se)
    return lo, hi

results = []
n = len(df_clean)
for roi_key, roi_name in zip(roi_normalized, roi_display):
    for cog in baseline_cognitive:
        res = partial_corr(data=df_clean, x=roi_key, y=cog,
                           covar=full_covariates, method='pearson')
        r = res['r'].values[0]
        p = res['p-val'].values[0]
        lo, hi = fisher_ci(r, n)
        results.append({
            'ROI': roi_name,
            'Cognitive': cog,
            'r': r,
            'p': p,
            'r_low': lo,
            'r_high': hi,
            'n': n
        })

corr_df = pd.DataFrame(results)

# FDR 보정
rej, p_adj, _, _ = multipletests(corr_df['p'], alpha=0.05, method='fdr_bh')
corr_df['p_adj'] = p_adj
corr_df['sig'] = corr_df['p_adj'] < 0.05

# 저장
corr_df[['ROI','Cognitive','r','r_low','r_high','p','p_adj','sig','n']].to_csv(
    'Table_Corr_Partial.csv', index=True)

# =============================
# 4) 히트맵
# =============================
corr_matrix = corr_df.pivot(index='ROI', columns='Cognitive', values='r')\
                     .reindex(index=roi_display, columns=baseline_cognitive)
p_matrix = corr_df.pivot(index='ROI', columns='Cognitive', values='p_adj')\
                  .reindex(index=roi_display, columns=baseline_cognitive)

labels = corr_matrix.copy()
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        r = corr_matrix.iloc[i, j]
        if pd.isna(r):
            labels.iloc[i, j] = ''
        else:
            star = '*' if (not pd.isna(p_matrix.iloc[i,j]) and p_matrix.iloc[i,j] < 0.05) else ''
            labels.iloc[i, j] = f'{r:.2f}{star}'

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

fig, ax = plt.subplots(figsize=(7,4.5))
sns.heatmap(corr_matrix, annot=labels, fmt='', cmap='RdBu_r', center=0,
            vmin=-0.15, vmax=0.15, cbar_kws={'label':'Partial r','shrink':0.8},
            linewidths=0.5, linecolor='gray', square=True, ax=ax)

ax.set_title('Adjusted Partial Correlations (FDR)', fontweight='bold')
ax.set_xlabel('Baseline Cognitive Domain')
ax.set_ylabel('Brain Region (TIV-normalized)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig('Figure_Corr_Heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig('Figure_Corr_Heatmap.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================
# 5) Barplots (옵션)
# =============================
for cog in baseline_cognitive:
    sub = corr_df[corr_df['Cognitive']==cog].set_index('ROI').loc[roi_display]
    colors = ['#999999' if not s else '#1f77b4' for s in sub['sig']]
    plt.figure(figsize=(6,3.5))
    bars = plt.bar(sub.index, sub['r'], color=colors, edgecolor='black')
    for idx, (val,sig) in enumerate(zip(sub['r'],sub['sig'])):
        label = f'{val:.2f}' + ('*' if sig else '')
        plt.text(idx, val + (0.005 if val>=0 else -0.005), label,
                 ha='center', va='bottom' if val>=0 else 'top', fontsize=9)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Partial r')
    plt.title(f'Partial Correlations for {cog} (FDR)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'Figure_Corr_Bars_{cog}.png', dpi=300, bbox_inches='tight')
    plt.close()
