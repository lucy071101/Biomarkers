import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# BASELINE CAROTID PLAQUE GROUP COMPARISON
# Age/Sex Adjusted, TIV Normalized
# ================================================================================

print("="*80)
print("BASELINE CAROTID PLAQUE GROUP COMPARISON")
print("Criteria: p-value < 0.05 AND |Cohen's d| > 0.2")
print("="*80)

# Load data
df = pd.read_csv(r'E:\Gisung\2025년연구\carotidplaque연구\250826\250826_data1.csv')

# Define variables
roi_cols = df.filter(regex=r"^AG1_(l|r)").columns.tolist()
adjustment_vars = ['AG1_AGE', 'AG1_SEX']

# Clean data
required_cols = ['AG1_plaque', 'AG1_TIV'] + roi_cols + adjustment_vars
df_clean = df[required_cols].dropna()

print(f"\nDataset: {df_clean.shape[0]} subjects, {len(roi_cols)} ROIs")
print(f"Groups: No plaque (n={sum(df_clean['AG1_plaque']==0)}), "
      f"Plaque (n={sum(df_clean['AG1_plaque']==1)})")

# TIV normalization
for roi in roi_cols:
    df_clean[f'{roi}_norm'] = df_clean[roi] / df_clean['AG1_TIV']

roi_normalized_cols = [f'{roi}_norm' for roi in roi_cols]

# ================================================================================
# AGE/SEX ADJUSTED COMPARISON
# ================================================================================

print("\n" + "="*60)
print("ANALYZING ALL 600 ROIs WITH AGE/SEX ADJUSTMENT")
print("="*60)

results = []

for roi in roi_normalized_cols:
    # ANCOVA: ROI ~ Plaque + Age + Sex
    formula = f'{roi} ~ C(AG1_plaque) + AG1_AGE + AG1_SEX'
    model = smf.ols(formula, data=df_clean).fit()
    
    # Get adjusted p-value and coefficient for plaque
    plaque_pval = model.pvalues['C(AG1_plaque)[T.1]']
    plaque_coef = model.params['C(AG1_plaque)[T.1]']
    
    # Calculate Cohen's d using residual standard error
    residual_se = np.sqrt(model.mse_resid)
    cohens_d = plaque_coef / residual_se if residual_se > 0 else 0
    
    # Determine significance
    is_significant = (plaque_pval < 0.05) and (abs(cohens_d) > 0.2)
    
    results.append({
        'ROI': roi.replace('_norm', '').replace('AG1_', ''),
        'P_value': plaque_pval,
        'Cohens_d': cohens_d,
        'Direction': 'Increased' if cohens_d > 0 else 'Decreased',
        'Significant': 'SIGNIFICANT' if is_significant else 'Non-significant'
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('P_value')

# ================================================================================
# SUMMARY STATISTICS
# ================================================================================

n_total = len(results_df)
n_sig = sum(results_df['Significant'] == 'SIGNIFICANT')
n_p_only = sum(results_df['P_value'] < 0.05)
n_d_only = sum(abs(results_df['Cohens_d']) > 0.2)
n_increased = sum((results_df['Significant'] == 'SIGNIFICANT') & 
                  (results_df['Direction'] == 'Increased'))
n_decreased = sum((results_df['Significant'] == 'SIGNIFICANT') & 
                  (results_df['Direction'] == 'Decreased'))

print(f"\n📊 RESULTS SUMMARY:")
print(f"  Total ROIs analyzed: {n_total}")
print(f"  ROIs with p < 0.05: {n_p_only} ({n_p_only/n_total*100:.1f}%)")
print(f"  ROIs with |d| > 0.2: {n_d_only} ({n_d_only/n_total*100:.1f}%)")
print(f"  SIGNIFICANT (p<0.05 AND |d|>0.2): {n_sig} ({n_sig/n_total*100:.1f}%)")
print(f"    - Increased in plaque: {n_increased}")
print(f"    - Decreased in plaque: {n_decreased}")
print(f"  Non-significant: {n_total - n_sig} ({(n_total-n_sig)/n_total*100:.1f}%)")

# ================================================================================
# DISPLAY SIGNIFICANT ROIs
# ================================================================================

print(f"\n🔍 SIGNIFICANT ROIs (p<0.05 AND |d|>0.2):")
print("-" * 60)

sig_rois = results_df[results_df['Significant'] == 'SIGNIFICANT']

if len(sig_rois) > 0:
    for idx, row in sig_rois.iterrows():
        direction_symbol = "↑" if row['Direction'] == 'Increased' else "↓"
        print(f"  {row['ROI']:30} | p={row['P_value']:.4f} | d={row['Cohens_d']:+.3f} {direction_symbol}")
else:
    print("  No ROIs meet both criteria")

# ================================================================================
# VISUALIZATION
# ================================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Volcano plot
ax1 = axes[0, 0]
colors = ['red' if s == 'SIGNIFICANT' else 'gray' 
          for s in results_df['Significant']]
sizes = [50 if s == 'SIGNIFICANT' else 10 
         for s in results_df['Significant']]

ax1.scatter(results_df['Cohens_d'], -np.log10(results_df['P_value']), 
           c=colors, s=sizes, alpha=0.6)
ax1.axhline(-np.log10(0.05), color='blue', linestyle='--', alpha=0.5, label='p=0.05')
ax1.axvline(0.2, color='green', linestyle='--', alpha=0.5, label='|d|=0.2')
ax1.axvline(-0.2, color='green', linestyle='--', alpha=0.5)
ax1.set_xlabel("Cohen's d")
ax1.set_ylabel('-log10(p-value)')
ax1.set_title(f'Volcano Plot (n={n_sig} significant)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. P-value distribution
ax2 = axes[0, 1]
ax2.hist(results_df['P_value'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(0.05, color='red', linestyle='--', label='p=0.05')
ax2.set_xlabel('P-value')
ax2.set_ylabel('Count')
ax2.set_title(f'P-value Distribution ({n_p_only}/{n_total} < 0.05)')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Effect size distribution
ax3 = axes[0, 2]
ax3.hist(results_df['Cohens_d'], bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(0.2, color='green', linestyle='--', alpha=0.5, label='|d|=0.2')
ax3.axvline(-0.2, color='green', linestyle='--', alpha=0.5)
ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
ax3.set_xlabel("Cohen's d")
ax3.set_ylabel('Count')
ax3.set_title(f"Effect Size Distribution ({n_d_only}/{n_total} |d|>0.2)")
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Significant ROIs bar plot
ax4 = axes[1, 0]
if len(sig_rois) > 0:
    top_20 = sig_rois.head(20)
    colors_bar = ['#E74C3C' if d > 0 else '#3498DB' 
                  for d in top_20['Cohens_d']]
    y_pos = range(len(top_20))
    
    ax4.barh(y_pos, top_20['Cohens_d'], color=colors_bar, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([r[:20] for r in top_20['ROI']], fontsize=8)
    ax4.set_xlabel("Cohen's d")
    ax4.set_title(f'Top {min(20, len(sig_rois))} Significant ROIs')
    ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax4.invert_yaxis()
else:
    ax4.text(0.5, 0.5, 'No significant ROIs', 
             ha='center', va='center', transform=ax4.transAxes)
ax4.grid(alpha=0.3)

# 5. Classification summary
ax5 = axes[1, 1]
categories = ['Significant\n(p<0.05 & |d|>0.2)', 'Non-significant']
counts = [n_sig, n_total - n_sig]
colors_pie = ['#E74C3C', '#95A5A6']

ax5.pie(counts, labels=categories, colors=colors_pie, autopct='%1.1f%%',
        startangle=90)
ax5.set_title('ROI Classification')

# 6. Summary table
ax6 = axes[1, 2]
ax6.axis('tight')
ax6.axis('off')

summary_data = [
    ['Total ROIs', str(n_total)],
    ['p < 0.05', f'{n_p_only} ({n_p_only/n_total*100:.1f}%)'],
    ['|d| > 0.2', f'{n_d_only} ({n_d_only/n_total*100:.1f}%)'],
    ['SIGNIFICANT', f'{n_sig} ({n_sig/n_total*100:.1f}%)'],
    ['  - Increased', f'{n_increased}'],
    ['  - Decreased', f'{n_decreased}'],
    ['Non-significant', f'{n_total-n_sig} ({(n_total-n_sig)/n_total*100:.1f}%)']
]

table = ax6.table(cellText=summary_data,
                  colLabels=['Metric', 'Value'],
                  cellLoc='center', loc='center',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Color code the table
for i in range(len(summary_data) + 1):
    if i == 0:  # Header
        table[(i, 0)].set_facecolor('#34495E')
        table[(i, 1)].set_facecolor('#34495E')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    elif i == 4:  # Significant row
        table[(i, 0)].set_facecolor('#FADBD8')
        table[(i, 1)].set_facecolor('#FADBD8')

plt.suptitle('Baseline Carotid Plaque Group Comparison (Age/Sex Adjusted, TIV Normalized)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('baseline_plaque_roi_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================================================================
# SAVE RESULTS
# ================================================================================

# Save all results
results_df.to_csv('all_roi_results.csv', index=False)

# Save only significant ROIs
sig_rois.to_csv('significant_rois.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print(f"Files saved: all_roi_results.csv, significant_rois.csv")
print("="*80)