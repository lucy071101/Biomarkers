import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import os

# 데이터 준비
data = """Variable	Effect	AG1_cplaque	time	Estimate	Lower	Upper
LMI	AG1_plaque*time	1	1	22.138	20.722	23.554
LMI	AG1_plaque*time	1	2	22.373	21.035	23.711
LMI	AG1_plaque*time	1	3	23.485	22.136	24.834
LMI	AG1_plaque*time	0	1	21.560	20.554	22.566
LMI	AG1_plaque*time	0	2	23.131	22.140	24.123
LMI	AG1_plaque*time	0	3	24.725	23.733	25.718
LMD	AG1_plaque*time	1	1	20.846	19.446	22.245
LMD	AG1_plaque*time	1	2	22.501	21.154	23.848
LMD	AG1_plaque*time	1	3	22.806	21.457	24.155
LMD	AG1_plaque*time	0	1	21.472	20.578	22.367
LMD	AG1_plaque*time	0	2	22.373	21.488	23.258
LMD	AG1_plaque*time	0	3	24.528	23.644	25.412
LMR	AG1_plaque*time	1	1	7.863	7.662	8.065
LMR	AG1_plaque*time	1	2	8.094	7.911	8.276
LMR	AG1_plaque*time	1	3	8.282	8.099	8.465
LMR	AG1_plaque*time	0	1	8.012	7.889	8.134
LMR	AG1_plaque*time	0	2	8.089	7.971	8.208
LMR	AG1_plaque*time	0	3	8.410	8.292	8.528
VRI	AG1_plaque*time	1	1	7.198	6.777	7.620
VRI	AG1_plaque*time	1	2	7.027	6.638	7.416
VRI	AG1_plaque*time	1	3	6.570	6.175	6.965
VRI	AG1_plaque*time	0	1	7.124	6.830	7.419
VRI	AG1_plaque*time	0	2	7.066	6.778	7.355
VRI	AG1_plaque*time	0	3	6.764	6.475	7.054
VRD	AG1_plaque*time	1	1	6.240	5.790	6.691
VRD	AG1_plaque*time	1	2	6.378	5.955	6.802
VRD	AG1_plaque*time	1	3	5.949	5.533	6.365
VRD	AG1_plaque*time	0	1	6.526	6.245	6.806
VRD	AG1_plaque*time	0	2	6.396	6.121	6.672
VRD	AG1_plaque*time	0	3	6.397	6.124	6.670
VRR	AG1_plaque*time	1	1	2.319	2.158	2.480
VRR	AG1_plaque*time	1	2	2.440	2.292	2.588
VRR	AG1_plaque*time	1	3	2.500	2.349	2.651
VRR	AG1_plaque*time	0	1	2.460	2.362	2.558
VRR	AG1_plaque*time	0	2	2.444	2.349	2.540
VRR	AG1_plaque*time	0	3	2.565	2.469	2.661
VFLET	AG1_plaque*time	1	1	22.031	20.544	23.519
VFLET	AG1_plaque*time	1	2	22.101	20.673	23.530
VFLET	AG1_plaque*time	1	3	22.848	21.394	24.302
VFLET	AG1_plaque*time	0	1	22.624	21.540	23.708
VFLET	AG1_plaque*time	0	2	22.731	21.658	23.805
VFLET	AG1_plaque*time	0	3	23.422	22.344	24.500
VFAN	AG1_plaque*time	1	1	14.023	13.438	14.608
VFAN	AG1_plaque*time	1	2	13.494	12.945	14.042
VFAN	AG1_plaque*time	1	3	13.053	12.495	13.611
VFAN	AG1_plaque*time	0	1	14.091	13.683	14.500
VFAN	AG1_plaque*time	0	2	13.726	13.324	14.127
VFAN	AG1_plaque*time	0	3	13.415	13.012	13.817
TMA	AG1_plaque*time	1	1	40.832	38.656	43.008
TMA	AG1_plaque*time	1	2	43.981	41.802	46.161
TMA	AG1_plaque*time	1	3	44.785	42.477	47.093
TMA	AG1_plaque*time	0	1	40.563	38.974	42.152
TMA	AG1_plaque*time	0	2	43.482	41.886	45.077
TMA	AG1_plaque*time	0	3	43.641	42.022	45.259
DS	AG1_plaque*time	1	1	50.946	48.784	53.109
DS	AG1_plaque*time	1	2	48.981	46.867	51.096
DS	AG1_plaque*time	1	3	47.259	45.142	49.375
DS	AG1_plaque*time	0	1	51.133	49.496	52.769
DS	AG1_plaque*time	0	2	49.830	48.202	51.459
DS	AG1_plaque*time	0	3	47.846	46.218	49.474
STR1	AG1_plaque*time	1	1	103.120	100.340	105.910
STR1	AG1_plaque*time	1	2	103.160	100.460	105.870
STR1	AG1_plaque*time	1	3	96.185	93.583	98.787
STR1	AG1_plaque*time	0	1	105.040	103.280	106.800
STR1	AG1_plaque*time	0	2	101.050	99.302	102.800
STR1	AG1_plaque*time	0	3	98.854	97.131	100.580
STR2	AG1_plaque*time	1	1	49.437	47.654	51.220
STR2	AG1_plaque*time	1	2	49.846	47.983	51.709
STR2	AG1_plaque*time	1	3	44.992	43.240	46.743
STR2	AG1_plaque*time	0	1	50.048	48.869	51.228
STR2	AG1_plaque*time	0	2	48.309	47.104	49.513
STR2	AG1_plaque*time	0	3	48.761	47.589	49.933"""

# DataFrame 생성
df = pd.read_csv(StringIO(data), sep='\t')

# 플롯 설정
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14

# 색상 설정
color_negative = '#2E8B57'  # 초록색 (plaque-negative)
color_positive = '#DC143C'  # 빨간색 (plaque-positive)

# 시간 포인트 설정
time_points = [1, 2, 3]
time_labels = ['Baseline', '4-year', '8-year']

# 변수별 제목 매핑
variable_titles = {
    'LMI': 'LMI: LSMEAN over time by carotid plaque status',
    'LMD': 'LMD: LSMEAN over time by carotid plaque status',
    'LMR': 'LMR: LSMEAN over time by carotid plaque status',
    'VRI': 'VRI: LSMEAN over time by carotid plaque status',
    'VRD': 'VRD: LSMEAN over time by carotid plaque status',
    'VRR': 'VRR: LSMEAN over time by carotid plaque status',
    'VFLET': 'VFLET: LSMEAN over time by carotid plaque status',
    'VFAN': 'VFAN: LSMEAN over time by carotid plaque status',
    'TMA': 'TMA: LSMEAN over time by carotid plaque status',
    'DS': 'DS1: LSMEAN over time by carotid plaque status',
    'STR1': 'STR1: LSMEAN over time by carotid plaque status',
    'STR2': 'STR2: LSMEAN over time by carotid plaque status'
}

# P-value 데이터
p_values = {
    'LMI': ['ns', 'ns', '*'],
    'LMD': ['ns', 'ns', '**'],
    'LMR': ['ns', 'ns', 'ns'],
    'VRI': ['ns', 'ns', 'ns'],
    'VRD': ['ns', 'ns', '*'],
    'VRR': ['ns', 'ns', 'ns'],
    'VFLET': ['ns', 'ns', 'ns'],
    'VFAN': ['ns', 'ns', 'ns'],
    'TMA': ['ns', 'ns', 'ns'],
    'DS': ['ns', 'ns', 'ns'],
    'STR1': ['ns', 'ns', '*'],
    'STR2': ['ns', 'ns', '***']
}

# Y축 범위 조정 함수
def calculate_y_range(var_data, variable, p_vals):
    """
    변수별로 적절한 y축 범위를 계산
    ns인 경우 범위를 넓게, 유의한 차이가 있는 경우 좁게 설정
    """
    all_lowers = var_data['Lower'].values
    all_uppers = var_data['Upper'].values
    all_estimates = var_data['Estimate'].values
    
    # 기본 범위 계산
    data_min = np.min(all_lowers)
    data_max = np.max(all_uppers)
    data_range = data_max - data_min
    
    # p-value에 따른 margin 조정
    # 모든 시점에서 ns인 경우 margin을 크게 설정
    if all(p == 'ns' for p in p_vals):
        # ns인 경우: 데이터 범위의 50-80% 추가 마진
        margin_factor = 0.6
    elif any(p != 'ns' for p in p_vals):
        # 일부 유의한 경우: 데이터 범위의 20-30% 추가 마진
        margin_factor = 0.25
    else:
        margin_factor = 0.3
    
    # 변수별 특별 조정 (필요시)
    special_adjustments = {
        'VRR': 0.8,  # VRR은 범위가 매우 작으므로 더 큰 마진
        'LMR': 0.5,   # LMR도 범위가 작음
        'STR1': 0.15,  # STR1은 값이 크므로 작은 마진
        'STR2': 0.2,
        'DS': 0.2,
        'TMA': 0.2
    }
    
    if variable in special_adjustments:
        margin_factor = special_adjustments[variable]
    
    margin = data_range * margin_factor
    
    # P-value 표시를 위한 상단 여백 추가
    top_margin_extra = data_range * 0.15 if any(p != 'ns' for p in p_vals) else data_range * 0.1
    
    y_min = data_min - margin
    y_max = data_max + margin + top_margin_extra
    
    return y_min, y_max

# 각 변수별로 개별 플롯 생성
for variable in df['Variable'].unique():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 데이터 필터링
    var_data = df[df['Variable'] == variable]
    
    # Plaque-negative (0) 데이터
    negative_data = var_data[var_data['AG1_cplaque'] == 0].sort_values('time')
    neg_means = negative_data['Estimate'].values
    neg_lower = negative_data['Lower'].values
    neg_upper = negative_data['Upper'].values
    
    # Plaque-positive (1) 데이터
    positive_data = var_data[var_data['AG1_cplaque'] == 1].sort_values('time')
    pos_means = positive_data['Estimate'].values
    pos_lower = positive_data['Lower'].values
    pos_upper = positive_data['Upper'].values
    
    # Y축 범위 계산 및 설정
    p_vals = p_values.get(variable, ['ns', 'ns', 'ns'])
    y_min, y_max = calculate_y_range(var_data, variable, p_vals)
    ax.set_ylim(y_min, y_max)
    
    # 신뢰구간 음영 (plaque-negative)
    ax.fill_between(time_points, neg_lower, neg_upper, 
                    color=color_negative, alpha=0.2, linewidth=0)
    
    # 신뢰구간 음영 (plaque-positive)
    ax.fill_between(time_points, pos_lower, pos_upper, 
                    color=color_positive, alpha=0.2, linewidth=0)
    
    # 중간 겹치는 영역 (더 진한 색)
    lower_bound = np.maximum(neg_lower, pos_lower)
    upper_bound = np.minimum(neg_upper, pos_upper)
    mask = lower_bound < upper_bound
    if mask.any():
        ax.fill_between(np.array(time_points)[mask], 
                       lower_bound[mask], upper_bound[mask],
                       color='#8B7355', alpha=0.3, linewidth=0)
    
    # 평균선 그리기
    ax.plot(time_points, neg_means, 'o-', color=color_negative, 
           linewidth=2.5, markersize=8, label='plaque-negative', zorder=5)
    ax.plot(time_points, pos_means, 'o-', color=color_positive,
           linewidth=2.5, markersize=8, label='plaque-positive', zorder=5)
    
    # P-value 표시
    if variable in p_values:
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        
        for i, (t, p_val) in enumerate(zip(time_points, p_values[variable])):
            # CI 상단에서 약간 위에 표시
            y_pos = max(neg_upper[i], pos_upper[i]) + y_range * 0.03
            ax.text(t, y_pos, p_val, ha='center', va='bottom', fontsize=10)
    
    # 축 설정
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel('')
    ax.set_ylabel('Estimate (LSMEAN with 95% CI) score')
    
    # 제목
    ax.set_title(variable_titles.get(variable, f'{variable}: LSMEAN over time by carotid plaque status'))
    
    # 범례
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
             edgecolor='none', framealpha=0.9)
    
    # 그리드 (y축만)
    ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 배경색
    ax.set_facecolor('#F8F8F8')
    
    # 스파인 제거 (상단, 우측)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 파일 저장
    filename = f'{variable}_lsmean_plot_adjusted.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    
    plt.show()

# Y축 범위 정보 출력
print("\n=== Y-axis Range Information ===")
for variable in df['Variable'].unique():
    var_data = df[df['Variable'] == variable]
    p_vals = p_values.get(variable, ['ns', 'ns', 'ns'])
    y_min, y_max = calculate_y_range(var_data, variable, p_vals)
    
    all_estimates = var_data['Estimate'].values
    data_min = np.min(var_data['Lower'].values)
    data_max = np.max(var_data['Upper'].values)
    
    print(f"\n{variable}:")
    print(f"  Data range: [{data_min:.2f}, {data_max:.2f}]")
    print(f"  Plot range: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  P-values: {p_vals}")
    print(f"  Range expansion: {((y_max - y_min) / (data_max - data_min) - 1) * 100:.1f}%")

print("\n모든 플롯이 성공적으로 생성되었습니다!")