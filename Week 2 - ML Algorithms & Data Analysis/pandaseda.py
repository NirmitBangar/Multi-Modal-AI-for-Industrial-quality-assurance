"""
Week 2: Pandas EDA for Industrial QA Data
==========================================
Exploratory Data Analysis (EDA) is the mandatory first step before any ML.
It reveals: class imbalance, missing values, outliers, feature distributions,
and correlations — all of which affect which models and preprocessing steps
are appropriate.

This file simulates a realistic industrial sensor log and performs full EDA.
Dataset structure (mimics a real factory log):
  - timestamp: when the reading was taken
  - station_id: which assembly station
  - shift: morning/afternoon/night
  - temperature, vibration, pressure, spindle_speed: sensor readings
  - product_type: categorical (A, B, C)
  - defect: 0 = normal, 1 = defective
  - defect_type: scratch, dent, stain, none
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# =============================================================================
# GENERATE SIMULATED INDUSTRIAL DATASET
# =============================================================================

def generate_industrial_log(n: int = 3000) -> pd.DataFrame:
    """Generate a realistic factory sensor log with intentional messiness."""
    timestamps = pd.date_range("2025-01-01", periods=n, freq="10min")
    stations = np.random.choice(['ST-01', 'ST-02', 'ST-03', 'ST-04'], n)
    shifts = np.where(pd.DatetimeIndex(timestamps).hour < 8, 'night',
                      np.where(pd.DatetimeIndex(timestamps).hour < 16, 'morning', 'afternoon'))
    product_types = np.random.choice(['TypeA', 'TypeB', 'TypeC'], n, p=[0.5, 0.3, 0.2])

    temperature   = np.random.normal(75, 10, n)
    vibration     = np.abs(np.random.normal(1.5, 0.8, n))
    pressure      = np.random.normal(9, 2, n)
    spindle_speed = np.random.normal(120, 15, n)

    # Inject station-specific biases (realistic — older machines run hotter)
    temp_bias = {'ST-01': 0, 'ST-02': 5, 'ST-03': -3, 'ST-04': 10}
    for st, bias in temp_bias.items():
        temperature[stations == st] += bias

    # Night shift has slightly higher vibration (fatigue, less maintenance)
    vibration[shifts == 'night'] *= 1.3

    # Compute defect probability (same non-linear model as before)
    logit = (-5 + 0.04*(temperature - 75) + 0.7*(vibration - 1.5)
             + 0.03*(temperature-75)*(vibration-1.5) + 0.08*(pressure-9))
    prob_defect = 1 / (1 + np.exp(-logit))
    defect = (np.random.rand(n) < prob_defect).astype(int)

    defect_types = np.where(defect == 0, 'none',
                            np.random.choice(['scratch', 'dent', 'stain', 'contamination'],
                                             n, p=[0.4, 0.3, 0.2, 0.1]))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'station_id': stations,
        'shift': shifts,
        'product_type': product_types,
        'temperature': temperature.round(2),
        'vibration': vibration.round(3),
        'pressure': pressure.round(2),
        'spindle_speed': spindle_speed.round(1),
        'defect': defect,
        'defect_type': defect_types,
    })

    # Inject realistic data quality issues
    # 1. Random missing values (~3% of sensor readings — sensor dropout)
    for col in ['temperature', 'vibration', 'pressure']:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    # 2. A few physically impossible values (firmware bugs → replace with NaN in cleaning)
    outlier_idx = np.random.choice(n, 15, replace=False)
    df.loc[outlier_idx[:5], 'temperature'] = -999     # Impossible sensor error
    df.loc[outlier_idx[5:10], 'vibration'] = 50.0     # Sensor overflow
    df.loc[outlier_idx[10:], 'spindle_speed'] = 9999  # Firmware glitch

    return df


df_raw = generate_industrial_log(3000)

# =============================================================================
# STEP 1: First Look
# =============================================================================

print("="*60)
print("STEP 1: First Look at the Dataset")
print("="*60)
print(f"\nShape: {df_raw.shape}")
print(f"\nFirst 5 rows:\n{df_raw.head()}")
print(f"\nData types:\n{df_raw.dtypes}")
print(f"\nBasic statistics:\n{df_raw.describe()}")


# =============================================================================
# STEP 2: Missing Values & Data Quality
# =============================================================================

print("\n" + "="*60)
print("STEP 2: Missing Values & Data Quality")
print("="*60)

null_summary = pd.DataFrame({
    'missing_count': df_raw.isnull().sum(),
    'missing_pct': (df_raw.isnull().sum() / len(df_raw) * 100).round(2),
    'dtype': df_raw.dtypes
}).query('missing_count > 0')
print(f"\nMissing value summary:\n{null_summary}")

# Detect physically impossible values (outliers that are data errors, not genuine extremes)
print("\nSuspiciously low temperatures (< 0°C):", (df_raw['temperature'] < 0).sum())
print("Suspiciously high vibration (> 20):", (df_raw['vibration'] > 20).sum())
print("Suspiciously high spindle speed (> 500):", (df_raw['spindle_speed'] > 500).sum())

# Data Cleaning
df = df_raw.copy()

# Replace impossible values with NaN
df.loc[df['temperature'] < 0, 'temperature'] = np.nan
df.loc[df['vibration'] > 20, 'vibration'] = np.nan
df.loc[df['spindle_speed'] > 500, 'spindle_speed'] = np.nan

# For time-series sensor data: forward fill is appropriate
# Rationale: the most recent valid reading is the best estimate of current state
df = df.set_index('timestamp').sort_index()
for col in ['temperature', 'vibration', 'pressure', 'spindle_speed']:
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

df = df.reset_index()
print(f"\nAfter cleaning — remaining nulls: {df.isnull().sum().sum()}")


# =============================================================================
# STEP 3: Target Variable Analysis — Class Imbalance
# =============================================================================

print("\n" + "="*60)
print("STEP 3: Target Variable Analysis")
print("="*60)

defect_counts = df['defect'].value_counts()
print(f"\nDefect distribution:\n{defect_counts}")
print(f"Defect rate: {df['defect'].mean()*100:.2f}%")
print("\nDefect TYPE breakdown:")
print(df[df['defect']==1]['defect_type'].value_counts())

# Key insight: If defects are rare (< 10%), we need class_weight='balanced'
# or oversampling (SMOTE) or undersampling to prevent the model from
# just predicting "normal" for everything and getting 90%+ accuracy trivially.


# =============================================================================
# STEP 4: Feature Distributions & Outlier Detection
# =============================================================================

print("\n" + "="*60)
print("STEP 4: Feature Distributions")
print("="*60)

numeric_cols = ['temperature', 'vibration', 'pressure', 'spindle_speed']
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

    skewness = df[col].skew()
    kurtosis = df[col].kurt()
    _, p_shapiro = stats.shapiro(df[col].dropna().sample(min(500, len(df)), random_state=42))

    print(f"\n{col}:")
    print(f"  Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, "
          f"Skew={skewness:.3f}, Kurt={kurtosis:.3f}")
    print(f"  Shapiro-Wilk p-value={p_shapiro:.4f} "
          f"({'Normal' if p_shapiro > 0.05 else 'NOT normal'} distribution)")
    print(f"  IQR outliers: {outlier_count} ({100*outlier_count/len(df):.2f}%)")


# =============================================================================
# STEP 5: Feature–Target Relationship
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Feature-Target Relationships")
print("="*60)

for col in numeric_cols:
    normal_vals   = df[df['defect'] == 0][col]
    defective_vals = df[df['defect'] == 1][col]
    t_stat, p_val = stats.ttest_ind(normal_vals, defective_vals)
    print(f"\n{col}:")
    print(f"  Normal:    mean={normal_vals.mean():.3f}, std={normal_vals.std():.3f}")
    print(f"  Defective: mean={defective_vals.mean():.3f}, std={defective_vals.std():.3f}")
    print(f"  t-test p-value: {p_val:.6f} → "
          f"{'SIGNIFICANT' if p_val < 0.05 else 'not significant'} difference")


# =============================================================================
# STEP 6: GroupBy Analysis — Defect Rate by Operational Context
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Defect Rate by Station & Shift")
print("="*60)

pivot = df.pivot_table(
    values='defect',
    index='station_id',
    columns='shift',
    aggfunc='mean'
).round(3) * 100

print("\nDefect Rate (%) by Station × Shift:")
print(pivot.to_string())
print("\n> If ST-04 night shift shows highest defect rate, it suggests")
print("> that Station 4 needs maintenance priority before the night shift.")

# By product type
print("\nDefect Rate by Product Type:")
print(df.groupby('product_type')['defect'].agg(['mean', 'count']).round(3))

# Correlation matrix
print("\nFeature Correlation Matrix:")
corr = df[numeric_cols + ['defect']].corr()
print(corr.round(3))


# =============================================================================
# STEP 7: Visualizations
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('EDA: Industrial QA Sensor Dataset', fontsize=14, fontweight='bold')

# 1. Defect rate distribution
ax = axes[0, 0]
df['defect'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], rot=0)
ax.set_title('Defect Distribution')
ax.set_xlabel('0=Normal, 1=Defective')
ax.set_ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=10)

# 2. Temperature distribution by defect status
ax = axes[0, 1]
df[df['defect']==0]['temperature'].hist(ax=ax, bins=40, alpha=0.6, color='#2ecc71', label='Normal')
df[df['defect']==1]['temperature'].hist(ax=ax, bins=40, alpha=0.6, color='#e74c3c', label='Defective')
ax.set_title('Temperature Distribution by Defect Status')
ax.set_xlabel('Temperature (°C)')
ax.legend()

# 3. Vibration distribution by defect status
ax = axes[0, 2]
df[df['defect']==0]['vibration'].hist(ax=ax, bins=40, alpha=0.6, color='#2ecc71', label='Normal')
df[df['defect']==1]['vibration'].hist(ax=ax, bins=40, alpha=0.6, color='#e74c3c', label='Defective')
ax.set_title('Vibration Distribution by Defect Status')
ax.set_xlabel('Vibration (mm/s)')
ax.legend()

# 4. Scatter plot: temperature vs vibration, coloured by defect
ax = axes[1, 0]
scatter_df = df.sample(500, random_state=42)
colors = scatter_df['defect'].map({0: '#2ecc71', 1: '#e74c3c'})
ax.scatter(scatter_df['temperature'], scatter_df['vibration'],
           c=colors, alpha=0.5, s=20)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Vibration (mm/s)')
ax.set_title('Temp vs Vibration (Green=Normal, Red=Defective)')
# This plot visually demonstrates the interaction: upper-right cluster is mostly red

# 5. Defect rate heatmap by station x shift
ax = axes[1, 1]
sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn_r',
            cbar_kws={'label': 'Defect Rate (%)'})
ax.set_title('Defect Rate (%) by Station × Shift')

# 6. Correlation heatmap
ax = axes[1, 2]
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nEDA visualizations saved to eda_plots.png")
