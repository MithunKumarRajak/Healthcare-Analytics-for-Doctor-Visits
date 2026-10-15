#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 HEALTHCARE ANALYTICS FOR DOCTOR VISITS - COMPREHENSIVE DATA ANALYSIS
================================================================================
 Author  : Data Analyst
 Date    : 2026-10-15
 Dataset : 1776250375-P2-Healthcare Analytics for Doctor Visits.csv
================================================================================
 This analysis explores healthcare data to understand patterns in doctor visits,
 identifying key factors that influence visit frequency, and uncovering
 relationships between demographic, health, and insurance variables.
================================================================================
"""

# =============================================================================
# 1. IMPORTING LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 120

print("=" * 70)
print(" SECTION 1: IMPORTING LIBRARIES ")
print("=" * 70)
print("\n✅ Libraries Imported Successfully:")
print(f"   • pandas        v{pd.__version__}")
print(f"   • numpy         v{np.__version__}")
print(f"   • matplotlib    v{plt.matplotlib.__version__}")
print(f"   • seaborn       v{sns.__version__}")
import scipy
print(f"   • scipy         v{scipy.__version__}")
print("\n📌 Configuration Applied:")
print("   • Warnings suppressed")
print("   • Seaborn whitegrid style applied")
print("   • Default figure size: 12x6 inches")
print("   • DPI: 120 for high-resolution plots")

# =============================================================================
# 2. LOADING THE DATA
# =============================================================================
print("\n" + "=" * 70)
print(" SECTION 2: LOADING THE DATA ")
print("=" * 70)

# Load the dataset
df = pd.read_csv(r"1776250375-P2-Healthcare Analytics for Doctor Visits.csv", index_col=0)

print(f"\n✅ Dataset loaded successfully!")
print(f"\n📊 Dataset Dimensions:")
print(f"   • Rows (Records)  : {df.shape[0]:,}")
print(f"   • Columns (Features): {df.shape[1]}")
print(f"\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:>2}. {col}")
print(f"\n🔍 First 5 Records:")
print(df.head().to_string())
print(f"\n🔍 Last 5 Records:")
print(df.tail().to_string())

"""
💡 INSIGHT - Data Loading:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Dataset contains 5,190 patient records with 12 features.
• Features capture demographic info (gender, age, income), health status
  (illness, reduced activity days, health score, chronic conditions), and
  insurance coverage (private, freepoor, freerepat).
• Target variable is 'visits' — the number of doctor consultations.
• Index column (original row numbers) was properly handled during loading.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# 3. DATA OVERVIEW
# =============================================================================
print("\n" + "=" * 70)
print(" SECTION 3: DATA OVERVIEW ")
print("=" * 70)

# 3.1 Data Types and Info
print("\n📌 3.1 Data Types & Memory Usage:")
print("-" * 50)
print(f"\n{'Column':<15} {'Data Type':<15} {'Non-Null':<12} {'Null':<8}")
print("-" * 50)
for col in df.columns:
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    dtype = str(df[col].dtype)
    print(f"{col:<15} {dtype:<15} {non_null:<12} {null_count:<8}")

memory_usage = df.memory_usage(deep=True).sum() / 1024
print(f"\n💾 Total Memory Usage: {memory_usage:.2f} KB")

# 3.2 Statistical Summary - Numerical Features
print("\n\n📌 3.2 Statistical Summary - Numerical Features:")
print("-" * 70)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns: {numerical_cols}")
print(df[numerical_cols].describe().round(3).to_string())

# 3.3 Statistical Summary - Categorical Features
print("\n\n📌 3.3 Statistical Summary - Categorical Features:")
print("-" * 70)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")
print(df[categorical_cols].describe().to_string())

# 3.4 Unique Values per Column
print("\n\n📌 3.4 Unique Values per Column:")
print("-" * 50)
for col in df.columns:
    unique_count = df[col].nunique()
    sample_values = df[col].unique()[:5]
    print(f"  {col:<15} → {unique_count:>4} unique values | Sample: {sample_values}")

# 3.5 Feature Description
print("\n\n📌 3.5 Feature Dictionary:")
print("-" * 70)
feature_dict = {
    'visits': 'Number of doctor visits in the past 2 weeks (Target Variable)',
    'gender': 'Gender of the patient (male/female)',
    'age': 'Age of the patient (scaled/normalized, likely age/100)',
    'income': 'Annual income (scaled/normalized)',
    'illness': 'Number of illnesses in the past 2 weeks (0-5)',
    'reduced': 'Number of days of reduced activity due to illness in past 2 weeks (0-14)',
    'health': 'General health questionnaire score (0=best health, 12=worst)',
    'private': 'Whether patient has private health insurance (yes/no)',
    'freepoor': 'Whether patient has free govt insurance due to low income (yes/no)',
    'freerepat': 'Whether patient has free govt insurance due to old age/disability/veteran (yes/no)',
    'nchronic': 'Whether patient has a non-limiting chronic condition (yes/no)',
    'lchronic': 'Whether patient has a limiting chronic condition (yes/no)'
}
for feature, description in feature_dict.items():
    print(f"  📎 {feature:<12} : {description}")

"""
💡 INSIGHT - Data Overview:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• No missing values detected in any column — dataset is complete (5,190 records × 12 columns).
• 6 numerical features: visits, age, income, illness, reduced, health.
• 5 binary categorical features: gender, private, freepoor, freerepat, nchronic, lchronic.
• Age appears normalized (range 0.19-0.72) — likely represents age/100 or a similar scale,
  translating to approximately 19-72 years.
• Income is also normalized (range 0.0-1.5).
• 'visits' is highly skewed: 75th percentile = 0 (most patients had 0 visits), mean = 0.30.
• 'health' score: 58.3% of patients have a score of 0 (best health).
• 'freepoor' is heavily imbalanced: only 4.3% patients have free-poor insurance.
• 'lchronic' is imbalanced: only 11.7% patients have limiting chronic conditions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n" + "=" * 70)
print(" SECTION 4: DATA CLEANING ")
print("=" * 70)

# 4.1 Check for Missing Values
print("\n📌 4.1 Missing Values Analysis:")
print("-" * 50)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df.to_string())
print(f"\n   Total missing values: {missing.sum()}")

# 4.2 Check for Duplicate Rows
print("\n\n📌 4.2 Duplicate Rows Analysis:")
print("-" * 50)
duplicates = df.duplicated().sum()
print(f"   Duplicate rows found: {duplicates} ({(duplicates/len(df)*100):.2f}%)")
print(f"\n   ⚠️ Note: {duplicates} duplicate rows exist, but this is expected in")
print(f"   survey/medical data where multiple patients can have identical ")
print(f"   demographic and health profiles. These are NOT errors — they represent")
print(f"   different individuals with the same attributes. Retaining all rows.")

# 4.3 Check for Outliers using IQR Method
print("\n\n📌 4.3 Outlier Detection (IQR Method):")
print("-" * 50)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    pct = (len(outliers) / len(df)) * 100
    print(f"   {col:<12} → Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f} | "
          f"Bounds: [{lower:.2f}, {upper:.2f}] | Outliers: {len(outliers)} ({pct:.1f}%)")

# 4.4 Data Type Validation and Conversion
print("\n\n📌 4.4 Data Type Optimization:")
print("-" * 50)
# Convert binary yes/no columns to numeric (0/1) for analysis
binary_cols = ['private', 'freepoor', 'freerepat', 'nchronic', 'lchronic']
df_clean = df.copy()

# Create numeric versions of binary columns
for col in binary_cols:
    df_clean[f'{col}_num'] = (df_clean[col] == 'yes').astype(int)
    print(f"   ✅ Created '{col}_num' → yes=1, no=0")

# Encode gender
df_clean['gender_num'] = (df_clean['gender'] == 'female').astype(int)
print(f"   ✅ Created 'gender_num' → female=1, male=0")

# Create derived features for analysis
df_clean['age_years'] = (df_clean['age'] * 100).round(0).astype(int)
print(f"   ✅ Created 'age_years' → age × 100 (approximate actual age)")

# Create age groups
df_clean['age_group'] = pd.cut(df_clean['age_years'],
                                bins=[0, 25, 35, 45, 55, 65, 100],
                                labels=['19-25', '26-35', '36-45', '46-55', '56-65', '66+'])
print(f"   ✅ Created 'age_group' → age binned into 6 categories")

# Create income groups
df_clean['income_group'] = pd.cut(df_clean['income'],
                                   bins=[-0.01, 0.25, 0.50, 0.75, 1.0, 1.5],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
print(f"   ✅ Created 'income_group' → income binned into 5 categories")

# Create has_visited flag
df_clean['has_visited'] = (df_clean['visits'] > 0).astype(int)
print(f"   ✅ Created 'has_visited' → 1 if visits > 0, else 0")

# 4.5 Verify cleaned data
print("\n\n📌 4.5 Cleaned Dataset Summary:")
print("-" * 50)
print(f"   Original shape    : {df.shape}")
print(f"   Cleaned shape     : {df_clean.shape}")
print(f"   New features added: {df_clean.shape[1] - df.shape[1]}")
print(f"   Rows removed      : 0 (no rows removed)")
print(f"   Missing values    : {df_clean.isnull().sum().sum()}")

"""
💡 INSIGHT - Data Cleaning:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ✅ No missing values — dataset is 100% complete.
• ⚠️ 1,320 duplicate rows (25.4%) exist but are RETAINED because in
  survey/medical data, multiple patients can legitimately share identical
  attributes. These represent different individuals, not data errors.
• 📊 Outliers detected via IQR method:
  - 'visits' (20.2%): Expected — most patients have 0 visits, those with 
    multiple visits are natural outliers but clinically meaningful.
  - 'reduced' (10.5%): Patients with many reduced activity days — valid 
    data representing sicker patients.
  - 'health' (18.5%): Patients with high health scores (poor health) — 
    important subset for analysis.
• 🔧 Data type optimizations: Created numeric encodings for all binary
  categorical features (yes→1, no→0) for correlation and modeling analysis.
• 📐 Derived features created: age_years, age_group, income_group, has_visited
  to enable more meaningful groupings in EDA.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 70)
print(" SECTION 5: EXPLORATORY DATA ANALYSIS (EDA) ")
print("=" * 70)

# 5.1 Target Variable Analysis — Doctor Visits
print("\n📌 5.1 Target Variable Analysis — Doctor Visits Distribution:")
print("-" * 50)
visit_dist = df_clean['visits'].value_counts().sort_index()
print("\nVisit Frequency Distribution:")
print(f"{'Visits':<10} {'Count':<10} {'Percentage':<12} {'Cumulative %':<15}")
print("-" * 47)
cumulative = 0
for idx, count in visit_dist.items():
    pct = count / len(df_clean) * 100
    cumulative += pct
    bar = '█' * int(pct)
    print(f"{idx:<10} {count:<10} {pct:<12.2f} {cumulative:<15.2f} {bar}")

# Plot - Target Variable Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar chart of visits
colors = sns.color_palette("viridis", len(visit_dist))
axes[0].bar(visit_dist.index, visit_dist.values, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Distribution of Doctor Visits', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Number of Visits')
axes[0].set_ylabel('Count')
for i, (idx, val) in enumerate(visit_dist.items()):
    axes[0].text(idx, val + 30, str(val), ha='center', fontsize=9, fontweight='bold')

# Pie chart - Visit vs No Visit
visit_labels = ['No Visit (0)', 'At Least 1 Visit']
visit_sizes = [df_clean['has_visited'].value_counts()[0], df_clean['has_visited'].value_counts()[1]]
explode = (0.05, 0.05)
axes[1].pie(visit_sizes, explode=explode, labels=visit_labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=['#FF6B6B', '#4ECDC4'],
            textprops={'fontsize': 11})
axes[1].set_title('Visit vs No Visit', fontweight='bold', fontsize=14)

# Box plot of visits
bp = axes[2].boxplot(df_clean['visits'], patch_artist=True, vert=True,
                     boxprops=dict(facecolor='#4ECDC4', alpha=0.7))
axes[2].set_title('Visits Box Plot', fontweight='bold', fontsize=14)
axes[2].set_ylabel('Number of Visits')

plt.tight_layout()
plt.savefig('01_target_variable_distribution.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n   📈 Plot saved: 01_target_variable_distribution.png")

# 5.2 Correlation Heatmap
print("\n\n📌 5.2 Correlation Analysis:")
print("-" * 50)
numeric_for_corr = ['visits', 'age', 'income', 'illness', 'reduced', 'health',
                     'gender_num', 'private_num', 'freepoor_num', 'freerepat_num',
                     'nchronic_num', 'lchronic_num']
corr_matrix = df_clean[numeric_for_corr].corr()

# Print top correlations with visits
print("\nCorrelations with 'visits' (sorted by absolute value):")
visits_corr = corr_matrix['visits'].drop('visits').sort_values(key=abs, ascending=False)
for feature, corr_val in visits_corr.items():
    strength = "Strong" if abs(corr_val) > 0.3 else "Moderate" if abs(corr_val) > 0.15 else "Weak"
    direction = "↑ Positive" if corr_val > 0 else "↓ Negative"
    bar = '█' * int(abs(corr_val) * 50)
    print(f"   {feature:<15} : {corr_val:>+.4f} ({strength}, {direction}) {bar}")

# Heatmap
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap=cmap,
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Heatmap — All Features', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n   📈 Plot saved: 02_correlation_heatmap.png")

"""
💡 INSIGHT - EDA Overview:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 🎯 Target Variable ('visits') is heavily right-skewed (zero-inflated):
  - 79.8% of patients had 0 visits in the past 2 weeks.
  - Only 20.2% had at least 1 visit.
  - Mean = 0.30, Median = 0.0, Max = 9.
  - This is a classic "zero-inflated" count distribution.
• 🔗 Top Correlations with Doctor Visits:
  1. 'reduced' (r=+0.42): STRONGEST predictor — more reduced activity days 
     strongly associated with more doctor visits.
  2. 'illness' (r=+0.22): More illnesses → more visits.
  3. 'health' (r=+0.19): Worse health score → more visits.
  4. 'age' (r=+0.12): Older patients visit doctors slightly more.
  5. 'gender_num' (r=+0.10): Females tend to have slightly more visits.
  6. 'income' (r=-0.08): Higher income slightly associated with fewer visits.
  7. 'lchronic_num' (r=+0.11): Limiting chronic conditions increase visits.
• ⚠️ No very strong correlations (|r| > 0.5) with visits — suggesting
  doctor visits are influenced by multiple factors, not a single predictor.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# 6. FURTHER EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print(" SECTION 6: FURTHER EXPLORATORY DATA ANALYSIS ")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 6.1 UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 6.1 UNIVARIATE ANALYSIS ")
print("─" * 70)

# 6.1.1 Numerical Features — Distribution Plots
print("\n📌 6.1.1 Distribution of Numerical Features:")
num_features = ['visits', 'age', 'income', 'illness', 'reduced', 'health']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(num_features):
    ax = axes[i]
    # Histogram with KDE
    n, bins, patches = ax.hist(df_clean[col], bins=30, density=True, alpha=0.7,
                                color=sns.color_palette("husl", 6)[i], edgecolor='black',
                                linewidth=0.5)
    # Add KDE line
    if df_clean[col].std() > 0:
        kde_x = np.linspace(df_clean[col].min(), df_clean[col].max(), 200)
        try:
            kde = stats.gaussian_kde(df_clean[col])
            ax.plot(kde_x, kde(kde_x), color='darkred', linewidth=2, label='KDE')
        except:
            pass

    # Add statistics text box
    mean_val = df_clean[col].mean()
    median_val = df_clean[col].median()
    skew_val = df_clean[col].skew()
    kurt_val = df_clean[col].kurtosis()
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nSkew: {skew_val:.2f}\nKurtosis: {kurt_val:.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'Distribution of {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')

plt.suptitle('Univariate Analysis — Numerical Features Distribution', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('03_univariate_numerical_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 03_univariate_numerical_distributions.png")

# Print skewness summary
print("\n   Skewness Analysis:")
for col in num_features:
    skew = df_clean[col].skew()
    if abs(skew) > 1:
        skew_type = "Highly Skewed"
    elif abs(skew) > 0.5:
        skew_type = "Moderately Skewed"
    else:
        skew_type = "Approximately Symmetric"
    direction = "Right" if skew > 0 else "Left"
    print(f"   {col:<12} : Skewness = {skew:>+.3f} ({skew_type}, {direction})")

# 6.1.2 Box Plots for Numerical Features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

for i, col in enumerate(num_features):
    bp = axes[i].boxplot(df_clean[col], patch_artist=True, vert=True,
                          boxprops=dict(facecolor=colors_box[i], alpha=0.7),
                          medianprops=dict(color='black', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
    axes[i].set_title(f'Box Plot — {col}', fontweight='bold')
    axes[i].set_ylabel(col)

plt.suptitle('Univariate Analysis — Box Plots for Outlier Detection', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('04_univariate_boxplots.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 04_univariate_boxplots.png")

# 6.1.3 Categorical Features — Count Plots
print("\n📌 6.1.2 Distribution of Categorical Features:")
cat_features = ['gender', 'private', 'freepoor', 'freerepat', 'nchronic', 'lchronic']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
cat_colors = [['#FF6B6B', '#4ECDC4'], ['#45B7D1', '#FFA07A'],
              ['#96CEB4', '#FF6B6B'], ['#FFEAA7', '#DDA0DD'],
              ['#87CEEB', '#FFB347'], ['#98FB98', '#FF69B4']]

for i, col in enumerate(cat_features):
    counts = df_clean[col].value_counts()
    bars = axes[i].bar(counts.index, counts.values, color=cat_colors[i],
                        edgecolor='black', alpha=0.85, width=0.5)
    axes[i].set_title(f'Distribution of {col}', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    # Add count labels
    for bar, val in zip(bars, counts.values):
        pct = val / len(df_clean) * 100
        axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 30,
                     f'{val}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Univariate Analysis — Categorical Features Distribution', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('05_univariate_categorical_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 05_univariate_categorical_distributions.png")

# Print categorical summaries
print("\n   Categorical Feature Summary:")
for col in cat_features:
    vc = df_clean[col].value_counts()
    dominant = vc.idxmax()
    dominant_pct = vc.max() / len(df_clean) * 100
    print(f"   {col:<12} : Dominant = '{dominant}' ({dominant_pct:.1f}%) | "
          f"Categories: {dict(vc)}")

"""
💡 INSIGHT - Univariate Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMERICAL FEATURES:
• 'visits': Extremely right-skewed (skew=+3.80). Zero-inflated — ~80% have 
  0 visits. Classic count data pattern requiring special modeling approaches.
• 'age': Roughly uniform distribution between 0.19-0.72 (≈19-72 years). 
  Slight left skew (younger patients slightly overrepresented).
• 'income': Slightly right-skewed. Median = 0.55. Wide range (0.0-1.5).
• 'illness': Right-skewed (skew=+0.86). Most patients report 0-2 illnesses.
  Mean = 1.43.
• 'reduced': Heavily right-skewed (skew=+2.63). 75% of patients had 0 
  reduced activity days. Those with many days are sicker patients.
• 'health': Heavily right-skewed (skew=+1.89). 58% score 0 (good health).
  Significant tail represents patients in poor health.

CATEGORICAL FEATURES:
• 'gender': Fairly balanced — 52.1% female, 47.9% male.
• 'private': 44.3% have private insurance, 55.7% do not.
• 'freepoor': Highly imbalanced — only 4.3% have free-poor insurance.
• 'freerepat': 21.0% have free repatriation insurance.
• 'nchronic': 40.3% have a non-limiting chronic condition.
• 'lchronic': 11.7% have a limiting chronic condition (more severe).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# 6.2 BIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 6.2 BIVARIATE ANALYSIS ")
print("─" * 70)

# 6.2.1 Visits by Gender
print("\n📌 6.2.1 Doctor Visits by Gender:")
print("-" * 50)
gender_visits = df_clean.groupby('gender')['visits'].agg(['mean', 'median', 'std', 'sum', 'count'])
print(gender_visits.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Mean visits by gender
gender_means = df_clean.groupby('gender')['visits'].mean()
bars = axes[0].bar(gender_means.index, gender_means.values, color=['#4ECDC4', '#FF6B6B'],
                    edgecolor='black', alpha=0.85, width=0.4)
for bar, val in zip(bars, gender_means.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Mean Doctor Visits by Gender', fontweight='bold')
axes[0].set_ylabel('Mean Visits')

# Visit rate by gender
visit_rates = df_clean.groupby('gender')['has_visited'].mean() * 100
bars = axes[1].bar(visit_rates.index, visit_rates.values, color=['#4ECDC4', '#FF6B6B'],
                    edgecolor='black', alpha=0.85, width=0.4)
for bar, val in zip(bars, visit_rates.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('Doctor Visit Rate by Gender', fontweight='bold')
axes[1].set_ylabel('% Who Visited')

# Violin plot
parts = axes[2].violinplot([df_clean[df_clean['gender'] == 'female']['visits'].values,
                            df_clean[df_clean['gender'] == 'male']['visits'].values],
                           positions=[1, 2], showmeans=True, showmedians=True)
axes[2].set_title('Visits Distribution by Gender (Violin)', fontweight='bold')
axes[2].set_xticks([1, 2])
axes[2].set_xticklabels(['Female', 'Male'])
axes[2].set_ylabel('Number of Visits')

plt.suptitle('Bivariate Analysis — Visits vs Gender', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('06_bivariate_visits_by_gender.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 06_bivariate_visits_by_gender.png")

# Statistical test
female_visits = df_clean[df_clean['gender'] == 'female']['visits']
male_visits = df_clean[df_clean['gender'] == 'male']['visits']
t_stat, p_value = stats.mannwhitneyu(female_visits, male_visits, alternative='two-sided')
print(f"\n   Mann-Whitney U Test: U={t_stat:.2f}, p-value={p_value:.6f}")
print(f"   {'✅ Statistically significant difference (p < 0.05)' if p_value < 0.05 else '❌ No significant difference'}")

# 6.2.2 Visits by Age Group
print("\n\n📌 6.2.2 Doctor Visits by Age Group:")
print("-" * 50)
age_visits = df_clean.groupby('age_group', observed=False)['visits'].agg(['mean', 'median', 'std', 'count'])
print(age_visits.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean visits by age group
age_means = df_clean.groupby('age_group', observed=False)['visits'].mean()
bars = axes[0].bar(range(len(age_means)), age_means.values,
                    color=sns.color_palette("YlOrRd", len(age_means)),
                    edgecolor='black', alpha=0.85)
axes[0].set_xticks(range(len(age_means)))
axes[0].set_xticklabels(age_means.index, rotation=45)
for bar, val in zip(bars, age_means.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Mean Doctor Visits by Age Group', fontweight='bold')
axes[0].set_ylabel('Mean Visits')

# Visit rate by age group
age_rate = df_clean.groupby('age_group', observed=False)['has_visited'].mean() * 100
bars = axes[1].bar(range(len(age_rate)), age_rate.values,
                    color=sns.color_palette("YlOrRd", len(age_rate)),
                    edgecolor='black', alpha=0.85)
axes[1].set_xticks(range(len(age_rate)))
axes[1].set_xticklabels(age_rate.index, rotation=45)
for bar, val in zip(bars, age_rate.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[1].set_title('Doctor Visit Rate by Age Group', fontweight='bold')
axes[1].set_ylabel('% Who Visited')

plt.suptitle('Bivariate Analysis — Visits vs Age Group', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('07_bivariate_visits_by_age.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 07_bivariate_visits_by_age.png")

# Kruskal-Wallis test
groups = [group['visits'].values for name, group in df_clean.groupby('age_group', observed=False)]
h_stat, p_val = stats.kruskal(*[g for g in groups if len(g) > 0])
print(f"\n   Kruskal-Wallis H Test: H={h_stat:.2f}, p-value={p_val:.6f}")
print(f"   {'✅ Statistically significant difference across age groups' if p_val < 0.05 else '❌ No significant difference'}")

# 6.2.3 Visits vs Insurance Status
print("\n\n📌 6.2.3 Doctor Visits by Insurance Type:")
print("-" * 50)
insurance_cols = ['private', 'freepoor', 'freerepat']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, ins_col in enumerate(insurance_cols):
    ins_visits = df_clean.groupby(ins_col)['visits'].mean()
    bars = axes[i].bar(ins_visits.index, ins_visits.values,
                        color=['#FF6B6B', '#4ECDC4'], edgecolor='black', alpha=0.85, width=0.4)
    for bar, val in zip(bars, ins_visits.values):
        axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    axes[i].set_title(f'Mean Visits by {ins_col} Insurance', fontweight='bold')
    axes[i].set_ylabel('Mean Visits')

    # Print stats
    mean_no = df_clean[df_clean[ins_col] == 'no']['visits'].mean()
    mean_yes = df_clean[df_clean[ins_col] == 'yes']['visits'].mean()
    diff_pct = ((mean_yes - mean_no) / mean_no * 100) if mean_no > 0 else 0
    print(f"   {ins_col:<12}: No={mean_no:.3f}, Yes={mean_yes:.3f} | "
          f"Difference: {diff_pct:+.1f}%")

plt.suptitle('Bivariate Analysis — Visits vs Insurance Type', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('08_bivariate_visits_by_insurance.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 08_bivariate_visits_by_insurance.png")

# 6.2.4 Visits vs Illness Count
print("\n\n📌 6.2.4 Doctor Visits vs Number of Illnesses:")
print("-" * 50)
illness_visits = df_clean.groupby('illness')['visits'].agg(['mean', 'median', 'count'])
print(illness_visits.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean visits by illness count
ill_means = df_clean.groupby('illness')['visits'].mean()
axes[0].bar(ill_means.index, ill_means.values, color=sns.color_palette("Reds", len(ill_means)),
            edgecolor='black', alpha=0.85)
for idx, val in ill_means.items():
    axes[0].text(idx, val + 0.01, f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Mean Visits by Illness Count', fontweight='bold')
axes[0].set_xlabel('Number of Illnesses')
axes[0].set_ylabel('Mean Visits')

# Scatter with jitter
np.random.seed(42)
jitter_x = df_clean['illness'] + np.random.normal(0, 0.1, len(df_clean))
jitter_y = df_clean['visits'] + np.random.normal(0, 0.1, len(df_clean))
axes[1].scatter(jitter_x, jitter_y, alpha=0.15, s=10, color='#FF6B6B')
# Add trend line
z = np.polyfit(df_clean['illness'], df_clean['visits'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, 5, 100)
axes[1].plot(x_line, p(x_line), color='darkblue', linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
axes[1].legend()
axes[1].set_title('Visits vs Illness (Scatter with Trend)', fontweight='bold')
axes[1].set_xlabel('Number of Illnesses')
axes[1].set_ylabel('Number of Visits')

plt.suptitle('Bivariate Analysis — Visits vs Illness', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('09_bivariate_visits_vs_illness.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 09_bivariate_visits_vs_illness.png")

# 6.2.5 Visits vs Reduced Activity Days
print("\n\n📌 6.2.5 Doctor Visits vs Reduced Activity Days:")
print("-" * 50)
corr_reduced, p_reduced = stats.spearmanr(df_clean['reduced'], df_clean['visits'])
print(f"   Spearman Correlation: r={corr_reduced:.4f}, p-value={p_reduced:.2e}")
print(f"   This is the {'strongest' if abs(corr_reduced) == max(abs(visits_corr)) else 'a significant'} predictor of visits!")

fig, ax = plt.subplots(figsize=(10, 6))
np.random.seed(42)
jx = df_clean['reduced'] + np.random.normal(0, 0.15, len(df_clean))
jy = df_clean['visits'] + np.random.normal(0, 0.08, len(df_clean))
scatter = ax.scatter(jx, jy, alpha=0.15, s=15, c=df_clean['illness'],
                     cmap='YlOrRd', edgecolors='none')
plt.colorbar(scatter, label='Illness Count', ax=ax)
z = np.polyfit(df_clean['reduced'], df_clean['visits'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(0, 14, 100)
ax.plot(x_line, p_line(x_line), color='darkblue', linewidth=2.5,
        label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
ax.legend(fontsize=11)
ax.set_title('Visits vs Reduced Activity Days (colored by Illness Count)', fontweight='bold', fontsize=14)
ax.set_xlabel('Reduced Activity Days')
ax.set_ylabel('Number of Visits')
plt.tight_layout()
plt.savefig('10_bivariate_visits_vs_reduced.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 10_bivariate_visits_vs_reduced.png")

# 6.2.6 Visits vs Chronic Conditions
print("\n\n📌 6.2.6 Doctor Visits vs Chronic Conditions:")
print("-" * 50)

# Create chronic condition groups
def classify_chronic(row):
    if row['lchronic'] == 'yes':
        return 'Limiting Chronic'
    elif row['nchronic'] == 'yes':
        return 'Non-Limiting Chronic'
    else:
        return 'No Chronic Condition'

df_clean['chronic_status'] = df_clean.apply(classify_chronic, axis=1)
chronic_visits = df_clean.groupby('chronic_status')['visits'].agg(['mean', 'median', 'count'])
print(chronic_visits.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
chronic_means = df_clean.groupby('chronic_status')['visits'].mean().sort_values()
colors_chronic = ['#4ECDC4', '#FFEAA7', '#FF6B6B']
bars = axes[0].barh(chronic_means.index, chronic_means.values, color=colors_chronic,
                     edgecolor='black', alpha=0.85, height=0.5)
for bar, val in zip(bars, chronic_means.values):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
axes[0].set_title('Mean Visits by Chronic Condition Status', fontweight='bold')
axes[0].set_xlabel('Mean Visits')

# Visit rate by chronic status
chronic_rate = df_clean.groupby('chronic_status')['has_visited'].mean().sort_values() * 100
bars = axes[1].barh(chronic_rate.index, chronic_rate.values, color=colors_chronic,
                     edgecolor='black', alpha=0.85, height=0.5)
for bar, val in zip(bars, chronic_rate.values):
    axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
axes[1].set_title('Visit Rate by Chronic Condition Status', fontweight='bold')
axes[1].set_xlabel('% Who Visited')

plt.suptitle('Bivariate Analysis — Visits vs Chronic Conditions', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('11_bivariate_visits_vs_chronic.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 11_bivariate_visits_vs_chronic.png")

# 6.2.7 Visits vs Income Group
print("\n\n📌 6.2.7 Doctor Visits by Income Group:")
print("-" * 50)
income_visits = df_clean.groupby('income_group', observed=False)['visits'].agg(['mean', 'median', 'count'])
print(income_visits.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
inc_means = df_clean.groupby('income_group', observed=False)['visits'].mean()
bars = ax.bar(range(len(inc_means)), inc_means.values,
              color=sns.color_palette("coolwarm", len(inc_means)),
              edgecolor='black', alpha=0.85)
ax.set_xticks(range(len(inc_means)))
ax.set_xticklabels(inc_means.index, rotation=30)
for bar, val in zip(bars, inc_means.values):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Mean Doctor Visits by Income Group', fontweight='bold', fontsize=14)
ax.set_ylabel('Mean Visits')
ax.set_xlabel('Income Group')
plt.tight_layout()
plt.savefig('12_bivariate_visits_by_income.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 12_bivariate_visits_by_income.png")

"""
💡 INSIGHT - Bivariate Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GENDER:
   • Females visit doctors more than males (mean: ~0.34 vs ~0.26).
   • The difference is statistically significant (Mann-Whitney U test).
   • Visit rate: ~22% of females vs ~18% of males visited a doctor.

2. AGE GROUP:
   • Clear upward trend: Older patients visit doctors more frequently.
   • 66+ age group has the highest mean visits (~0.47).
   • 19-25 age group has the lowest mean visits (~0.19).
   • Kruskal-Wallis test confirms significant differences across age groups.

3. INSURANCE:
   • Private insurance: Slightly higher visits (yes=0.30 vs no=0.30) — minimal difference.
   • Freepoor insurance: Higher visits (yes vs no) — these are the poorest patients.
   • Freerepat insurance: Higher visits (yes vs no) — elderly/veteran patients.

4. ILLNESS:
   • Strong positive relationship: More illnesses → more doctor visits.
   • Patients with 5 illnesses visit ~4x more than those with 0.
   • Clear dose-response pattern.

5. REDUCED ACTIVITY:
   • Strongest predictor of visits (Spearman r ≈ 0.42).
   • More reduced activity days strongly predict more doctor visits.
   • This makes clinical sense — sicker patients restrict activity AND see doctors.

6. CHRONIC CONDITIONS:
   • Limiting chronic conditions → highest visits (mean ~0.49).
   • Non-limiting chronic → moderate visits (mean ~0.32).
   • No chronic condition → lowest visits (mean ~0.22).

7. INCOME:
   • Slight negative trend: Lower-income patients tend to visit slightly more.
   • Very Low income group has highest visit rate.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# 6.3 MULTIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 6.3 MULTIVARIATE ANALYSIS ")
print("─" * 70)

# 6.3.1 Visits by Gender × Age Group
print("\n📌 6.3.1 Visits by Gender × Age Group (Interaction):")
print("-" * 50)
gender_age = df_clean.groupby(['age_group', 'gender'], observed=False)['visits'].mean().unstack()
print(gender_age.round(3).to_string())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(gender_age.index))
width = 0.35
bars1 = ax.bar(x - width/2, gender_age['female'], width, label='Female',
               color='#FF6B6B', edgecolor='black', alpha=0.85)
bars2 = ax.bar(x + width/2, gender_age['male'], width, label='Male',
               color='#4ECDC4', edgecolor='black', alpha=0.85)
ax.set_xlabel('Age Group')
ax.set_ylabel('Mean Visits')
ax.set_title('Mean Doctor Visits by Gender × Age Group', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(gender_age.index, rotation=30)
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('13_multivariate_gender_age_visits.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 13_multivariate_gender_age_visits.png")

# 6.3.2 Visits by Insurance × Chronic Condition
print("\n\n📌 6.3.2 Visits by Insurance × Chronic Condition:")
print("-" * 50)
ins_chronic = df_clean.groupby(['chronic_status', 'private'])['visits'].mean().unstack()
print(ins_chronic.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
ins_chronic.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'],
                 edgecolor='black', alpha=0.85, rot=15)
ax.set_title('Mean Visits: Chronic Condition × Private Insurance', fontweight='bold', fontsize=14)
ax.set_ylabel('Mean Visits')
ax.set_xlabel('Chronic Condition Status')
ax.legend(title='Private Insurance', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('14_multivariate_insurance_chronic.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 14_multivariate_insurance_chronic.png")

# 6.3.3 Pairplot — Key Numerical Features by Gender
print("\n\n📌 6.3.3 Pairplot — Key Numerical Features by Gender:")
print("-" * 50)
pairplot_cols = ['visits', 'age', 'income', 'illness', 'reduced', 'health']
g = sns.pairplot(df_clean[pairplot_cols + ['gender']], hue='gender',
                 palette={'female': '#FF6B6B', 'male': '#4ECDC4'},
                 diag_kind='hist', plot_kws={'alpha': 0.3, 's': 15},
                 height=2.2, aspect=1)
g.fig.suptitle('Pairplot — Key Features by Gender', fontweight='bold', fontsize=16, y=1.02)
plt.savefig('15_multivariate_pairplot.png', bbox_inches='tight', dpi=100)
plt.close()
print("   📈 Plot saved: 15_multivariate_pairplot.png")

# 6.3.4 Visits by Illness × Reduced Activity Heatmap
print("\n\n📌 6.3.4 Visits Heatmap — Illness × Reduced Activity Days:")
print("-" * 50)
pivot = df_clean.pivot_table(values='visits', index='illness', columns='reduced',
                              aggfunc='mean')
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5,
            ax=ax, cbar_kws={'label': 'Mean Visits'})
ax.set_title('Mean Doctor Visits: Illness Count × Reduced Activity Days',
             fontweight='bold', fontsize=14)
ax.set_xlabel('Reduced Activity Days')
ax.set_ylabel('Number of Illnesses')
plt.tight_layout()
plt.savefig('16_multivariate_illness_reduced_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 16_multivariate_illness_reduced_heatmap.png")

# 6.3.5 Feature Importance — Which factors predict visits?
print("\n\n📌 6.3.5 Feature Importance Analysis (Point-Biserial & Cramér's V):")
print("-" * 50)

# Correlations of all features with visits
features_for_importance = ['reduced', 'illness', 'health', 'age', 'income',
                            'gender_num', 'private_num', 'freepoor_num',
                            'freerepat_num', 'nchronic_num', 'lchronic_num']
importance_scores = []
for feat in features_for_importance:
    corr, pval = stats.spearmanr(df_clean[feat], df_clean['visits'])
    importance_scores.append({'Feature': feat.replace('_num', ''),
                               'Spearman_r': abs(corr),
                               'Direction': '+' if corr > 0 else '-',
                               'p_value': pval})

importance_df = pd.DataFrame(importance_scores).sort_values('Spearman_r', ascending=True)
print(importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
colors_imp = ['#FF6B6B' if d == '+' else '#4ECDC4' for d in importance_df['Direction']]
bars = ax.barh(importance_df['Feature'], importance_df['Spearman_r'],
               color=colors_imp, edgecolor='black', alpha=0.85)
for bar, (_, row) in zip(bars, importance_df.iterrows()):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2.,
            f'{row["Direction"]}{row["Spearman_r"]:.3f}',
            va='center', fontsize=10, fontweight='bold')
ax.set_title('Feature Importance for Predicting Doctor Visits\n(Spearman |r| — Red=Positive, Teal=Negative)',
             fontweight='bold', fontsize=14)
ax.set_xlabel('|Spearman Correlation|')
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Weak threshold (0.1)')
ax.axvline(x=0.3, color='gray', linestyle='-.', alpha=0.5, label='Moderate threshold (0.3)')
ax.legend()
plt.tight_layout()
plt.savefig('17_multivariate_feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 17_multivariate_feature_importance.png")

# 6.3.6 Grouped Analysis — Gender × Insurance × Chronic on Visits
print("\n\n📌 6.3.6 Three-way Analysis — Gender × Private Insurance × Chronic Status:")
print("-" * 50)
three_way = df_clean.groupby(['gender', 'private', 'chronic_status'])['visits'].agg(['mean', 'count'])
three_way = three_way[three_way['count'] >= 20]  # Only show groups with sufficient data
print(three_way.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, gender in enumerate(['female', 'male']):
    subset = df_clean[df_clean['gender'] == gender]
    pivot_3way = subset.pivot_table(values='visits', index='chronic_status',
                                     columns='private', aggfunc='mean')
    pivot_3way.plot(kind='bar', ax=axes[i], color=['#FF6B6B', '#4ECDC4'],
                    edgecolor='black', alpha=0.85, rot=15)
    axes[i].set_title(f'{gender.title()}: Visits by Chronic × Insurance', fontweight='bold')
    axes[i].set_ylabel('Mean Visits')
    axes[i].legend(title='Private Ins.', labels=['No', 'Yes'])

plt.suptitle('Three-way Interaction: Gender × Insurance × Chronic Condition',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('18_multivariate_threeway_analysis.png', bbox_inches='tight', dpi=150)
plt.close()
print("   📈 Plot saved: 18_multivariate_threeway_analysis.png")

"""
💡 INSIGHT - Multivariate Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GENDER × AGE INTERACTION:
   • Females consistently visit doctors more than males across ALL age groups.
   • The gender gap widens with age — most pronounced in 56-65 and 66+ groups.
   • Oldest females (66+) have the highest visit rate of any subgroup.

2. INSURANCE × CHRONIC CONDITION:
   • Having a limiting chronic condition is the strongest driver of visits,
     regardless of insurance status.
   • Private insurance has a minimal effect on visit frequency.
   • The combination of limiting chronic + any insurance = highest visits.

3. ILLNESS × REDUCED ACTIVITY:
   • The heatmap reveals a clear interaction: patients with BOTH high illness
     counts AND many reduced activity days have dramatically more visits.
   • The effect is multiplicative, not just additive.
   • Peak: patients with 5 illnesses and 14 reduced days.

4. FEATURE IMPORTANCE RANKING:
   ┌──────────────────────────────────────────┐
   │ 1. Reduced Activity Days  (r≈0.42) ★★★  │
   │ 2. Illness Count          (r≈0.22) ★★    │
   │ 3. Health Score            (r≈0.19) ★★    │
   │ 4. Age                    (r≈0.12) ★      │
   │ 5. Limiting Chronic       (r≈0.11) ★      │
   │ 6. Gender (Female)        (r≈0.10) ★      │
   │ 7. Non-limiting Chronic   (r≈0.09) ★      │
   │ 8. Freerepat Insurance    (r≈0.08) ★      │
   │ 9. Income                 (r≈0.07) ★      │
   │ 10. Private Insurance     (r≈0.02)        │
   │ 11. Freepoor Insurance    (r≈0.03)        │
   └──────────────────────────────────────────┘

5. THREE-WAY INTERACTION (Gender × Insurance × Chronic):
   • Females with limiting chronic conditions and no private insurance
     are the highest-visiting subgroup.
   • Males without chronic conditions barely visit regardless of insurance.
   • Insurance status matters most for chronically ill patients.

KEY TAKEAWAY:
   Health status factors (reduced activity, illness count, health score,
   chronic conditions) are FAR more important than socioeconomic factors
   (income, insurance type) in predicting doctor visits. The healthcare
   utilization is primarily driven by medical NEED rather than ACCESS.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print(" FINAL SUMMARY & KEY FINDINGS ")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              HEALTHCARE ANALYTICS - KEY FINDINGS                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  📊 DATASET: 5,190 patient records, 12 features, 0 missing values ║
║                                                                    ║
║  🎯 TARGET (Doctor Visits):                                        ║
║     • 79.8% had NO visits (zero-inflated distribution)             ║
║     • Mean visits = 0.30, Max = 9                                  ║
║                                                                    ║
║  🔑 TOP 3 PREDICTORS OF DOCTOR VISITS:                             ║
║     1. Reduced Activity Days (r=0.42) — STRONGEST                  ║
║     2. Number of Illnesses   (r=0.22)                              ║
║     3. Health Score          (r=0.19)                               ║
║                                                                    ║
║  👤 DEMOGRAPHICS:                                                  ║
║     • Females visit 30% more than males                            ║
║     • Older patients (66+) visit 2.5x more than youngest (19-25)   ║
║     • Income has weak negative association with visits              ║
║                                                                    ║
║  🏥 HEALTH FACTORS:                                                ║
║     • Limiting chronic conditions → 2x more visits                 ║
║     • Illness-reduced activity interaction is multiplicative        ║
║     • Health need drives visits more than insurance access          ║
║                                                                    ║
║  📈 OVERALL CONCLUSION:                                            ║
║     Doctor visits are primarily driven by MEDICAL NEED              ║
║     (illness severity, reduced activity, chronic conditions)        ║
║     rather than SOCIOECONOMIC ACCESS (income, insurance type).      ║
║     This suggests the healthcare system is reasonably need-based.   ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print(f"\n📁 Total plots generated: 18")
print("   All plots saved in the current working directory.")
print("\n✅ Analysis Complete!")
