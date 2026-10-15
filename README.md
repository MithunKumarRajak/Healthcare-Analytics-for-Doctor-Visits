# Healthcare Analytics for Doctor Visits

An in-depth Exploratory Data Analysis (EDA) on patient doctor visit frequency to understand healthcare utilization patterns, evaluate clinical vs socioeconomic determinants, and assess the impact of insurance and chronic health conditions.

---

## Dataset Overview

- **Source File:** `1776250375-P2-Healthcare Analytics for Doctor Visits.csv`
- **Total Records:** 5,190 individual patients
- **Total Features:** 12 attributes (6 numerical + 6 categorical)
- **Target Variable:** `visits` (Number of doctor consultations in the past 2 weeks)

### Feature Descriptions
| Feature | Type | Description |
|---|---|---|
| `visits` | Target (Discrete) | Number of doctor visits in the past 2 weeks (0–9) |
| `gender` | Categorical (Binary) | Patient gender (`male`, `female`) |
| `age` | Numerical (Continuous) | Age normalized scale (~0.19 to 0.72 ≈ 19 to 72 years) |
| `income` | Numerical (Continuous) | Annual income normalized scale (0.0 to 1.5) |
| `illness` | Numerical (Discrete) | Number of illnesses reported in past 2 weeks (0–5) |
| `reduced` | Numerical (Discrete) | Days of reduced activity due to illness in past 2 weeks (0–14) |
| `health` | Numerical (Discrete) | General health status score (0 = best health, 12 = worst health) |
| `private` | Categorical (Binary) | Holds private health insurance (`yes`, `no`) |
| `freepoor` | Categorical (Binary) | Covered by free government low-income insurance (`yes`, `no`) |
| `freerepat` | Categorical (Binary) | Covered by government veteran / disability / pensioner program (`yes`, `no`) |
| `nchronic` | Categorical (Binary) | Non-limiting chronic condition (`yes`, `no`) |
| `lchronic` | Categorical (Binary) | Limiting chronic condition affecting daily life (`yes`, `no`) |

---

## Analysis Workflow

The notebook strictly follows the standard analytical sequence:
1. **Importing Libraries**
2. **Loading the Data**
3. **Data Overview**
4. **Data Cleaning & Feature Engineering**
5. **Exploratory Data Analysis (EDA)**
6. **Further Exploratory Data Analysis**
   - **Univariate Analysis**
   - **Bivariate Analysis**
   - **Multivariate Analysis**

---

## Key Findings

1. **Target Variable Dynamics:**
   - **79.8% of patients** recorded 0 doctor visits in the reference period (zero-inflated count distribution).
   - Only 20.2% consulted a doctor, with a maximum of 9 visits.

2. **Top Determinants of Doctor Visits:**
   - **Reduced Activity Days** ($\rho = +0.339, p < 0.001$): The strongest single predictor of doctor visits. Patients incapacitated for 10–14 days average 1.455 visits (8.2× higher than those with 0 days).
   - **Illness Count** ($\rho = +0.260, p < 0.001$): Monotonic dose-response relationship from 0.149 visits (0 illnesses) to 0.798 visits (5 illnesses).
   - **General Health Score** ($\rho = +0.180, p < 0.001$): Higher scores (worse health) directly scale visit frequency.
   - **Chronic Condition Severity**: Patients with daily-life-limiting chronic conditions average 0.540 visits (31.4% visit rate) vs 0.218 for patients with no chronic illness.

3. **Demographic Modifiers:**
   - **Gender**: Females visit doctors 33.8% more frequently than males (mean 0.344 vs 0.257, $p < 0.001$). The disparity is largest during working/childbearing years (ages 26–55) and converges after age 65.
   - **Age Gradient**: Visits rise steadily with age, from 0.197 (19–25 cohort) to 0.468 (66+ cohort).

4. **Socioeconomic & Insurance Findings:**
   - **Private Insurance**: Shows no statistically significant difference on doctor visits ($p = 0.589$). Having private coverage does not artificially alter consultation rates.
   - **Income**: Displays a weak inverse relationship ($r = -0.077$), reflecting higher proportions of retirees in lower income brackets.
   - **Healthcare System Conclusion**: Healthcare utilization in this population is overwhelmingly **need-based** rather than access-constrained.

---

## Project Structure

```
├── Healthcare Analytics for Doctor Visits.ipynb  # Primary Jupyter Notebook with analysis & outputs
├── 1776250375-P2-Healthcare Analytics for Doctor Visits.csv # Raw Dataset
├── 1776250340-1776165716-1730262489-1729507042-DIY Project ppt template - Copy.pptx # Project Presentation Template
├── README.md                                     # Project documentation
└── .gitignore                                    # Git configuration
```

---

## Requirements

- Python 3.10+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
