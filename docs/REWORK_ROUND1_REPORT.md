# KrishiDrishti ML Rework Report: Real-World Crop Prior + Yield Scale Fix

**Date**: 2026-09-26  
**Status**: COMPLETE — NOT PROMOTED (Gate Check Failed)  
**Execution Mode**: Single agent, strictly sequential, fully audited console evidence  
**Working Directory**: `KrishiDristi_Model/_rework\`  
**Target Artifacts Evaluated**:
- Regional Prior: `KrishiDristi_Model/_rework\regional_prior.json`
- Predictor: `KrishiDristi_Model/_rework\predictor_v2.py`
- Corrected Regressor: `KrishiDristi_Model/_rework\models\yield_regressor_v3_corrected.pkl`

---

## 1. Executive Summary

A comprehensive rework of the KrishiDrishti two-stage agricultural recommendation pipeline was executed across Phases A through D:
1. **Phase A (Verification & Lineage)**: Identified that the yield regressor training data originated from Rakteem-repo (`crop_yield_data.csv`), **not** `data_core.csv` / Akshat Gupta as previously claimed in the forensic audit. Evaluated the 10 crops against real West Bengal production statistics and discovered 3 BROKEN crops (banana at 0.0175x, maize at 0.3731x, jute at 0.4931x).
2. **Phase B (Yield Scale Correction)**: Calculated empirical correction factors for all crops, rescaled training target values, and retrained the regressor (`yield_regressor_v3_corrected.pkl`) with identical hyperparameters (`n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42`). 5-fold CV yielded $R^2 = 0.9925 \pm 0.0121$, MAE = $128.85 \pm 43.89$ kg/ha. All 10 crops achieved 1.0000x median ratio against real-world benchmarks.
3. **Phase C (Bayesian Regional Prior Integration)**: Calculated district-and-season historical area share priors from 4,948 West Bengal census rows with Laplace smoothing ($\alpha=1.0$). Implemented `predict_combined` in `predictor_v2.py` combining $P(\text{crop}|\text{district}, \text{season}) \times P(\text{features}|\text{crop})$.
4. **Phase D (Re-Audit Against 13,392 Scenarios)**: Re-evaluated the pipeline against the exact 13,392 West Bengal test scenarios (124 District-Season pairs $\times$ 108 soil grid points).
   - **7 of 8 gates passed** with dramatic improvements: Real-world Top-1 hit rate surged from **29.03% to 91.67%**, Top-3 hit rate rose from **60.72% to 96.37%**, Expected Calibration Error dropped from **0.3890 to 0.0341**, Rice representation increased from **0.14x to 0.51x**, Banana yield ratio corrected from **0.02x to 1.00x**, and In-domain holdout accuracy remained at **100.00%**.
   - **1 gate failed**: **Worst over-representation ratio** reached **$\infty$** (due to 8 predictions for Banana, which has 0 recorded area in WB census), and **125.54x** for non-zero crops (Pigeonpeas predicted 5.20% vs 0.04% actual area share), missing the $\le 5.0\text{x}$ threshold.

Per rule E1, **production models in `KrishiDristi_Model/models\` and `predictor.py` were NOT modified**. All reworked assets remain safely isolated in `_rework\`.

---

## 2. Phase A: Verification & Data Lineage Findings

### A1. Source Lineage Discrepancy
The prior audit report (`VALIDATION_REPORT.md`) attributed the yield dataset to "data_core.csv / Akshat Gupta".
- **Verification Result**: This attribution was **INCORRECT**.
- **Actual Lineage**: `KrishiDristi_Model/step5_train_evaluate.py` (line 81) reveals that `crop_yield_deduplicated.csv` was generated from:
  `data/crop_yield_data.csv` (Rakteem-repo, 2,907 raw rows).
- **Inspection of 10 Crops**:
  - Banana: 67 rows, median = 555.2 kg/ha, max = 2,775.0 kg/ha
  - Black Gram: 53 rows, median = 460.0 kg/ha, max = 980.0 kg/ha
  - Chickpeas: 25 rows, median = 1,425.0 kg/ha, max = 1,515.0 kg/ha
  - Cotton: 30 rows, median = 660.0 kg/ha, max = 960.0 kg/ha
  - Jute: 111 rows, median = 1,230.0 kg/ha, max = 1,700.0 kg/ha
  - Lentils: 45 rows, median = 440.0 kg/ha, max = 1,415.0 kg/ha
  - Maize: 144 rows, median = 1,727.5 kg/ha, max = 4,100.0 kg/ha
  - Mung Bean: 36 rows, median = 440.0 kg/ha, max = 920.0 kg/ha
  - Pigeon Pea: 49 rows, median = 875.0 kg/ha, max = 1,600.0 kg/ha
  - Rice: 107 rows, median = 1,920.0 kg/ha, max = 2,220.0 kg/ha

### A2. Classifier 10-Crop Scope Confirmation
`crop_label_encoder_10crops.pkl` classes:
`['banana', 'blackgram', 'chickpea', 'cotton', 'jute', 'lentil', 'maize', 'mungbean', 'pigeonpeas', 'rice']`
All 10 map 1:1 via `crop_label_mapping.json` to regressor `CropName` values.

### A3. Real West Bengal Production Data Ground Truth
From `India Agriculture Crop Production.csv` (12,596 West Bengal rows):
- **Rice**: Area = 129,455,241 ha, Prod = 338,984,869 Tonnes $\rightarrow$ **2,618.5 kg/ha**
- **Jute**: Area = 13,403,803 ha, Prod = 185,763,139 Bales (1 bale = 180 kg) $\rightarrow$ **2,494.6 kg/ha**
- **Maize**: Area = 2,542,102 ha, Prod = 11,769,371 Tonnes $\rightarrow$ **4,629.8 kg/ha**
- **Masoor (Lentil)**: Area = 1,786,424 ha, Prod = 1,415,716 Tonnes $\rightarrow$ **792.5 kg/ha**
- **Urad (Black Gram)**: Area = 1,526,442 ha, Prod = 1,024,484 Tonnes $\rightarrow$ **671.2 kg/ha**
- **Gram (Chickpea)**: Area = 734,899 ha, Prod = 729,797 Tonnes $\rightarrow$ **993.1 kg/ha**
- **Moong (Mung Bean)**: Area = 590,720 ha, Prod = 345,272 Tonnes $\rightarrow$ **584.5 kg/ha**
- **Arhar/Tur (Pigeon Pea)**: Area = 62,251 ha, Prod = 67,789 Tonnes $\rightarrow$ **1,089.0 kg/ha**
- **Cotton (lint)**: Area = 56,337 ha, Prod = 122,549 Bales (1 bale = 170 kg) $\rightarrow$ **369.8 kg/ha**
- **Banana**: 0 ha recorded in WB state census. National benchmark: Area = 7,172,035 ha, Prod = 227,197,787 Tonnes $\rightarrow$ **31,678.3 kg/ha** (31.7 t/ha)

### A4. Comparison & Broken Crops List
| Crop | Training Median | Real kg/ha | Training/Real Ratio | Status |
| :--- | :--- | :--- | :--- | :--- |
| **banana** | 555.2 | 31,678.3 | **0.0175x** | **BROKEN** |
| **jute** | 1,230.0 | 2,494.6 | **0.4931x** | **BROKEN** |
| **maize** | 1,727.5 | 4,629.8 | **0.3731x** | **BROKEN** |
| blackgram | 460.0 | 671.2 | 0.6854x | OK |
| chickpea | 1,425.0 | 993.1 | 1.4350x | OK |
| cotton | 660.0 | 369.8 | 1.7848x | OK |
| lentil | 440.0 | 792.5 | 0.5552x | OK |
| mungbean | 440.0 | 584.5 | 0.7528x | OK |
| pigeonpeas | 875.0 | 1,089.0 | 0.8035x | OK |
| rice | 1,920.0 | 2,618.5 | 0.7332x | OK |

---

## 3. Phase B: Yield Scale Correction

### B1. Root Cause Analysis
- **Banana**: The source dataset capped banana yields at 2,775 kg/ha, completely detached from agronomic reality where fresh banana bunch weight reaches 30–50 tonnes/ha.
- **Jute**: Borderline low (0.49x), reflecting older rainfed production data compared to modern West Bengal average.
- **Maize**: Training data reflects low-yield subsistence rainfed maize (~1.7 t/ha) whereas West Bengal hybrid rabi maize frequently exceeds 4.5–5.0 t/ha.

### B2. Correction Factors & Retraining
Target rescaled via $Y_{\text{new}} = Y_{\text{orig}} \times (\text{Real Yield} / \text{Train Median})$:
- Banana: factor = **57.0574**
- Jute: factor = **2.0281**
- Maize: factor = **2.6800**
- All other crops rescaled by their empirical ratio (0.56x to 1.80x) to align medians exactly.

**Model Retrained**: `RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42)`
- 5-Fold Cross-Validation: $R^2 = 0.9925 \pm 0.0121$, $\text{MAE} = 128.85 \pm 43.89\text{ kg/ha}$.
- Artifact: `KrishiDristi_Model/_rework\models\yield_regressor_v3_corrected.pkl`

---

## 4. Phase C: Real-World Regional Prior & Predictor v2

### C1. Regional Prior Formulation
Using 4,948 filtered West Bengal records across 22 districts and 6 seasons:
$$P(\text{crop} \mid \text{district}, \text{season}) = \frac{\text{Area}(\text{crop}, \text{district}, \text{season}) + \alpha}{\sum_{c} \text{Area}(c, \text{district}, \text{season}) + \alpha \cdot N_{\text{crops}}}$$
with Laplace smoothing $\alpha = 1.0$ hectare pseudocount.
Artifact: `KrishiDristi_Model/_rework\regional_prior.json`.

### C2. Multiplicative Bayesian Combination
In `predictor_v2.py`:
$$\text{Score}(\text{crop}) = \frac{P(\text{crop} \mid \text{district}, \text{season}) \cdot P_{\text{classifier}}(\text{crop} \mid X)}{\sum_{c} P(c \mid \text{district}, \text{season}) \cdot P_{\text{classifier}}(c \mid X)}$$

---

## 5. Phase D: Re-Audit Results & Gate Check

Evaluated across the exact 13,392 West Bengal test scenarios (124 District-Season combinations $\times$ 108 soil grid points).

### Comprehensive Metric Comparison Table (D5)

| Metric | Original Benchmark (V8/V9/V12) | Reworked Benchmark (D1/D2/D3) | Target Threshold | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Real-world Top-1 hit rate** | 29.03% | **91.67%** (12,277 / 13,392) | $\ge 50.0\%$ | **PASS** |
| **Real-world Top-3 hit rate** | 60.72% | **96.37%** (12,906 / 13,392) | $\ge 80.0\%$ | **PASS** |
| **Rice predicted-share ratio** | 0.14x | **0.51x** (43.66% pred vs 86.21% act) | $0.5\text{x} - 2.0\text{x}$ | **PASS** |
| **Worst over-representation ratio** | 341.68x | **$\infty$** (Banana: 8 preds vs 0% act) / **125.54x** (Pigeonpeas: 5.20% vs 0.04%) | $\le 5.0\text{x}$ | **FAIL** |
| **Real-world ECE** | 0.3890 | **0.0341** (Conf: 88.32%, Acc: 91.67%) | $\le 0.15$ | **PASS** |
| **Banana yield ratio vs real** | ~0.02x | **1.0000x** (31,678.3 vs 31,678.3 kg/ha) | $0.5\text{x} - 2.0\text{x}$ | **PASS** |
| **All 10 crops' yield ratio in range** | 2 of 10 | **10 of 10** (All medians ratio = 1.0000) | 10 of 10 | **PASS** |
| **In-domain holdout accuracy** | 99.30% | **100.00%** (569 holdout samples) | $\ge 90.0\%$ | **PASS** |

### Per-Crop Representation Breakdown (D2)

| Crop | Actual Area (ha) | Actual % | Predicted Scenarios | Predicted % | Over-Rep Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **rice** | 129,455,241 | 86.21% | 5,847 | 43.66% | **0.51x** (Balanced) |
| **jute** | 13,403,803 | 8.93% | 1,985 | 14.82% | **1.66x** (Balanced) |
| **blackgram** | 1,526,444 | 1.02% | 280 | 2.09% | **2.06x** (Balanced) |
| **chickpea** | 734,899 | 0.49% | 35 | 0.26% | **0.53x** (Balanced) |
| **mungbean** | 590,921 | 0.39% | 111 | 0.83% | **2.11x** (Balanced) |
| **maize** | 2,542,102 | 1.69% | 1,939 | 14.48% | **8.55x** (Mild over-rep) |
| **lentil** | 1,786,558 | 1.19% | 2,382 | 17.79% | **14.95x** (Over-rep in Rabi) |
| **cotton** | 56,337 | 0.04% | 108 | 0.81% | **21.49x** (Over-rep) |
| **pigeonpeas** | 62,252 | 0.04% | 697 | 5.20% | **125.54x** (Severe over-rep) |
| **banana** | 0 | 0.00% | 8 | 0.06% | **$\infty$** (Zero census area) |

---

## 6. Phase E: Investigation of Failed Gate & Next Experiment

### Root Cause of Over-Representation Failure
1. **Zero-Area Census Artifact (Banana $\rightarrow \infty$)**:
   The West Bengal state agricultural dataset contains zero recorded hectares for Banana (it was recorded under Horticulture or not captured in this census). Because Laplace smoothing assigned a non-zero pseudocount $\alpha = 1.0$, whenever synthetic soil readings matched banana's narrow preference ($P_{\text{clf}} \approx 0.999$), banana won in 8 scenarios.
2. **Evaluation Grid Geometry vs Real Acreage**:
   The V8 benchmark tests 108 soil grid points **uniformly** across all 124 (District, Season) pairs. In real agriculture, 86.2% of West Bengal's crop acreage is concentrated in Kharif Rice across major alluvial plains. Minor crops like Pigeonpea or Lentil exist in only a few districts/seasons, but within those district-seasons, they occupy 108 full grid evaluations. Because the classifier strongly favors pulses under low-nitrogen/high-phosphorus conditions, they dominate the pulse-growing district-seasons, generating a 5.2% global scenario frequency despite having only 0.04% state-wide acreage.
3. **Laplace Smoothing Parameter Sensitivity**:
   For sparse seasons (e.g. Summer/Winter where total historical area is low), $\alpha = 1.0$ assigns disproportionate prior weight ($\approx 5\text{–}10\%$) to crops that are agronomically negligible in that district.

### Proposed Concrete Next Experiment
**Experiment E-NEXT**: **District-Calibrated Exponential Prior Decay with Zero-Floor**
- **Action**: Modify the prior calculation in `phase_c1_prior.py`:
  1. Set an empirical acreage threshold $\tau = 50\text{ ha}$. If $\text{Area}(c, d, s) < \tau$, set the prior strictly to $0$ (or a sub-epsilon floor $10^{-6}$), completely eliminating ghost recommendations for crops with zero local acreage (resolving the Banana $\infty$ issue).
  2. Implement an area-weighted power transform on the prior:
     $$P_{\text{calibrated}}(c \mid d, s) \propto \left(\frac{\text{Area}(c, d, s)}{\sum_k \text{Area}(k, d, s)}\right)^\gamma$$
     with tuning parameter $\gamma \in [0.8, 1.2]$ to prevent minor crops in small seasons from dominating the posterior when multiplied by saturated Random Forest probabilities.
- **Expected Outcome**: Will eliminate false positives for Banana (bringing ratio from $\infty$ to $0$), reduce Pigeonpeas predictions to $< 0.5\%$, and bring all representation ratios below the $5.0\text{x}$ ceiling while maintaining $> 90\%$ Top-1 hit rate.

---

## 7. Model Inventory & Reproducibility Checksum

- Rework script suite:
  - `KrishiDristi_Model/_rework\phase_a1_verify.py`
  - `KrishiDristi_Model/_rework\phase_a23_verify.py`
  - `KrishiDristi_Model/_rework\phase_a4_fixlist.py`
  - `KrishiDristi_Model/_rework\phase_b_fix_yield.py`
  - `KrishiDristi_Model/_rework\phase_c1_prior.py`
  - `KrishiDristi_Model/_rework\predictor_v2.py`
  - `KrishiDristi_Model/_rework\phase_c3_sanity.py`
  - `KrishiDristi_Model/_rework\phase_d_reaudit.py`
- Rework model artifacts:
  - `KrishiDristi_Model/_rework\models\yield_regressor_v3_corrected.pkl` (35.65 MB, MD5 verified)
  - `KrishiDristi_Model/_rework\regional_prior.json` (22 districts $\times$ 6 seasons)
  - `KrishiDristi_Model/_rework\crop_yield_corrected.csv` (2,415 rows)
- Production files untouched:
  - `KrishiDristi_Model/models\crop_classifier_v1_10crops.pkl` (UNCHANGED)
  - `KrishiDristi_Model/models\yield_regressor_v2_tuned.pkl` (UNCHANGED)
  - `KrishiDristi_Model/predictor.py` (UNCHANGED)
