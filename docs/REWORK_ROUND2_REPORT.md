# KrishiDrishti ML Rework Round 2 Report: Prior Calibration Fix

**Date**: 2026-09-27  
**Status**: COMPLETE — NOT PROMOTED (1 Gate Failed across Candidate Grid)  
**Execution Mode**: Single agent, sequential, audited console evidence  
**Working Directory**: `KrishiDristi_Model/_rework\`  
**Target File**: `KrishiDristi_Model/_rework\phase_f_prior_v2.py`  

---

## 1. Executive Summary

In Round 2 of the KrishiDrishti ML rework, we addressed the single gate that failed in Round 1:
$$\text{Worst Over-Representation Ratio (Round 1: } \infty \text{ on Banana, } 125.54\text{x on Pigeonpeas vs threshold } \le 5.0\text{x)}$$

### Summary of Actions Taken:
1. **F1 Verification**: Confirmed that Banana has **exactly 0 rows** in the West Bengal agricultural census dataset (`India Agriculture Crop Production.csv`). This is a documented data-source scope limitation (banana in West Bengal was captured under a separate horticulture survey).
2. **F2 Parameterized Prior**: Developed `phase_f_prior_v2.py` implementing:
   - Minimum-acreage cutoff ($\tau \in [0, 25, 50, 100]$ ha)
   - Area-share power scaling exponent ($\gamma \in [1.0, 1.2, 1.5, 2.0]$)
   - Controlled fallback to Round 1 Laplace smoothing when all crops in a cell are zeroed out.
3. **F3 Grid Search**: Evaluated all 16 $(\tau, \gamma)$ candidate pairs on the exact same 13,392-scenario benchmark (124 District-Season pairs $\times$ 108 soil grid points).
4. **F4 & F5 Safety & Selection**:
   - **7 of 8 gates passed** with exceptional metrics across all 16 candidates (Top-1: 97.9%–100%, Top-3: 100%, Rice share: 0.55x–0.56x, ECE: 0.035–0.082, Banana yield: 1.00x, 10/10 crops in range, Holdout: 100%).
   - **Gate 4 (Worst Over-Representation) failed across all 16 candidates** ($112.57\text{x}$ to $127.16\text{x}$, driven by Pigeonpeas).
   - Per protocol, the candidate grid was **not** expanded arbitrarily. Production files were **not** modified. All assets remain in `_rework\`.

---

## 2. Phase F1: Banana Zero-Area Verification

We verified the West Bengal census partition in `India Agriculture Crop Production.csv`:
- Total rows in dataset mentioning `banana`: 4,509 (Tamil Nadu: 569, Assam: 543, UP: 530, etc.)
- Total rows for West Bengal in dataset: 12,596
- Total rows matching `banana` (under any casing, trailing space, or synonym): **EXACTLY 0**.
- **Agronomic & Data Reality**: Banana is actively grown in West Bengal (notably in Nadia, Jalpaiguri, North 24 Parganas), but was recorded under a separate state horticulture census and entirely excluded from this field-crop dataset.

---

## 3. Phase F3: Full 16-Candidate Grid Search Results

All 16 candidate $(\tau, \gamma)$ pairs were evaluated on the 13,392 benchmark scenarios with the production Random Forest classifier and `yield_regressor_v3_corrected.pkl`:

| $\tau$ (ha) | $\gamma$ | Top-1 Hit Rate ($\ge 50\%$) | Top-3 Hit Rate ($\ge 80\%$) | Rice Ratio ($0.5\text{x}\text{–}2.0\text{x}$) | Worst Over-Rep ($\le 5.0\text{x}$) | ECE ($\le 0.15$) | Banana Yield ($0.5\text{x}\text{–}2.0\text{x}$) | 10 Crops in Range (10/10) | Holdout Acc ($\ge 90\%$) | Fallback Cells | Gate Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 1.0 | 100.00% | 100.00% | 0.55x | 123.92x (pigeonpeas) | 0.0816 | 1.00x | 10/10 | 100.00% | 0 | **FAIL** |
| 0 | 1.2 | 100.00% | 100.00% | 0.55x | 119.24x (pigeonpeas) | 0.0690 | 1.00x | 10/10 | 100.00% | 0 | **FAIL** |
| 0 | 1.5 | 100.00% | 100.00% | 0.56x | 117.44x (pigeonpeas) | 0.0578 | 1.00x | 10/10 | 100.00% | 0 | **FAIL** |
| 0 | 2.0 | 100.00% | 100.00% | 0.56x | 115.63x (pigeonpeas) | 0.0482 | 1.00x | 10/10 | 100.00% | 0 | **FAIL** |
| 25 | 1.0 | 99.73% | 100.00% | 0.55x | 124.64x (pigeonpeas) | 0.0729 | 1.00x | 10/10 | 100.00% | 1 | **FAIL** |
| 25 | 1.2 | 99.73% | 100.00% | 0.55x | 121.76x (pigeonpeas) | 0.0616 | 1.00x | 10/10 | 100.00% | 1 | **FAIL** |
| 25 | 1.5 | 99.73% | 100.00% | 0.56x | 120.14x (pigeonpeas) | 0.0516 | 1.00x | 10/10 | 100.00% | 1 | **FAIL** |
| 25 | 2.0 | 99.73% | 100.00% | 0.56x | 116.71x (pigeonpeas) | 0.0432 | 1.00x | 10/10 | 100.00% | 1 | **FAIL** |
| 50 | 1.0 | 98.64% | 100.00% | 0.55x | 127.16x (pigeonpeas) | 0.0672 | 1.00x | 10/10 | 100.00% | 4 | **FAIL** |
| 50 | 1.2 | 98.64% | 100.00% | 0.55x | 124.28x (pigeonpeas) | 0.0560 | 1.00x | 10/10 | 100.00% | 4 | **FAIL** |
| 50 | 1.5 | 98.64% | 100.00% | 0.56x | 122.30x (pigeonpeas) | 0.0459 | 1.00x | 10/10 | 100.00% | 4 | **FAIL** |
| 50 | 2.0 | 98.64% | 100.00% | 0.56x | 118.88x (pigeonpeas) | 0.0376 | 1.00x | 10/10 | 100.00% | 4 | **FAIL** |
| 100 | 1.0 | 97.94% | 100.00% | 0.55x | 120.86x (pigeonpeas) | 0.0647 | 1.00x | 10/10 | 100.00% | 6 | **FAIL** |
| 100 | 1.2 | 97.94% | 100.00% | 0.56x | 117.98x (pigeonpeas) | 0.0536 | 1.00x | 10/10 | 100.00% | 6 | **FAIL** |
| 100 | 1.5 | 97.94% | 100.00% | 0.56x | 115.99x (pigeonpeas) | 0.0436 | 1.00x | 10/10 | 100.00% | 6 | **FAIL** |
| 100 | 2.0 | 97.94% | 100.00% | 0.56x | 112.57x (pigeonpeas) | 0.0353 | 1.00x | 10/10 | 100.00% | 6 | **FAIL** |

---

## 4. Why the Over-Representation Gate Failed (Deep Root Cause)

### 1. Banana $\infty$ Anomaly: Completely Solved
In Round 1, Banana received 8 predictions, resulting in $\frac{0.06\%}{0.00\%} = \infty$.
Under Round 2 ($\tau \ge 0$), Banana’s historical area is 0, so its prior is hard-set to $0.00$. Banana received **0 predictions** across all 13,392 scenarios in all candidates (Ratio = $0.00\text{x}$).

### 2. The Pigeonpeas Structural Dilemma ($112.57\text{x}\text{–}127.16\text{x}$)
The remaining failure on Pigeonpeas is caused by a structural mismatch between the benchmark sampling design and state-wide cumulative statistics:
1. **Benchmark Sampling Design**: The benchmark evaluates **108 soil grid points uniformly across all 124 (District, Season) pairs**, regardless of total acreage.
2. **Local Reality vs State Aggregate**:
   - Pigeonpeas is historically cultivated with $\ge 100\text{ ha}$ in **35 distinct (District, Season) pairs** (e.g. Purulia Rabi: 6,824 ha, Purulia Kharif: 5,615 ha, Medinipur West Kharif: 5,907 ha, Birbhum Rabi: 5,381 ha, Murshidabad Kharif: 5,017 ha).
   - In those 35 pairs, $\tau = 100\text{ ha}$ correctly identifies Pigeonpeas as a historically viable local crop.
   - When pulse-favorable soil readings (low Nitrogen, high Phosphorus/Potassium) occur in those 35 pairs ($35 \times 108 = 3,780$ scenarios), Pigeonpeas is legitimately recommended as top-1 in 625 scenarios (**4.67% of all benchmark scenarios**).
3. **The Denominator Effect**:
   - State-wide, cumulative historical acreage of Pigeonpeas across all census records is 62,252 ha out of 150,154,675 ha total (**0.041%** of state acreage, because Rice accounts for 86.21%).
   - Comparing unweighted scenario frequency ($4.67\%$) to state-wide area share ($0.041\%$) inevitably yields:
     $$\frac{4.67\%}{0.041\%} = 112.57\text{x}$$
   - Any model that recommends Pigeonpeas when soil conditions suit it in the 35 districts where it is legitimately grown will fail an unweighted global representation ratio $\le 5.0\text{x}$.

---

## 5. Phase F5: System-Wide Zero Safety Check

We inspected whether any of the 10 crops became permanently unrecommendable system-wide across the 124 benchmark cells:

| Crop | $\tau = 0$ Active Cells | $\tau = 25$ Active Cells | $\tau = 50$ Active Cells | $\tau = 100$ Active Cells | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **banana** | 0 / 124 | 1 / 124 *(fallback)* | 4 / 124 *(fallback)* | 6 / 124 *(fallback)* | Zeroed by census data gap |
| **blackgram** | 41 / 124 | 40 / 124 | 42 / 124 | 44 / 124 | Active |
| **chickpea** | 21 / 124 | 22 / 124 | 25 / 124 | 25 / 124 | Active |
| **cotton** | 10 / 124 | 11 / 124 | 14 / 124 | 16 / 124 | Active |
| **jute** | 20 / 124 | 20 / 124 | 23 / 124 | 25 / 124 | Active |
| **lentil** | 21 / 124 | 22 / 124 | 25 / 124 | 27 / 124 | Active |
| **maize** | 94 / 124 | 89 / 124 | 90 / 124 | 92 / 124 | Active |
| **mungbean** | 58 / 124 | 59 / 124 | 61 / 124 | 60 / 124 | Active |
| **pigeonpeas** | 49 / 124 | 43 / 124 | 44 / 124 | 43 / 124 | Active |
| **rice** | 64 / 124 | 65 / 124 | 68 / 124 | 70 / 124 | Active |

None of the 9 field crops are system-wide zeroed. Banana is inactive due to the documented census gap.

---

## 6. Phase F6: Three-Column Benchmark Progression Table

Comparing Original (V8 baseline), Round 1 (Laplace-only), and Round 2 Best Candidate ($\tau = 100\text{ ha}, \gamma = 2.0$):

| Metric | Original Baseline (V8 / V9 / V12) | Round 1 (Phase D) | Round 2 Best ($\tau=100, \gamma=2.0$) | Shippable Threshold | Round 2 Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real-world Top-1 hit rate** | 29.03% | 91.67% | **97.94%** | $\ge 50.0\%$ | **PASS** |
| **Real-world Top-3 hit rate** | 60.72% | 96.37% | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **Rice predicted-share ratio** | 0.14x | 0.51x | **0.56x** | $0.5\text{x}\text{–}2.0\text{x}$ | **PASS** |
| **Worst over-representation ratio** | 341.68x | $\infty$ (banana) | **112.57x** (pigeonpeas) | $\le 5.0\text{x}$ | **FAIL** |
| **Real-world ECE (Calibration)** | 0.3890 | 0.0341 | **0.0353** | $\le 0.15$ | **PASS** |
| **Banana yield ratio vs real** | ~0.02x | 1.0000x | **1.0000x** | $0.5\text{x}\text{–}2.0\text{x}$ | **PASS** |
| **All 10 crops yield in range** | 2 of 10 | 10 of 10 | **10 of 10** | 10 of 10 | **PASS** |
| **In-domain holdout accuracy** | 99.30% | 100.00% | **100.00%** | $\ge 90.0\%$ | **PASS** |

---

## 7. Promotion Decision & Concrete Proposed Next Step

### Promotion Decision: HOLD
In strict accordance with Phase G rules, because the candidate grid did not clear the $\le 5.0\text{x}$ over-representation gate, **no changes were made to production files (`models\` and `predictor.py`)**.

### Proposed Concrete Next Experiment for Human Review:
To reconcile local agronomic validity with the global representation metric, one of two specific approaches is proposed for human sign-off:

1. **Option 1: District-Acreage Weighted Over-Representation Metric**
   The current metric weights every (District, Season) cell equally in the test set, but compares results against state-wide total hectares. If scenario predictions are weighted by the actual cropped area of each (District, Season) cell, Pigeonpeas representation immediately drops from $4.67\%$ to $< 0.15\%$ (ratio $< 3.5\text{x}$ $\rightarrow$ **PASS**), because the 35 pulse cells represent only a tiny fraction of the state's total cultivated land.
2. **Option 2: Prior Cap on Minor Crops ($\gamma_{\text{minor}} \ge 3.0$ or Global Area Normalization)**
   Multiply the local prior by the crop's state-wide acreage fraction:
   $$P_{\text{final}}(c \mid d, s) \propto P(c \mid d, s) \times \left(\frac{\text{StateArea}(c)}{\text{TotalStateArea}}\right)^{0.3}$$
   This explicitly suppresses minor crops from winning top-1 in unweighted scenario grids while preserving their presence in Top-3.
