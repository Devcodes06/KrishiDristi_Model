# Input Contract — Crop Recommendation & Yield Prediction System

> **Version**: 2.1
> **Date**: 2026-09-22
> **Status**: Decision made — Option B (keep collecting soil_type for yield only)

---

## 1. Plain-Language Summary

**Crop recommendation does not use soil type.** The classifier recommends
crops based only on soil chemistry (N, P, K, pH), climate (temperature,
humidity, rainfall), and season. Entering a different soil type will not
change which crop is recommended.

**Soil type only affects the yield estimate for the recommended crop.** The
yield regressor uses soil type, along with district and crop identity, to
predict expected kg/ha. If soil type is omitted, yield predictions still run
but with reduced accuracy.

**Crop recommendation does not use district.** The classifier does not use
the farmer's district at all. District only affects the yield estimate, not
which crop is recommended.

---

## 2. Input Fields Summary

| Field | Type | Required By | Source | Notes |
|---|---|---|---|---|
| `N` | float (mg/kg) | Classifier + Regressor | **Pi sensor (live)** | Soil nitrogen |
| `P` | float (mg/kg) | Classifier + Regressor | **Pi sensor (live)** | Soil phosphorus |
| `K` | float (mg/kg) | Classifier + Regressor | **Pi sensor (live)** | Soil potassium |
| `ph` | float (0–14) | Classifier + Regressor | **Pi sensor (live)** | Soil pH |
| `temperature` | float (°C) | Classifier + Regressor | **Climate lookup** by district + season | NOT a live Pi sensor reading (see Section 5) |
| `humidity` | float (%) | Classifier + Regressor | **Climate lookup** by district + season | NOT a live Pi sensor reading (see Section 5) |
| `rainfall` | float (mm) | Classifier only (seasonal scale) | **Climate lookup** by district + season | Seasonal ~31–299 mm |
| `annual_rainfall` | float (mm) | Regressor only (annual scale) | **Climate lookup** (annual total for district) | Annual ~450–2300 mm. See Section 3. |
| `season` | string | Classifier only | **Farmer selection** | One of: `Kharif`, `Rabi`, `Summer`, `Autumn`, `Winter`, `Whole Year` |
| `soil_type` | string | **Regressor only** | **Farmer selection / lookup** | Does NOT affect crop recommendation — yield only |
| `district` | string | **Regressor only** | **One-time farm configuration** | Does NOT affect crop recommendation — yield only. See Section 4. |

---

## 3. Rainfall Scale Difference (CRITICAL)

The two models were trained on datasets using **different rainfall scales**:

| Model | Rainfall Column | Scale | Training Range |
|---|---|---|---|
| Classifier | `rainfall` | **Seasonal** (mm per season) | 30.92 – 298.56 mm |
| Regressor | `Rainfall` | **Annual** (mm per year) | 450 – 2,300 mm |

**Rule**: The caller must supply TWO separate rainfall values:
- `rainfall` — the seasonal value, used by the classifier as-is.
- `annual_rainfall` — the annual value, used by the regressor.

If `annual_rainfall` is not provided, the function falls back to using the
seasonal `rainfall` value for the regressor and emits a warning. **This
fallback produces wrong-scale input** (e.g., 200mm seasonal fed into a
model trained on 450–2300mm annual). The yield prediction under this
fallback is unreliable and should not be presented to the farmer without
flagging it.

**There is no internal conversion formula.** A principled conversion
requires the climate lookup component (`climate_lookup.py`) to return both
seasonal and annual rainfall for each district+season. Until that component
is integrated, the caller is responsible for providing both values.

---

## 4. District and Soil Type — Decision (Option B Implemented)

### Soil Type

**Decision: Keep collecting soil_type (Option B).**

Soil type is collected from the farmer and passed to the yield regressor.
The crop recommendation classifier does not use it. The farmer-facing UI
should clearly label the soil type field as:

> *"Used for yield estimation only — does not affect crop recommendation."*

The valid soil type values (from the regressor's training data, all West
Bengal) are:
`Alluvial Soil`, `Alluvial & Red/Yellow Soil`, `Black Soil`,
`Coastal Alluvial Soil`, `Laterite Soil`, `Mountainous Soil`,
`Red & Laterite Soil`, `Red Soil`, `Red and Laterite Soil`,
`Sandy Soil`, `Terai Alluvial Soil`

If the farmer does not know their soil type, leave it as `None`. The
regressor will still produce a prediction, but with degraded accuracy
(the SoilType dimension will be encoded as all-zeros).

### District

**District works the same way as soil type:** the classifier does NOT use
district. District only affects the yield regressor.

**Recommendation:** If a single deployment only ever serves one fixed farm
or location, `district` should be **set once as a fixed configuration
value** during device setup, rather than asked of the farmer on every
prediction. This avoids repetitive input and reduces user friction. The
Pi could store the district in a `config.json` at first boot.

The valid district values (23 districts, all in West Bengal) are:
`Bardhaman` (note: leading space in training data — use `" Bardhaman"`),
`Alipurduar`, `Bankura`, `Birbhum`, `Cooch Behar`, `Dakshin Dinajpur`,
`Darjeeling`, `Hooghly`, `Howrah`, `Jalpaiguri`, `Jhargram`, `Kalimpong`,
`Kolkata`, `Malda`, `Murshidabad`, `Nadia`, `North 24 Parganas`,
`Paschim Medinipur (West Medinipur)`, `Purba Medinipur (East Medinipur)`,
`Purulia`, `Siliguri`, `South 24 Parganas`,
`Uttar Dinajpur (North Dinajpur)`

---

## 5. Data Source Rules (Explicit)

This section restates which values come from where, so this rule is never
lost in downstream implementation.

### From Pi sensors (live, per-prediction)
- **N, P, K, pH** — read from the soil sensor hardware at prediction time.
  These are the only values that change between predictions at the same farm.

### From farmer selection (infrequent / one-time)
- **season** — selected once per planting cycle. Determines which classifier
  OHE column is active and which climate lookup row is used.
- **soil_type** — selected once per farm setup (or left blank). Only affects
  yield estimate, not crop recommendation.
- **district** — set once at device setup. Only affects yield estimate, not
  crop recommendation.

### From climate lookup (computed, not live)
- **temperature, humidity, rainfall, annual_rainfall** — these are
  **looked-up historical climate averages** for the farmer's district +
  season, NOT live sensor readings from a DHT11 or rain gauge.

A DHT11 sensor on the Pi measures ambient temperature and humidity, but
these readings represent one point in time at one spot, not the seasonal
average the models were trained on. Using live DHT11 readings as a proxy
for seasonal climate averages is a **known limitation** (see Section 6)
and should be documented as such in any user-facing material.

---

## 6. Known Limitations and Gaps

1. **Rainfall scale gap**: The classifier and regressor expect different
   rainfall scales (seasonal vs. annual). No internal conversion exists.
   The climate lookup component must return both values.

2. **Temperature/humidity lookup gap**: The original raw datasets
   do NOT contain temperature or humidity columns at
   district+season granularity (see Task 3 findings). These values
   currently have no real historical-lookup source. Options:
   - (a) Use live DHT11 readings as an imperfect proxy (documented limitation).
   - (b) Source an external climate dataset (e.g., IMD district-level normals).

3. **Geographic scope**: The yield regressor is trained only on West Bengal
   data (23 districts). Predictions for other states/districts will use
   degraded all-zero encoding.

4. **Gap crops**: `coconut` and `mothbeans` can be recommended by the
   classifier but have no yield training data. The predictor returns
   `yield=None` with an explanatory note for these crops.

5. **Dataset separability**: The classifier's ~99%+ accuracy reflects the
   synthetic separability of the Crop Recommendation Dataset, not real-world
   sensor noise. Expect lower accuracy with actual Pi sensor readings.
