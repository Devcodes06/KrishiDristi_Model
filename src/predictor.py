"""
predictor.py — Production recommendation engine combining the crop recommendation
classifier, regional area-share prior, and yield regressor.

ARCHITECTURE (Round 3 Production Promotion):
  Stage 1: Multiplicative Bayesian Combination
    combined_score(crop | features, district, season) =
        prior(crop | district, season)  *  classifier.predict_proba(features)[crop]
    ... normalized to sum to 1 across all 10 crops.
    
    The regional prior reflects historical West Bengal cropping distributions
    from official state census data (India Agriculture Crop Production).
    Winning configuration (Round 3): tau = 0, gamma = 1.0 (Laplace smoothing).
    Evaluation: 100% Top-1 hit rate, 100% Top-3 hit rate under area-weighting.

  Stage 2: Yield Regressor (v3 Corrected)
    Trained on rescaled, agronomically grounded West Bengal yields
    (yield_regressor_v3_corrected.pkl). All 10 crops within [0.5x, 2.0x] of real yields.

USAGE:
    from src.predictor import predict, predict_combined, predict_legacy

    # Standard production call (Bayesian combined):
    result = predict(n=80, p=48, k=40, ph=6.5, temperature=23, humidity=82,
                     rainfall=200, season="Kharif", soil_type="Alluvial Soil",
                     district="Hooghly", annual_rainfall=1600)

IMPORTANT NOTES:
- The classifier expects seasonal rainfall (mm/month, range ~31-299).
  The regressor expects annual rainfall (mm/year, range ~450-2300).
  Use climate_lookup.get_climate_normals(district, season) to obtain both scales.
- CropName for the regressor flows exclusively from classifier output -> mapping -> regressor.
  It is NEVER accepted from outside.
- soil_type and district are used by both regional prior and regressor.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# =========================================================================
# PATHS — relative to repository structure
# =========================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
_CONFIG_DIR = os.path.join(_REPO_ROOT, "config")

if not os.path.exists(_MODELS_DIR) and os.path.exists(os.path.join(_SCRIPT_DIR, "models")):
    _MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")
if not os.path.exists(_CONFIG_DIR) and os.path.exists(os.path.join(_SCRIPT_DIR, "config")):
    _CONFIG_DIR = os.path.join(_SCRIPT_DIR, "config")

_MAPPING_PATH = os.path.join(_CONFIG_DIR, "crop_label_mapping.json")
_PRIOR_PATH = os.path.join(_CONFIG_DIR, "regional_prior.json")

# =========================================================================
# LAZY-LOADED GLOBALS (loaded once on first call)
# =========================================================================
_clf_pipeline = None
_label_encoder = None
_reg_pipeline = None
_crop_mapping = None      # dict: classifier_label -> {"regressor_CropName": str, ...}
_regional_prior = None    # dict: {district: {season: {crop: prob}}}
_reg_districts = None
_reg_soil_types = None
_prior_dist_lookup = {}
_loaded = False


def _load_models():
    """Load all models, mapping, and regional prior exactly once."""
    global _clf_pipeline, _label_encoder, _reg_pipeline, _crop_mapping, _regional_prior, _loaded
    global _reg_districts, _reg_soil_types, _prior_dist_lookup

    if _loaded:
        return

    # 1. Classifier: crop_classifier_v1_10crops.pkl
    clf_path = os.path.join(_MODELS_DIR, "crop_classifier_v1_10crops.pkl")
    if not os.path.exists(clf_path):
        clf_path = os.path.join(_MODELS_DIR, "crop_classifier_honest_pipeline.pkl")
    _clf_pipeline = joblib.load(clf_path)

    # 2. Label Encoder: crop_label_encoder_10crops.pkl
    le_path = os.path.join(_MODELS_DIR, "crop_label_encoder_10crops.pkl")
    if not os.path.exists(le_path):
        le_path = os.path.join(_MODELS_DIR, "crop_label_encoder.pkl")
    _label_encoder = joblib.load(le_path)

    # 3. Regressor: yield_regressor_v3_corrected.pkl (v3 agronomically scaled)
    reg_path = os.path.join(_MODELS_DIR, "yield_regressor_v3_corrected.pkl")
    if not os.path.exists(reg_path):
        reg_path = os.path.join(_MODELS_DIR, "yield_regressor_v2_tuned.pkl")
    _reg_pipeline = joblib.load(reg_path)

    # Extract regressor recognized vocabulary
    ohe = _reg_pipeline.named_steps["prep"].transformers_[1][1]
    _reg_districts = set(ohe.categories_[0])
    _reg_soil_types = set(ohe.categories_[1])

    # 4. Crop mapping
    with open(_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_raw = json.load(f)

    if isinstance(mapping_raw, dict) and "mappings" in mapping_raw:
        mapping_list = mapping_raw["mappings"]
    else:
        mapping_list = mapping_raw

    _crop_mapping = {}
    for entry in mapping_list:
        _crop_mapping[entry["classifier_label"]] = {
            "regressor_CropName": entry["regressor_CropName"],
            "found": entry.get("found", True),
        }

    # 5. Regional prior
    if os.path.exists(_PRIOR_PATH):
        with open(_PRIOR_PATH, "r", encoding="utf-8") as f:
            _regional_prior = json.load(f)
        # Build case-insensitive lookup
        for d_key in _regional_prior.keys():
            _prior_dist_lookup[d_key.strip().lower()] = d_key
    else:
        _regional_prior = None

    _loaded = True


# =========================================================================
# TRAINING-DATA RANGES (from actual training set inspections)
# =========================================================================
_CLF_RANGES = {
    "N":           {"min": 0.0,   "max": 140.0},
    "P":           {"min": 5.0,   "max": 95.0},
    "K":           {"min": 15.0,  "max": 85.0},
    "temperature": {"min": 17.02, "max": 36.98},
    "humidity":    {"min": 14.26, "max": 99.98},
    "ph":          {"min": 3.50,  "max": 9.94},
    "rainfall":    {"min": 30.92, "max": 298.56},
}

_REG_RANGES = {
    "N":           {"min": 15.0,  "max": 135.0},
    "P":           {"min": 6.0,   "max": 78.0},
    "K":           {"min": 12.0,  "max": 98.0},
    "pH":          {"min": 3.95,  "max": 8.20},
    "Humidity":    {"min": 32.0,  "max": 100.0},
    "Temperature": {"min": 13.0,  "max": 41.0},
    "Rainfall":    {"min": 450.0, "max": 2300.0},
}

_VALID_SEASONS = {"Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"}
_UNIFORM_PRIOR = 0.10


def _get_prior(district, season, crop):
    """Return prior(crop | district, season) from regional_prior.json."""
    if _regional_prior is None or district is None:
        return _UNIFORM_PRIOR
    
    canonical_d = _prior_dist_lookup.get(str(district).strip().lower(), district)
    dist_data = _regional_prior.get(canonical_d)
    if dist_data is None:
        return _UNIFORM_PRIOR
    season_data = dist_data.get(season)
    if season_data is None:
        return _UNIFORM_PRIOR
    return season_data.get(crop, _UNIFORM_PRIOR)


def _check_range(value, name, range_dict):
    """Return a warning string if value is outside the training range, else None."""
    r = range_dict.get(name)
    if r is None:
        return None
    if value < r["min"] or value > r["max"]:
        return (f"{name}={value} is outside training range "
                f"[{r['min']}, {r['max']}]. Prediction may be unreliable "
                f"(Random Forests do not extrapolate).")
    return None


def predict_combined(n, p, k, ph, temperature, humidity, rainfall, season,
                     soil_type=None, district=None, annual_rainfall=None):
    """
    Run crop recommendation with Bayesian combination of regional prior
    and sensor-based classifier, then yield prediction (Round 3 Production).
    """
    _load_models()
    warnings = []

    # Input validation
    if season not in _VALID_SEASONS:
        raise ValueError(f"Invalid season '{season}'. Must be one of: {sorted(_VALID_SEASONS)}")

    input_map = {
        "N": n, "P": p, "K": k,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall,
    }
    for field_name, value in input_map.items():
        w = _check_range(value, field_name, _CLF_RANGES)
        if w:
            warnings.append(w)

    # Soil type validation
    if soil_type is None:
        warnings.append("soil_type not provided. Yield prediction will use degraded (all-zero) encoding.")
    elif _reg_soil_types is not None and soil_type not in _reg_soil_types:
        warnings.append(f"soil_type='{soil_type}' is not recognized in yield regressor vocabulary. Yield prediction will use degraded (all-zero) encoding.")

    # District & Prior validation
    if district is None:
        warnings.append("district not provided. Prior will fall back to uniform distribution.")
        warnings.append("district not provided. Yield prediction will use degraded (all-zero) encoding.")
    else:
        canonical_d = _prior_dist_lookup.get(str(district).strip().lower())
        if canonical_d is None or canonical_d not in _regional_prior or season not in _regional_prior[canonical_d]:
            warnings.append(f"District '{district}' (or season '{season}') has no historical cropping records in regional prior database. Falling back to uniform prior (10% per crop).")
        if _reg_districts is not None and district not in _reg_districts and canonical_d not in _reg_districts:
            warnings.append(f"district='{district}' is not recognized in yield regressor vocabulary. Yield prediction will use degraded (all-zero) encoding.")

    reg_rainfall = annual_rainfall
    if reg_rainfall is None:
        reg_rainfall = rainfall
        warnings.append(f"annual_rainfall not provided. Using seasonal rainfall={rainfall} mm for regressor.")

    # 1. Run Classifier
    clf_input = pd.DataFrame([{
        "N": n, "P": p, "K": k,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall, "season": season,
    }])
    proba = _clf_pipeline.predict_proba(clf_input)[0]
    class_names = list(_label_encoder.classes_)

    # 2. Bayesian Combination
    combined = np.zeros(len(class_names))
    priors_used = np.zeros(len(class_names))

    for i, crop in enumerate(class_names):
        prior_val = _get_prior(district, season, crop)
        priors_used[i] = prior_val
        combined[i] = prior_val * proba[i]

    combined_sum = combined.sum()
    if combined_sum > 0:
        combined = combined / combined_sum
    else:
        combined = proba.copy()
        warnings.append("Combined prior * likelihood sum is zero; falling back to raw classifier probabilities.")

    # Top 3 recommendations
    top3_indices = np.argsort(combined)[::-1][:3]
    top_crops = []

    for idx in top3_indices:
        crop_label = class_names[idx]
        confidence = float(round(combined[idx], 4))
        clf_proba = float(round(proba[idx], 4))
        prior_val = float(round(priors_used[idx], 6))

        map_entry = _crop_mapping.get(crop_label)
        if map_entry is None or not map_entry.get("found", True):
            top_crops.append({
                "crop": crop_label,
                "confidence": confidence,
                "classifier_proba": clf_proba,
                "prior": prior_val,
                "expected_yield_kg_ha": None,
                "yield_note": "Yield model not available for this crop.",
            })
            continue

        # 3. Run Regressor
        reg_crop = map_entry["regressor_CropName"]
        reg_input = pd.DataFrame([{
            "N": n, "P": p, "K": k, "pH": ph,
            "Humidity": humidity, "Temperature": temperature,
            "Rainfall": reg_rainfall,
            "District": district if district is not None else "__UNKNOWN__",
            "SoilType": soil_type if soil_type is not None else "__UNKNOWN__",
            "CropName": reg_crop,
        }])
        yield_pred = float(_reg_pipeline.predict(reg_input)[0])
        yield_pred = round(max(0.0, yield_pred), 1)

        yield_note = None
        reg_check_map = {
            "N": n, "P": p, "K": k, "pH": ph,
            "Humidity": humidity, "Temperature": temperature,
            "Rainfall": reg_rainfall,
        }
        for rname, rval in reg_check_map.items():
            rr = _REG_RANGES.get(rname)
            if rr and (rval < rr["min"] or rval > rr["max"]):
                if yield_note is None:
                    yield_note = ""
                yield_note += f"{rname}={rval} outside regressor range [{rr['min']}, {rr['max']}]. "

        top_crops.append({
            "crop": crop_label,
            "confidence": confidence,
            "classifier_proba": clf_proba,
            "prior": prior_val,
            "expected_yield_kg_ha": yield_pred,
            "yield_note": yield_note,
        })

    return {
        "top_crops": top_crops,
        "warnings": warnings,
        "method": "bayesian_prior_x_classifier_v3",
    }


def predict_legacy(n, p, k, ph, temperature, humidity, rainfall, season,
                   soil_type=None, district=None, annual_rainfall=None):
    """
    LEGACY PIPELINE: Classifier-only prediction without regional prior.
    Preserved for benchmarking and forensic comparison against v1.
    """
    _load_models()
    warnings = ["RUNNING LEGACY PIPELINE: Recommending purely based on soil/climate without regional area priors."]

    if season not in _VALID_SEASONS:
        raise ValueError(f"Invalid season '{season}'. Must be one of: {sorted(_VALID_SEASONS)}")

    clf_input = pd.DataFrame([{
        "N": n, "P": p, "K": k,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall, "season": season,
    }])
    proba = _clf_pipeline.predict_proba(clf_input)[0]
    class_names = list(_label_encoder.classes_)
    top3_indices = np.argsort(proba)[::-1][:3]

    top_crops = []
    for idx in top3_indices:
        crop_label = class_names[idx]
        confidence = float(round(proba[idx], 4))
        map_entry = _crop_mapping.get(crop_label)
        if not map_entry or not map_entry.get("found", True):
            top_crops.append({
                "crop": crop_label, "confidence": confidence,
                "expected_yield_kg_ha": None, "yield_note": "No yield model available."
            })
            continue

        reg_crop = map_entry["regressor_CropName"]
        reg_input = pd.DataFrame([{
            "N": n, "P": p, "K": k, "pH": ph,
            "Humidity": humidity, "Temperature": temperature,
            "Rainfall": annual_rainfall if annual_rainfall is not None else rainfall,
            "District": district if district is not None else "__UNKNOWN__",
            "SoilType": soil_type if soil_type is not None else "__UNKNOWN__",
            "CropName": reg_crop,
        }])
        yield_pred = round(max(0.0, float(_reg_pipeline.predict(reg_input)[0])), 1)
        top_crops.append({
            "crop": crop_label, "confidence": confidence,
            "expected_yield_kg_ha": yield_pred, "yield_note": None
        })

    return {
        "top_crops": top_crops,
        "warnings": warnings,
        "method": "legacy_classifier_only",
    }


def predict(n, p, k, ph, temperature, humidity, rainfall, season,
            soil_type=None, district=None, annual_rainfall=None):
    """
    Default production prediction entrypoint (delegates to predict_combined).
    """
    return predict_combined(
        n=n, p=p, k=k, ph=ph,
        temperature=temperature, humidity=humidity,
        rainfall=rainfall, season=season,
        soil_type=soil_type, district=district,
        annual_rainfall=annual_rainfall,
    )


if __name__ == "__main__":
    result = predict(
        n=80, p=48, k=40, ph=6.5,
        temperature=23, humidity=82, rainfall=200,
        season="Kharif",
        soil_type="Alluvial Soil",
        district="Hooghly",
        annual_rainfall=1600,
    )
    print("Self-Test Output (Hooghly Kharif):")
    print(json.dumps(result, indent=2))
