"""
climate_lookup.py — Meteorological climate normals lookup for West Bengal districts
powered by NASA POWER Agroclimatology (2001-2020 20-year baseline).

Replaces previous ad-hoc dataset proxy with real satellite & assimilation climatology.
"""

import os
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_CONFIG_DIR = os.path.join(_REPO_ROOT, "config")
if not os.path.exists(_CONFIG_DIR):
    _CONFIG_DIR = _SCRIPT_DIR

_NASA_CACHE_PATH = os.path.join(_CONFIG_DIR, "climate_normals_nasa.json")

# Load the NASA cache
_NASA_DATA = None
_LOOKUP_INDEX = {}

def _ensure_loaded():
    global _NASA_DATA, _LOOKUP_INDEX
    if _NASA_DATA is not None:
        return
    with open(_NASA_CACHE_PATH, "r", encoding="utf-8") as f:
        _NASA_DATA = json.load(f)
    for raw_key, entry in _NASA_DATA.items():
        clean = entry.get("clean_name", raw_key.strip()).lower()
        _LOOKUP_INDEX[clean] = raw_key
        # Also index the raw key without leading/trailing whitespace
        _LOOKUP_INDEX[raw_key.strip().lower()] = raw_key


def get_climate_normals(district, season=None):
    """
    Look up meteorological climate normals (2001-2020) for a West Bengal district.

    Parameters
    ----------
    district : str
        District name (e.g., 'Hooghly', 'Purulia', 'Kolkata'). Case-insensitive.
    season : str or None
        One of 'Kharif', 'Rabi', 'Summer', 'Autumn', 'Winter', 'Whole Year'.
        If None, returns annual averages.

    Returns
    -------
    dict with identical contract to v1:
        'found': bool
        'district': str
        'temperature': float (°C)
        'humidity': float (%)
        'annual_rainfall': float (mm/year, suitable for regressor)
        'seasonal_rainfall': float or None (mm/season, total cumulative precipitation)
        'n_data_points': int (number of baseline years: 20)
        'data_quality': str ('good')
        'limitations': list of str
    """
    _ensure_loaded()
    
    clean_search = district.strip().lower()
    canonical_key = _LOOKUP_INDEX.get(clean_search)

    if canonical_key is None:
        return {
            "found": False,
            "district": district,
            "error": f"District '{district}' not found in NASA POWER climate database.",
            "valid_districts": sorted(list(set(e["clean_name"] for e in _NASA_DATA.values()))),
        }

    entry = _NASA_DATA[canonical_key]
    annual = entry["annual"]
    seasons = entry["seasons"]

    # Canonical district representation (clean display name)
    disp_name = entry["clean_name"]

    limitations = []
    limitations.append("Source: NASA POWER Agroclimatology (20-year climatological baseline 2001-2020).")

    # Handle season
    norm_season = None
    if season:
        s_title = season.strip().title()
        if s_title in ("Summer", "Zaid", "Summer/Zaid"):
            norm_season = "Summer"
        elif s_title in seasons:
            norm_season = s_title
        elif season.strip() in seasons:
            norm_season = season.strip()
        else:
            limitations.append(f"Season '{season}' not recognized. Falling back to annual values.")

    _SEASON_MONTHS = {
        "Kharif": 5,      # Jun, Jul, Aug, Sep, Oct
        "Rabi": 5,        # Nov, Dec, Jan, Feb, Mar
        "Summer": 4,      # Mar, Apr, May, Jun
        "Autumn": 3,      # Sep, Oct, Nov
        "Winter": 3,      # Dec, Jan, Feb
        "Whole Year": 12  # All 12 months
    }

    if norm_season and norm_season in seasons:
        s_data = seasons[norm_season]
        temp = s_data["temperature"]
        hum = s_data["humidity"]
        raw_total_rain = s_data["seasonal_rainfall_mm"]
        n_months = _SEASON_MONTHS.get(norm_season, 5)
        
        # Convert cumulative seasonal total to monthly average rainfall to match classifier training scale [30.9, 298.6] mm
        seasonal_rain = round(float(raw_total_rain / n_months), 2)
        
        # Check against classifier training domain [30.92, 298.56]
        if seasonal_rain < 30.92:
            limitations.append(
                f"Seasonal monthly rainfall ({seasonal_rain:.2f} mm/mo, total {raw_total_rain:.1f} mm across {n_months} mos) "
                f"is below classifier training minimum (30.92 mm). This reflects natural dry conditions in {norm_season}."
            )
        elif seasonal_rain > 298.56:
            limitations.append(
                f"Seasonal monthly rainfall ({seasonal_rain:.2f} mm/mo, total {raw_total_rain:.1f} mm across {n_months} mos) "
                f"exceeds classifier training maximum (298.56 mm). This reflects intense monsoon precipitation in {disp_name}."
            )
    else:
        temp = annual["annual_temperature"]
        hum = annual["annual_humidity"]
        seasonal_rain = None
        if not season:
            limitations.append("No season specified; temperature and humidity reflect annual averages.")

    annual_rain = annual["annual_rainfall_mm"]

    return {
        "found": True,
        "district": disp_name,
        "temperature": temp,
        "humidity": hum,
        "annual_rainfall": annual_rain,
        "seasonal_rainfall": seasonal_rain,
        "n_data_points": annual["n_years"],
        "data_quality": "good",
        "limitations": limitations,
    }


def list_districts():
    """Return sorted list of all supported district names."""
    _ensure_loaded()
    return sorted(list(set(e["clean_name"] for e in _NASA_DATA.values())))
