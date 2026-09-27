"""
test_end_to_end.py — End-to-end integration test simulating the real product chain:
  [Pi Soil Sensor: N, P, K, pH]
  + [Farmer UI: District, Season, Soil Type]
  -> [NASA POWER Climate Lookup: Temperature, Humidity, Monthly Rainfall, Annual Rainfall]
  -> [Predictor: 10-Crop Tuned+Calibrated Classifier -> Crop Label Mapping -> Tuned Yield Regressor]
  -> [Final User Output: Top-3 Crops, Confidence %, Predicted Yield kg/ha, Warnings]

Run: python tests/test_end_to_end.py
"""

import sys
import os
import json

# Ensure src/ is importable
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from climate_lookup import get_climate_normals
from predictor import predict


def run_simulation(session_name, district, season, soil_type, sensor_readings, description):
    print("=" * 80)
    print(f"SIMULATION: {session_name}")
    print(f"Description: {description}")
    print("=" * 80)

    # 1. Farmer Inputs
    print("\n[Step 1] Farmer Context:")
    print(f"  District:  {district}")
    print(f"  Season:    {season}")
    print(f"  Soil Type: {soil_type}")

    # 2. Live Pi Sensor Readings
    n = sensor_readings["N"]
    p = sensor_readings["P"]
    k = sensor_readings["K"]
    ph = sensor_readings["ph"]
    print("\n[Step 2] Live Pi Sensor Readings:")
    print(f"  Nitrogen (N):   {n} mg/kg")
    print(f"  Phosphorus (P): {p} mg/kg")
    print(f"  Potassium (K):  {k} mg/kg")
    print(f"  pH:             {ph}")

    # 3. Auto Climate Lookup (NASA POWER 20-year normals)
    print("\n[Step 3] Automatic Climate Normals Lookup (NASA POWER):")
    climate = get_climate_normals(district, season)
    if not climate["found"]:
        print(f"  ERROR: {climate['error']}")
        return False

    temp = climate["temperature"]
    hum = climate["humidity"]
    seasonal_rain = climate["seasonal_rainfall"]
    annual_rain = climate["annual_rainfall"]
    print(f"  Temperature:       {temp} °C")
    print(f"  Relative Humidity: {hum} %")
    print(f"  Monthly Rainfall:  {seasonal_rain} mm/month (fed to classifier)")
    print(f"  Annual Rainfall:   {annual_rain} mm/year (fed to yield regressor)")
    if climate["limitations"]:
        print("  Climate Notes:")
        for lim in climate["limitations"]:
            print(f"    * {lim}")

    # 4. Pipeline Execution
    print("\n[Step 4] Executing Prediction Pipeline (predict)...")
    result = predict(
        n=n,
        p=p,
        k=k,
        ph=ph,
        temperature=temp,
        humidity=hum,
        rainfall=seasonal_rain,
        season=season,
        soil_type=soil_type,
        district=district,
        annual_rainfall=annual_rain
    )

    # 5. Output Verification
    print("\n[Step 5] Final Recommendations:")
    top_crops = result["top_crops"]
    for i, c in enumerate(top_crops, 1):
        crop = c["crop"]
        conf = c["confidence"]
        y_val = c["expected_yield_kg_ha"]
        y_str = f"{y_val:.1f} kg/ha" if y_val is not None else "None"
        note = f" ({c['yield_note']})" if c["yield_note"] else ""
        print(f"  #{i} {crop:<14} | Confidence: {conf*100:>5.1f}% | Expected Yield: {y_str:<15}{note}")

    print("\n  Warnings / Diagnostics:")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"    ! {w}")
    else:
        print("    (None — all parameters strictly inside training distributions)")

    # Assertions
    assert len(top_crops) > 0, "No crops returned!"
    for c in top_crops:
        assert c["expected_yield_kg_ha"] is not None, f"Missing yield for {c['crop']}!"
        assert c["crop"] not in ("coconut", "mothbeans"), f"Unsupported crop {c['crop']} returned in v1!"

    print("\n  VERIFICATION: PASS (valid top crops, non-null yields, 10-crop v1 compliant)")
    return True


def main():
    print("=" * 80)
    print("END-TO-END PRODUCT CHAIN VERIFICATION SUITE")
    print("=" * 80)

    # Test Case 1: Monsoon Rice Profile in Hooghly (Alluvial plain)
    t1 = run_simulation(
        session_name="Session 1: Monsoon Rice Profile",
        district="Hooghly",
        season="Kharif",
        soil_type="Alluvial Soil",
        sensor_readings={"N": 85.0, "P": 45.0, "K": 42.0, "ph": 6.4},
        description="High-fertility alluvial soil during Kharif monsoon. Expected: Rice or Jute as top crop."
    )

    # Test Case 2: Dry-Season Lentil Profile in Bankura (Laterite zone)
    t2 = run_simulation(
        session_name="Session 2: Dry-Season Pulse/Lentil Profile",
        district="Bankura",
        season="Rabi",
        soil_type="Laterite Soil",
        sensor_readings={"N": 24.0, "P": 65.0, "K": 25.0, "ph": 6.8},
        description="Low N, high P/K pulse profile during dry Rabi season. Expected: Lentil/Chickpea/Blackgram."
    )

    # Test Case 3: Ambiguous / Transition Profile in Purulia (Red Soil plateau)
    t3 = run_simulation(
        session_name="Session 3: Ambiguous Summer Profile",
        district="Purulia",
        season="Summer",
        soil_type="Red Soil",
        sensor_readings={"N": 55.0, "P": 42.0, "K": 38.0, "ph": 6.7},
        description="Moderate balanced NPK in hot summer plateau conditions. Expected: Well-distributed confidences."
    )

    print("\n" + "=" * 80)
    print("END-TO-END SUITE RESULT")
    print("=" * 80)
    if t1 and t2 and t3:
        print("ALL 3 END-TO-END SIMULATIONS PASSED SUCCESSFULLY.")
    else:
        print("FAILURES DETECTED IN END-TO-END SUITE.")


if __name__ == "__main__":
    main()
