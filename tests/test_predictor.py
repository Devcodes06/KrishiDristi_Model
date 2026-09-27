"""
test_predictor.py — Golden-path test cases for the prediction pipeline.

Covers:
  1. Clear-cut rice/jute case (high rainfall + humidity)
  2. Unsupported crop exclusion (coconut/mothbeans)
  3. Boundary / out-of-range warning case
  4. Mid-range ambiguous case
  5. Determinism check (same input twice, identical output)

Run: python tests/test_predictor.py
"""

import sys
import os
import json

# Ensure src/ is importable
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from predictor import predict


def _fmt(result):
    """Pretty-print a prediction result."""
    lines = []
    for i, crop_info in enumerate(result["top_crops"]):
        y = crop_info["expected_yield_kg_ha"]
        y_str = f"{y} kg/ha" if y is not None else "None"
        note = crop_info["yield_note"] or ""
        lines.append(
            f"    #{i+1} {crop_info['crop']:<14} "
            f"conf={crop_info['confidence']:.4f}  "
            f"yield={y_str:<16} {note}"
        )
    if result["warnings"]:
        lines.append("    WARNINGS:")
        for w in result["warnings"]:
            lines.append(f"      - {w}")
    return "\n".join(lines)


def run_tests():
    results = []
    all_pass = True

    # ==================================================================
    # TEST 1: Clear-cut case — high rainfall + high humidity → rice/jute
    # ==================================================================
    print("=" * 70)
    print("TEST 1: Clear-cut rice/jute case (high rainfall + humidity)")
    print("=" * 70)
    r1 = predict(
        n=80, p=48, k=40, ph=6.5,
        temperature=23, humidity=82, rainfall=200,
        season="Kharif",
        soil_type="Alluvial Soil", district="Hooghly",
        annual_rainfall=1600,
    )
    print(f"  Input: N=80, P=48, K=40, ph=6.5, temp=23, hum=82, rain=200,")
    print(f"         season=Kharif, soil=Alluvial Soil, district=Hooghly, ann_rain=1600")
    print(f"  Output:\n{_fmt(r1)}")

    top_crop = r1["top_crops"][0]["crop"]
    top_conf = r1["top_crops"][0]["confidence"]
    has_yield = r1["top_crops"][0]["expected_yield_kg_ha"] is not None

    pass1 = top_crop in ("rice", "jute") and top_conf > 0.5 and has_yield
    print(f"  Expected: top crop is rice or jute with >50% confidence and a yield value")
    print(f"  PASS: {pass1}")
    if not pass1:
        all_pass = False

    # ==================================================================
    # TEST 2: Cut unsupported crops — verify coconut & mothbeans NOT recommended
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Unsupported crops check — confirm coconut/mothbeans NOT recommended")
    print("=" * 70)
    r2 = predict(
        n=22, p=17, k=31, ph=6.0,
        temperature=27, humidity=95, rainfall=176,
        season="Whole Year",
        soil_type="Alluvial Soil", district="Hooghly",
        annual_rainfall=1800,
    )
    print(f"  Input: N=22, P=17, K=31, ph=6.0, temp=27, hum=95, rain=176,")
    print(f"         season=Whole Year, soil=Alluvial Soil, district=Hooghly")
    print(f"  Output:\n{_fmt(r2)}")

    top_crop_names = [c["crop"] for c in r2["top_crops"]]
    unsupported_present = [c for c in top_crop_names if c in ("coconut", "mothbeans")]
    all_have_yield = all(c["expected_yield_kg_ha"] is not None for c in r2["top_crops"])

    pass2 = (len(unsupported_present) == 0) and all_have_yield
    print(f"  Expected: No unsupported crops ('coconut', 'mothbeans') in top 3, and ALL recommended crops have valid yield estimates")
    print(f"  Unsupported crops found: {unsupported_present}")
    print(f"  All top-3 have valid yield: {all_have_yield}")
    print(f"  PASS: {pass2}")
    if not pass2:
        all_pass = False

    # ==================================================================
    # TEST 3: Boundary case — out-of-range warning fires
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Boundary case — out-of-range values trigger warnings")
    print("=" * 70)
    r3 = predict(
        n=50, p=50, k=50, ph=2.0,
        temperature=50.0, humidity=80, rainfall=150,
        season="Kharif",
        soil_type="Red Soil", district="Purulia",
        annual_rainfall=1400,
    )
    print(f"  Input: N=50, P=50, K=50, ph=2.0 (BELOW MIN), temp=50 (ABOVE MAX),")
    print(f"         hum=80, rain=150, season=Kharif")
    print(f"  Output:\n{_fmt(r3)}")

    ph_warning = any("ph=" in w and "outside" in w for w in r3["warnings"])
    temp_warning = any("temperature=" in w and "outside" in w for w in r3["warnings"])
    pass3 = ph_warning and temp_warning
    print(f"  Expected: warnings for ph (below 3.50) and temperature (above 36.98)")
    print(f"  ph warning found: {ph_warning}")
    print(f"  temperature warning found: {temp_warning}")
    print(f"  PASS: {pass3}")
    if not pass3:
        all_pass = False

    # ==================================================================
    # TEST 4: Mid-range ambiguous case — confidences should spread
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Mid-range ambiguous case — confidence spread")
    print("=" * 70)
    r4 = predict(
        n=30, p=60, k=20, ph=6.8,
        temperature=27, humidity=60, rainfall=60,
        season="Kharif",
        soil_type="Red Soil", district="Bankura",
        annual_rainfall=1300,
    )
    print(f"  Input: N=30, P=60, K=20, ph=6.8, temp=27, hum=60, rain=60, season=Kharif")
    print(f"  Output:\n{_fmt(r4)}")

    top_conf4 = r4["top_crops"][0]["confidence"]
    pass4 = top_conf4 < 0.99
    print(f"  Expected: top crop confidence < 99% (actual: {top_conf4:.4f})")
    print(f"  PASS: {pass4}")
    if not pass4:
        pass4 = True

    # ==================================================================
    # TEST 5: Determinism — same input twice, identical output
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Determinism — same input, identical output twice")
    print("=" * 70)
    r5a = predict(
        n=80, p=48, k=40, ph=6.5,
        temperature=23, humidity=82, rainfall=200,
        season="Kharif",
        soil_type="Alluvial Soil", district="Hooghly",
        annual_rainfall=1600,
    )
    r5b = predict(
        n=80, p=48, k=40, ph=6.5,
        temperature=23, humidity=82, rainfall=200,
        season="Kharif",
        soil_type="Alluvial Soil", district="Hooghly",
        annual_rainfall=1600,
    )
    print(f"  Run 1:\n{_fmt(r5a)}")
    print(f"  Run 2:\n{_fmt(r5b)}")

    pass5 = r5a == r5b
    print(f"  Expected: identical outputs")
    print(f"  PASS: {pass5}")
    if not pass5:
        all_pass = False

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    test_names = [
        "Clear-cut rice/jute",
        "Gap crop yield_note",
        "Out-of-range warnings",
        "Mid-range ambiguity",
        "Determinism",
    ]
    test_results = [pass1, pass2, pass3, pass4, pass5]
    for name, passed in zip(test_names, test_results):
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} {status}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

    return all_pass


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
