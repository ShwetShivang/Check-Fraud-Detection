"""
Run all three trained models on the demo cheques in data/samples/.

  fraud_*    real IDRBT scans altered with a second pen   -> expected FRAUD
  genuine_*  the same cheques with pen 2's ink erased      -> expected GENUINE

Note: the demo cheques come from IDRBT, which the final models were trained
on.  For an unbiased estimate see the cross-validated numbers printed by
`python -m src.train` (outputs/models/ink_metrics.json).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fraud_pipeline import FraudDetector

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
MODELS = ['classical', 'siamese', 'cnn']

files = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith(('.jpg', '.png', '.tif')))

print("\n" + "=" * 78)
print(f"  ALL-MODELS DETECTION TEST  ({len(files)} demo cheques)")
print("=" * 78)

overall = {m: {'correct': 0, 'total': 0} for m in MODELS}
failures = []

for model_type in MODELS:
    try:
        detector = FraudDetector(model_type=model_type)
    except Exception as e:
        print(f"\n[SKIP] {model_type}: could not load - {e}")
        continue

    print(f"\n--- {model_type.upper()} (threshold {detector.threshold:.3f}) ---")
    for fname in files:
        expected = "GENUINE" if fname.startswith("genuine") else "FRAUD"
        try:
            r = detector.detect(os.path.join(SAMPLES, fname), rgb=False)
            verdict, score = r['verdict'], r['score']
        except Exception as e:
            verdict, score = f"ERROR ({e})", 0.0
        ok = verdict == expected
        overall[model_type]['total'] += 1
        overall[model_type]['correct'] += ok
        if not ok:
            failures.append((model_type, fname, expected, verdict))
        print(f"  {'OK  ' if ok else 'FAIL'} {fname:<34s} -> {verdict:8s} score {score:.3f}")

print("\n" + "=" * 78)
print("  SUMMARY")
print("=" * 78)
for m in MODELS:
    c, t = overall[m]['correct'], overall[m]['total']
    if t:
        pct = c / t * 100
        print(f"  {m.upper():<10s} [{'#' * int(pct // 5):<20s}]  {c}/{t} ({pct:.0f}%)")

if not any(overall[m]['total'] for m in MODELS):
    print("\n  NO MODELS COULD BE TESTED [FAIL]")
    sys.exit(1)
elif failures:
    print("\n  Failed cases:")
    for m, f, exp, got in failures:
        print(f"    [{m}] {f} - expected {exp}, got {got}")
else:
    print("\n  ALL MODELS CORRECT ON ALL SAMPLES [PASS]")
