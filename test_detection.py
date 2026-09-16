"""Test ALL THREE models on the 20 curated samples: 5 genuine + 15 fraud."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fraud_pipeline import FraudDetector

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
MODELS  = ['siamese', 'cnn', 'classical']

files = sorted([f for f in os.listdir(SAMPLES) if f.endswith('.png')])

print("\n" + "="*75)
print("  ALL-MODELS DETECTION TEST  (5 genuine + 15 fraud = 20 samples)")
print("="*75)

overall = {m: {'correct': 0, 'total': 0} for m in MODELS}
failures = []

for model_type in MODELS:
    try:
        detector = FraudDetector(model_type=model_type)
    except Exception as e:
        print(f"\n[SKIP] {model_type}: could not load - {e}")
        continue

    print(f"\n--- {model_type.upper()} MODEL ---")
    for fname in files:
        path = os.path.join(SAMPLES, fname)
        expected = "GENUINE" if fname.startswith("genuine") else "FRAUD"
        try:
            result = detector.detect(path)
            verdict = result['verdict']
            conf    = result['confidence']
        except Exception as e:
            verdict, conf = "ERROR", 0.0

        match   = verdict == expected
        symbol  = "OK  " if match else "FAIL"
        overall[model_type]['total']   += 1
        if match:
            overall[model_type]['correct'] += 1
        else:
            failures.append((model_type, fname, expected, verdict))

        print(f"  {symbol} {fname:<40s} -> {verdict:8s} ({conf:.0%})")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  SUMMARY")
print("="*75)
for m in MODELS:
    c = overall[m]['correct']
    t = overall[m]['total']
    if t == 0:
        continue
    pct = c / t * 100
    bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
    print(f"  {m.upper():<12s} [{bar}]  {c}/{t} ({pct:.0f}%)")

if failures:
    print(f"\n  Failed cases:")
    for m, f, exp, got in failures:
        print(f"    [{m}] {f} - expected {exp}, got {got}")
else:
    print("\n  ALL MODELS CORRECT ON ALL SAMPLES! [PASS]")
