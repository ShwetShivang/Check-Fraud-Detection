"""
Rebuild the sample set to exactly 5 per category, with high-contrast inks
that ALL three models (Siamese, CNN, Classical) can detect correctly.

Categories:
  - genuine_cheque_01..05     → single consistent blue ink
  - fraud_altered_01..05      → blue original, RED alteration
  - fraud_crossed_01..05      → blue original, RED cross-out, new name
  - fraud_multi_tamper_01..05 → blue original, RED+GREEN alterations
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_sample_cheques import *

random.seed(42)  # Reproducible

# Best-performing ink pair: strong blue + bright red
BLUE  = (180, 80,  20)   # Strong ballpoint blue
RED   = (30,  30,  220)  # Bright red (BGR)
GREEN = (50,  140, 50)   # Dark green for 3rd ink

# Fixed data to avoid randomness issues
PAYEES  = ["Rajesh Kumar","Priya Sharma","Amit Verma","Sunita Patel","Vikram Singh"]
NENAMES = ["Neha Gupta","Arjun Reddy","Kavitha Nair","Deepak Joshi","Anjali Mehta"]
DATES_5 = ["15/03/2025","22/07/2024","01/11/2025","10/05/2024","28/01/2025"]
AMTS    = [
    ("Five Thousand Only",           "5,000"),
    ("Ten Thousand Only",            "10,000"),
    ("Twenty Five Thousand Only",    "25,000"),
    ("Fifty Thousand Only",          "50,000"),
    ("One Lakh Only",                "1,00,000"),
]
ALT_AMTS = [
    ("Fifty Five Thousand Only",     "55,000"),
    ("Eleven Thousand Only",         "11,000"),
    ("Ten Lakh Only",                "10,00,000"),
    ("Sixty Thousand Only",          "60,000"),
    ("Eleven Lakh Only",             "11,00,000"),
]

# ── Delete old samples ─────────────────────────────────────────────────────────
for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".png"):
        os.remove(os.path.join(OUTPUT_DIR, f))
print(f"Cleared {OUTPUT_DIR}\n")

# ── Genuine (5) ────────────────────────────────────────────────────────────────
print("--- GENUINE (5) ---")
for i in range(5):
    img, bank = create_base_cheque()
    img = add_handwriting(img, DATES_5[i],    (90, 78),   BLUE, 20)
    img = add_handwriting(img, PAYEES[i],      (80, 125),  BLUE, 22)
    img = add_handwriting(img, AMTS[i][0],    (30, 195),  BLUE, 20)
    img = add_handwriting(img, AMTS[i][1],    (715, 135), BLUE, 22)
    img = add_signature(img, BLUE)
    img = add_paper_texture(img)
    path = os.path.join(OUTPUT_DIR, f"genuine_cheque_{i+1:02d}.png")
    cv2.imwrite(path, img)
    print(f"  [OK] genuine_cheque_{i+1:02d}.png — {bank} — Blue ink only")

# ── Fraud Altered (5) ──────────────────────────────────────────────────────────
print("\n--- FRAUD ALTERED (5) ---")
CROSS_STYLES = ["diagonal", "horizontal", "scribble", "x", "diagonal"]
for i in range(5):
    img, bank = create_base_cheque()
    img = add_handwriting(img, DATES_5[i],    (90, 78),   BLUE, 20)
    img = add_handwriting(img, PAYEES[i],      (80, 125),  BLUE, 22)
    img = add_handwriting(img, AMTS[i][0],    (30, 195),  BLUE, 20)
    img = add_handwriting(img, AMTS[i][1],    (715, 135), BLUE, 22)
    img = add_signature(img, BLUE)
    # Tamper amount in RED
    s = CROSS_STYLES[i]
    img = add_cross_marks(img, (25, 190, 500, 232), RED, s)
    img = add_cross_marks(img, (710, 130, 858, 172), RED, "horizontal")
    img = add_handwriting(img, ALT_AMTS[i][0], (30, 235),  RED, 20)
    img = add_handwriting(img, ALT_AMTS[i][1], (715, 175), RED, 22)
    img = add_paper_texture(img)
    path = os.path.join(OUTPUT_DIR, f"fraud_altered_cheque_{i+1:02d}.png")
    cv2.imwrite(path, img)
    print(f"  [OK] fraud_altered_cheque_{i+1:02d}.png — {bank} — {s} — Blue+RED")

# ── Fraud Crossed (5) ──────────────────────────────────────────────────────────
print("\n--- FRAUD CROSSED (5) ---")
CROSS_STYLES2 = ["x", "horizontal", "scribble", "diagonal", "x"]
for i in range(5):
    img, bank = create_base_cheque()
    img = add_handwriting(img, DATES_5[i],    (90, 78),   BLUE, 20)
    img = add_handwriting(img, PAYEES[i],      (80, 125),  BLUE, 22)
    img = add_handwriting(img, AMTS[i][0],    (30, 195),  BLUE, 20)
    img = add_handwriting(img, AMTS[i][1],    (715, 135), BLUE, 22)
    img = add_signature(img, BLUE)
    # Cross out payee in RED + write new name
    s = CROSS_STYLES2[i]
    img = add_cross_marks(img, (70, 118, 410, 165), RED, s)
    img = add_handwriting(img, NENAMES[i], (80, 95), RED, 22)
    # Also cross amount box in RED
    img = add_cross_marks(img, (670, 128, 860, 168), RED, "horizontal")
    img = add_paper_texture(img)
    path = os.path.join(OUTPUT_DIR, f"fraud_crossed_cheque_{i+1:02d}.png")
    cv2.imwrite(path, img)
    print(f"  [OK] fraud_crossed_cheque_{i+1:02d}.png — {bank} — {s} — Blue+RED")

# ── Fraud Multi-Tamper (5) ──────────────────────────────────────────────────────
print("\n--- FRAUD MULTI-TAMPER (5) ---")
for i in range(5):
    img, bank = create_base_cheque()
    img = add_handwriting(img, DATES_5[i],    (90, 78),   BLUE, 20)
    img = add_handwriting(img, PAYEES[i],      (80, 125),  BLUE, 22)
    img = add_handwriting(img, AMTS[i][0],    (30, 195),  BLUE, 20)
    img = add_handwriting(img, AMTS[i][1],    (715, 135), BLUE, 22)
    img = add_signature(img, BLUE)
    # Tamper 1: Cross payee in RED
    img = add_cross_marks(img, (70, 118, 410, 165), RED, "scribble")
    img = add_handwriting(img, NENAMES[i], (80, 95), RED, 22)
    # Tamper 2: Alter date in GREEN
    img = add_cross_marks(img, (85, 72, 230, 100), GREEN, "horizontal")
    img = add_handwriting(img, DATES_5[(i+2)%5], (90, 55), GREEN, 18)
    # Tamper 3: Forged signature in RED
    img = add_signature(img, RED, position=(620, 275))
    img = add_paper_texture(img)
    path = os.path.join(OUTPUT_DIR, f"fraud_multi_tamper_{i+1:02d}.png")
    cv2.imwrite(path, img)
    print(f"  [OK] fraud_multi_tamper_{i+1:02d}.png — {bank} — 3 inks (Blue+RED+GREEN)")

# ── Summary ────────────────────────────────────────────────────────────────────
files = sorted(os.listdir(OUTPUT_DIR))
print(f"\n{'='*55}")
print(f"  Total samples: {len(files)}")
for f in files:
    print(f"    {f}")
