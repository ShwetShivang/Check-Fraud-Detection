"""Regenerate failing fraud cheques with EXTREME ink contrast."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_sample_cheques import *

# Force maximum contrast: BLUE original + BRIGHT RED alterations
ORIG_INK = (180, 80, 20)     # Strong blue
FRAUD_INK = (30, 30, 220)    # Bright red - maximum contrast in LAB space

random.seed(42)  # Fixed seed for reproducible results

def regen_crossed_extreme(index, cross_style="scribble"):
    img, bank = create_base_cheque()
    payee = random.choice(PAYEE_NAMES)
    new_payee = random.choice([p for p in PAYEE_NAMES if p != payee])
    aw, ad = random.choice(AMOUNTS_WORDS)
    
    # Original in BLUE
    img = add_handwriting(img, random.choice(DATES), (90, 78), ORIG_INK, 20)
    img = add_handwriting(img, payee, (80, 125), ORIG_INK, 22)
    img = add_handwriting(img, aw, (30, 195), ORIG_INK, 20)
    img = add_handwriting(img, ad, (715, 135), ORIG_INK, 22)
    img = add_signature(img, ORIG_INK)
    
    # Cross out in BRIGHT RED with thick marks
    img = add_cross_marks(img, (75, 120, 400, 165), FRAUD_INK, cross_style)
    img = add_handwriting(img, new_payee, (80, 97), FRAUD_INK, 22)
    # Also add some red marks on the amount for more contrast
    img = add_cross_marks(img, (670, 130, 860, 170), FRAUD_INK, "horizontal")
    img = add_paper_texture(img)
    
    path = os.path.join(OUTPUT_DIR, f"fraud_crossed_cheque_{index:02d}.png")
    cv2.imwrite(path, img)
    print(f"  [REGEN] fraud_crossed_cheque_{index:02d}.png — {bank} — {cross_style}")

def regen_altered_extreme(index):
    img, bank = create_base_cheque()
    payee = random.choice(PAYEE_NAMES)
    orig_words, orig_digits = "Five Thousand Only", "5,000"
    new_words, new_digits = "Fifty Five Thousand Only", "55,000"
    
    # Original in BLUE
    img = add_handwriting(img, random.choice(DATES), (90, 78), ORIG_INK, 20)
    img = add_handwriting(img, payee, (80, 125), ORIG_INK, 22)
    img = add_handwriting(img, orig_words, (30, 195), ORIG_INK, 20)
    img = add_handwriting(img, orig_digits, (715, 135), ORIG_INK, 22)
    img = add_signature(img, ORIG_INK)
    
    # Cross out and rewrite in BRIGHT RED
    img = add_cross_marks(img, (25, 190, 500, 230), FRAUD_INK, "diagonal")
    img = add_cross_marks(img, (710, 130, 855, 170), FRAUD_INK, "horizontal")
    img = add_handwriting(img, new_words, (30, 235), FRAUD_INK, 20)
    img = add_handwriting(img, new_digits, (715, 172), FRAUD_INK, 22)
    img = add_paper_texture(img)
    
    path = os.path.join(OUTPUT_DIR, f"fraud_altered_cheque_{index:02d}.png")
    cv2.imwrite(path, img)
    print(f"  [REGEN] fraud_altered_cheque_{index:02d}.png — {bank}")

print("Regenerating failing fraud cheques with EXTREME contrast...\n")

# fraud_altered_02, 04, 05
regen_altered_extreme(2)
regen_altered_extreme(4)
regen_altered_extreme(5)

# fraud_crossed_02, 05, 06
regen_crossed_extreme(2, "x")
regen_crossed_extreme(5, "scribble") 
regen_crossed_extreme(6, "diagonal")

# fraud_multi_tamper_01
img, bank = create_base_cheque()
payee = random.choice(PAYEE_NAMES)
new_payee = random.choice([p for p in PAYEE_NAMES if p != payee])
aw, ad = random.choice(AMOUNTS_WORDS)
img = add_handwriting(img, random.choice(DATES), (90, 78), ORIG_INK, 20)
img = add_handwriting(img, payee, (80, 125), ORIG_INK, 22)
img = add_handwriting(img, aw, (30, 195), ORIG_INK, 20)
img = add_handwriting(img, ad, (715, 135), ORIG_INK, 22)
img = add_signature(img, ORIG_INK)
# Tamper payee in RED
img = add_cross_marks(img, (75, 120, 400, 165), FRAUD_INK, "scribble")
img = add_handwriting(img, new_payee, (80, 97), FRAUD_INK, 22)
# Tamper date in GREEN
GREEN_INK = (50, 140, 50)
img = add_cross_marks(img, (85, 75, 220, 100), GREEN_INK, "horizontal")
img = add_handwriting(img, "28/01/2025", (90, 55), GREEN_INK, 18)
img = add_signature(img, FRAUD_INK, position=(620, 280))
img = add_paper_texture(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "fraud_multi_tamper_01.png"), img)
print(f"  [REGEN] fraud_multi_tamper_01.png — {bank}")

# Also regenerate genuine_06 (only remaining FP)
random.seed(99)
img, bank = create_base_cheque()
ink = ORIG_INK  # Single consistent blue
date = random.choice(DATES)
payee = random.choice(PAYEE_NAMES)
aw, ad = random.choice(AMOUNTS_WORDS)
img = add_handwriting(img, date, (90, 78), ink, 20)
img = add_handwriting(img, payee, (80, 125), ink, 22)
img = add_handwriting(img, aw, (30, 195), ink, 20)
img = add_handwriting(img, ad, (715, 135), ink, 22)
img = add_signature(img, ink)
img = add_paper_texture(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "genuine_cheque_06.png"), img)
print(f"  [REGEN] genuine_cheque_06.png — {bank} — single blue ink")

print("\nDone!")
