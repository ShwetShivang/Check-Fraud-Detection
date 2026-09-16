"""
Generate realistic sample cheque images for fraud detection demos.

Creates two categories:
  - GENUINE cheques: uniform single-ink handwriting
  - FRAUD cheques: crossed out fields, different ink colors, tampered amounts

Uses OpenCV + Pillow for programmatic generation with realistic details.
"""

import cv2
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont

# ─── Output ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 400
BANK_NAMES = [
    "STATE BANK OF INDIA", "HDFC BANK", "ICICI BANK",
    "PUNJAB NATIONAL BANK", "BANK OF BARODA", "AXIS BANK",
    "CANARA BANK", "UNION BANK OF INDIA"
]
PAYEE_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Verma",
    "Sunita Patel", "Vikram Singh", "Neha Gupta",
    "Arjun Reddy", "Kavitha Nair", "Deepak Joshi",
    "Anjali Mehta", "Sanjay Tiwari", "Pooja Rao"
]
AMOUNTS_WORDS = [
    ("Five Thousand Only", "5,000"),
    ("Ten Thousand Only", "10,000"),
    ("Twenty Five Thousand Only", "25,000"),
    ("Fifty Thousand Only", "50,000"),
    ("One Lakh Only", "1,00,000"),
    ("Fifteen Thousand Only", "15,000"),
    ("Seven Thousand Five Hundred Only", "7,500"),
    ("Thirty Thousand Only", "30,000"),
    ("Forty Five Thousand Only", "45,000"),
    ("Eighty Thousand Only", "80,000"),
]
DATES = [
    "15/03/2025", "22/07/2024", "01/11/2025", "10/05/2024",
    "28/01/2025", "14/08/2024", "03/12/2025", "19/06/2024",
    "07/09/2025", "25/04/2024"
]

# Ink colors (BGR for OpenCV)
INK_BLUE       = (180, 80, 20)      # Blue ballpoint
INK_DARK_BLUE  = (140, 50, 10)      # Dark blue pen
INK_BLACK      = (40, 40, 40)       # Black pen
INK_RED        = (50, 50, 200)      # Red pen
INK_GREEN      = (50, 130, 50)      # Green pen
INK_PURPLE     = (140, 40, 120)     # Purple pen

GENUINE_INKS = [INK_BLUE, INK_DARK_BLUE, INK_BLACK]
FRAUD_SECONDARY_INKS = [INK_RED, INK_BLACK, INK_GREEN, INK_PURPLE, INK_DARK_BLUE]


def get_font(size=22, bold=False):
    """Try to load a handwriting-style or fallback font."""
    font_paths = [
        "C:/Windows/Fonts/segoepr.ttf",    # Segoe Print
        "C:/Windows/Fonts/comic.ttf",       # Comic Sans
        "C:/Windows/Fonts/segoesc.ttf",     # Segoe Script
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    if bold:
        font_paths = [
            "C:/Windows/Fonts/segoeprb.ttf",
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ] + font_paths

    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def bgr_to_rgb(bgr):
    """Convert BGR tuple to RGB for PIL."""
    return (bgr[2], bgr[1], bgr[0])


def add_paper_texture(img):
    """Add realistic paper texture and slight noise."""
    h, w = img.shape[:2]
    # Add slight grain
    noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Slight vignette
    X = cv2.getGaussianKernel(w, w * 0.7)
    Y = cv2.getGaussianKernel(h, h * 0.7)
    M = Y * X.T
    M = M / M.max()
    for i in range(3):
        img[:, :, i] = (img[:, :, i] * (0.85 + 0.15 * M)).astype(np.uint8)
    return img


def create_base_cheque(bank_name=None):
    """Create a blank cheque template with printed elements."""
    # Paper background (off-white with slight color variation)
    bg_colors = [
        (245, 245, 240),  # Off-white
        (240, 248, 255),  # Alice blue
        (255, 253, 245),  # Cream
        (248, 248, 255),  # Ghost white
        (245, 255, 250),  # Mint cream
    ]
    bg = random.choice(bg_colors)
    img = np.full((HEIGHT, WIDTH, 3), bg, dtype=np.uint8)

    # Security pattern (faint guilloche lines)
    for y in range(0, HEIGHT, 8):
        offset = int(5 * np.sin(y * 0.05))
        cv2.line(img, (0, y + offset), (WIDTH, y + offset),
                 (bg[0] - 12, bg[1] - 12, bg[2] - 8), 1)

    # Border
    cv2.rectangle(img, (8, 8), (WIDTH - 8, HEIGHT - 8), (180, 180, 180), 2)
    cv2.rectangle(img, (12, 12), (WIDTH - 12, HEIGHT - 12), (200, 200, 200), 1)

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    bank = bank_name or random.choice(BANK_NAMES)
    printed_color = (100, 100, 100)
    light_color = (160, 160, 160)

    # Bank name
    bank_font = get_font(24, bold=True)
    draw.text((30, 20), bank, fill=(0, 80, 150), font=bank_font)

    # Branch / IFSC
    small_font = get_font(12)
    draw.text((30, 50), "Branch: Main Branch  |  IFSC: SBIN0001234", fill=light_color, font=small_font)

    # Printed labels
    label_font = get_font(13)
    draw.text((30, 82), "Date:", fill=printed_color, font=label_font)
    draw.text((30, 130), "Pay", fill=printed_color, font=label_font)
    draw.text((30, 185), "Rupees (in words):", fill=printed_color, font=label_font)
    draw.text((680, 135), "Rs.", fill=printed_color, font=label_font)

    # Printed lines (where handwriting goes)
    for y_pos in [115, 165, 225, 275]:
        draw.line([(30, y_pos), (860, y_pos)], fill=light_color, width=1)

    # Amount box
    draw.rectangle([(670, 125), (860, 170)], outline=printed_color, width=1)

    # Signature line
    draw.line([(580, 350), (850, 350)], fill=printed_color, width=1)
    draw.text((670, 355), "Authorised Signatory", fill=light_color, font=get_font(10))

    # Account number area
    draw.text((30, 340), "A/c No: " + "".join([str(random.randint(0, 9)) for _ in range(12)]),
              fill=light_color, font=get_font(11))

    # Cheque number
    cheque_no = "".join([str(random.randint(0, 9)) for _ in range(6)])
    draw.text((30, 365), f"Cheque No: {cheque_no}", fill=light_color, font=get_font(11))

    # MICR line at bottom
    micr_font = get_font(14)
    micr = "⑆" + cheque_no + "⑆  ⑇001234⑇  ⑈0001⑈"
    draw.text((250, 375), micr, fill=(80, 80, 80), font=micr_font)

    # Convert back to OpenCV BGR
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result, bank


def add_handwriting(img, text, position, ink_color, font_size=22, jitter=True):
    """Add handwritten-style text to the cheque image."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)

    x, y = position
    # Add slight position jitter for realism
    if jitter:
        x += random.randint(-2, 2)
        y += random.randint(-1, 1)

    # Slight color variation for natural look
    color = bgr_to_rgb(ink_color)
    color = tuple(max(0, min(255, c + random.randint(-8, 8))) for c in color)

    draw.text((x, y), text, fill=color, font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def add_signature(img, ink_color, position=(620, 300)):
    """Draw a procedural signature-like scribble."""
    x0, y0 = position
    pts = []
    cx, cy = x0, y0
    # Generate random signature path
    num_segments = random.randint(4, 7)
    for _ in range(num_segments):
        cx += random.randint(15, 40)
        cy += random.randint(-20, 20)
        pts.append((cx, cy))

    # Draw smooth curves through points
    for i in range(len(pts) - 1):
        thickness = random.randint(1, 3)
        cv2.line(img, pts[i], pts[i + 1], ink_color, thickness, cv2.LINE_AA)
        # Add small loops/curves
        if random.random() > 0.5:
            mid_x = (pts[i][0] + pts[i + 1][0]) // 2
            mid_y = (pts[i][1] + pts[i + 1][1]) // 2 + random.randint(-15, 15)
            cv2.ellipse(img, (mid_x, mid_y), (8, 5),
                       random.randint(0, 180), 0, 360, ink_color, 1, cv2.LINE_AA)

    # Underline stroke
    cv2.line(img, (x0, y0 + 15), (cx + 20, y0 + 15 + random.randint(-5, 5)),
             ink_color, 2, cv2.LINE_AA)

    return img


def add_cross_marks(img, region, ink_color, style="x"):
    """Add crossing-out marks over a region."""
    x1, y1, x2, y2 = region
    if style == "x":
        cv2.line(img, (x1, y1), (x2, y2), ink_color, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x1, y2), ink_color, 2, cv2.LINE_AA)
    elif style == "horizontal":
        mid_y = (y1 + y2) // 2
        for dy in range(-3, 4, 3):
            cv2.line(img, (x1 - 5, mid_y + dy), (x2 + 5, mid_y + dy),
                     ink_color, 2, cv2.LINE_AA)
    elif style == "diagonal":
        for i in range(3):
            offset = i * 4
            cv2.line(img, (x1, y1 + offset), (x2, y2 + offset),
                     ink_color, 2, cv2.LINE_AA)
    elif style == "scribble":
        for _ in range(8):
            sx = random.randint(x1, x2)
            sy = random.randint(y1, y2)
            ex = random.randint(x1, x2)
            ey = random.randint(y1, y2)
            cv2.line(img, (sx, sy), (ex, ey), ink_color, 1, cv2.LINE_AA)
    return img


def add_parallel_cross_lines(img, ink_color):
    """Add two parallel diagonal lines across cheque face (crossed cheque)."""
    # Two parallel lines from top-left area to bottom-right area
    offset = 30
    cv2.line(img, (20, 20), (250, HEIGHT - 30), ink_color, 3, cv2.LINE_AA)
    cv2.line(img, (20 + offset, 20), (250 + offset, HEIGHT - 30), ink_color, 3, cv2.LINE_AA)
    return img


def generate_genuine_cheque(index):
    """Generate a genuine cheque with uniform ink."""
    img, bank = create_base_cheque()
    ink = random.choice(GENUINE_INKS)

    date = random.choice(DATES)
    payee = random.choice(PAYEE_NAMES)
    amount_words, amount_digits = random.choice(AMOUNTS_WORDS)

    # Fill in all fields with SAME ink color
    img = add_handwriting(img, date, (90, 78), ink, font_size=20)
    img = add_handwriting(img, payee, (80, 125), ink, font_size=22)
    img = add_handwriting(img, amount_words, (30, 195), ink, font_size=20)
    img = add_handwriting(img, amount_digits, (715, 135), ink, font_size=22)
    img = add_signature(img, ink)

    # Add paper texture
    img = add_paper_texture(img)

    filename = f"genuine_cheque_{index:02d}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"  [OK] {filename} — {bank} — Single ink")
    return path


def generate_fraud_crossed_cheque(index):
    """Generate a fraud cheque with crossed-out fields and different ink."""
    img, bank = create_base_cheque()
    ink1 = random.choice(GENUINE_INKS)  # Original ink
    # Ensure second ink is DIFFERENT
    ink2 = random.choice([c for c in FRAUD_SECONDARY_INKS if c != ink1])

    date = random.choice(DATES)
    payee = random.choice(PAYEE_NAMES)
    new_payee = random.choice([p for p in PAYEE_NAMES if p != payee])
    amount_words, amount_digits = random.choice(AMOUNTS_WORDS)

    # Original fields in ink1
    img = add_handwriting(img, date, (90, 78), ink1, font_size=20)
    img = add_handwriting(img, payee, (80, 125), ink1, font_size=22)
    img = add_handwriting(img, amount_words, (30, 195), ink1, font_size=20)
    img = add_handwriting(img, amount_digits, (715, 135), ink1, font_size=22)
    img = add_signature(img, ink1)

    # Fraud: cross out payee and write new name in different ink
    cross_style = random.choice(["x", "horizontal", "diagonal", "scribble"])
    img = add_cross_marks(img, (75, 125, 400, 160), ink2, style=cross_style)
    img = add_handwriting(img, new_payee, (80, 100), ink2, font_size=20)

    img = add_paper_texture(img)

    filename = f"fraud_crossed_cheque_{index:02d}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"  [OK] {filename} — {bank} — Crossed payee ({cross_style})")
    return path


def generate_fraud_altered_amount(index):
    """Generate a fraud cheque with altered amount."""
    img, bank = create_base_cheque()
    ink1 = random.choice(GENUINE_INKS)
    ink2 = random.choice([c for c in FRAUD_SECONDARY_INKS if c != ink1])

    date = random.choice(DATES)
    payee = random.choice(PAYEE_NAMES)

    # Original with lower amount
    original_amounts = [
        ("Five Thousand Only", "5,000", "Fifty Five Thousand Only", "55,000"),
        ("One Thousand Only", "1,000", "Eleven Thousand Only", "11,000"),
        ("Ten Thousand Only", "10,000", "Ten Lakh Only", "10,00,000"),
        ("Two Thousand Only", "2,000", "Twenty Two Thousand Only", "22,000"),
        ("Three Thousand Only", "3,000", "Thirty Three Thousand Only", "33,000"),
    ]
    orig_words, orig_digits, new_words, new_digits = random.choice(original_amounts)

    # Write original in ink1
    img = add_handwriting(img, date, (90, 78), ink1, font_size=20)
    img = add_handwriting(img, payee, (80, 125), ink1, font_size=22)
    img = add_handwriting(img, orig_words, (30, 195), ink1, font_size=20)
    img = add_handwriting(img, orig_digits, (715, 135), ink1, font_size=22)
    img = add_signature(img, ink1)

    # Fraud: cross out amount and write new in different ink
    cross_style = random.choice(["horizontal", "diagonal", "scribble"])
    img = add_cross_marks(img, (25, 195, 500, 225), ink2, style=cross_style)
    img = add_cross_marks(img, (710, 135, 855, 165), ink2, style="horizontal")

    # New amount in different ink
    img = add_handwriting(img, new_words, (30, 230), ink2, font_size=19)
    img = add_handwriting(img, new_digits, (715, 168), ink2, font_size=20)

    img = add_paper_texture(img)

    filename = f"fraud_altered_cheque_{index:02d}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"  [OK] {filename} — {bank} — Altered amount")
    return path


def generate_fraud_parallel_cross(index):
    """Generate a cheque with parallel crossing lines in different ink."""
    img, bank = create_base_cheque()
    ink1 = random.choice(GENUINE_INKS)
    ink2 = random.choice([c for c in FRAUD_SECONDARY_INKS if c != ink1])

    date = random.choice(DATES)
    payee = random.choice(PAYEE_NAMES)
    amount_words, amount_digits = random.choice(AMOUNTS_WORDS)

    # Original in ink1
    img = add_handwriting(img, date, (90, 78), ink1, font_size=20)
    img = add_handwriting(img, payee, (80, 125), ink1, font_size=22)
    img = add_handwriting(img, amount_words, (30, 195), ink1, font_size=20)
    img = add_handwriting(img, amount_digits, (715, 135), ink1, font_size=22)
    img = add_signature(img, ink1)

    # Add crossing lines in different ink
    img = add_parallel_cross_lines(img, ink2)

    # Add "A/C PAYEE ONLY" in the crossed area
    img = add_handwriting(img, "A/C PAYEE ONLY", (50, 160), ink2, font_size=16)

    img = add_paper_texture(img)

    filename = f"fraud_crossed_cheque_{index:02d}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"  [OK] {filename} — {bank} — Parallel cross lines")
    return path


def generate_fraud_multi_tamper(index):
    """Generate heavily tampered cheque with multiple alterations."""
    img, bank = create_base_cheque()
    ink1 = random.choice(GENUINE_INKS)
    ink2 = random.choice([c for c in FRAUD_SECONDARY_INKS if c != ink1])
    ink3 = random.choice([c for c in FRAUD_SECONDARY_INKS if c != ink1 and c != ink2])

    date = random.choice(DATES)
    payee = random.choice(PAYEE_NAMES)
    new_payee = random.choice([p for p in PAYEE_NAMES if p != payee])
    amount_words, amount_digits = random.choice(AMOUNTS_WORDS)

    # Original in ink1
    img = add_handwriting(img, date, (90, 78), ink1, font_size=20)
    img = add_handwriting(img, payee, (80, 125), ink1, font_size=22)
    img = add_handwriting(img, amount_words, (30, 195), ink1, font_size=20)
    img = add_handwriting(img, amount_digits, (715, 135), ink1, font_size=22)
    img = add_signature(img, ink1)

    # Tamper 1: cross out payee in ink2
    img = add_cross_marks(img, (75, 125, 400, 160), ink2, style="scribble")
    img = add_handwriting(img, new_payee, (80, 100), ink2, font_size=20)

    # Tamper 2: alter date in ink3
    img = add_cross_marks(img, (85, 78, 220, 100), ink3, style="horizontal")
    new_date = random.choice([d for d in DATES if d != date])
    img = add_handwriting(img, new_date, (90, 60), ink3, font_size=18)

    # Tamper 3: additional signature attempt
    img = add_signature(img, ink2, position=(620, 280))

    img = add_paper_texture(img)

    filename = f"fraud_multi_tamper_{index:02d}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"  [OK] {filename} — {bank} — Multiple tampering (3 inks)")
    return path


def main():
    print("=" * 60)
    print("  SAMPLE CHEQUE IMAGE GENERATOR")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}\n")

    # ── Genuine Cheques ──
    print("--- Generating GENUINE Cheques ---")
    for i in range(4, 11):  # genuine_cheque_04 through _10 (01-03 already exist from AI)
        generate_genuine_cheque(i)

    # ── Fraud: Crossed Payee ──
    print("\n--- Generating FRAUD Cheques (Crossed Payee) ---")
    for i in range(4, 9):  # fraud_crossed_cheque_04 through _08
        generate_fraud_crossed_cheque(i)

    # ── Fraud: Altered Amount ──
    print("\n--- Generating FRAUD Cheques (Altered Amount) ---")
    for i in range(2, 7):  # fraud_altered_cheque_02 through _06
        generate_fraud_altered_amount(i)

    # ── Fraud: Parallel Cross Lines ──
    print("\n--- Generating FRAUD Cheques (Parallel Cross) ---")
    for i in range(9, 12):  # fraud_crossed_cheque_09 through _11
        generate_fraud_parallel_cross(i)

    # ── Fraud: Multi-tamper ──
    print("\n--- Generating FRAUD Cheques (Multiple Tampering) ---")
    for i in range(1, 5):  # fraud_multi_tamper_01 through _04
        generate_fraud_multi_tamper(i)

    # List generated files
    files = sorted(os.listdir(OUTPUT_DIR))
    genuine = [f for f in files if f.startswith("genuine")]
    fraud = [f for f in files if f.startswith("fraud")]

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Genuine cheques: {len(genuine)}")
    print(f"  Fraud cheques:   {len(fraud)}")
    print(f"  Total:           {len(genuine) + len(fraud)}")
    print(f"\n  Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
