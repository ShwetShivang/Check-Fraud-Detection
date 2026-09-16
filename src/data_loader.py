"""
Data Loader for the Cheque Fraud Detection project.

Handles:
  - Loading IDRBT cheque images and metadata from data/raw/
  - Synthetic cheque generation for development/testing
"""
import os
import glob
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Real Dataset Loader ─────────────────────────────────────────────────

class ChequeDataset:
    """
    Loads cheque images and metadata from the IDRBT Cheque Dataset.

    Expected structure in data/raw/:
        - Cheque images (*.jpg, *.png, *.bmp, *.tif)
        - Metadata Excel file (*.xlsx or *.xls) with pen/writer info
    """

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or config.RAW_DATA_DIR
        self.images = []
        self.metadata = None
        self.pen_info = {}

    def load(self):
        """Load all images and metadata from the data directory."""
        print(f"[DataLoader] Scanning: {self.data_dir}")

        # Find image files recursively
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        for ext in extensions:
            self.images.extend(glob.glob(
                os.path.join(self.data_dir, '**', ext), recursive=True
            ))

        # Sort for consistency
        self.images.sort()
        print(f"  Found {len(self.images)} cheque images")

        # Find and load metadata Excel
        excels = glob.glob(os.path.join(self.data_dir, '**', '*.xlsx'), recursive=True)
        excels += glob.glob(os.path.join(self.data_dir, '**', '*.xls'), recursive=True)

        if excels:
            self._load_metadata(excels[0])
        else:
            print("  ⚠ No metadata Excel found. Using filename-based pen assignment.")
            self._assign_pens_from_filenames()

        return self

    def _load_metadata(self, excel_path):
        """Parse the metadata Excel file for pen and writer information."""
        print(f"  Loading metadata: {excel_path}")
        try:
            self.metadata = pd.read_excel(excel_path)
            print(f"  Metadata columns: {list(self.metadata.columns)}")
            print(f"  Metadata rows: {len(self.metadata)}")

            # Try to map images to pen info from metadata
            # The Excel may have columns like: Image, Pen1, Pen2, Writer, etc.
            # We'll handle multiple possible column name formats
            self._parse_pen_columns()
        except Exception as e:
            print(f"  ⚠ Error reading metadata: {e}")
            self._assign_pens_from_filenames()

    def _parse_pen_columns(self):
        """Extract pen info from metadata columns (flexible parsing)."""
        if self.metadata is None:
            return

        cols = [c.lower().strip() for c in self.metadata.columns]

        # Look for pen-related columns
        pen_cols = [c for c in self.metadata.columns
                    if any(k in c.lower() for k in ['pen', 'ink', 'instrument'])]
        img_cols = [c for c in self.metadata.columns
                    if any(k in c.lower() for k in ['image', 'file', 'name', 'cheque', 'check'])]

        if pen_cols and img_cols:
            img_col = img_cols[0]
            for _, row in self.metadata.iterrows():
                img_name = str(row[img_col]).strip()
                pens = [str(row[pc]).strip() for pc in pen_cols if pd.notna(row[pc])]
                self.pen_info[img_name] = pens
            print(f"  Mapped {len(self.pen_info)} images to pen info")
        else:
            # If columns don't match expected patterns, use index-based mapping
            print("  ⚠ Could not identify pen columns. Using index-based assignment.")
            self._assign_pens_from_filenames()

    def _assign_pens_from_filenames(self):
        """
        Fallback: assign pen IDs based on image index.
        Each cheque gets two pens: pen_A and pen_B.
        """
        for i, img_path in enumerate(self.images):
            name = Path(img_path).stem
            pen_a = f"pen_{(i % 14) + 1}"
            pen_b = f"pen_{((i + 7) % 14) + 1}"
            if pen_a == pen_b:
                pen_b = f"pen_{((i + 3) % 14) + 1}"
            self.pen_info[name] = [pen_a, pen_b]

    def get_image(self, idx):
        """Load and return an image by index."""
        img = cv2.imread(self.images[idx])
        if img is None:
            raise FileNotFoundError(f"Could not read: {self.images[idx]}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_pen_info(self, idx):
        """Get pen IDs for an image by index."""
        name = Path(self.images[idx]).stem
        return self.pen_info.get(name, ["pen_unknown_1", "pen_unknown_2"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.get_image(idx), self.get_pen_info(idx)


# ─── Synthetic Data Generator ────────────────────────────────────────────

class SyntheticChequeGenerator:
    """
    Generates synthetic cheque-like images with controlled ink variations
    for development and testing when the real dataset is unavailable.

    Each synthetic cheque has:
    - A light background (mimicking cheque paper)
    - Text-like ink strokes in two regions with different "inks"
    - Known pen assignments for ground-truth labeling
    """

    # Define a palette of ink colors (BGR format for OpenCV)
    INK_COLORS = [
        (180, 50, 30),    # Dark blue
        (160, 40, 40),    # Navy
        (200, 60, 20),    # Blue
        (140, 30, 30),    # Very dark blue
        (50, 50, 50),     # Dark gray (black ink)
        (60, 40, 40),     # Near-black
        (170, 80, 40),    # Medium blue
        (100, 30, 20),    # Dark blue-black
        (40, 40, 60),     # Dark charcoal
        (180, 70, 50),    # Blue
        (150, 50, 35),    # Steel blue
        (70, 50, 45),     # Dark gray-blue
        (190, 65, 25),    # Royal blue
        (130, 45, 50),    # Indigo-ish
    ]

    def __init__(self, output_dir=None, num_images=None):
        self.output_dir = output_dir or config.SYNTHETIC_DIR
        self.num_images = num_images or config.SYNTHETIC_NUM_IMAGES
        self.image_size = config.SYNTHETIC_IMAGE_SIZE  # (H, W)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self):
        """Generate all synthetic cheque images. Returns list of (path, pen_info)."""
        print(f"[SyntheticGen] Generating {self.num_images} synthetic cheques...")
        results = []

        for i in range(self.num_images):
            pen_a_idx = i % len(self.INK_COLORS)
            pen_b_idx = (i + 3 + i // len(self.INK_COLORS)) % len(self.INK_COLORS)
            if pen_a_idx == pen_b_idx:
                pen_b_idx = (pen_b_idx + 1) % len(self.INK_COLORS)

            img = self._create_cheque(
                self.INK_COLORS[pen_a_idx],
                self.INK_COLORS[pen_b_idx],
                seed=i
            )

            fname = f"synthetic_cheque_{i:03d}.png"
            path = os.path.join(self.output_dir, fname)
            cv2.imwrite(path, img)

            pen_info = [f"pen_{pen_a_idx + 1}", f"pen_{pen_b_idx + 1}"]
            results.append((path, pen_info))

        print(f"  [OK] Generated {len(results)} synthetic cheques in {self.output_dir}")
        return results

    def _create_cheque(self, color_a, color_b, seed=0):
        """Create a single synthetic cheque image."""
        rng = np.random.RandomState(seed)
        H, W = self.image_size

        # Create paper-like background
        bg_val = rng.randint(230, 250)
        img = np.full((H, W, 3), bg_val, dtype=np.uint8)

        # Add subtle paper texture
        noise = rng.normal(0, 3, (H, W, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw printed lines (like a real cheque)
        line_color = (200, 200, 200)
        for y in range(60, H - 30, 40):
            cv2.line(img, (30, y), (W - 30, y), line_color, 1)

        # Region A: Left portion (original writing — pen A)
        self._draw_handwriting(img, color_a, region=(30, 30, W // 2 - 20, H - 60), rng=rng)

        # Region B: Right portion (potentially altered — pen B)
        self._draw_handwriting(img, color_b, region=(W // 2 + 20, 30, W - 60, H - 60), rng=rng)

        # Add a border
        cv2.rectangle(img, (5, 5), (W - 5, H - 5), (150, 150, 150), 2)

        return img

    def _draw_handwriting(self, img, color, region, rng):
        """Draw simulated handwriting strokes in a region."""
        x1, y1, x2, y2 = region
        num_strokes = rng.randint(15, 30)

        for _ in range(num_strokes):
            # Random starting point within region
            sx = rng.randint(x1, max(x1 + 1, x2 - 50))
            sy = rng.randint(y1, max(y1 + 1, y2 - 20))

            # Draw a series of connected line segments (simulating cursive)
            pts = [(sx, sy)]
            for _ in range(rng.randint(3, 8)):
                dx = rng.randint(5, 25)
                dy = rng.randint(-10, 10)
                new_x = min(max(pts[-1][0] + dx, x1), x2)
                new_y = min(max(pts[-1][1] + dy, y1), y2)
                pts.append((new_x, new_y))

            # Add slight color variation (simulating ink pressure changes)
            c_var = tuple(
                int(min(255, max(0, c + rng.randint(-10, 10))))
                for c in color
            )

            thickness = rng.randint(1, 3)
            for j in range(len(pts) - 1):
                cv2.line(img, pts[j], pts[j + 1], c_var, thickness, cv2.LINE_AA)

            # Occasionally add dots (like i-dots or periods)
            if rng.random() > 0.7:
                dx = rng.randint(0, max(1, x2 - x1))
                dy = rng.randint(0, max(1, y2 - y1))
                cv2.circle(img, (x1 + dx, y1 + dy), rng.randint(1, 3), c_var, -1)


# ─── Convenience Functions ───────────────────────────────────────────────

def load_dataset():
    """
    Load the dataset. Uses real data if available, falls back to synthetic.
    Returns a ChequeDataset-like object.
    """
    # Try real dataset first
    dataset = ChequeDataset()
    dataset.load()

    if len(dataset) > 0:
        print(f"[DataLoader] Using real dataset: {len(dataset)} images")
        return dataset

    # Fall back to synthetic
    print("[DataLoader] No real data found. Generating synthetic dataset...")
    gen = SyntheticChequeGenerator()
    results = gen.generate()

    # Create a ChequeDataset wrapper for synthetic data
    dataset = ChequeDataset(data_dir=config.SYNTHETIC_DIR)
    dataset.images = [r[0] for r in results]
    for path, pens in results:
        name = Path(path).stem
        dataset.pen_info[name] = pens
    print(f"[DataLoader] Using synthetic dataset: {len(dataset)} images")
    return dataset


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"\nDataset size: {len(dataset)}")
    if len(dataset) > 0:
        img, pens = dataset[0]
        print(f"First image shape: {img.shape}")
        print(f"First image pens: {pens}")
