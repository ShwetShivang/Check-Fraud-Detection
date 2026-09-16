"""
Patch Generator for cheque fraud detection.

Extracts patches from cheque images and generates
positive/negative pairs for Siamese network training.
"""
import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import (
    resize_and_normalize, to_lab, extract_ink_mask, apply_augmentation
)


def extract_patches_sliding(img, patch_size=None, stride=None, ink_mask=None):
    """
    Extract patches using a sliding window approach.
    Only keeps patches with sufficient ink content.

    Args:
        img: Input image (H, W, C) float32 [0,1]
        patch_size: Size of square patches
        stride: Step size between patches
        ink_mask: Optional binary mask; if provided, only patches with
                  enough ink pixels are kept

    Returns:
        List of (patch, (row, col)) tuples
    """
    patch_size = patch_size or config.PATCH_SIZE
    stride = stride or config.PATCH_STRIDE
    H, W = img.shape[:2]

    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]

            # Check ink content if mask is available
            if ink_mask is not None:
                mask_patch = ink_mask[y:y + patch_size, x:x + patch_size]
                ink_ratio = mask_patch.sum() / mask_patch.size
                if ink_ratio < config.MIN_INK_RATIO:
                    continue

            patches.append((patch, (y, x)))

    return patches


def extract_patches_contour(img, patch_size=None):
    """
    Extract patches centered on detected ink contours.
    Uses contour detection to find regions with writing.

    Args:
        img: Input image (H, W, C) float32 [0,1]
        patch_size: Size of square patches

    Returns:
        List of (patch, (row, col)) tuples
    """
    patch_size = patch_size or config.PATCH_SIZE
    half = patch_size // 2
    H, W = img.shape[:2]

    # Get ink mask
    mask = extract_ink_mask(img)
    mask_uint8 = (mask.astype(np.uint8)) * 255

    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    patches = []
    seen_positions = set()

    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue

        # Get bounding box center
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Snap to grid to avoid too many overlapping patches
        grid_x = (cx // (patch_size // 2)) * (patch_size // 2)
        grid_y = (cy // (patch_size // 2)) * (patch_size // 2)

        if (grid_y, grid_x) in seen_positions:
            continue
        seen_positions.add((grid_y, grid_x))

        # Extract patch (ensure within bounds)
        y1 = max(0, grid_y - half)
        x1 = max(0, grid_x - half)
        y2 = min(H, y1 + patch_size)
        x2 = min(W, x1 + patch_size)

        if y2 - y1 != patch_size or x2 - x1 != patch_size:
            continue

        patch = img[y1:y2, x1:x2]
        patches.append((patch, (y1, x1)))

    return patches


def assign_patch_regions(patches_with_pos, image_width):
    """
    Assign patches to left/right regions of the cheque.
    In the dataset, each cheque has two pens — we assume the left half
    is pen A and the right half is pen B.

    Args:
        patches_with_pos: List of (patch, (row, col))
        image_width: Width of the original image

    Returns:
        dict: {'pen_a': [...], 'pen_b': [...]} with patches
    """
    mid = image_width // 2
    regions = {'pen_a': [], 'pen_b': []}

    for patch, (row, col) in patches_with_pos:
        if col + config.PATCH_SIZE // 2 < mid:
            regions['pen_a'].append((patch, (row, col)))
        else:
            regions['pen_b'].append((patch, (row, col)))

    return regions


def generate_pairs(dataset, max_pairs_per_image=100, augment=True):
    """
    Generate patch pairs for Siamese network training.

    For each cheque:
      - Same-pen pairs (label=1): patches from the same region
      - Different-pen pairs (label=0): patches from different regions

    Args:
        dataset: ChequeDataset with loaded images
        max_pairs_per_image: Maximum pairs to generate per image
        augment: Whether to apply augmentation

    Returns:
        List of (patch_a, patch_b, label) tuples
    """
    all_pairs = []
    rng = np.random.RandomState(42)

    for idx in range(len(dataset)):
        try:
            img, pens = dataset[idx]
        except Exception as e:
            print(f"  ⚠ Skipping image {idx}: {e}")
            continue

        # Preprocess
        img_resized = resize_and_normalize(img)
        ink_mask = extract_ink_mask(img_resized)

        # Extract patches
        patches = extract_patches_sliding(img_resized, ink_mask=ink_mask)
        if len(patches) < 4:
            patches = extract_patches_sliding(img_resized, ink_mask=None)

        if len(patches) < 4:
            continue

        # Assign to regions
        regions = assign_patch_regions(patches, img_resized.shape[1])

        pen_a_patches = [p for p, _ in regions['pen_a']]
        pen_b_patches = [p for p, _ in regions['pen_b']]

        if len(pen_a_patches) < 2 or len(pen_b_patches) < 2:
            continue

        pairs_count = 0

        # Generate SAME-PEN pairs (label=1)
        # Pairs from region A
        for _ in range(max_pairs_per_image // 4):
            i, j = rng.choice(len(pen_a_patches), 2, replace=False)
            pa = pen_a_patches[i]
            pb = pen_a_patches[j]
            if augment:
                pa = apply_augmentation(pa, rng)
                pb = apply_augmentation(pb, rng)
            all_pairs.append((pa, pb, 1))
            pairs_count += 1

        # Pairs from region B
        for _ in range(max_pairs_per_image // 4):
            i, j = rng.choice(len(pen_b_patches), 2, replace=False)
            pa = pen_b_patches[i]
            pb = pen_b_patches[j]
            if augment:
                pa = apply_augmentation(pa, rng)
                pb = apply_augmentation(pb, rng)
            all_pairs.append((pa, pb, 1))
            pairs_count += 1

        # Generate DIFFERENT-PEN pairs (label=0)
        for _ in range(max_pairs_per_image // 2):
            i = rng.randint(0, len(pen_a_patches))
            j = rng.randint(0, len(pen_b_patches))
            pa = pen_a_patches[i]
            pb = pen_b_patches[j]
            if augment:
                pa = apply_augmentation(pa, rng)
                pb = apply_augmentation(pb, rng)
            all_pairs.append((pa, pb, 0))
            pairs_count += 1

    rng.shuffle(all_pairs)
    print(f"[PatchGen] Generated {len(all_pairs)} patch pairs")
    return all_pairs


def generate_single_patches(dataset, augment=True):
    """
    Generate labeled single patches for CNN classifier training.

    Each patch gets a label based on which pen region it came from.

    Returns:
        List of (patch, label) where label is pen region ID (0 or 1)
    """
    all_patches = []
    rng = np.random.RandomState(42)

    for idx in range(len(dataset)):
        try:
            img, pens = dataset[idx]
        except Exception:
            continue

        img_resized = resize_and_normalize(img)
        ink_mask = extract_ink_mask(img_resized)
        patches = extract_patches_sliding(img_resized, ink_mask=ink_mask)

        if len(patches) < 2:
            patches = extract_patches_sliding(img_resized, ink_mask=None)

        regions = assign_patch_regions(patches, img_resized.shape[1])

        for p, _ in regions['pen_a']:
            if augment:
                p = apply_augmentation(p, rng)
            all_patches.append((p, 0))

        for p, _ in regions['pen_b']:
            if augment:
                p = apply_augmentation(p, rng)
            all_patches.append((p, 1))

    rng.shuffle(all_patches)
    print(f"[PatchGen] Generated {len(all_patches)} single patches")
    return all_patches


# ─── PyTorch Datasets ────────────────────────────────────────────────────

class PatchPairDataset(Dataset):
    """PyTorch Dataset for Siamese network patch pairs."""

    def __init__(self, pairs):
        """
        Args:
            pairs: List of (patch_a, patch_b, label) tuples
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pa, pb, label = self.pairs[idx]

        # Convert to torch tensors: (C, H, W)
        pa_tensor = torch.from_numpy(pa.transpose(2, 0, 1)).float()
        pb_tensor = torch.from_numpy(pb.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return pa_tensor, pb_tensor, label_tensor


class SinglePatchDataset(Dataset):
    """PyTorch Dataset for CNN classifier with single patches."""

    def __init__(self, patches):
        """
        Args:
            patches: List of (patch, label) tuples
        """
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, label = self.patches[idx]
        patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return patch_tensor, label_tensor


if __name__ == "__main__":
    from data_loader import load_dataset

    dataset = load_dataset()
    print("\n--- Generating patch pairs ---")
    pairs = generate_pairs(dataset, max_pairs_per_image=20, augment=False)
    print(f"Total pairs: {len(pairs)}")
    if pairs:
        pa, pb, label = pairs[0]
        print(f"Patch A shape: {pa.shape}, Patch B shape: {pb.shape}, Label: {label}")

    print("\n--- Generating single patches ---")
    singles = generate_single_patches(dataset, augment=False)
    print(f"Total patches: {len(singles)}")
