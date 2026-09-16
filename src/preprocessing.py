"""
Image preprocessing for cheque fraud detection.

Handles:
  - Resize and normalization
  - Color space conversions (Grayscale, LAB)
  - Data augmentation (rotation, noise, brightness)
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def resize_and_normalize(img, size=None):
    """
    Resize image and normalize pixel values to [0, 1].

    Args:
        img: Input image (H, W, C) in uint8
        size: Target (height, width). Defaults to config.IMAGE_SIZE.

    Returns:
        Normalized float32 image
    """
    size = size or config.IMAGE_SIZE
    resized = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def to_grayscale(img):
    """
    Convert image to grayscale.

    Args:
        img: Input RGB image (H, W, 3)

    Returns:
        Grayscale image (H, W)
    """
    if len(img.shape) == 2:
        return img
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    if img.dtype == np.float32 or img.dtype == np.float64:
        return gray.astype(np.float32) / 255.0
    return gray


def to_lab(img):
    """
    Convert RGB image to LAB color space.
    LAB separates luminance (L) from color (A, B),
    making it ideal for ink color differentiation.

    Args:
        img: Input RGB image (H, W, 3)

    Returns:
        LAB image (H, W, 3) as float32
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = img
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32) / 255.0


def apply_augmentation(img, rng=None):
    """
    Apply random augmentations to an image patch.

    Augmentations:
      - Random rotation (±15°)
      - Gaussian noise (σ=0.01)
      - Brightness variation (±20%)

    Args:
        img: Input image (H, W, C) as float32 [0,1]
        rng: Numpy random state for reproducibility

    Returns:
        Augmented image (same shape)
    """
    if rng is None:
        rng = np.random.RandomState()

    result = img.copy()
    h, w = result.shape[:2]

    # 1. Random rotation
    if rng.random() > 0.5:
        angle = rng.uniform(
            -config.AUG_ROTATION_RANGE,
            config.AUG_ROTATION_RANGE
        )
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT)

    # 2. Gaussian noise
    if rng.random() > 0.5:
        noise = rng.normal(0, config.AUG_NOISE_STD, result.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 1)

    # 3. Brightness variation
    if rng.random() > 0.5:
        factor = 1.0 + rng.uniform(
            -config.AUG_BRIGHTNESS_RANGE,
            config.AUG_BRIGHTNESS_RANGE
        )
        result = np.clip(result * factor, 0, 1)

    return result.astype(np.float32)


def preprocess_cheque(img_or_path, size=None):
    """
    Full preprocessing pipeline for a cheque image.

    Args:
        img_or_path: Image array (RGB) or path to image file
        size: Target size. Defaults to config.IMAGE_SIZE.

    Returns:
        dict with keys: 'rgb', 'grayscale', 'lab', 'original'
    """
    # Load if path
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_or_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = img_or_path.copy()

    original = img.copy()

    # Resize and normalize
    rgb = resize_and_normalize(img, size)

    # Color space conversions
    grayscale = to_grayscale(rgb)
    lab = to_lab(rgb)

    return {
        'rgb': rgb,
        'grayscale': grayscale,
        'lab': lab,
        'original': original
    }


def extract_ink_mask(img, threshold=0.6):
    """
    Extract a binary mask of ink pixels from a cheque image.
    Uses adaptive thresholding on the grayscale image.

    Args:
        img: Preprocessed image (float32, [0,1], RGB or grayscale)
        threshold: Ink darkness threshold

    Returns:
        Binary mask (H, W) where True = ink pixel
    """
    if len(img.shape) == 3:
        gray = to_grayscale(img)
    else:
        gray = img

    if gray.dtype == np.float32 or gray.dtype == np.float64:
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = gray

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray_uint8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 5
    )

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary > 0


if __name__ == "__main__":
    # Quick test with synthetic data
    from data_loader import load_dataset
    dataset = load_dataset()
    if len(dataset) > 0:
        img, pens = dataset[0]
        result = preprocess_cheque(img)
        print(f"RGB shape: {result['rgb'].shape}")
        print(f"Grayscale shape: {result['grayscale'].shape}")
        print(f"LAB shape: {result['lab'].shape}")
        mask = extract_ink_mask(result['rgb'])
        ink_ratio = mask.sum() / mask.size
        print(f"Ink pixel ratio: {ink_ratio:.3f}")
