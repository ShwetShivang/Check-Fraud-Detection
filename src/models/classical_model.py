"""
Classical Machine Learning model for ink classification.

Extracts hand-crafted features:
  - GLCM (Gray-Level Co-occurrence Matrix) texture features
  - LBP (Local Binary Pattern) histogram
  - LAB color space histogram

Trains SVM and Random Forest classifiers.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cv2
import os
import sys
import warnings
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Suppress convergence warnings for small datasets
warnings.filterwarnings('ignore', category=UserWarning)


def extract_glcm_features(patch):
    """
    Extract GLCM (Gray-Level Co-occurrence Matrix) texture features.

    Features: contrast, dissimilarity, homogeneity, energy, correlation
    Computed at multiple distances and angles.

    Args:
        patch: Grayscale image patch (H, W) as float32 or uint8

    Returns:
        Feature vector (numpy array)
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        from skimage.feature import graycomatrix, graycoprops

    # Convert to uint8 if needed
    if patch.dtype == np.float32 or patch.dtype == np.float64:
        patch_uint8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    else:
        patch_uint8 = patch

    # Ensure 2D
    if len(patch_uint8.shape) == 3:
        patch_uint8 = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2GRAY)

    # Quantize to 64 levels to reduce computation
    patch_quantized = (patch_uint8 // 4).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        patch_quantized, 
        distances=config.GLCM_DISTANCES,
        angles=config.GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True
    )

    # Extract properties
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        vals = graycoprops(glcm, prop).flatten()
        features.extend(vals)

    return np.array(features, dtype=np.float32)


def extract_lbp_features(patch):
    """
    Extract LBP (Local Binary Pattern) histogram features.

    LBP captures local texture patterns around each pixel.

    Args:
        patch: Grayscale image patch

    Returns:
        LBP histogram feature vector
    """
    from skimage.feature import local_binary_pattern

    if patch.dtype == np.float32 or patch.dtype == np.float64:
        patch_uint8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    else:
        patch_uint8 = patch

    if len(patch_uint8.shape) == 3:
        patch_uint8 = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(
        patch_uint8,
        P=config.LBP_N_POINTS,
        R=config.LBP_RADIUS,
        method='uniform'
    )

    # Compute histogram
    n_bins = config.LBP_N_POINTS + 2  # uniform LBP has P+2 bins
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return hist.astype(np.float32)


def extract_color_histogram(patch):
    """
    Extract color histogram features from LAB color space.

    LAB is ideal because:
      - L: luminance (ink darkness)
      - A: green-red (ink tint)
      - B: blue-yellow (ink color)

    Args:
        patch: RGB image patch (H, W, 3)

    Returns:
        Concatenated L, A, B histogram feature vector
    """
    if patch.dtype == np.float32 or patch.dtype == np.float64:
        patch_uint8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    else:
        patch_uint8 = patch

    if len(patch_uint8.shape) == 2:
        patch_uint8 = cv2.cvtColor(patch_uint8, cv2.COLOR_GRAY2RGB)

    # Convert to LAB
    lab = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2LAB)

    features = []
    n_bins = config.COLOR_HIST_BINS

    for channel in range(3):
        hist = cv2.calcHist(
            [lab], [channel], None,
            [n_bins], [0, 256]
        ).flatten()
        # Normalize
        hist = hist / (hist.sum() + 1e-8)
        features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_all_features(patch):
    """
    Extract combined feature vector from a patch.

    Combines GLCM + LBP + Color Histogram features.

    Args:
        patch: RGB image patch (H, W, 3) or grayscale (H, W)

    Returns:
        Combined feature vector (numpy array)
    """
    # Ensure RGB for color histogram
    if len(patch.shape) == 2:
        patch_rgb = cv2.cvtColor(
            (patch * 255).astype(np.uint8) if patch.dtype == np.float32 else patch,
            cv2.COLOR_GRAY2RGB
        )
    else:
        patch_rgb = patch

    # Grayscale for GLCM and LBP
    if len(patch.shape) == 3:
        if patch.dtype == np.float32 or patch.dtype == np.float64:
            gray = cv2.cvtColor(
                (np.clip(patch, 0, 1) * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        if patch.dtype == np.float32:
            gray = (patch * 255).astype(np.uint8)
        else:
            gray = patch

    glcm_feat = extract_glcm_features(gray)
    lbp_feat = extract_lbp_features(gray)
    color_feat = extract_color_histogram(patch_rgb)

    return np.concatenate([glcm_feat, lbp_feat, color_feat])


class ClassicalInkClassifier:
    """
    Classical ML classifier using hand-crafted features.

    Supports both SVM and Random Forest backends.
    """

    def __init__(self, method='svm'):
        """
        Args:
            method: 'svm' or 'rf' (random forest)
        """
        self.method = method

        if method == 'svm':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    C=config.SVM_C,
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced'
                ))
            ])
        else:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=config.RF_N_ESTIMATORS,
                    class_weight='balanced',
                    random_state=42
                ))
            ])

    def extract_features_batch(self, patches):
        """Extract features from a batch of patches."""
        features = []
        for patch in patches:
            feat = extract_all_features(patch)
            features.append(feat)
        return np.array(features)

    def fit(self, X_patches, y_labels):
        """
        Train the classifier.

        Args:
            X_patches: List of image patches
            y_labels: List of labels (0 or 1)
        """
        print(f"[Classical-{self.method.upper()}] Extracting features...")
        X_features = self.extract_features_batch(X_patches)
        print(f"  Feature matrix shape: {X_features.shape}")

        print(f"[Classical-{self.method.upper()}] Training classifier...")
        self.model.fit(X_features, y_labels)
        train_acc = self.model.score(X_features, y_labels)
        print(f"  Training accuracy: {train_acc:.4f}")

    def predict(self, patches):
        """Predict labels for a batch of patches."""
        X_features = self.extract_features_batch(patches)
        return self.model.predict(X_features)

    def predict_proba(self, patches):
        """Predict probabilities for a batch of patches."""
        X_features = self.extract_features_batch(patches)
        return self.model.predict_proba(X_features)

    def score(self, patches, labels):
        """Compute accuracy on a batch."""
        X_features = self.extract_features_batch(patches)
        return self.model.score(X_features, labels)

    def save(self, path):
        """Save the trained model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"  [OK] Saved classical model: {path}")

    def load(self, path):
        """Load a trained model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"  [OK] Loaded classical model: {path}")


if __name__ == "__main__":
    # Quick test with random data
    patch = np.random.rand(64, 64, 3).astype(np.float32)

    print("Extracting features from a test patch...")
    features = extract_all_features(patch)
    print(f"Feature vector size: {features.shape[0]}")
    print(f"  GLCM features: {extract_glcm_features(patch).shape[0]}")
    print(f"  LBP features: {extract_lbp_features(patch).shape[0]}")
    print(f"  Color histogram features: {extract_color_histogram(patch).shape[0]}")
