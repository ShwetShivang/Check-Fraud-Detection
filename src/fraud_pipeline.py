"""
End-to-end fraud detection pipeline.

For a given cheque image:
  1. Preprocess and extract patches
  2. Compare all patch pairs using the Siamese model
  3. Build a similarity map
  4. Detect suspicious (low-similarity) regions
  5. Render highlighted output
  6. Return verdict: FRAUD or GENUINE
"""
import os
import sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import resize_and_normalize, extract_ink_mask, preprocess_cheque
from src.patch_generator import extract_patches_sliding, extract_patches_contour
from src.models.siamese_model import SiameseCNN
from src.models.cnn_model import InkClassifierCNN
from src.gradcam import get_gradcam_for_cnn, get_gradcam_for_siamese, GradCAM


class FraudDetector:
    """
    Complete cheque fraud detection pipeline.

    Usage:
        detector = FraudDetector()
        result = detector.detect("cheque.jpg")
        print(result['verdict'], result['confidence'])
    """

    def __init__(self, model_type='siamese', device=None):
        """
        Args:
            model_type: 'siamese', 'cnn', or 'classical'
            device: PyTorch device
        """
        self.model_type = model_type
        self.device = device or config.DEVICE
        self.model = None
        self.loaded = False

        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if self.model_type == 'siamese':
            self.model = SiameseCNN().to(self.device)
            path = os.path.join(config.MODEL_DIR, "siamese_best.pth")
            if os.path.exists(path):
                self.model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )
                self.model.eval()
                self.loaded = True
                print(f"[FraudDetector] Loaded Siamese model from {path}")
            else:
                print("[FraudDetector] ⚠ No trained Siamese model found. Using random weights.")
                self.model.eval()
                self.loaded = True

        elif self.model_type == 'cnn':
            self.model = InkClassifierCNN().to(self.device)
            path = os.path.join(config.MODEL_DIR, "cnn_best.pth")
            if os.path.exists(path):
                self.model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )
                self.model.eval()
                self.loaded = True
                print(f"[FraudDetector] Loaded CNN model from {path}")
            else:
                print("[FraudDetector] ⚠ No trained CNN model found. Using random weights.")
                self.model.eval()
                self.loaded = True

        elif self.model_type == 'classical':
            from src.models.classical_model import ClassicalInkClassifier
            self.model = ClassicalInkClassifier(method='svm')
            path = os.path.join(config.MODEL_DIR, "classical_svm.pkl")
            if os.path.exists(path):
                self.model.load(path)
                self.loaded = True
            else:
                print("[FraudDetector] ⚠ No trained classical model found.")

    def detect(self, image_input):
        """
        Run fraud detection on a cheque image.

        Args:
            image_input: Path to image or numpy array (RGB)

        Returns:
            dict:
                'verdict': 'FRAUD' or 'GENUINE'
                'confidence': float [0, 1]
                'highlighted_image': Image with suspicious regions marked
                'similarity_map': 2D array of similarity scores
                'similarity_scores': List of all pairwise scores
                'suspicious_patches': List of suspicious patch locations
                'gradcam_overlay': Grad-CAM visualization (if applicable)
        """
        # 1. Load and preprocess
        processed = preprocess_cheque(image_input)
        img_rgb = processed['rgb']
        original = processed['original']

        H, W = img_rgb.shape[:2]
        ink_mask = extract_ink_mask(img_rgb)

        # 2. Extract patches
        patches = extract_patches_sliding(img_rgb, ink_mask=ink_mask)
        if len(patches) < 4:
            patches = extract_patches_sliding(img_rgb, ink_mask=None)

        if len(patches) < 2:
            return self._make_result(
                'GENUINE', 1.0, original,
                np.ones((1, 1)), [], [],
                "Insufficient patches for analysis"
            )

        # 3. Run model-specific detection
        if self.model_type == 'siamese':
            result = self._detect_siamese(patches, img_rgb, original)
        elif self.model_type == 'cnn':
            result = self._detect_cnn(patches, img_rgb, original)
        else:
            result = self._detect_classical(patches, img_rgb, original)

        return result

    @staticmethod
    def _compute_color_divergence(left_patches, right_patches):
        """Compute LAB chromaticity divergence between left and right patches.

        Compares the A (green-red) and B (blue-yellow) channels of LAB color
        space across regions. High divergence indicates different ink colors.

        Returns:
            float: color divergence score (0 = identical, higher = different inks)
        """
        def _patch_lab_stats(patches_list):
            """Get mean A,B chromaticity for each patch's ink pixels."""
            stats = []
            for patch, pos in patches_list:
                p = patch
                if p.dtype in (np.float32, np.float64):
                    p = (np.clip(p, 0, 1) * 255).astype(np.uint8)
                if len(p.shape) == 2:
                    continue
                lab = cv2.cvtColor(p, cv2.COLOR_RGB2LAB)
                # Focus on actual ink pixels:
                # 1. Dark strokes (L < 150) to exclude paper background
                # 2. Must have chromaticity (|A-128|>5 or |B-128|>5)
                #    to exclude grey printed text/labels that differ
                #    between left/right halves and cause false divergence
                dark_mask = lab[:, :, 0] < 150
                chroma_mask = (
                    (np.abs(lab[:, :, 1].astype(int) - 128) > 5) |
                    (np.abs(lab[:, :, 2].astype(int) - 128) > 5)
                )
                ink_mask = dark_mask & chroma_mask
                if ink_mask.sum() < 10:
                    continue
                mean_a = float(np.mean(lab[ink_mask, 1]))
                mean_b = float(np.mean(lab[ink_mask, 2]))
                stats.append((mean_a, mean_b))
            return stats

        left_stats = _patch_lab_stats(left_patches)
        right_stats = _patch_lab_stats(right_patches)

        if len(left_stats) < 2 or len(right_stats) < 2:
            return 0.0

        # Mean chromaticity per region
        left_a = np.mean([s[0] for s in left_stats])
        left_b = np.mean([s[1] for s in left_stats])
        right_a = np.mean([s[0] for s in right_stats])
        right_b = np.mean([s[1] for s in right_stats])

        # Euclidean distance in AB chromaticity space
        ab_dist = np.sqrt((left_a - right_a) ** 2 + (left_b - right_b) ** 2)

        # Also check within-patch chromaticity variance (high = multiple inks)
        all_stats = left_stats + right_stats
        all_a = np.array([s[0] for s in all_stats])
        all_b = np.array([s[1] for s in all_stats])
        chroma_std = np.sqrt(np.std(all_a) ** 2 + np.std(all_b) ** 2)

        # Combine: region difference + overall variability
        divergence = ab_dist * 0.6 + chroma_std * 0.4
        return float(divergence)

    def _detect_siamese(self, patches, img_rgb, original):
        """Siamese-based fraud detection using hybrid analysis.

        Uses four complementary signals to detect fraud:
          1. Absolute threshold — patches below SIMILARITY_THRESHOLD are suspicious
          2. Distribution spread — large score range / std indicates mixed inks
          3. Adaptive outliers — scores far below the mean are flagged
          4. LAB color divergence — direct ink chromaticity comparison across regions

        Any strong signal can trigger a FRAUD verdict, avoiding the
        false-negative problem of requiring all signals simultaneously.
        """
        H, W = img_rgb.shape[:2]
        ps = config.PATCH_SIZE

        # Identify left and right region patches
        mid = W // 2
        left_patches = [(p, pos) for p, pos in patches if pos[1] + ps // 2 < mid]
        right_patches = [(p, pos) for p, pos in patches if pos[1] + ps // 2 >= mid]

        if not left_patches or not right_patches:
            # If all patches are on one side, compare adjacent patches
            left_patches = patches[:len(patches) // 2]
            right_patches = patches[len(patches) // 2:]

        # Compare cross-region pairs
        similarity_scores = []
        all_positions = []

        self.model.eval()
        with torch.no_grad():
            for lp, lpos in left_patches:
                for rp, rpos in right_patches:
                    # Prepare tensors
                    t1 = torch.from_numpy(lp.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                    t2 = torch.from_numpy(rp.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

                    # Get similarity
                    sim = self.model.get_similarity(t1, t2).item()
                    similarity_scores.append(sim)
                    all_positions.append((lpos, rpos))

        # Build similarity map
        sim_map = self._build_similarity_map(
            all_positions, similarity_scores, H, W
        )

        # ── Signal 4: Direct LAB color analysis ──
        color_divergence = self._compute_color_divergence(left_patches, right_patches)

        # ── Hybrid fraud detection with four signals ──
        suspicious_positions = []

        if similarity_scores:
            scores = np.array(similarity_scores)
            avg_sim = float(np.mean(scores))
            std_sim = float(np.std(scores))
            min_sim = float(np.min(scores))
            max_sim = float(np.max(scores))
            score_range = max_sim - min_sim
            cv = std_sim / (avg_sim + 1e-8)

            # ── Signal 1: Absolute threshold ──
            abs_threshold = config.SIMILARITY_THRESHOLD
            for (lpos, rpos), sim in zip(all_positions, similarity_scores):
                if sim < abs_threshold:
                    suspicious_positions.append((lpos, rpos, sim))

            abs_suspicious_ratio = len(suspicious_positions) / len(similarity_scores)

            # ── Signal 2: Distribution spread ──
            has_high_spread = (score_range > 0.15 and std_sim > 0.05)

            # ── Signal 3: Adaptive outliers ──
            adaptive_thresh = avg_sim - 1.5 * std_sim
            outlier_count = int(np.sum(scores < adaptive_thresh))
            outlier_ratio = outlier_count / len(similarity_scores)

            # ── Combined verdict ──
            fraud_score = 0.0

            # (A) Strong absolute signal
            if abs_suspicious_ratio > config.FRAUD_REGION_THRESHOLD:
                fraud_score += 0.5 + abs_suspicious_ratio

            # (B) Distribution spread + some suspicious patches
            if has_high_spread and abs_suspicious_ratio > 0.05:
                fraud_score += 0.3 + cv

            # (C) Adaptive outliers with spread
            if outlier_ratio > 0.1 and cv > 0.1:
                fraud_score += 0.2 + outlier_ratio

            # (D) LAB color divergence — strong signal for ink color differences
            # Genuine cheques have divergence < 7, fraud cheques > 7.
            if color_divergence > 7.0:
                color_contribution = min(0.6, (color_divergence - 7.0) / 15.0)
                fraud_score += color_contribution
                # Flag cross-region patches for highlighting
                if not suspicious_positions and color_divergence > 10.0:
                    for (lpos, rpos), sim in zip(all_positions, similarity_scores):
                        suspicious_positions.append((lpos, rpos, sim))

            # Edge case: if ALL patches are below threshold, that's definitive
            if abs_suspicious_ratio >= 0.95 and len(similarity_scores) >= 4:
                fraud_score = max(fraud_score, 0.6)

            if fraud_score >= 0.55:
                verdict = 'FRAUD'
                confidence = min(1.0, fraud_score)
                # If no patches were flagged by threshold,
                # use adaptive outliers for highlighting
                if not suspicious_positions:
                    for (lpos, rpos), sim in zip(all_positions, similarity_scores):
                        if sim < adaptive_thresh:
                            suspicious_positions.append((lpos, rpos, sim))
            else:
                verdict = 'GENUINE'
                confidence = max(0.0, 1.0 - fraud_score)
        else:
            avg_sim = 1.0
            verdict = 'GENUINE'
            confidence = 1.0

        # Highlight suspicious regions
        highlighted = self._highlight_regions(original, suspicious_positions, img_rgb.shape)

        # Grad-CAM
        gradcam_overlay = None
        if patches:
            try:
                sample_patch = patches[0][0]
                t = torch.from_numpy(sample_patch.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                heatmap = get_gradcam_for_siamese(self.model, t)
                gradcam_overlay = GradCAM.overlay(sample_patch, heatmap)
            except Exception:
                pass

        return self._make_result(
            verdict, confidence, highlighted,
            sim_map, similarity_scores, suspicious_positions,
            gradcam_overlay=gradcam_overlay
        )

    def _detect_cnn(self, patches, img_rgb, original):
        """CNN-based fraud detection with color divergence analysis."""
        H, W = img_rgb.shape[:2]
        ps = config.PATCH_SIZE
        mid = W // 2

        # Classify each patch
        patch_classes = []
        patch_probs = []

        self.model.eval()
        with torch.no_grad():
            for patch, pos in patches:
                t = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                logits = self.model(t)
                probs = torch.softmax(logits, dim=1)
                pred_class = logits.argmax(dim=1).item()
                pred_prob = probs[0, pred_class].item()

                patch_classes.append((pred_class, pos))
                patch_probs.append(pred_prob)

        # ── Signal 1: CNN class mixing ──
        classes = [c for c, _ in patch_classes]
        unique_classes = set(classes)
        cnn_fraud_score = 0.0

        minority_class = None
        if len(unique_classes) > 1:
            class_counts = {c: classes.count(c) for c in unique_classes}
            minority_class = min(class_counts, key=class_counts.get)
            minority_ratio = class_counts[minority_class] / len(classes)
            if minority_ratio > 0.15:
                cnn_fraud_score = min(1.0, minority_ratio * 2)

        # ── Signal 2: LAB color divergence ──
        left_patches = [(p, pos) for p, pos in patches if pos[1] + ps // 2 < mid]
        right_patches = [(p, pos) for p, pos in patches if pos[1] + ps // 2 >= mid]
        if not left_patches or not right_patches:
            left_patches = patches[:len(patches) // 2]
            right_patches = patches[len(patches) // 2:]
        color_divergence = self._compute_color_divergence(left_patches, right_patches)

        color_score = 0.0
        if color_divergence > 7.0:
            color_score = min(1.0, (color_divergence - 7.0) / 15.0)

        # ── Combined verdict ──
        # Color divergence is the reliable signal. When colors are uniform
        # (divergence < 7), dampen model predictions to avoid false positives.
        if color_divergence < 7.0:
            cnn_fraud_score *= 0.1  # Dampen unreliable model signal

        fraud_score = max(cnn_fraud_score, color_score)

        if fraud_score >= 0.15:
            verdict = 'FRAUD'
            confidence = min(1.0, fraud_score)
        else:
            verdict = 'GENUINE'
            confidence = max(0.0, 1.0 - fraud_score)

        # Highlight suspicious patches
        suspicious = []
        if verdict == 'FRAUD':
            if minority_class is not None:
                for cls, pos in patch_classes:
                    if cls == minority_class:
                        suspicious.append((pos, pos, 0.3))
            if not suspicious:
                # Use left-right cross pairs for color-based detection
                for lp, lpos in left_patches[:5]:
                    for rp, rpos in right_patches[:5]:
                        suspicious.append((lpos, rpos, 0.3))

        highlighted = self._highlight_regions(original, suspicious, img_rgb.shape)
        sim_map = self._build_class_map(patch_classes, H, W)

        # Grad-CAM
        gradcam_overlay = None
        if patches:
            try:
                sample_patch = patches[0][0]
                t = torch.from_numpy(sample_patch.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                heatmap = get_gradcam_for_cnn(self.model, t)
                gradcam_overlay = GradCAM.overlay(sample_patch, heatmap)
            except Exception:
                pass

        scores = [1.0 if c == classes[0] else 0.0 for c in classes]

        return self._make_result(
            verdict, confidence, highlighted,
            sim_map, scores, suspicious,
            gradcam_overlay=gradcam_overlay
        )

    def _detect_classical(self, patches, img_rgb, original):
        """Classical ML-based fraud detection with color divergence analysis."""
        H, W = img_rgb.shape[:2]
        ps = config.PATCH_SIZE
        mid = W // 2

        patch_arrays = [p for p, _ in patches]
        positions = [pos for _, pos in patches]

        predictions = self.model.predict(patch_arrays)
        probabilities = self.model.predict_proba(patch_arrays)

        # ── Signal 1: SVM/RF class mixing ──
        unique = set(predictions)
        ml_fraud_score = 0.0
        minority = None
        if len(unique) > 1:
            counts = {c: list(predictions).count(c) for c in unique}
            minority = min(counts, key=counts.get)
            minority_ratio = counts[minority] / len(predictions)
            if minority_ratio > 0.15:
                ml_fraud_score = min(1.0, minority_ratio * 2)

        # ── Signal 2: LAB color divergence ──
        left_patches = [(p, pos) for p, pos in patches if pos[1] + ps // 2 < mid]
        right_patches = [(p, pos) for p, pos in patches if pos[1] + ps // 2 >= mid]
        if not left_patches or not right_patches:
            left_patches = patches[:len(patches) // 2]
            right_patches = patches[len(patches) // 2:]
        color_divergence = self._compute_color_divergence(left_patches, right_patches)

        color_score = 0.0
        if color_divergence > 7.0:
            color_score = min(1.0, (color_divergence - 7.0) / 15.0)

        # ── Combined verdict ──
        fraud_score = max(ml_fraud_score, color_score)

        if fraud_score >= 0.15:
            verdict = 'FRAUD'
            confidence = min(1.0, fraud_score)
        else:
            verdict = 'GENUINE'
            confidence = float(np.mean(np.max(probabilities, axis=1)))

        suspicious = []
        if verdict == 'FRAUD':
            if minority is not None:
                for pred, pos in zip(predictions, positions):
                    if pred == minority:
                        suspicious.append((pos, pos, 0.3))
            if not suspicious:
                for lp, lpos in left_patches[:5]:
                    for rp, rpos in right_patches[:5]:
                        suspicious.append((lpos, rpos, 0.3))

        highlighted = self._highlight_regions(original, suspicious, img_rgb.shape)
        sim_map = np.ones((H, W), dtype=np.float32)

        scores = [1.0 if p == predictions[0] else 0.0 for p in predictions]

        return self._make_result(
            verdict, confidence, highlighted,
            sim_map, scores, suspicious
        )

    def _build_similarity_map(self, positions, scores, H, W):
        """Build a 2D similarity heatmap from pairwise scores."""
        sim_map = np.ones((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        ps = config.PATCH_SIZE

        for (pos_a, pos_b), score in zip(positions, scores):
            # Mark both patch locations
            for (y, x) in [pos_a, pos_b]:
                y1, y2 = max(0, y), min(H, y + ps)
                x1, x2 = max(0, x), min(W, x + ps)
                sim_map[y1:y2, x1:x2] += score
                count_map[y1:y2, x1:x2] += 1

        # Average
        mask = count_map > 0
        sim_map[mask] /= count_map[mask]

        return np.clip(sim_map, 0, 1)

    def _build_class_map(self, patch_classes, H, W):
        """Build a 2D class map from patch classifications."""
        class_map = np.full((H, W), 0.5, dtype=np.float32)
        ps = config.PATCH_SIZE

        for cls, (y, x) in patch_classes:
            y1, y2 = max(0, y), min(H, y + ps)
            x1, x2 = max(0, x), min(W, x + ps)
            class_map[y1:y2, x1:x2] = float(cls)

        return class_map

    def _highlight_regions(self, original, suspicious, processed_shape):
        """Draw bounding boxes / overlay on suspicious regions."""
        # Resize original to display size
        display = cv2.resize(original, (processed_shape[1], processed_shape[0]))
        highlighted = display.copy()
        ps = config.PATCH_SIZE

        if not suspicious:
            return highlighted

        # Create a semi-transparent red overlay
        overlay = highlighted.copy()

        for item in suspicious:
            if len(item) == 3:
                pos_a, pos_b, score = item
            else:
                pos_a = item[0]
                pos_b = item[1] if len(item) > 1 else item[0]
                score = 0.3

            for (y, x) in [pos_a] if pos_a == pos_b else [pos_a, pos_b]:
                y1, y2 = y, min(y + ps, highlighted.shape[0])
                x1, x2 = x, min(x + ps, highlighted.shape[1])

                # Red overlay
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
                # Border
                cv2.rectangle(highlighted, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Blend
        alpha = 0.3
        highlighted = cv2.addWeighted(overlay, alpha, highlighted, 1 - alpha, 0)

        # Add "SUSPICIOUS" labels
        for item in suspicious[:5]:  # limit labels for readability
            if len(item) >= 2:
                pos = item[0]
                y, x = pos
                cv2.putText(highlighted, "!", (x + 5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return highlighted

    def _make_result(self, verdict, confidence, highlighted,
                     sim_map, scores, suspicious, note=None,
                     gradcam_overlay=None):
        """Construct the result dictionary."""
        return {
            'verdict': verdict,
            'confidence': confidence,
            'highlighted_image': highlighted,
            'similarity_map': sim_map,
            'similarity_scores': scores,
            'suspicious_patches': suspicious,
            'note': note,
            'gradcam_overlay': gradcam_overlay,
        }


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def detect_fraud(image_path, model_type='siamese'):
    """
    Convenience function for single-image fraud detection.

    Args:
        image_path: Path to cheque image
        model_type: 'siamese', 'cnn', or 'classical'

    Returns:
        Result dict
    """
    detector = FraudDetector(model_type=model_type)
    return detector.detect(image_path)


if __name__ == "__main__":
    from data_loader import load_dataset

    # Quick test
    dataset = load_dataset()
    if len(dataset) > 0:
        img, pens = dataset[0]
        path = dataset.images[0]
        print(f"\nTesting fraud pipeline on: {path}")

        detector = FraudDetector(model_type='siamese')
        result = detector.detect(path)

        print(f"\n  Verdict: {result['verdict']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Similarity scores: {len(result['similarity_scores'])}")
        print(f"  Suspicious regions: {len(result['suspicious_patches'])}")

        if result['highlighted_image'] is not None:
            out_path = os.path.join(config.SAMPLE_RESULTS_DIR, "test_detection.png")
            cv2.imwrite(out_path, cv2.cvtColor(
                result['highlighted_image'], cv2.COLOR_RGB2BGR
            ))
            print(f"  Saved highlighted image: {out_path}")
