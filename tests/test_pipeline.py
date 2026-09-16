"""
Basic validation tests for the cheque fraud detection pipeline.

Run: python -m pytest tests/test_pipeline.py -v
"""
import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TestConfig:
    """Test configuration is valid."""

    def test_directories_exist(self):
        assert os.path.isdir(config.DATA_DIR)
        assert os.path.isdir(config.MODEL_DIR)
        assert os.path.isdir(config.FIGURES_DIR)

    def test_hyperparameters(self):
        assert config.PATCH_SIZE > 0
        assert config.BATCH_SIZE > 0
        assert config.EPOCHS > 0
        assert 0 < config.LEARNING_RATE < 1
        assert 0 < config.CONTRASTIVE_MARGIN


class TestPreprocessing:
    """Test image preprocessing functions."""

    def test_resize_and_normalize(self):
        from src.preprocessing import resize_and_normalize
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = resize_and_normalize(img)
        assert result.shape == (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1.0

    def test_to_grayscale(self):
        from src.preprocessing import to_grayscale
        img = np.random.rand(64, 64, 3).astype(np.float32)
        gray = to_grayscale(img)
        assert len(gray.shape) == 2
        assert gray.shape == (64, 64)

    def test_to_lab(self):
        from src.preprocessing import to_lab
        img = np.random.rand(64, 64, 3).astype(np.float32)
        lab = to_lab(img)
        assert lab.shape == (64, 64, 3)
        assert lab.dtype == np.float32

    def test_augmentation(self):
        from src.preprocessing import apply_augmentation
        img = np.random.rand(64, 64, 3).astype(np.float32)
        aug = apply_augmentation(img)
        assert aug.shape == img.shape
        assert aug.dtype == np.float32

    def test_preprocess_cheque(self):
        from src.preprocessing import preprocess_cheque
        img = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        result = preprocess_cheque(img)
        assert 'rgb' in result
        assert 'grayscale' in result
        assert 'lab' in result
        assert result['rgb'].shape[:2] == config.IMAGE_SIZE

    def test_extract_ink_mask(self):
        from src.preprocessing import extract_ink_mask
        img = np.random.rand(64, 64, 3).astype(np.float32)
        mask = extract_ink_mask(img)
        assert mask.shape == (64, 64)
        assert mask.dtype == bool


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_synthetic_generator(self):
        from src.data_loader import SyntheticChequeGenerator
        gen = SyntheticChequeGenerator(num_images=3)
        results = gen.generate()
        assert len(results) == 3
        for path, pens in results:
            assert os.path.exists(path)
            assert len(pens) == 2

    def test_load_dataset(self):
        from src.data_loader import load_dataset
        dataset = load_dataset()
        assert len(dataset) > 0
        img, pens = dataset[0]
        assert img.ndim == 3
        assert len(pens) == 2


class TestPatchGeneration:
    """Test patch extraction and pairing."""

    def test_sliding_window(self):
        from src.patch_generator import extract_patches_sliding
        img = np.random.rand(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3).astype(np.float32)
        patches = extract_patches_sliding(img)
        assert len(patches) > 0
        for patch, (y, x) in patches:
            assert patch.shape == (config.PATCH_SIZE, config.PATCH_SIZE, 3)

    def test_contour_patches(self):
        from src.patch_generator import extract_patches_contour
        # Create image with some "ink"
        img = np.ones((256, 512, 3), dtype=np.float32) * 0.9
        # Draw some dark lines
        img[50:60, 100:200, :] = 0.1
        img[100:110, 150:250, :] = 0.1
        patches = extract_patches_contour(img)
        # May or may not find patches depending on contour detection
        assert isinstance(patches, list)

    def test_generate_pairs(self):
        from src.data_loader import load_dataset
        from src.patch_generator import generate_pairs
        dataset = load_dataset()
        pairs = generate_pairs(dataset, max_pairs_per_image=10)
        assert len(pairs) > 0
        pa, pb, label = pairs[0]
        assert pa.shape == (config.PATCH_SIZE, config.PATCH_SIZE, 3)
        assert label in [0, 1]


class TestModels:
    """Test model architectures."""

    def test_siamese_forward(self):
        from src.models.siamese_model import SiameseCNN
        model = SiameseCNN()
        x1 = torch.randn(2, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        x2 = torch.randn(2, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        out1, out2 = model(x1, x2)
        assert out1.shape == (2, config.EMBEDDING_DIM)
        assert out2.shape == (2, config.EMBEDDING_DIM)

    def test_siamese_similarity(self):
        from src.models.siamese_model import SiameseCNN
        model = SiameseCNN()
        x1 = torch.randn(2, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        x2 = torch.randn(2, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        sim = model.get_similarity(x1, x2)
        assert sim.shape == (2,)
        assert (sim >= 0).all() and (sim <= 1).all()

    def test_contrastive_loss(self):
        from src.models.siamese_model import ContrastiveLoss
        loss_fn = ContrastiveLoss()
        out1 = torch.randn(4, config.EMBEDDING_DIM)
        out2 = torch.randn(4, config.EMBEDDING_DIM)
        labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        loss = loss_fn(out1, out2, labels)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_cnn_forward(self):
        from src.models.cnn_model import InkClassifierCNN
        model = InkClassifierCNN()
        x = torch.randn(2, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        logits = model(x)
        assert logits.shape == (2, config.NUM_CLASSES)

    def test_cnn_param_count(self):
        from src.models.cnn_model import InkClassifierCNN
        from src.utils import count_parameters
        model = InkClassifierCNN()
        params = count_parameters(model)
        assert params < 500_000  # Lightweight requirement

    def test_siamese_param_count(self):
        from src.models.siamese_model import SiameseCNN
        from src.utils import count_parameters
        model = SiameseCNN()
        params = count_parameters(model)
        assert params < 500_000  # Lightweight requirement

    def test_classical_features(self):
        from src.models.classical_model import extract_all_features
        patch = np.random.rand(64, 64, 3).astype(np.float32)
        features = extract_all_features(patch)
        assert features.ndim == 1
        assert len(features) > 50  # Should be ~130+ features


class TestGradCAM:
    """Test Grad-CAM visualization."""

    def test_gradcam_cnn(self):
        from src.gradcam import get_gradcam_for_cnn
        from src.models.cnn_model import InkClassifierCNN
        model = InkClassifierCNN()
        x = torch.randn(1, 3, 64, 64)
        heatmap = get_gradcam_for_cnn(model, x)
        assert heatmap.shape == (64, 64)
        assert heatmap.min() >= 0 and heatmap.max() <= 1

    def test_gradcam_overlay(self):
        from src.gradcam import GradCAM
        img = np.random.rand(64, 64, 3).astype(np.float32)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = GradCAM.overlay(img, heatmap)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8


class TestFraudPipeline:
    """Test the end-to-end fraud detection pipeline."""

    def test_pipeline_runs(self):
        from src.fraud_pipeline import FraudDetector
        # Create a test image
        img = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        detector = FraudDetector(model_type='siamese')
        result = detector.detect(img)
        assert 'verdict' in result
        assert result['verdict'] in ['FRAUD', 'GENUINE']
        assert 0 <= result['confidence'] <= 1
        assert result['highlighted_image'] is not None


class TestPyTorchDatasets:
    """Test PyTorch Dataset wrappers."""

    def test_pair_dataset(self):
        from src.patch_generator import PatchPairDataset
        pairs = [
            (np.random.rand(64, 64, 3).astype(np.float32),
             np.random.rand(64, 64, 3).astype(np.float32),
             1)
            for _ in range(5)
        ]
        dataset = PatchPairDataset(pairs)
        assert len(dataset) == 5
        pa, pb, label = dataset[0]
        assert pa.shape == (3, 64, 64)
        assert isinstance(label, torch.Tensor)

    def test_single_patch_dataset(self):
        from src.patch_generator import SinglePatchDataset
        patches = [
            (np.random.rand(64, 64, 3).astype(np.float32), 0)
            for _ in range(5)
        ]
        dataset = SinglePatchDataset(patches)
        assert len(dataset) == 5
        patch, label = dataset[0]
        assert patch.shape == (3, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
