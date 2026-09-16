"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Provides visual explanations for CNN predictions by highlighting
the regions that most influenced the model's decision.

Custom implementation using PyTorch hooks (no external dependencies).
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM visualization for CNN models.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(input_tensor, target_class)
        overlay = gradcam.overlay(original_image, heatmap)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch CNN model
            target_layer: The convolutional layer to visualize
                         (typically the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class to visualize. If None, uses predicted class.

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations. "
                             "Check that target_layer is correct.")

        # Global average pooling of gradients → weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            image: Original image (H, W, 3) as float32 [0,1] or uint8 [0,255]
            heatmap: Grad-CAM heatmap (H, W) normalized to [0, 1]
            alpha: Transparency of the heatmap overlay
            colormap: OpenCV colormap to use

        Returns:
            Overlay image (H, W, 3) as uint8
        """
        # Convert image to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()

        # Resize heatmap to match image
        h, w = img_uint8.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )

        # Convert heatmap from BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay


def get_gradcam_for_cnn(model, input_tensor, target_class=None):
    """
    Convenience function to generate Grad-CAM for the InkClassifierCNN.

    Args:
        model: InkClassifierCNN instance
        input_tensor: (1, 3, 64, 64) tensor
        target_class: Optional target class index

    Returns:
        heatmap: (H, W) numpy array
    """
    # Find last conv layer in the features Sequential
    target_layer = None
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("No Conv2d layer found in model.features")

    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, target_class)

    return heatmap


def get_gradcam_for_siamese(model, input_tensor, target_class=None):
    """
    Generate Grad-CAM for one branch of the Siamese network.

    Since Siamese doesn't have class outputs, we visualize the
    embedding magnitude (which regions contribute most to the embedding).

    Args:
        model: SiameseCNN instance
        input_tensor: (1, 3, 64, 64) tensor

    Returns:
        heatmap: (H, W) numpy array
    """
    # Find last conv layer in the branch
    target_layer = None
    for module in model.branch.features:
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("No Conv2d layer found in model.branch.features")

    gradcam_obj = GradCAM.__new__(GradCAM)
    gradcam_obj.model = model.branch
    gradcam_obj.target_layer = target_layer
    gradcam_obj.gradients = None
    gradcam_obj.activations = None
    gradcam_obj._register_hooks()

    model.branch.eval()
    embedding = model.branch(input_tensor)

    # Use the L2 norm of the embedding as the "target"
    target = embedding.norm(dim=1)
    model.branch.zero_grad()
    target.backward(retain_graph=True)

    if gradcam_obj.gradients is None or gradcam_obj.activations is None:
        return np.zeros(input_tensor.shape[2:])

    weights = gradcam_obj.gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * gradcam_obj.activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:],
                        mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


if __name__ == "__main__":
    from models.cnn_model import InkClassifierCNN

    model = InkClassifierCNN()
    x = torch.randn(1, 3, 64, 64)

    heatmap = get_gradcam_for_cnn(model, x)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
