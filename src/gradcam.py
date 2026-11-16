"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the ink CNNs.

Shows which ink pixels of a unit drove the model's decision:
  * CNN pen classifier - evidence for the predicted pen
  * Siamese network    - evidence that a suspicious unit's ink differs
                         from a reference unit of the cheque's main ink

Custom implementation with PyTorch hooks; hooks are always removed after
use so repeated calls never accumulate them on the model.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Captures activations and gradients of one conv layer."""

    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def heatmap(self, size):
        if self.gradients is None or self.activations is None:
            return np.zeros(size, np.float32)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(cam)

    @staticmethod
    def overlay(image, heatmap, alpha=0.5):
        """Blend a [0,1] heatmap onto an RGB image (float [0,1] or uint8)."""
        img = image if image.dtype == np.uint8 else (np.clip(image, 0, 1) * 255).astype(np.uint8)
        h, w = img.shape[:2]
        hm = cv2.resize(heatmap.astype(np.float32), (w, h))
        col = cv2.cvtColor(cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 1 - alpha, col, alpha, 0)


def _run(encoder, patch, objective):
    """Generic Grad-CAM: objective(net_output) -> scalar to explain."""
    cam = GradCAM(encoder.last_conv)
    try:
        x = torch.as_tensor(patch[None], dtype=torch.float32,
                            device=next(encoder.parameters()).device)
        with torch.enable_grad():
            x.requires_grad_(True)
            target = objective(x)
            target.backward()
        hm = cam.heatmap(patch.shape[1:])
    finally:
        cam.remove()
    ink = patch.sum(0) > 0
    return hm * ink


def gradcam_pen_classifier(net, patch):
    """Heat-map of the evidence for the CNN's predicted pen."""
    net.eval()

    def objective(x):
        logits = net(x)
        return logits[0, logits.argmax(1).item()]

    return _run(net.encoder, patch, objective)


def gradcam_siamese(net, patch, reference_patch):
    """Heat-map of the evidence that `patch` and `reference_patch` differ in ink."""
    net.eval()
    ref = torch.as_tensor(reference_patch[None], dtype=torch.float32,
                          device=next(net.parameters()).device)
    with torch.no_grad():
        e_ref = net.embed(ref)

    def objective(x):
        return net.pair_logit(net.embed(x), e_ref)[0]

    return _run(net.encoder, patch, objective)


def patch_to_rgb(patch):
    """Render an optical-density patch (3, H, W) as an RGB ink image on white."""
    od = patch.transpose(1, 2, 0) * 2.0                 # undo OD_SCALE
    bgr = np.exp(-od) * 255
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
