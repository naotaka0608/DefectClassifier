
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

def get_target_layer(model):
    """Grad-CAMのターゲット層を自動特定"""
    # model is DefectClassifier, backbone is model.backbone
    if not hasattr(model, 'backbone'):
        raise ValueError("Model does not have a backbone attribute")
        
    backbone = model.backbone
    
    # ResNet series
    if hasattr(backbone, 'layer4'):
        return backbone.layer4[-1]
    
    # EfficientNet series
    if hasattr(backbone, 'features'):
        return backbone.features[-1]
        
    # Fallback: Last Conv2d
    for module in reversed(list(backbone.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
            
    raise ValueError("Could not find suitable target layer for Grad-CAM")

class GradCAM:
    """簡易Grad-CAM実装"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer else get_target_layer(model)
        self.gradients = None
        self.activations = None
        
        # フックの登録
        self.handle_f = self.target_layer.register_forward_hook(self._save_activations)
        self.handle_b = self.target_layer.register_full_backward_hook(self._save_gradients)

    def remove_hooks(self):
        """フックを削除"""
        self.handle_f.remove()
        self.handle_b.remove()

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None, task_type=None):
        """
        Args:
            x: Input tensor (1, C, H, W)
            class_idx: Target class index. If None, predicted class is used.
            task_type: TaskType enum (CAUSE, SHAPE, DEPTH)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(x)
        
        # タスクごとの出力取得
        if task_type is None:
            # デフォルトは 'cause'
            from src.core.types import TaskType
            task_type = TaskType.CAUSE

        if isinstance(outputs, dict):
            logits = outputs[task_type]
        else:
            logits = getattr(outputs, task_type)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        score = logits[0, class_idx]
        score.backward()
        
        if self.gradients is None:
            return None, class_idx
             
        # GAP (Global Average Pooling) on gradients
        weights = torch.mean(self.gradients, dim=(2, 3))[0]  # (channels,)
        
        # Weighted sum of activations
        activations = self.activations[0]  # (channels, h, w)
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy(), class_idx

def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """ヒートマップを画像に重ねる"""
    img_np = np.array(img)
    
    # ヒートマップを画像サイズにリサイズ
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # カラーマップ適用 (0-255に変換)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # RGB変換 (OpenCVはBGR)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 重ね合わせ
    overlay = (alpha * heatmap_color + (1 - alpha) * img_np).astype(np.uint8)
    
    return Image.fromarray(overlay)
