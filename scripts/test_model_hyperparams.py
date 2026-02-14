
import sys
import unittest
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.defect_classifier import DefectClassifier
from src.core.types import TaskType

class TestModelHyperparams(unittest.TestCase):
    def setUp(self):
        self.task_config = {
            TaskType.CAUSE: 3,
            TaskType.SHAPE: 2,
            TaskType.DEPTH: 2
        }

    def test_dropout_application(self):
        """Test if dropout parameter is applied to the model heads"""
        dropout_rate = 0.75
        model = DefectClassifier(
            task_config=self.task_config,
            dropout=dropout_rate,
            pretrained=False # Speed up init
        )
        
        # Check shared layer dropout
        # The shared layer in MultiTaskHead is defined as:
        # self.shared_layer = nn.Sequential(..., nn.Dropout(dropout), ...)
        # So we look for Dropout modules in shared_layer
        shared_dropouts = [m for m in model.multi_task_head.shared_layer if isinstance(m, nn.Dropout)]
        self.assertTrue(len(shared_dropouts) > 0)
        self.assertAlmostEqual(shared_dropouts[0].p, dropout_rate)
        
        # Check task head dropout
        # Each head has: nn.Sequential(..., nn.Dropout(dropout), ...)
        for task_name, head_module in model.multi_task_head.heads.items():
            head_dropouts = [m for m in head_module.head if isinstance(m, nn.Dropout)]
            self.assertTrue(len(head_dropouts) > 0, f"No dropout found in {task_name} head")
            self.assertAlmostEqual(head_dropouts[0].p, dropout_rate)
            
        print(f"✅ Dropout rate {dropout_rate} verified in all heads")

    def test_freeze_backbone(self):
        """Test if backbone freezing works"""
        # Default freeze is 6 layers
        model = DefectClassifier(
            task_config=self.task_config,
            freeze_backbone_layers=6,
            pretrained=False
        )
        
        # Check if some parameters are frozen (requires_grad=False)
        frozen_params = [p for p in model.backbone.parameters() if not p.requires_grad]
        trainable_params = [p for p in model.backbone.parameters() if p.requires_grad]
        
        self.assertTrue(len(frozen_params) > 0, "Should have frozen parameters")
        self.assertTrue(len(trainable_params) > 0, "Should have trainable parameters")
        
        # Now unfreeze all
        model.unfreeze_backbone()
        frozen_params_after = [p for p in model.backbone.parameters() if not p.requires_grad]
        self.assertEqual(len(frozen_params_after), 0, "All parameters should be trainable after unfreeze")
        
        print("✅ Backbone freezing/unfreezing verified")

if __name__ == "__main__":
    unittest.main()
