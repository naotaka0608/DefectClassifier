
import sys
import unittest
from pathlib import Path
import albumentations as A
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import AugmentationConfig
from src.core.category_manager import CategoryManager
from src.core.constants import CATEGORIES_CONFIG_PATH
from src.training.dataset import DefectDataset

class TestAugmentationConfig(unittest.TestCase):
    def setUp(self):
        self.category_manager = CategoryManager(CATEGORIES_CONFIG_PATH)
        self.dummy_samples = [
            {
                "image_path": "dummy.jpg", 
                "cause": "擦り傷", "shape": "線状", "depth": "表層",
                "file_name": "dummy.jpg"
            }
        ]
        # Create a dummy image directory that won't be used because we mock loading
        self.dummy_dir = Path("dummy_dir")

    def test_resize_config_application(self):
        """Test if resize parameter in config is applied to dataset transform"""
        # Config with specific resize
        aug_config = AugmentationConfig(resize=(128, 128))
        
        dataset = DefectDataset(
            data_dir=self.dummy_dir,
            category_manager=self.category_manager,
            samples=self.dummy_samples,
            aug_config=aug_config,
            is_training=True
        )
        
        # Check transform pipeline
        transforms = dataset.transform.transforms
        resize_op = next((t for t in transforms if isinstance(t, A.Resize)), None)
        
        self.assertIsNotNone(resize_op, "Resize operation should strictly exist")
        self.assertEqual(resize_op.height, 128)
        self.assertEqual(resize_op.width, 128)
        print("✅ Augmentation Config (Resize) verified successfully")

    def test_horizontal_flip_config_application(self):
        """Test if horizontal_flip parameter is applied"""
        aug_config = AugmentationConfig(horizontal_flip=1.0) # Always flip
        
        dataset = DefectDataset(
            data_dir=self.dummy_dir,
            category_manager=self.category_manager,
            samples=self.dummy_samples,
            aug_config=aug_config,
            is_training=True
        )
        
        transforms = dataset.transform.transforms
        flip_op = next((t for t in transforms if isinstance(t, A.HorizontalFlip)), None)
        
        self.assertIsNotNone(flip_op)
        self.assertEqual(flip_op.p, 1.0)
        print("✅ Augmentation Config (HorizontalFlip) verified successfully")

if __name__ == "__main__":
    unittest.main()
