import torch
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import sys
import numpy as np

# Add project root to path so we can import modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from torch.utils.data import DataLoader
from src.core.category_manager import CategoryManager
from src.models.defect_classifier import DefectClassifier
from src.training.dataset import DefectDataset, collate_fn
from src.training.trainer import Trainer, TrainingConfig

def main():
    # Setup paths
    data_root = project_root / "data" / "processed"
    config_dir = project_root / "config"
    checkpoints_dir = project_root / "checkpoints"
    
    train_dir = data_root / "train" / "images"
    train_anno = data_root / "train" / "annotations.json"
    val_dir = data_root / "val" / "images"
    val_anno = data_root / "val" / "annotations.json"
    
    model_config_path = config_dir / "model_config.yaml"
    category_config_path = config_dir / "categories.yaml"

    # Verify paths exist
    if not train_dir.exists() or not train_anno.exists():
        print(f"Training data not found at {train_dir} or {train_anno}")
        return
    if not val_dir.exists() or not val_anno.exists():
        print(f"Validation data not found at {val_dir} or {val_anno}")
        return

    # Load Category Manager
    print("Loading categories...")
    category_manager = CategoryManager(category_config_path)
    
    # Load Model Config
    print("Loading model config...")
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config_dict = yaml.safe_load(f)
    
    # Create Datasets
    print("Creating datasets...")
    train_dataset = DefectDataset(
        data_dir=data_root / "train", # dataset expects relative paths from this dir
        annotation_file=train_anno,
        category_manager=category_manager,
        is_training=True
    )
    
    val_dataset = DefectDataset(
        data_dir=data_root / "val",
        annotation_file=val_anno,
        category_manager=category_manager,
        is_training=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config_dict["training"]["batch_size"],
        shuffle=True,
        num_workers=0, # optimized for windows
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config_dict["inference"]["batch_size"],
        shuffle=False,
        num_workers=0, # optimized for windows
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Initialize Model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DefectClassifier(
        backbone_name=model_config_dict["model"]["backbone"],
        pretrained=model_config_dict["model"]["pretrained"],
        num_cause_classes=category_manager.get_num_classes("cause"),
        num_shape_classes=category_manager.get_num_classes("shape"),
        num_depth_classes=category_manager.get_num_classes("depth"),
        dropout=model_config_dict["model"]["dropout"],
        shared_features=model_config_dict["model"]["shared_features"],
        head_hidden_features=model_config_dict["model"]["head_hidden_features"],
        freeze_backbone_layers=model_config_dict["model"]["freeze_backbone_layers"]
    )

    # Setup Training Config
    training_config = TrainingConfig(
        epochs=10, # Short run for demonstration
        batch_size=model_config_dict["training"]["batch_size"],
        learning_rate=model_config_dict["training"]["learning_rate"],
        weight_decay=model_config_dict["training"]["weight_decay"],
        warmup_epochs=model_config_dict["training"]["warmup_epochs"],
        patience=5,
        min_delta=model_config_dict["training"]["min_delta"],
        gradient_clip=model_config_dict["training"]["gradient_clip"],
        mixed_precision=model_config_dict["training"]["mixed_precision"] and device.type == "cuda",
        num_workers=0,
        loss_weights=model_config_dict["training"]["loss_weights"],
        label_smoothing=model_config_dict["training"]["label_smoothing"]
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoints_dir
    )
    
    # Start Training
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Visualization
    print("Visualizing results...")
    plot_training_history(history, project_root / "training_results.png")
    print(f"Training complete. Results saved to {project_root / 'training_results.png'}")

def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Training Loss')
    plt.plot(epochs, history["val_loss"], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Metrics (Accuracy)
    # Extract mean accuracy from metrics dict in history
    val_acc = [m['mean_accuracy'] for m in history['metrics']]
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, 'g-', label='Validation Mean Accuracy')
    
    # Also plot individual task accuracies if available
    if len(history['metrics']) > 0:
        first_metric = history['metrics'][0]
        if 'cause_accuracy' in first_metric:
             cause_acc = [m['cause_accuracy'] for m in history['metrics']]
             plt.plot(epochs, cause_acc, '--', label='Cause Acc', alpha=0.7)
        if 'shape_accuracy' in first_metric:
             shape_acc = [m['shape_accuracy'] for m in history['metrics']]
             plt.plot(epochs, shape_acc, '--', label='Shape Acc', alpha=0.7)
        if 'depth_accuracy' in first_metric:
             depth_acc = [m['depth_accuracy'] for m in history['metrics']]
             plt.plot(epochs, depth_acc, '--', label='Depth Acc', alpha=0.7)

    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
