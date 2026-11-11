"""
Baseline Model 2: Fine-Tuned Deep CNN on Fashion MNIST
Author: Antigravity
Description: PyTorch implementation of a high-performance Convolutional Neural Network
             for Fashion MNIST classification.
             Controlled and parameterized via config.yaml.
             Features real-time data augmentation, AdamW + Cosine Annealing,
             label smoothing, validation monitoring with best-checkpointing,
             and full hyperparameter persistence.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config_utils import load_config, save_hyperparameters, format_hyperparams_for_display
from model_architecture import build_cnn_model, EnhancedCNN, BasicCNN

def apply_data_augmentation(images: torch.Tensor, aug_cfg: dict) -> torch.Tensor:
    """
    Applies real-time pure PyTorch data augmentation to mini-batch tensors.
    No torchvision dependency required; fast vectorized GPU/CPU operations.
    """
    if not aug_cfg.get("enabled", True):
        return images

    B, C, H, W = images.shape
    device = images.device

    # 1. Random Horizontal Flip (50% probability)
    if aug_cfg.get("horizontal_flip", True):
        flip_mask = torch.rand(B, 1, 1, 1, device=device) < 0.5
        images = torch.where(flip_mask, torch.flip(images, dims=[-1]), images)

    # 2. Random Crop / Shift with reflection padding
    crop_padding = aug_cfg.get("random_crop_padding", 2)
    if crop_padding > 0:
        padded = F.pad(images, (crop_padding, crop_padding, crop_padding, crop_padding), mode='replicate')
        # Random offset per batch
        dx = torch.randint(0, crop_padding * 2 + 1, (1,)).item()
        dy = torch.randint(0, crop_padding * 2 + 1, (1,)).item()
        images = padded[:, :, dy:dy + H, dx:dx + W]

    return images

def load_data(pipe_cfg: dict, cnn_cfg: dict):
    """Loads CSVs and sets up Train, Validation, and Test DataLoaders."""
    print("=" * 65)
    print("STEP 1: Loading & Preprocessing Fashion MNIST Data for CNN")
    print("=" * 65)

    train_csv = pipe_cfg.get("train_csv", "data/fashion-mnist_train.csv")
    test_csv = pipe_cfg.get("test_csv", "data/fashion-mnist_test.csv")
    val_split = pipe_cfg.get("val_split", 0.1)
    batch_size = cnn_cfg.get("batch_size", 128)
    random_seed = pipe_cfg.get("random_seed", 42)

    print(f"Loading training data from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    y_full = train_df['label'].values
    X_full = train_df.drop('label', axis=1).values.astype(np.float32) / 255.0
    X_full = X_full.reshape(-1, 1, 28, 28)

    print(f"Loading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values.astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28)

    # Stratified validation split to preserve class distribution
    if val_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=val_split, random_state=random_seed, stratify=y_full
        )
    else:
        X_train, y_train = X_full, y_full
        X_val, y_val = X_test, y_test

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Samples: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print(f"Batch Size: {batch_size} (Train Batches: {len(train_loader)})\n")

    return train_loader, val_loader, test_loader, X_test, y_test

def train_cnn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cnn_cfg: dict, device: str = "cpu"):
    """
    Trains the CNN with AdamW, Cosine Annealing, Label Smoothing,
    and checkpointing of the best validation model.
    """
    epochs = cnn_cfg.get("epochs", 10)
    lr = cnn_cfg.get("learning_rate", 0.001)
    weight_decay = cnn_cfg.get("weight_decay", 0.0001)
    label_smoothing = cnn_cfg.get("label_smoothing", 0.05)
    optimizer_type = cnn_cfg.get("optimizer", "adamw").lower()
    lr_scheduler_type = cnn_cfg.get("lr_scheduler", "cosine").lower()
    min_lr = cnn_cfg.get("min_lr", 0.00001)
    aug_cfg = cnn_cfg.get("augmentation", {})
    early_stop_cfg = cnn_cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 4) if early_stop_cfg.get("enabled", True) else 9999

    print("=" * 65)
    print(f"STEP 2: Training {cnn_cfg.get('model_name', 'CNN')} for {epochs} Epochs on {device.upper()}")
    print("=" * 65)
    print("Active Hyperparameters:")
    print(format_hyperparams_for_display(cnn_cfg))
    print()

    model.to(device)

    # Loss function with optional label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer selection
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    if lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif lr_scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    else:
        scheduler = None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0
    best_weights = None
    no_improve_epochs = 0

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        current_lr = optimizer.param_groups[0]['lr']

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Apply batch augmentation
            images = apply_data_augmentation(images, aug_cfg)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                val_running_loss += v_loss.item() * val_images.size(0)
                _, val_pred = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_pred == val_labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['learning_rate'].append(float(current_lr))

        epoch_time = time.time() - epoch_start
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            tag = " [BEST]"
        else:
            no_improve_epochs += 1
            tag = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - "
              f"Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.5f}{tag}", flush=True)

        if early_stop_cfg.get("enabled", True) and no_improve_epochs >= patience:
            print(f"\n[Early Stopping] No validation improvement for {patience} epochs. Stopping early at epoch {epoch}.")
            break

    total_train_time = time.time() - total_start
    print(f"\nTraining completed in {total_train_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")

    # Load best weights if configured
    if cnn_cfg.get("save_best_only", True) and best_weights is not None:
        print("[Checkpoint] Restoring best validation weights for final evaluation.")
        model.load_state_dict(best_weights)

    return history, total_train_time

def evaluate_cnn(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, history: dict, class_names: list, cnn_cfg: dict, device: str = "cpu"):
    """Evaluates the model on test data, exports charts, metrics JSON, and hyperparameter logs."""
    print("=" * 65)
    print("STEP 3: Model Evaluation (Fine-Tuned Deep CNN)")
    print("=" * 65)

    curves_path = cnn_cfg.get("training_curves_path", "outputs/baseline_model_2_training_curves.png")
    cm_path = cnn_cfg.get("confusion_matrix_path", "outputs/baseline_model_2_confusion_matrix.png")
    metrics_path = cnn_cfg.get("metrics_output_path", "outputs/baseline_model_2_metrics.json")
    hp_path = cnn_cfg.get("hyperparams_output_path", "outputs/cnn_hyperparameters.json")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    model.eval()
    model.to(device)

    # Benchmark test inference
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    start_time = time.time()
    with torch.no_grad():
        logits = model(test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)
    inference_time = time.time() - start_time

    acc = accuracy_score(y_test, y_pred)
    print(f"Inference on {len(X_test)} test samples took {inference_time:.2f} seconds ({inference_time/len(X_test)*1000:.3f} ms/sample).")
    print(f"\n>>> Fine-Tuned Deep CNN Test Accuracy: {acc * 100:.2f}% <<<\n")

    cls_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Plot 1: Training & Validation Curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    plt.title('Deep CNN - Loss per Epoch', fontsize=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Cross Entropy Loss', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history['train_acc']], label='Train Acc', marker='o', linewidth=2)
    plt.plot([a * 100 for a in history['val_acc']], label='Val Acc', marker='s', linewidth=2)
    plt.axhline(y=acc * 100, color='r', linestyle='--', label=f'Test Acc ({acc*100:.2f}%)')
    plt.title('Deep CNN - Accuracy (%) per Epoch', fontsize=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {curves_path}")

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Purples',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f"Fashion MNIST - {cnn_cfg.get('model_name', 'Deep CNN')} Confusion Matrix (Acc: {acc*100:.2f}%)", fontsize=13, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Save metrics JSON (including full hyperparameter config)
    metrics = {
        "model_name": cnn_cfg.get("model_name", "Fine-Tuned Deep CNN"),
        "architecture": cnn_cfg.get("architecture", "EnhancedCNN"),
        "trainable_parameters": total_params,
        "accuracy": float(acc),
        "inference_time_seconds": float(inference_time),
        "total_test_samples": int(len(y_test)),
        "hyperparameters": {k: (str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v) for k, v in cnn_cfg.items()},
        "history": history,
        "classification_report": cls_report
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {metrics_path}")

    # Save dedicated hyperparameters file
    save_hyperparameters(cnn_cfg, hp_path, model_section="cnn")

    return acc, metrics

def save_model(model: nn.Module, filepath: str, cnn_cfg: dict, acc: float):
    """Saves model weights along with architecture hyperparameters for zero-friction inference."""
    print("=" * 65)
    print("STEP 4: Saving Fine-Tuned CNN Model Checkpoint")
    print("=" * 65)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "hyperparameters": cnn_cfg,
        "accuracy": float(acc),
        "saved_at": time.time()
    }
    torch.save(bundle, filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved model checkpoint to: {filepath} ({file_size_mb:.2f} MB)\n")

def main():
    parser = argparse.ArgumentParser(description="Train Fine-Tuned CNN on Fashion MNIST")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    args = parser.parse_args()

    config = load_config(args.config)
    pipe_cfg = config.get("pipeline", {})
    cnn_cfg = config.get("cnn", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, X_test, y_test = load_data(pipe_cfg, cnn_cfg)
    model = build_cnn_model(cnn_cfg)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {cnn_cfg.get('architecture', 'EnhancedCNN')} with {param_count:,} trainable parameters.")

    history, train_time = train_cnn(model, train_loader, val_loader, cnn_cfg, device=device)
    acc, metrics = evaluate_cnn(model, X_test, y_test, history, pipe_cfg.get("class_names", []), cnn_cfg, device=device)
    save_model(model, cnn_cfg.get("output_model_path", "models/baseline_model_2.pt"), cnn_cfg, acc)

    print("=" * 65)
    print("FINE-TUNED DEEP CNN PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final Test Accuracy: {acc * 100:.2f}% | Training Duration: {train_time:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
