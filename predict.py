"""
Unified Inference Script for Fashion MNIST Fine-Tuned Models
Supports:
  - Model 1: Fine-Tuned Random Forest Classifier (models/baseline_model_1.joblib)
  - Model 2: Fine-Tuned Deep CNN (models/baseline_model_2.pt)
Configurable via config.yaml.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch

from config_utils import load_config
from model_architecture import load_cnn_checkpoint, build_cnn_model, BasicCNN, EnhancedCNN

def predict_with_rf(model_path: str, pixels: np.ndarray):
    """Runs inference using Fine-Tuned Random Forest."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train it first using train_baseline_1_rf.py.")
    print(f"Loading Model 1 (RF) from: {model_path}")
    model = joblib.load(model_path)
    preds = model.predict(pixels)
    probs = model.predict_proba(pixels)
    confidences = np.max(probs, axis=1) * 100
    return preds, confidences

def predict_with_cnn(model_path: str, pixels: np.ndarray, cnn_cfg: dict = None):
    """Runs inference using Fine-Tuned Deep CNN."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train it first using train_baseline_2_cnn.py.")
    print(f"Loading Model 2 (CNN) from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_cnn_checkpoint(model_path, cnn_config=cnn_cfg, device=device)

    # Reshape (N, 784) -> (N, 1, 28, 28)
    tensor_data = torch.tensor(pixels.reshape(-1, 1, 28, 28), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor_data)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1) * 100
    return preds, confidences

def main():
    parser = argparse.ArgumentParser(description="Inference for Fashion MNIST Models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, default="2", choices=["1", "2", "rf", "cnn"],
                        help="Choose model: '1' or 'rf' for Random Forest, '2' or 'cnn' for CNN")
    parser.add_argument("--data", type=str, default=None, help="Path to test CSV (defaults to config)")
    parser.add_argument("--num", type=int, default=10, help="Number of test samples to predict")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    config = load_config(args.config)
    pipe_cfg = config.get("pipeline", {})
    rf_cfg = config.get("random_forest", {})
    cnn_cfg = config.get("cnn", {})

    class_names = pipe_cfg.get("class_names", [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ])

    data_path = args.data or pipe_cfg.get("test_csv", "data/fashion-mnist_test.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found at: {data_path}")

    # Load test dataset
    df = pd.read_csv(data_path)
    np.random.seed(args.seed)
    sample_indices = np.random.choice(len(df), size=args.num, replace=False)
    sample_df = df.iloc[sample_indices]

    labels = sample_df['label'].values
    pixels = sample_df.drop('label', axis=1).values.astype(np.float32) / 255.0

    if args.model in ["1", "rf"]:
        model_name = rf_cfg.get("model_name", "Fine-Tuned Random Forest")
        model_path = rf_cfg.get("output_model_path", "models/baseline_model_1.joblib")
        output_image = os.path.join(pipe_cfg.get("output_dir", "outputs"), "sample_predictions_model_1.png")
        preds, confidences = predict_with_rf(model_path, pixels)
    else:
        model_name = cnn_cfg.get("model_name", "Fine-Tuned Deep CNN")
        model_path = cnn_cfg.get("output_model_path", "models/baseline_model_2.pt")
        output_image = os.path.join(pipe_cfg.get("output_dir", "outputs"), "sample_predictions_model_2.png")
        preds, confidences = predict_with_cnn(model_path, pixels, cnn_cfg)

    print(f"\n{model_name} - Prediction Results:")
    print("-" * 75)
    print(f"{'Index':<8} | {'True Label':<15} | {'Predicted Label':<15} | {'Confidence':<10} | {'Status'}")
    print("-" * 75)
    correct_count = 0
    for idx, true_lbl, pred_lbl, conf in zip(sample_indices, labels, preds, confidences):
        is_correct = (true_lbl == pred_lbl)
        if is_correct:
            correct_count += 1
        status = "CORRECT" if is_correct else "MISMATCH"
        print(f"{idx:<8} | {class_names[true_lbl]:<15} | {class_names[pred_lbl]:<15} | {conf:6.2f}%    | {status}")
    print("-" * 75)
    print(f"Sample Accuracy: {correct_count}/{args.num} ({correct_count/args.num*100:.1f}%)")

    # Plot sample images collage
    rows = 2
    cols = (args.num + 1) // 2
    plt.figure(figsize=(cols * 2.6, rows * 2.8))

    for i in range(args.num):
        ax = plt.subplot(rows, cols, i + 1)
        img = pixels[i].reshape(28, 28)
        ax.imshow(img, cmap='gray_r')

        is_correct = (labels[i] == preds[i])
        color = '#15803D' if is_correct else '#B91C1C'
        title = f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]} ({confidences[i]:.1f}%)"
        ax.set_title(title, fontsize=9, color=color, pad=4, weight='bold')
        ax.axis('off')

    plt.suptitle(f"{model_name} Predictions", fontsize=13, weight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=200)
    plt.close()
    print(f"\nVisual predictions saved to: {output_image}")

if __name__ == "__main__":
    main()
