"""
Baseline Model 1: Fine-Tuned Random Forest on Fashion MNIST
Author: Antigravity
Description: Fine-tuned Random Forest Classifier for Fashion MNIST classification.
             Controlled and parameterized via config.yaml.
             Evaluates on test data, generates confusion matrix, pixel feature
             importance heatmap, persists hyperparameters, and saves model.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from config_utils import load_config, save_hyperparameters, format_hyperparams_for_display

def load_and_preprocess_data(train_path: str, test_path: str):
    """Loads CSV files and normalizes pixel intensities into [0, 1]."""
    print("=" * 65)
    print("STEP 1: Loading & Preprocessing Fashion MNIST Data (Random Forest)")
    print("=" * 65)

    start_time = time.time()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values.astype(np.float32) / 255.0

    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values.astype(np.float32) / 255.0

    elapsed = time.time() - start_time
    print(f"Loaded {len(X_train)} train and {len(X_test)} test samples in {elapsed:.2f}s.\n")
    return X_train, y_train, X_test, y_test

def tune_or_train_rf(X_train: np.ndarray, y_train: np.ndarray, rf_cfg: dict):
    """Trains Random Forest using configured hyperparameters with optional search."""
    print("=" * 65)
    print("STEP 2: Training Fine-Tuned Random Forest Model")
    print("=" * 65)

    tune_search = rf_cfg.get("tune_search", False)

    params = {
        "n_estimators": rf_cfg.get("n_estimators", 200),
        "criterion": rf_cfg.get("criterion", "gini"),
        "max_depth": rf_cfg.get("max_depth", None),
        "min_samples_split": rf_cfg.get("min_samples_split", 2),
        "min_samples_leaf": rf_cfg.get("min_samples_leaf", 1),
        "max_features": rf_cfg.get("max_features", "sqrt"),
        "bootstrap": rf_cfg.get("bootstrap", True),
        "random_state": rf_cfg.get("random_state", 42),
        "n_jobs": rf_cfg.get("n_jobs", -1),
        "verbose": rf_cfg.get("verbose", 1)
    }

    if tune_search:
        print("[Hyperparameter Search] Running RandomizedSearchCV over parameter grid...")
        param_dist = {
            "n_estimators": [150, 200, 250],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.2]
        }
        # Run search on a subset for efficiency
        search_subset_size = min(15000, len(X_train))
        sub_indices = np.random.choice(len(X_train), size=search_subset_size, replace=False)
        base_estimator = RandomForestClassifier(random_state=rf_cfg.get("random_state", 42), n_jobs=-1)
        search = RandomizedSearchCV(
            base_estimator,
            param_distributions=param_dist,
            n_iter=rf_cfg.get("tune_n_iter", 8),
            cv=rf_cfg.get("tune_cv_folds", 3),
            scoring="accuracy",
            random_state=rf_cfg.get("random_state", 42),
            verbose=1
        )
        search.fit(X_train[sub_indices], y_train[sub_indices])
        print(f"[Hyperparameter Search] Best Score: {search.best_score_ * 100:.2f}%")
        print(f"[Hyperparameter Search] Best Params Found: {search.best_params_}\n")
        params.update(search.best_params_)

    print("Active Hyperparameters for Training:")
    print(format_hyperparams_for_display(params))
    print()

    rf = RandomForestClassifier(**params)

    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds.\n")
    return rf, params, train_time

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, class_names: list, rf_cfg: dict, active_params: dict):
    """Evaluates the Random Forest model, computes metrics, and generates visualizations."""
    print("=" * 65)
    print("STEP 3: Model Evaluation (Fine-Tuned Random Forest)")
    print("=" * 65)

    metrics_path = rf_cfg.get("metrics_output_path", "outputs/baseline_model_1_metrics.json")
    cm_path = rf_cfg.get("confusion_matrix_path", "outputs/baseline_model_1_confusion_matrix.png")
    fi_path = rf_cfg.get("feature_importance_path", "outputs/baseline_model_1_feature_importance.png")
    hp_path = rf_cfg.get("hyperparams_output_path", "outputs/rf_hyperparameters.json")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time

    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> Fine-Tuned Random Forest Test Accuracy: {acc * 100:.2f}% <<<\n")

    cls_report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f"Fashion MNIST - {rf_cfg.get('model_name', 'Random Forest')} Confusion Matrix (Acc: {acc*100:.2f}%)", fontsize=13, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # Pixel Feature Importance Heatmap
    importances = model.feature_importances_
    importance_map = importances.reshape(28, 28)
    plt.figure(figsize=(7, 6))
    plt.imshow(importance_map, cmap='magma', interpolation='nearest')
    plt.title('Pixel Feature Importance Heatmap (Random Forest)', fontsize=13, pad=15)
    plt.colorbar(label='Gini Importance')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved feature importance heatmap to: {fi_path}")

    # Save metrics JSON (including hyperparameter configuration)
    metrics = {
        "model_name": rf_cfg.get("model_name", "Fine-Tuned Random Forest"),
        "accuracy": float(acc),
        "inference_time_seconds": float(pred_time),
        "total_test_samples": int(len(y_test)),
        "hyperparameters": {k: (str(v) if not isinstance(v, (int, float, bool, list, type(None))) else v) for k, v in active_params.items()},
        "classification_report": cls_report_dict
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {metrics_path}")

    # Save dedicated hyperparameters file
    save_hyperparameters(active_params, hp_path, model_section="random_forest")

    return acc, metrics

def save_model(model, filepath: str):
    """Saves the trained model artifact to disk."""
    print("=" * 65)
    print("STEP 4: Saving Fine-Tuned Random Forest Model")
    print("=" * 65)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath, compress=3)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Model saved to: {filepath} ({file_size_mb:.2f} MB)\n")

def main():
    parser = argparse.ArgumentParser(description="Train Fine-Tuned Random Forest on Fashion MNIST")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    args = parser.parse_args()

    config = load_config(args.config)
    pipe_cfg = config.get("pipeline", {})
    rf_cfg = config.get("random_forest", {})

    train_path = pipe_cfg.get("train_csv", "data/fashion-mnist_train.csv")
    test_path = pipe_cfg.get("test_csv", "data/fashion-mnist_test.csv")
    class_names = pipe_cfg.get("class_names", [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ])
    model_output_path = rf_cfg.get("output_model_path", "models/baseline_model_1.joblib")

    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)
    model, active_params, train_time = tune_or_train_rf(X_train, y_train, rf_cfg)
    acc, metrics = evaluate_model(model, X_test, y_test, class_names, rf_cfg, active_params)
    save_model(model, model_output_path)

    print("=" * 65)
    print("FINE-TUNED RANDOM FOREST PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final Test Accuracy: {acc * 100:.2f}% | Training Duration: {train_time:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
