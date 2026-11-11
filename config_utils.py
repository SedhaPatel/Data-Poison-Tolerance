"""
Configuration Utilities for Fashion MNIST ML Pipeline
Provides loading, validating, and saving functions for hyperparameters and pipeline settings.
"""

import os
import json
import yaml
from datetime import datetime
from typing import Any, Dict

DEFAULT_CONFIG_FILE = "config.yaml"

DEFAULT_CONFIG = {
    "pipeline": {
        "random_seed": 42,
        "data_dir": "data",
        "train_csv": "data/fashion-mnist_train.csv",
        "test_csv": "data/fashion-mnist_test.csv",
        "val_split": 0.1,
        "output_dir": "outputs",
        "models_dir": "models",
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    },
    "random_forest": {
        "model_name": "Fine-Tuned Random Forest",
        "output_model_path": "models/baseline_model_1.joblib",
        "metrics_output_path": "outputs/baseline_model_1_metrics.json",
        "hyperparams_output_path": "outputs/rf_hyperparameters.json",
        "confusion_matrix_path": "outputs/baseline_model_1_confusion_matrix.png",
        "feature_importance_path": "outputs/baseline_model_1_feature_importance.png",
        "n_estimators": 200,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 1,
        "tune_search": False,
        "tune_n_iter": 8,
        "tune_cv_folds": 3
    },
    "cnn": {
        "model_name": "Fine-Tuned Deep CNN",
        "output_model_path": "models/baseline_model_2.pt",
        "metrics_output_path": "outputs/baseline_model_2_metrics.json",
        "hyperparams_output_path": "outputs/cnn_hyperparameters.json",
        "training_curves_path": "outputs/baseline_model_2_training_curves.png",
        "confusion_matrix_path": "outputs/baseline_model_2_confusion_matrix.png",
        "architecture": "EnhancedCNN",
        "in_channels": 1,
        "num_classes": 10,
        "conv_channels": [32, 64, 128],
        "fc_units": 256,
        "dropout_rate": 0.35,
        "use_batch_norm": True,
        "activation": "relu",
        "augmentation": {
            "enabled": True,
            "horizontal_flip": True,
            "random_crop_padding": 2
        },
        "epochs": 10,
        "batch_size": 128,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "lr_scheduler": "cosine",
        "min_lr": 0.00001,
        "label_smoothing": 0.05,
        "early_stopping": {
            "enabled": True,
            "patience": 4,
            "min_delta": 0.001
        },
        "save_best_only": True
    }
}

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dictionary `update` into dictionary `base`."""
    result = dict(base)
    for key, val in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result

def load_config(config_path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """
    Loads YAML configuration from disk and deep-merges with defaults.
    Falls back gracefully to defaults if the file cannot be read.
    """
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config = deep_merge(config, loaded)
                    print(f"[Config] Successfully loaded configuration from: {config_path}")
                else:
                    print(f"[Config] Warning: {config_path} was empty or invalid. Using defaults.")
        except Exception as e:
            print(f"[Config] Warning: Failed to parse {config_path}: {e}. Using defaults.")
    else:
        print(f"[Config] Configuration file '{config_path}' not found. Using default configurations.")

    return config

def save_hyperparameters(hyperparams: Dict[str, Any], output_path: str, model_section: str = ""):
    """
    Saves used hyperparameters to a designated JSON or YAML file,
    recording runtime metadata such as timestamp and origin model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model_section": model_section,
        "saved_at": datetime.now().isoformat(),
        "hyperparameters": hyperparams
    }

    if output_path.endswith(".yaml") or output_path.endswith(".yml"):
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    print(f"[Config] Saved hyperparameters to: {output_path}")

def format_hyperparams_for_display(hyperparams: Dict[str, Any]) -> str:
    """Formats hyperparameter dictionary as readable key-value text."""
    lines = []
    for k, v in hyperparams.items():
        if isinstance(v, (str, int, float, bool, list)) or v is None:
            lines.append(f"  • {k}: {v}")
    return "\n".join(lines)
