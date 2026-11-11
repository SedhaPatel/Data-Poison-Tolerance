"""
Generate Comparative Visualizations and Metrics between Fine-Tuned Models
Compares Random Forest and Deep CNN performance, F1-scores per class, and inference latency.
Controlled via config.yaml.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from config_utils import load_config

def generate_comparison(config_path: str = "config.yaml"):
    config = load_config(config_path)
    pipe_cfg = config.get("pipeline", {})
    rf_cfg = config.get("random_forest", {})
    cnn_cfg = config.get("cnn", {})

    m1_path = rf_cfg.get("metrics_output_path", "outputs/baseline_model_1_metrics.json")
    m2_path = cnn_cfg.get("metrics_output_path", "outputs/baseline_model_2_metrics.json")

    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"Model 1 metrics file not found: {m1_path}. Please train Model 1 first.")
    if not os.path.exists(m2_path):
        raise FileNotFoundError(f"Model 2 metrics file not found: {m2_path}. Please train Model 2 first.")

    with open(m1_path, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(m2_path, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    classes = pipe_cfg.get("class_names", [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ])

    f1_m1 = [m1['classification_report'][c]['f1-score'] * 100 for c in classes]
    f1_m2 = [m2['classification_report'][c]['f1-score'] * 100 for c in classes]

    m1_name = m1.get("model_name", "Random Forest")
    m2_name = m2.get("model_name", "Deep CNN")

    x = np.arange(len(classes))
    width = 0.38

    # Create figure with 2 subplots: (1) Per-Class F1 comparison, (2) Summary KPIs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.8, 1]})

    acc1 = m1["accuracy"] * 100
    acc2 = m2["accuracy"] * 100

    rects1 = ax1.bar(x - width/2, f1_m1, width, label=f'{m1_name} ({acc1:.2f}%)', color='#2563EB', alpha=0.9, edgecolor='black', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, f1_m2, width, label=f'{m2_name} ({acc2:.2f}%)', color='#7C3AED', alpha=0.9, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('F1-Score (%)', fontsize=12)
    ax1.set_title('Per-Class F1-Score Performance Comparison', fontsize=13, pad=15, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=30, ha='right', fontsize=10)
    ax1.set_ylim(50, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(fontsize=11, loc='lower right')

    for bar in rects1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    for bar in rects2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)

    # Subplot 2: Overall Accuracy & Latency Summary Cards
    ax2.axis('off')
    summary_text = (
        f"MODEL PERFORMANCE SUMMARY\n"
        f"═══════════════════════════════\n\n"
        f"● {m1_name}\n"
        f"  ├─ Accuracy: {acc1:.2f}%\n"
        f"  ├─ Inference: {m1['inference_time_seconds']:.3f}s\n"
        f"  └─ Trees: {m1.get('hyperparameters', {}).get('n_estimators', 200)}\n\n"
        f"● {m2_name}\n"
        f"  ├─ Accuracy: {acc2:.2f}%\n"
        f"  ├─ Inference: {m2['inference_time_seconds']:.3f}s\n"
        f"  └─ Params: {m2.get('trainable_parameters', 'N/A'):,}\n\n"
        f"═══════════════════════════════\n"
        f"Accuracy Delta: +{(acc2 - acc1):+.2f}% ({'CNN Leads' if acc2 > acc1 else 'RF Leads'})\n"
        f"Controlled via: {config_path}"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5))

    plt.tight_layout()
    comp_plot_path = os.path.join(pipe_cfg.get("output_dir", "outputs"), "models_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to: {comp_plot_path}")

    summary = {
        'model_1': {
            'name': m1_name,
            'accuracy': m1['accuracy'],
            'model_path': rf_cfg.get("output_model_path", "models/baseline_model_1.joblib"),
            'inference_time_seconds': m1['inference_time_seconds'],
            'hyperparameters': m1.get('hyperparameters', {})
        },
        'model_2': {
            'name': m2_name,
            'accuracy': m2['accuracy'],
            'model_path': cnn_cfg.get("output_model_path", "models/baseline_model_2.pt"),
            'inference_time_seconds': m2['inference_time_seconds'],
            'trainable_parameters': m2.get('trainable_parameters'),
            'hyperparameters': m2.get('hyperparameters', {})
        },
        'accuracy_delta': float(acc2 - acc1),
        'superior_model': m2_name if acc2 > acc1 else m1_name
    }

    comp_summary_path = os.path.join(pipe_cfg.get("output_dir", "outputs"), "comparison_summary.json")
    with open(comp_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved comparison summary to: {comp_summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Fine-Tuned Fashion MNIST Models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    generate_comparison(args.config)

if __name__ == '__main__':
    main()
