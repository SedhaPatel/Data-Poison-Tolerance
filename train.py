"""
Master Training Pipeline for Fashion MNIST Models
Coordinates training of Fine-Tuned Random Forest, Fine-Tuned Deep CNN,
and comparative evaluation. Controlled and parameterized via config.yaml.
"""

import os
import sys
import argparse
import subprocess
from config_utils import load_config

def get_python_executable():
    """Returns the python executable that has the necessary ML packages installed."""
    # Check current sys.executable
    try:
        res = subprocess.run([sys.executable, "-c", "import torch; print(1)"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return sys.executable
    except Exception:
        pass

    # Try py launcher with 3.13
    for candidate in [["py", "-3.13"], ["C:\\Python313\\python.exe"]]:
        try:
            cmd = candidate + ["-c", "import torch; print(1)"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                return candidate
        except Exception:
            pass

    return sys.executable

def run_script(script_name: str, config_path: str):
    """Executes a training or evaluation sub-script with config parameter."""
    print(f"\n{'='*70}")
    print(f">>> EXECUTING: {script_name} --config {config_path}")
    print(f"{'='*70}\n")

    py_exec = get_python_executable()
    if isinstance(py_exec, list):
        cmd = py_exec + [script_name, "--config", config_path]
    else:
        cmd = [py_exec, script_name, "--config", config_path]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Execution failed for {script_name} (exit code: {result.returncode})")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline for Fashion MNIST Fine-Tuned Models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["1", "2", "all", "rf", "cnn"],
        help="Model to train: '1'/'rf' for Random Forest, '2'/'cnn' for Deep CNN, 'all' for both"
    )
    args = parser.parse_args()

    # Validate config file
    config = load_config(args.config)

    print("\n" + "=" * 70)
    print("FASHION MNIST MACHINE LEARNING TRAINING PIPELINE")
    print(f"Config File: {args.config}")
    print(f"Target Selection: {args.model.upper()}")
    print("=" * 70)

    if args.model in ["1", "rf"]:
        run_script("train_baseline_1_rf.py", args.config)
    elif args.model in ["2", "cnn"]:
        run_script("train_baseline_2_cnn.py", args.config)
    else:
        run_script("train_baseline_1_rf.py", args.config)
        run_script("train_baseline_2_cnn.py", args.config)
        run_script("compare_models.py", args.config)
        print("\n" + "=" * 70)
        print("ALL MODELS TRAINED, EVALUATED, AND COMPARED SUCCESSFULLY!")
        print("Hyperparameters, metrics, and plots saved to 'outputs/' directory.")
        print("=" * 70)

if __name__ == "__main__":
    main()
