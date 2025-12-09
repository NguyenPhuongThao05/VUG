"""
Kaggle Setup Script for VUG Experiments
=======================================

This script sets up the environment on Kaggle for running VUG experiments.
Run this in the first cell of your Kaggle notebook.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_kaggle_environment():
    """Setup the environment for running VUG experiments on Kaggle"""
    
    print("🚀 Setting up VUG environment on Kaggle...")
    
    # 1. Install required packages
    print("📦 Installing dependencies...")
    
    # Install RecBole first (main dependency)
    subprocess.run([sys.executable, "-m", "pip", "install", "recbole>=1.1.1"], check=True)
    
    # Install other essential packages
    essential_packages = [
        "torch>=1.9.0", 
        "scipy>=1.7.0", 
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "PyYAML>=5.4.0",
        "colorlog>=6.4.0",
        "tqdm>=4.62.0"
    ]
    
    for package in essential_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")
    
    # 2. Setup directories
    print("📁 Setting up directories...")
    
    # Create working directory
    work_dir = Path("/kaggle/working/VUG")
    work_dir.mkdir(exist_ok=True)
    
    # Change to working directory
    os.chdir(work_dir)
    
    # 3. Copy dataset from input (if available)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("📂 Looking for datasets in /kaggle/input...")
        
        # Look for VUG source code dataset
        vug_datasets = list(kaggle_input.glob("*vug*")) + list(kaggle_input.glob("*VUG*"))
        if vug_datasets:
            source_path = vug_datasets[0]
            print(f"📋 Found VUG dataset: {source_path}")
            
            # Copy source code
            if (source_path / "recbole_cdr").exists():
                print("📥 Copying VUG source code...")
                shutil.copytree(source_path / "recbole_cdr", work_dir / "recbole_cdr", 
                              dirs_exist_ok=True)
                
                # Copy other important files
                for file in ["run_*.py", "*.py", "*.yaml", "*.txt"]:
                    for src_file in source_path.glob(file):
                        shutil.copy2(src_file, work_dir)
                        print(f"📄 Copied {src_file.name}")
        
        # Look for datasets
        dataset_paths = list(kaggle_input.glob("*amazon*")) + list(kaggle_input.glob("*douban*"))
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.name
            dest_path = work_dir / "recbole_cdr" / "dataset" / dataset_name
            if dataset_path.is_dir() and not dest_path.exists():
                shutil.copytree(dataset_path, dest_path)
                print(f"📊 Copied dataset: {dataset_name}")
    
    # 4. Add current directory to Python path
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))
    
    print("✅ Kaggle environment setup complete!")
    print(f"📍 Working directory: {work_dir}")
    
    return work_dir

def check_gpu_availability():
    """Check if GPU is available on Kaggle"""
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️  No GPU available - experiments will run on CPU")
        return False

def optimize_for_kaggle():
    """Apply Kaggle-specific optimizations"""
    
    print("⚙️  Applying Kaggle optimizations...")
    
    # Set environment variables for better performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA calls
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
    
    # Kaggle memory management
    import gc
    gc.collect()
    
    if check_gpu_availability():
        import torch
        torch.cuda.empty_cache()
    
    print("✅ Optimizations applied!")

if __name__ == "__main__":
    # Run setup
    work_dir = setup_kaggle_environment()
    check_gpu_availability()
    optimize_for_kaggle()
    
    print("\n" + "="*50)
    print("🎯 Ready to run VUG experiments!")
    print("="*50)