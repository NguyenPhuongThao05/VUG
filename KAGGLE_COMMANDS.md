# VUG Kaggle Commands - Chi tiết các câu lệnh thực thi

## 📋 Tổng quan câu lệnh
Đây là danh sách đầy đủ các câu lệnh để chạy thực nghiệm VUG trên Kaggle platform.

---

## 🔧 1. Chuẩn bị môi trường local (trước khi upload)

### Tạo zip file cho Kaggle Dataset:
```bash
# Di chuyển đến thư mục VUG
cd /Users/socnhi/Documents/VUG

# Tạo zip file chứa toàn bộ source code
zip -r VUG_Source_Code.zip . \
    -x "*.git*" "*__pycache__*" "*.pyc" "*.DS_Store" \
    -x "*/.vscode/*" "*/logs/*" "*/checkpoints/*"

# Kiểm tra nội dung zip
unzip -l VUG_Source_Code.zip
```

### Tạo requirements file tối ưu:
```bash
# Tạo requirements cho Kaggle
cat > kaggle_requirements.txt << EOF
recbole>=1.1.1
torch>=1.9.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
PyYAML>=5.4.0
colorlog>=6.4.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
EOF
```

---

## 🌐 2. Kaggle Dataset Upload Commands

### Sử dụng Kaggle CLI (nếu đã setup):
```bash
# Cài đặt Kaggle CLI
pip install kaggle

# Setup API credentials (cần kaggle.json từ Account settings)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Tạo dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "VUG Cross-Domain Recommendation Source Code",
  "id": "your-username/vug-cross-domain-recommendation",
  "description": "VUG (Virtual User Generation) source code for cross-domain recommendation experiments including VUG_CMF, VUG_CLFM, VUG_BiTGCF models and ablation studies.",
  "isPrivate": false,
  "licenses": [{"name": "MIT"}],
  "keywords": ["recommendation-system", "cross-domain", "virtual-user-generation", "machine-learning"],
  "collaborators": [],
  "data": []
}
EOF

# Upload dataset
kaggle datasets create -p . --dir-mode zip
```

### Hoặc upload thủ công:
```
1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload VUG_Source_Code.zip
4. Title: "VUG Cross-Domain Recommendation Source Code"
5. Description: "VUG source code for experiments"
6. Tags: machine-learning, recommendation-system
7. Click "Create"
```

---

## 💻 3. Kaggle Notebook Setup Commands

### Cell 1 - Install Dependencies:
```python
# Install essential packages
import subprocess
import sys

packages = [
    "recbole>=1.1.1",
    "torch>=1.9.0", 
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "PyYAML>=5.4.0",
    "colorlog>=6.4.0",
    "tqdm>=4.62.0"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"✅ Installed {package}")
```

### Cell 2 - Setup Environment:
```python
# Setup Kaggle workspace
import os
import shutil
from pathlib import Path

# Create working directory
work_dir = Path("/kaggle/working/VUG")
work_dir.mkdir(exist_ok=True)
os.chdir(work_dir)

# Copy VUG source from input
kaggle_input = Path("/kaggle/input")
vug_dataset = next(kaggle_input.glob("*vug*"), None)

if vug_dataset:
    # Copy all files
    for item in vug_dataset.iterdir():
        if item.is_file():
            shutil.copy2(item, work_dir)
        elif item.is_dir():
            shutil.copytree(item, work_dir / item.name, dirs_exist_ok=True)
    print("✅ VUG source code copied")
else:
    print("❌ VUG dataset not found in inputs")

# Add to Python path
import sys
if str(work_dir) not in sys.path:
    sys.path.insert(0, str(work_dir))
```

### Cell 3 - Check GPU and Optimize:
```python
# Check GPU availability
import torch
import gc

def setup_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.cuda.empty_cache()
        return True
    else:
        print("⚠️ No GPU available")
        return False

has_gpu = setup_gpu()

# Environment optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
gc.collect()
```

---

## 🚀 4. Experiment Execution Commands

### Cell 4 - Import và Verify Models:
```python
# Import VUG models
try:
    from recbole_cdr.quick_start import run_recbole_cdr
    from recbole_cdr.model.cross_domain_recommender.vug_cmf import VUG_CMF
    from recbole_cdr.model.cross_domain_recommender.vug_clfm import VUG_CLFM
    from recbole_cdr.model.cross_domain_recommender.vug_bitgcf import VUG_BiTGCF
    from recbole_cdr.model.cross_domain_recommender.vug import VUG
    print("✅ All VUG models imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

### Cell 5 - Run VUG_CMF:
```python
# Run VUG_CMF experiment
import time
import json

def run_vug_cmf():
    print("🚀 Running VUG_CMF...")
    start_time = time.time()
    
    try:
        result = run_recbole_cdr(
            model='VUGCMF',
            config_file_list=[
                './recbole_cdr/properties/dataset/Amazon.yaml',
                './recbole_cdr/properties/model/VUG_CMF.yaml'
            ],
            config_dict={
                'train_epochs': ['BOTH:30', 'TARGET:15'],
                'embedding_size': 32,
                'train_batch_size': 512,
                'eval_batch_size': 1024,
                'eval_step': 5,
                'stopping_step': 5
            }
        )
        
        runtime = time.time() - start_time
        
        # Extract results
        metrics = {}
        if 'test_result' in result and 'rec' in result['test_result']:
            test_metrics = result['test_result']['rec']
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
                if metric in test_metrics:
                    metrics[metric] = float(test_metrics[metric])
        
        # Save results
        vug_cmf_result = {
            'model': 'VUG_CMF',
            'runtime_minutes': runtime/60,
            'metrics': metrics
        }
        
        with open('/kaggle/working/vug_cmf_results.json', 'w') as f:
            json.dump(vug_cmf_result, f, indent=2)
        
        print(f"✅ VUG_CMF completed in {runtime/60:.1f} minutes")
        print("📊 Results:", metrics)
        
        return vug_cmf_result
        
    except Exception as e:
        print(f"❌ VUG_CMF failed: {e}")
        return None

# Execute
vug_cmf_result = run_vug_cmf()
```

### Cell 6 - Run VUG_CLFM:
```python
# Run VUG_CLFM experiment
def run_vug_clfm():
    print("🚀 Running VUG_CLFM...")
    start_time = time.time()
    
    try:
        result = run_recbole_cdr(
            model='VUGCLFM',
            config_file_list=[
                './recbole_cdr/properties/dataset/Amazon.yaml',
                './recbole_cdr/properties/model/VUG_CLFM.yaml'
            ],
            config_dict={
                'train_epochs': ['BOTH:30', 'TARGET:15'],
                'user_embedding_size': 32,
                'share_embedding_size': 16,
                'source_item_embedding_size': 32,
                'train_batch_size': 512,
                'eval_batch_size': 1024,
                'eval_step': 5,
                'stopping_step': 5
            }
        )
        
        runtime = time.time() - start_time
        
        # Extract results
        metrics = {}
        if 'test_result' in result and 'rec' in result['test_result']:
            test_metrics = result['test_result']['rec']
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
                if metric in test_metrics:
                    metrics[metric] = float(test_metrics[metric])
        
        # Save results
        vug_clfm_result = {
            'model': 'VUG_CLFM',
            'runtime_minutes': runtime/60,
            'metrics': metrics
        }
        
        with open('/kaggle/working/vug_clfm_results.json', 'w') as f:
            json.dump(vug_clfm_result, f, indent=2)
        
        print(f"✅ VUG_CLFM completed in {runtime/60:.1f} minutes")
        print("📊 Results:", metrics)
        
        return vug_clfm_result
        
    except Exception as e:
        print(f"❌ VUG_CLFM failed: {e}")
        return None

# Execute
vug_clfm_result = run_vug_clfm()

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Cell 7 - Run VUG_BiTGCF:
```python
# Run VUG_BiTGCF experiment
def run_vug_bitgcf():
    print("🚀 Running VUG_BiTGCF...")
    start_time = time.time()
    
    try:
        result = run_recbole_cdr(
            model='VUGBiTGCF',
            config_file_list=[
                './recbole_cdr/properties/dataset/Amazon.yaml',
                './recbole_cdr/properties/model/VUG_BiTGCF.yaml'
            ],
            config_dict={
                'train_epochs': ['BOTH:25', 'TARGET:12'],  # Reduced for BiTGCF
                'embedding_size': 32,
                'n_layers': 1,  # Reduced layers for speed
                'train_batch_size': 256,  # Smaller batch for BiTGCF
                'eval_batch_size': 512,
                'eval_step': 5,
                'stopping_step': 5,
                'drop_rate': 0.2
            }
        )
        
        runtime = time.time() - start_time
        
        # Extract results
        metrics = {}
        if 'test_result' in result and 'rec' in result['test_result']:
            test_metrics = result['test_result']['rec']
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
                if metric in test_metrics:
                    metrics[metric] = float(test_metrics[metric])
        
        # Save results
        vug_bitgcf_result = {
            'model': 'VUG_BiTGCF',
            'runtime_minutes': runtime/60,
            'metrics': metrics
        }
        
        with open('/kaggle/working/vug_bitgcf_results.json', 'w') as f:
            json.dump(vug_bitgcf_result, f, indent=2)
        
        print(f"✅ VUG_BiTGCF completed in {runtime/60:.1f} minutes")
        print("📊 Results:", metrics)
        
        return vug_bitgcf_result
        
    except Exception as e:
        print(f"❌ VUG_BiTGCF failed: {e}")
        return None

# Execute
vug_bitgcf_result = run_vug_bitgcf()

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## 🧪 5. Ablation Study Commands

### Cell 8 - Run Ablation Studies:
```python
# Run ablation studies
ablation_configs = {
    'VUG_wo_constrain': {'gen_weight': 0.0},
    'VUG_wo_super': {'gen_weight': 0.0, 'enhance_weight': 0.0},
    'VUG_wo_user_attn': {'user_weight_attn': 0.0},
    'VUG_wo_item_attn': {'user_weight_attn': 1.0},
    'VUG_full': {}
}

ablation_results = {}

def run_ablation_study():
    print("🧪 Running Ablation Studies...")
    
    for variant_name, config_updates in ablation_configs.items():
        print(f"\n🔬 Running {variant_name}...")
        start_time = time.time()
        
        try:
            # Base config
            base_config = {
                'train_epochs': ['BOTH:20', 'TARGET:10'],  # Further reduced
                'embedding_size': 32,
                'n_layers': 1,
                'train_batch_size': 512,
                'eval_batch_size': 1024,
                'eval_step': 5,
                'stopping_step': 3
            }
            
            # Merge with variant config
            full_config = {**base_config, **config_updates}
            
            result = run_recbole_cdr(
                model='VUG',
                config_file_list=[
                    './recbole_cdr/properties/dataset/Amazon.yaml',
                    './recbole_cdr/properties/model/VUG.yaml'
                ],
                config_dict=full_config
            )
            
            runtime = time.time() - start_time
            
            # Extract results
            metrics = {}
            if 'test_result' in result and 'rec' in result['test_result']:
                test_metrics = result['test_result']['rec']
                for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
                    if metric in test_metrics:
                        metrics[metric] = float(test_metrics[metric])
            
            # Store results
            ablation_results[variant_name] = {
                'variant': variant_name,
                'runtime_minutes': runtime/60,
                'metrics': metrics
            }
            
            # Save individual result
            with open(f'/kaggle/working/{variant_name}_results.json', 'w') as f:
                json.dump(ablation_results[variant_name], f, indent=2)
            
            print(f"✅ {variant_name} completed in {runtime/60:.1f} minutes")
            print("📊 Results:", metrics)
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ {variant_name} failed: {e}")
            continue

# Execute ablation studies
run_ablation_study()
```

---

## 📊 6. Results Analysis Commands

### Cell 9 - Create Results Summary:
```python
# Create comprehensive results summary
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_results_summary():
    """Create and display results summary"""
    
    all_results = []
    
    # Collect VUG model results
    for result in [vug_cmf_result, vug_clfm_result, vug_bitgcf_result]:
        if result:
            row = {'Model': result['model'], 'Type': 'VUG_Combination'}
            row.update(result['metrics'])
            row['Runtime_min'] = result['runtime_minutes']
            all_results.append(row)
    
    # Collect ablation results
    for variant_name, result in ablation_results.items():
        row = {'Model': variant_name, 'Type': 'Ablation'}
        row.update(result['metrics'])
        row['Runtime_min'] = result['runtime_minutes']
        all_results.append(row)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Display table
        print("📊 Experimental Results Summary")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Save CSV
        df.to_csv('/kaggle/working/experiment_summary.csv', index=False)
        print(f"\n💾 Results saved to /kaggle/working/experiment_summary.csv")
        
        return df
    else:
        print("⚠️ No results available")
        return None

# Create summary
results_df = create_results_summary()
```

### Cell 10 - Create Visualizations:
```python
# Create result visualizations
def create_visualizations(results_df):
    """Create performance comparison plots"""
    
    if results_df is None or len(results_df) < 2:
        print("📊 Insufficient data for visualization")
        return
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VUG Experiments Results Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        if metric in results_df.columns:
            # Create bar plot
            sns.barplot(data=results_df, x='Model', y=metric, hue='Type', ax=ax)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'{metric}\nNo Data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved to /kaggle/working/results_comparison.png")

# Create visualizations
create_visualizations(results_df)
```

---

## 📦 7. Results Export Commands

### Cell 11 - Package Results:
```python
# Create downloadable results package
import zipfile
from datetime import datetime

def create_results_package():
    """Package all results for download"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"/kaggle/working/VUG_Results_{timestamp}.zip"
    
    # Create comprehensive report
    report_content = f"""VUG Cross-Domain Recommendation Experiments
{'='*50}

Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platform: Kaggle
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

VUG Combination Models:
{'-'*30}
"""
    
    # Add VUG model results
    for result in [vug_cmf_result, vug_clfm_result, vug_bitgcf_result]:
        if result:
            report_content += f"\\n{result['model']}:\\n"
            report_content += f"  Runtime: {result['runtime_minutes']:.1f} minutes\\n"
            for metric, value in result['metrics'].items():
                report_content += f"  {metric}: {value:.4f}\\n"
    
    # Add ablation results
    if ablation_results:
        report_content += f"\\n\\nAblation Study Results:\\n{'-'*30}\\n"
        for variant_name, result in ablation_results.items():
            report_content += f"\\n{variant_name}:\\n"
            report_content += f"  Runtime: {result['runtime_minutes']:.1f} minutes\\n"
            for metric, value in result['metrics'].items():
                report_content += f"  {metric}: {value:.4f}\\n"
    
    # Save report
    with open('/kaggle/working/experiment_report.txt', 'w') as f:
        f.write(report_content)
    
    # Create zip package
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add all JSON results
        for file_path in Path('/kaggle/working').glob('*_results.json'):
            zipf.write(file_path, file_path.name)
        
        # Add CSV summary
        if Path('/kaggle/working/experiment_summary.csv').exists():
            zipf.write('/kaggle/working/experiment_summary.csv', 'experiment_summary.csv')
        
        # Add visualization
        if Path('/kaggle/working/results_comparison.png').exists():
            zipf.write('/kaggle/working/results_comparison.png', 'results_comparison.png')
        
        # Add report
        zipf.write('/kaggle/working/experiment_report.txt', 'experiment_report.txt')
    
    return zip_path

# Create package
if any([vug_cmf_result, vug_clfm_result, vug_bitgcf_result]) or ablation_results:
    zip_path = create_results_package()
    
    print("📦 Results Package Created!")
    print("="*40)
    print(f"📁 Download: {zip_path}")
    print(f"📊 Contains: JSON results, CSV summary, visualizations, report")
    print("💡 Click on the zip file name above to download")
else:
    print("⚠️ No results to package")
```

---

## ⚡ 8. Quick Commands cho Testing

### Test nhanh single model:
```python
# Quick test - chỉ chạy 1 model với config tối thiểu
def quick_test():
    result = run_recbole_cdr(
        model='VUGCMF',
        config_file_list=[
            './recbole_cdr/properties/dataset/Amazon.yaml',
            './recbole_cdr/properties/model/VUG_CMF.yaml'
        ],
        config_dict={
            'train_epochs': ['BOTH:5', 'TARGET:3'],  # Rất ngắn để test
            'embedding_size': 16,
            'train_batch_size': 256,
            'eval_step': 1,
            'stopping_step': 2
        }
    )
    print("✅ Quick test completed")
    return result

# Uncomment để test
# quick_test()
```

### Check memory usage:
```python
# Monitor memory usage
import psutil
import GPUtil

def check_system_resources():
    # CPU and RAM
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"💾 CPU Usage: {cpu_percent}%")
    print(f"💾 RAM Usage: {memory.percent}% ({memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f} GB)")
    
    # GPU if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB allocated, {gpu_cached:.1f} GB cached")

check_system_resources()
```

---

## 🎯 Summary: Quy trình hoàn chỉnh

### Thứ tự thực hiện:
```
1. Zip source code → Upload to Kaggle Dataset
2. Tạo Kaggle Notebook với GPU
3. Cell 1-3: Setup environment  
4. Cell 4: Import và verify models
5. Cell 5-7: Run VUG models (CMF, CLFM, BiTGCF)
6. Cell 8: Run ablation studies (optional)
7. Cell 9-10: Create analysis và visualization
8. Cell 11: Package và download results
```

### Estimated Runtime:
- Setup: 5-10 phút
- VUG_CMF: 30-45 phút  
- VUG_CLFM: 30-45 phút
- VUG_BiTGCF: 45-60 phút
- Ablation studies: 100-150 phút (5 variants × 20-30 phút)
- **Total**: 3-5 giờ (trong giới hạn 9 giờ của Kaggle)

Tất cả commands đã được tối ưu cho Kaggle environment! 🚀