# VUG Ablation Studies - Kaggle Commands

## 🧪 Chỉ chạy Ablation Studies cho VUG

Đây là hướng dẫn chi tiết để chạy **chỉ Ablation Studies** của VUG trên Kaggle, bỏ qua các model kết hợp.

---

## 📋 Overview: VUG Ablation Components

### 5 Variants cần test:
1. **VUG_wo_constrain** - Loại bỏ constraint loss (L_constrain)
2. **VUG_wo_super** - Loại bỏ supervision loss (L_super) 
3. **VUG_wo_user_attn** - Loại bỏ user-level attention (α_u^user)
4. **VUG_wo_item_attn** - Loại bỏ item-level attention (α_u^item)
5. **VUG_full** - Model đầy đủ (baseline)

### Expected Runtime: ~2-3 giờ total (20-30 phút mỗi variant)

---

## 🚀 Kaggle Setup Commands

### Cell 1 - Quick Setup:
```python
# Fast setup for ablation studies only
import subprocess
import sys
import os
import shutil
from pathlib import Path
import time
import json
import gc

# Install essential packages only
essential_packages = ["recbole>=1.1.1", "torch>=1.9.0", "pandas>=1.3.0", "matplotlib>=3.4.0"]
for package in essential_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Setup workspace
work_dir = Path("/kaggle/working/VUG_Ablation")
work_dir.mkdir(exist_ok=True)
os.chdir(work_dir)

# Copy VUG source
kaggle_input = Path("/kaggle/input")
vug_dataset = next(kaggle_input.glob("*vug*"), None)
if vug_dataset:
    for item in vug_dataset.iterdir():
        if item.is_file():
            shutil.copy2(item, work_dir)
        elif item.is_dir():
            shutil.copytree(item, work_dir / item.name, dirs_exist_ok=True)
    print("✅ VUG source copied for ablation studies")

# Add to path
sys.path.insert(0, str(work_dir))

# Check GPU
import torch
has_gpu = torch.cuda.is_available()
print(f"🎮 GPU Available: {has_gpu}")
if has_gpu:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2 - Verify VUG Import:
```python
# Import và verify VUG model (base model for ablation)
try:
    from recbole_cdr.quick_start import run_recbole_cdr
    from recbole_cdr.model.cross_domain_recommender.vug import VUG
    print("✅ VUG model imported successfully for ablation studies")
    
    # Check config files exist
    dataset_config = Path('./recbole_cdr/properties/dataset/Amazon.yaml')
    model_config = Path('./recbole_cdr/properties/model/VUG.yaml')
    
    print(f"📋 Dataset config: {'✅' if dataset_config.exists() else '❌'}")
    print(f"📋 VUG model config: {'✅' if model_config.exists() else '❌'}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("💡 Make sure VUG dataset is properly added to notebook inputs")
```

---

## 🧪 Ablation Study Execution

### Cell 3 - Define Ablation Configurations:
```python
# Define all ablation study variants
ablation_configs = {
    'VUG_wo_constrain': {
        'description': 'Remove constraint loss (L_constrain)',
        'config': {'gen_weight': 0.0}  # Disable generator constraint loss
    },
    
    'VUG_wo_super': {
        'description': 'Remove supervision loss (L_super)',
        'config': {'gen_weight': 0.0, 'enhance_weight': 0.0}  # Disable both generator and enhancement
    },
    
    'VUG_wo_user_attn': {
        'description': 'Remove user-level attention (α_u^user = 0)',
        'config': {'user_weight_attn': 0.0}  # Only item-level attention
    },
    
    'VUG_wo_item_attn': {
        'description': 'Remove item-level attention (α_u^item = 0)', 
        'config': {'user_weight_attn': 1.0}  # Only user-level attention
    },
    
    'VUG_full': {
        'description': 'Full VUG model (baseline)',
        'config': {}  # No modifications
    }
}

# Kaggle-optimized base configuration
kaggle_base_config = {
    'train_epochs': ['BOTH:25', 'TARGET:12'],  # Reduced for Kaggle
    'embedding_size': 32,  # Smaller embedding
    'n_layers': 1,  # Reduced layers 
    'train_batch_size': 512,
    'eval_batch_size': 1024,
    'eval_step': 5,  # More frequent evaluation
    'stopping_step': 5,  # Early stopping
    'learning_rate': 0.001,
    'reg_weight': 1e-3,
    'lambda_source': 0.8,
    'lambda_target': 0.8,
    'drop_rate': 0.2,
    'connect_way': 'concat',
    'is_transfer': True,
    'enhance_mode': 'asrealsource'
}

print("📋 Ablation Study Configurations:")
for name, info in ablation_configs.items():
    print(f"  {name}: {info['description']}")
    
print(f"\n⚙️ Kaggle base config: {kaggle_base_config}")
```

### Cell 4 - Ablation Study Runner:
```python
# Ablation study execution function
def run_single_ablation(variant_name, variant_config, description):
    """Run a single ablation study variant"""
    
    print(f"\n{'='*60}")
    print(f"🔬 Running {variant_name}")
    print(f"📝 Description: {description}")
    print(f"⚙️ Config modifications: {variant_config}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Merge base config with variant-specific config
        full_config = {**kaggle_base_config, **variant_config}
        
        print("🚀 Starting training...")
        
        # Run VUG with modified configuration
        result = run_recbole_cdr(
            model='VUG',
            config_file_list=[
                './recbole_cdr/properties/dataset/Amazon.yaml',
                './recbole_cdr/properties/model/VUG.yaml'
            ],
            config_dict=full_config
        )
        
        runtime = time.time() - start_time
        
        # Extract metrics
        metrics = {}
        if 'test_result' in result and 'rec' in result['test_result']:
            test_metrics = result['test_result']['rec']
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20', 'MRR@10', 'Recall@10']:
                if metric in test_metrics:
                    metrics[metric] = float(test_metrics[metric])
        
        # Prepare result summary
        result_summary = {
            'variant': variant_name,
            'description': description,
            'config_modifications': variant_config,
            'runtime_minutes': round(runtime/60, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'status': 'success'
        }
        
        # Save individual result
        result_file = work_dir / f"{variant_name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result_summary, f, indent=2)
        
        # Display results
        print(f"\n✅ {variant_name} completed successfully!")
        print(f"⏱️ Runtime: {runtime/60:.1f} minutes")
        print(f"📊 Key Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result_summary
        
    except Exception as e:
        error_summary = {
            'variant': variant_name,
            'description': description,
            'runtime_minutes': round((time.time() - start_time)/60, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'status': 'failed'
        }
        
        print(f"❌ {variant_name} failed: {e}")
        
        # Save error info
        error_file = work_dir / f"{variant_name}_error.json"
        with open(error_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        return error_summary

print("🛠️ Ablation study runner function defined")
```

### Cell 5 - Execute All Ablation Studies:
```python
# Run all ablation studies sequentially
ablation_results = {}
total_start_time = time.time()

print("🧪 Starting VUG Ablation Studies")
print("="*70)

for variant_name, variant_info in ablation_configs.items():
    
    # Check remaining time (Kaggle 9-hour limit)
    elapsed_hours = (time.time() - total_start_time) / 3600
    remaining_hours = 8.5 - elapsed_hours  # 8.5 hours to be safe
    
    if remaining_hours < 0.5:  # Less than 30 minutes
        print(f"⏰ Time limit approaching. Stopping at {variant_name}")
        break
    
    print(f"\n⏱️ Time remaining: {remaining_hours:.1f} hours")
    
    # Run the variant
    result = run_single_ablation(
        variant_name, 
        variant_info['config'], 
        variant_info['description']
    )
    
    # Store result
    ablation_results[variant_name] = result
    
    print(f"✅ Progress: {len(ablation_results)}/{len(ablation_configs)} variants completed")

total_runtime = time.time() - total_start_time
print(f"\n🎉 Ablation studies completed!")
print(f"⏱️ Total runtime: {total_runtime/3600:.2f} hours")
print(f"📊 Successful variants: {sum(1 for r in ablation_results.values() if r['status'] == 'success')}")
```

---

## 📊 Results Analysis & Comparison

### Cell 6 - Create Ablation Results Table:
```python
# Create comprehensive ablation results table
import pandas as pd

def create_ablation_table():
    """Create detailed ablation study results table"""
    
    table_data = []
    
    for variant_name, result in ablation_results.items():
        if result['status'] == 'success':
            row = {
                'Variant': variant_name,
                'Description': result['description'],
                'Runtime (min)': result['runtime_minutes'],
                **result['metrics']
            }
            table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Reorder columns for better readability
        metric_cols = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
        base_cols = ['Variant', 'Description', 'Runtime (min)']
        other_cols = [col for col in df.columns if col not in base_cols + metric_cols]
        
        final_cols = base_cols + metric_cols + other_cols
        df = df[[col for col in final_cols if col in df.columns]]
        
        return df
    else:
        return None

# Create and display results table
ablation_df = create_ablation_table()

if ablation_df is not None:
    print("📊 VUG Ablation Study Results")
    print("="*80)
    
    # Display with proper formatting
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print(ablation_df.to_string(index=False))
    
    # Save to CSV
    csv_file = work_dir / "ablation_results.csv"
    ablation_df.to_csv(csv_file, index=False)
    print(f"\n💾 Results saved to: {csv_file}")
    
else:
    print("⚠️ No successful ablation results to display")
```

### Cell 7 - Create Ablation Analysis:
```python
# Detailed ablation analysis and component importance
def analyze_component_importance():
    """Analyze the importance of each VUG component"""
    
    if ablation_df is None or len(ablation_df) == 0:
        print("⚠️ No data available for analysis")
        return
    
    print("🔍 VUG Component Importance Analysis")
    print("="*50)
    
    # Find baseline (full model)
    full_model_row = ablation_df[ablation_df['Variant'] == 'VUG_full']
    if len(full_model_row) == 0:
        print("⚠️ VUG_full baseline not found")
        return
    
    baseline_metrics = full_model_row.iloc[0]
    
    # Calculate performance drops for each ablation
    analysis_results = {}
    
    for _, row in ablation_df.iterrows():
        if row['Variant'] != 'VUG_full':
            variant_analysis = {
                'variant': row['Variant'],
                'description': row['Description'],
                'performance_drops': {}
            }
            
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
                if metric in ablation_df.columns:
                    baseline_val = baseline_metrics[metric]
                    current_val = row[metric]
                    drop_percent = ((baseline_val - current_val) / baseline_val) * 100
                    variant_analysis['performance_drops'][metric] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'drop_percent': drop_percent
                    }
            
            analysis_results[row['Variant']] = variant_analysis
    
    # Display analysis
    print(f"📋 Baseline (VUG_full) performance:")
    for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
        if metric in baseline_metrics:
            print(f"   {metric}: {baseline_metrics[metric]:.4f}")
    
    print(f"\n📉 Performance drops when removing components:")
    print("-" * 60)
    
    for variant_name, analysis in analysis_results.items():
        print(f"\n🔬 {variant_name}:")
        print(f"   {analysis['description']}")
        
        for metric, drop_info in analysis['performance_drops'].items():
            print(f"   {metric}: {drop_info['current']:.4f} "
                  f"({drop_info['drop_percent']:+.2f}%)")
    
    # Find most important components
    print(f"\n🏆 Component Importance Ranking:")
    print("-" * 40)
    
    avg_drops = {}
    for variant_name, analysis in analysis_results.items():
        drops = [info['drop_percent'] for info in analysis['performance_drops'].values()]
        avg_drops[variant_name] = sum(drops) / len(drops) if drops else 0
    
    # Sort by average drop (higher drop = more important component)
    sorted_components = sorted(avg_drops.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (variant, avg_drop) in enumerate(sorted_components, 1):
        component_name = variant.replace('VUG_wo_', '').replace('_', ' ').title()
        print(f"   {rank}. {component_name}: {avg_drop:.2f}% average drop")

# Run analysis
analyze_component_importance()
```

### Cell 8 - Create Ablation Visualization:
```python
# Create visualization for ablation study results
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_ablation_visualization():
    """Create comprehensive ablation study visualizations"""
    
    if ablation_df is None or len(ablation_df) == 0:
        print("⚠️ No data available for visualization")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VUG Ablation Study Results - Component Impact Analysis', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        if metric in ablation_df.columns:
            # Create bar plot
            variant_labels = [v.replace('VUG_wo_', '').replace('VUG_', '').replace('_', '\n') 
                             for v in ablation_df['Variant']]
            
            bars = ax.bar(variant_labels, ablation_df[metric], 
                         color=['red' if 'wo_' in v else 'green' for v in ablation_df['Variant']],
                         alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, ablation_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight baseline
            full_idx = list(ablation_df['Variant']).index('VUG_full') if 'VUG_full' in list(ablation_df['Variant']) else -1
            if full_idx >= 0:
                bars[full_idx].set_color('darkgreen')
                bars[full_idx].set_alpha(1.0)
        
        else:
            ax.text(0.5, 0.5, f'{metric}\nNo Data Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title(f'{metric} - No Data')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = work_dir / 'ablation_study_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Visualization saved to: {plot_file}")
    
    # Create performance drop heatmap
    if len(ablation_df) > 1:
        create_performance_drop_heatmap()

def create_performance_drop_heatmap():
    """Create heatmap showing performance drops"""
    
    # Find baseline
    baseline_row = ablation_df[ablation_df['Variant'] == 'VUG_full']
    if len(baseline_row) == 0:
        return
    
    baseline = baseline_row.iloc[0]
    
    # Calculate drops
    drop_data = []
    variants = []
    
    for _, row in ablation_df.iterrows():
        if row['Variant'] != 'VUG_full':
            drops = []
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
                if metric in ablation_df.columns:
                    drop = ((baseline[metric] - row[metric]) / baseline[metric]) * 100
                    drops.append(drop)
                else:
                    drops.append(0)
            
            drop_data.append(drops)
            variants.append(row['Variant'].replace('VUG_wo_', '').replace('_', ' ').title())
    
    if drop_data:
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        drop_matrix = np.array(drop_data)
        im = ax.imshow(drop_matrix, cmap='Reds', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'])))
        ax.set_xticklabels(['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'])
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Drop (%)', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(variants)):
            for j in range(len(['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'])):
                text = ax.text(j, i, f'{drop_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Performance Drop Heatmap\n(Compared to VUG Full Model)', 
                    fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        heatmap_file = work_dir / 'performance_drop_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🔥 Heatmap saved to: {heatmap_file}")

# Create visualizations
create_ablation_visualization()
```

---

## 📦 Export Ablation Results

### Cell 9 - Package Ablation Results:
```python
# Create comprehensive ablation results package
import zipfile
from datetime import datetime

def package_ablation_results():
    """Package all ablation study results for download"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"VUG_Ablation_Results_{timestamp}.zip"
    zip_path = work_dir / zip_filename
    
    # Create detailed report
    report_content = f"""VUG Ablation Study Results
{'='*50}

Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platform: Kaggle
Total Runtime: {(time.time() - total_start_time)/3600:.2f} hours
GPU Used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

Ablation Study Overview:
{'-'*30}
This study analyzes the impact of individual VUG components by systematically
removing each component and measuring the performance drop.

Components Tested:
"""
    
    # Add detailed results for each variant
    for variant_name, result in ablation_results.items():
        if result['status'] == 'success':
            report_content += f"""
{variant_name}:
  Description: {result['description']}
  Runtime: {result['runtime_minutes']:.1f} minutes
  Configuration: {result['config_modifications']}
  Results:
"""
            for metric, value in result['metrics'].items():
                report_content += f"    {metric}: {value:.4f}\n"
    
    # Add analysis summary
    if ablation_df is not None and len(ablation_df) > 1:
        report_content += f"""

Component Importance Summary:
{'-'*30}
Based on performance drops when components are removed:
"""
        
        # Calculate and rank importance
        baseline_row = ablation_df[ablation_df['Variant'] == 'VUG_full']
        if len(baseline_row) > 0:
            baseline = baseline_row.iloc[0]
            
            avg_drops = {}
            for _, row in ablation_df.iterrows():
                if row['Variant'] != 'VUG_full':
                    drops = []
                    for metric in ['HR@10', 'NDCG@10']:
                        if metric in ablation_df.columns:
                            drop = ((baseline[metric] - row[metric]) / baseline[metric]) * 100
                            drops.append(drop)
                    if drops:
                        avg_drops[row['Variant']] = sum(drops) / len(drops)
            
            sorted_importance = sorted(avg_drops.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (variant, avg_drop) in enumerate(sorted_importance, 1):
                component = variant.replace('VUG_wo_', '').replace('_', ' ').title()
                report_content += f"\n{rank}. {component}: {avg_drop:.2f}% average performance drop"
    
    # Save report
    report_file = work_dir / 'ablation_study_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    # Create zip package
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        
        # Add individual result files
        for result_file in work_dir.glob('*_result.json'):
            zipf.write(result_file, result_file.name)
        
        # Add CSV summary
        if (work_dir / 'ablation_results.csv').exists():
            zipf.write(work_dir / 'ablation_results.csv', 'ablation_results.csv')
        
        # Add visualizations
        for plot_file in work_dir.glob('*.png'):
            zipf.write(plot_file, plot_file.name)
        
        # Add comprehensive report
        zipf.write(report_file, 'ablation_study_report.txt')
        
        # Add configuration summary
        config_summary = {
            'total_variants': len(ablation_configs),
            'successful_runs': len([r for r in ablation_results.values() if r['status'] == 'success']),
            'failed_runs': len([r for r in ablation_results.values() if r['status'] == 'failed']),
            'total_runtime_hours': (time.time() - total_start_time) / 3600,
            'kaggle_config': kaggle_base_config,
            'ablation_configs': ablation_configs
        }
        
        config_file = work_dir / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_summary, f, indent=2)
        zipf.write(config_file, 'experiment_config.json')
    
    return zip_path

# Create results package
if ablation_results:
    zip_path = package_ablation_results()
    
    print("📦 VUG Ablation Results Package Created!")
    print("="*50)
    print(f"📁 Package: {zip_path.name}")
    print(f"📊 Contains:")
    print(f"   - Individual JSON results for each variant")
    print(f"   - CSV summary table")  
    print(f"   - Performance comparison plots")
    print(f"   - Performance drop heatmap")
    print(f"   - Comprehensive text report")
    print(f"   - Experiment configuration details")
    
    print(f"\n📈 Summary:")
    successful = len([r for r in ablation_results.values() if r['status'] == 'success'])
    total = len(ablation_results)
    print(f"   ✅ Successful variants: {successful}/{total}")
    print(f"   ⏱️ Total runtime: {(time.time() - total_start_time)/3600:.2f} hours")
    print(f"   💾 Package size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    print(f"\n💡 To download: Click on '{zip_path.name}' in the output files")

else:
    print("⚠️ No ablation results to package")
```

---

## 🎯 Quick Ablation Commands Summary

### For rapid execution, copy these cells sequentially:

1. **Setup** (Cell 1): Install + Import + GPU check
2. **Verify** (Cell 2): Check VUG import và configs  
3. **Configure** (Cell 3): Define ablation variants
4. **Execute** (Cell 4-5): Run all 5 ablation studies
5. **Analyze** (Cell 6-8): Create tables + analysis + plots
6. **Export** (Cell 9): Package results for download

### Expected timeline:
- Setup: 5 phút
- Each ablation: 20-30 phút  
- Total: 2-3 giờ cho 5 variants
- Analysis + Export: 10 phút

### Key metrics tracked:
- **HR@10, HR@20**: Hit Rate at top-K
- **NDCG@10, NDCG@20**: Normalized DCG at top-K
- **Component importance**: Ranked by performance drop

Chỉ copy và chạy các cells theo thứ tự để có kết quả ablation study hoàn chỉnh! 🚀