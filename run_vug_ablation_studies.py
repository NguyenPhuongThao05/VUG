"""
VUG Ablation Studies
####################

This script runs all ablation studies as shown in the VUG paper table:
1. w/o L_constrain - without constraint loss
2. w/o L_super - without supervision loss  
3. w/o α_u^user - without user-level attention weight
4. w/o α_u^item - without item-level attention weight
5. VUG - full model

Reproduces the results table showing HR@10, HR@20, NDCG@10, NDCG@20 metrics.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from recbole_cdr.quick_start import run_recbole_cdr


def _to_float(x):
    """Convert various numeric types to float"""
    if isinstance(x, (np.generic,)):
        return float(x.item())
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return float(x.item())
    if isinstance(x, (int, float)):
        return float(x)
    return None


def flatten_numeric(d, prefix=""):
    """Flatten nested dict and extract numeric values"""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            name = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            items.extend(flatten_numeric(v, name))
    else:
        val = _to_float(d)
        if val is not None:
            items.append((prefix, val))
    return items


def extract_metrics(result, metrics=['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']):
    """Extract specific metrics from result"""
    test_items = flatten_numeric(result.get("test_result", {}))
    metric_dict = {}
    
    for metric in metrics:
        for name, value in test_items:
            if metric.lower() in name.lower():
                metric_dict[metric] = value
                break
        if metric not in metric_dict:
            metric_dict[metric] = 0.0
    
    return metric_dict


def run_ablation_study(dataset_config='./recbole_cdr/properties/dataset/Amazon.yaml'):
    """Run all ablation studies"""
    
    results = {}
    
    print("="*80)
    print("VUG ABLATION STUDIES")
    print("="*80)
    
    # 1. w/o L_constrain (without constraint loss)
    print("\n1. Running w/o L_constrain...")
    config_dict_1 = {
        'gen_weight': 0.0,  # Turn off generator/constraint loss
        'use_virtual': True,
        'enhance_mode': 'add',
        'enhance_weight': 1.0,
        'user_weight_attn': 0.5,
    }
    res1 = run_recbole_cdr(
        model='VUG',
        config_file_list=[dataset_config, './recbole_cdr/properties/model/VUG.yaml'],
        config_dict=config_dict_1
    )
    results['w/o L_constrain'] = extract_metrics(res1)
    
    # 2. w/o L_super (without supervision loss)
    print("\n2. Running w/o L_super...")
    config_dict_2 = {
        'gen_weight': 0.0,  # Turn off supervision
        'use_virtual': False,  # No virtual enhancement
        'enhance_weight': 0.0,
    }
    res2 = run_recbole_cdr(
        model='VUG',
        config_file_list=[dataset_config, './recbole_cdr/properties/model/VUG.yaml'],
        config_dict=config_dict_2
    )
    results['w/o L_super'] = extract_metrics(res2)
    
    # 3. w/o α_u^user (without user-level attention)
    print("\n3. Running w/o α_u^user...")
    config_dict_3 = {
        'user_weight_attn': 0.0,  # Only item-level attention
        'gen_weight': 0.1,
        'use_virtual': True,
        'enhance_mode': 'add',
        'enhance_weight': 1.0,
    }
    res3 = run_recbole_cdr(
        model='VUG',
        config_file_list=[dataset_config, './recbole_cdr/properties/model/VUG.yaml'],
        config_dict=config_dict_3
    )
    results['w/o α_u^user'] = extract_metrics(res3)
    
    # 4. w/o α_u^item (without item-level attention)
    print("\n4. Running w/o α_u^item...")
    config_dict_4 = {
        'user_weight_attn': 1.0,  # Only user-level attention
        'gen_weight': 0.1,
        'use_virtual': True,
        'enhance_mode': 'add',
        'enhance_weight': 1.0,
    }
    res4 = run_recbole_cdr(
        model='VUG',
        config_file_list=[dataset_config, './recbole_cdr/properties/model/VUG.yaml'],
        config_dict=config_dict_4
    )
    results['w/o α_u^item'] = extract_metrics(res4)
    
    # 5. VUG (full model)
    print("\n5. Running VUG (full model)...")
    res5 = run_recbole_cdr(
        model='VUG',
        config_file_list=[dataset_config, './recbole_cdr/properties/model/VUG.yaml']
    )
    results['VUG'] = extract_metrics(res5)
    
    return results


def create_results_table(results):
    """Create results table similar to the paper"""
    
    # Create DataFrame
    methods = ['w/o L_constrain', 'w/o L_super', 'w/o α_u^user', 'w/o α_u^item', 'VUG']
    metrics = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
    
    # Accuracy metrics (larger is better)
    accuracy_data = []
    for method in methods:
        if method in results:
            row = [results[method].get(metric, 0.0) for metric in metrics]
            accuracy_data.append(row)
        else:
            accuracy_data.append([0.0, 0.0, 0.0, 0.0])
    
    accuracy_df = pd.DataFrame(accuracy_data, 
                              index=methods, 
                              columns=metrics)
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print("\nAccuracy (larger is better)")
    print("-"*60)
    print(accuracy_df.round(4))
    
    # Save to file
    accuracy_df.to_csv('vug_ablation_results.csv')
    print(f"\nResults saved to 'vug_ablation_results.csv'")
    
    # Create comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        ax = [ax1, ax2, ax3, ax4][i]
        values = accuracy_df[metric].values
        bars = ax.bar(methods, values)
        
        # Highlight best result
        best_idx = np.argmax(values)
        bars[best_idx].set_color('red')
        
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(j, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('vug_ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy_df


def main():
    """Main function to run all ablation studies"""
    
    print("Starting VUG Ablation Studies...")
    print("This will take some time as we need to train 5 different model variants.")
    
    # Run all experiments
    results = run_ablation_study()
    
    # Create and display results table
    results_df = create_results_table(results)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED!")
    print("="*80)
    print("Files generated:")
    print("- vug_ablation_results.csv: Raw results")
    print("- vug_ablation_comparison.png: Comparison chart")
    
    return results_df


if __name__ == '__main__':
    results_df = main()