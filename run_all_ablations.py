"""
Run All VUG Ablation Studies and Generate Results Table
#######################################################

This script will run all 5 ablation studies and generate a results table
similar to the one in the VUG paper.
"""

import subprocess
import json
import pandas as pd
import os


def run_all_ablations(dataset='Amazon'):
    """Run all ablation studies"""
    
    variants = ['wo_constrain', 'wo_super', 'wo_user', 'wo_item', 'full']
    variant_names = {
        'wo_constrain': 'w/o L_constrain',
        'wo_super': 'w/o L_super', 
        'wo_user': 'w/o α_u^user',
        'wo_item': 'w/o α_u^item',
        'full': 'VUG'
    }
    
    all_results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Running {variant_names[variant]}")
        print(f"{'='*60}")
        
        try:
            # Run the individual ablation script
            result = subprocess.run([
                'python', 'run_individual_ablation.py',
                '--variant', variant,
                '--dataset', dataset
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print(f"✅ {variant} completed successfully")
                
                # Load results
                result_file = f'results_{variant}_{dataset}.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        all_results[variant_names[variant]] = data['metrics']
                else:
                    print(f"⚠️  Result file not found: {result_file}")
            else:
                print(f"❌ {variant} failed with error:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {variant} timed out after 1 hour")
        except Exception as e:
            print(f"❌ Error running {variant}: {e}")
    
    return all_results


def create_results_table(results, save_csv=True):
    """Create results table similar to paper"""
    
    # Define metrics and method order
    metrics = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
    method_order = ['w/o L_constrain', 'w/o L_super', 'w/o α_u^user', 'w/o α_u^item', 'VUG']
    
    # Create table data
    table_data = []
    for method in method_order:
        if method in results:
            row = [results[method].get(metric, 0.0) for metric in metrics]
        else:
            row = [0.0, 0.0, 0.0, 0.0]
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data, index=method_order, columns=metrics)
    
    print("\n" + "="*70)
    print("VUG ABLATION STUDY RESULTS")
    print("="*70)
    print("Accuracy (larger is better)")
    print("-"*50)
    print(df.round(4).to_string())
    
    # Highlight best results
    print(f"\nBest Results:")
    for metric in metrics:
        best_method = df[metric].idxmax()
        best_value = df[metric].max()
        print(f"{metric}: {best_method} ({best_value:.4f})")
    
    if save_csv:
        df.to_csv('vug_ablation_results_final.csv')
        print(f"\n📁 Results saved to: vug_ablation_results_final.csv")
    
    return df


def main():
    print("Starting VUG Ablation Studies...")
    print("This will run 5 experiments and may take several hours.")
    
    # Run all ablation studies
    results = run_all_ablations(dataset='Amazon')
    
    if results:
        # Create results table
        results_df = create_results_table(results)
        
        print(f"\n🎉 All ablation studies completed!")
        print(f"📊 {len(results)} experiments were successful")
        
        return results_df
    else:
        print("❌ No successful experiments to report")
        return None


if __name__ == '__main__':
    main()