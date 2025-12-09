"""
Run Individual Ablation Studies
################################

Run each ablation study individually to reproduce the VUG paper results.
Usage: python run_individual_ablation.py --variant <variant_name>

Variants:
- wo_constrain: w/o L_constrain
- wo_super: w/o L_super  
- wo_user: w/o α_u^user
- wo_item: w/o α_u^item
- full: VUG (full model)
"""

import argparse
import json
from recbole_cdr.quick_start import run_recbole_cdr


def run_variant(variant, dataset='Amazon'):
    """Run specific ablation variant"""
    
    dataset_config = f'./recbole_cdr/properties/dataset/{dataset}.yaml'
    
    variant_configs = {
        'wo_constrain': './recbole_cdr/properties/model/VUG_wo_constrain.yaml',
        'wo_super': './recbole_cdr/properties/model/VUG_wo_super.yaml', 
        'wo_user': './recbole_cdr/properties/model/VUG_wo_user_attn.yaml',
        'wo_item': './recbole_cdr/properties/model/VUG_wo_item_attn.yaml',
        'full': './recbole_cdr/properties/model/VUG.yaml'
    }
    
    if variant not in variant_configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_configs.keys())}")
    
    print(f"Running VUG variant: {variant}")
    print(f"Config: {variant_configs[variant]}")
    print("="*60)
    
    # Run experiment
    result = run_recbole_cdr(
        model='VUG',
        config_file_list=[dataset_config, variant_configs[variant]]
    )
    
    # Extract key metrics
    test_result = result.get('test_result', {})
    
    metrics = {}
    if 'rec' in test_result:
        rec_metrics = test_result['rec']
        # Extract common metrics
        for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
            if metric in rec_metrics:
                metrics[metric] = float(rec_metrics[metric])
    
    print(f"\nResults for {variant}:")
    print("-"*30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    with open(f'results_{variant}_{dataset}.json', 'w') as f:
        json.dump({
            'variant': variant,
            'dataset': dataset,
            'metrics': metrics,
            'full_result': result
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: results_{variant}_{dataset}.json")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run VUG ablation studies')
    parser.add_argument('--variant', type=str, required=True,
                        choices=['wo_constrain', 'wo_super', 'wo_user', 'wo_item', 'full'],
                        help='Ablation variant to run')
    parser.add_argument('--dataset', type=str, default='Amazon',
                        choices=['Amazon', 'Douban'],
                        help='Dataset to use (default: Amazon)')
    
    args = parser.parse_args()
    
    try:
        metrics = run_variant(args.variant, args.dataset)
        print(f"\n✅ Successfully completed {args.variant} ablation study!")
        return metrics
    except Exception as e:
        print(f"\n❌ Error running {args.variant}: {e}")
        return None


if __name__ == '__main__':
    main()