"""
Run VUG_BiTGCF Model
####################

This script trains and evaluates the VUG_BiTGCF model, which combines:
1. VUG (Virtual User Generation) - generates virtual source embeddings for non-overlapping users
2. BiTGCF (Bi-partite Graph Convolutional Framework) - graph convolution with bidirectional transfer

Usage:
    python run_vug_bitgcf.py --config_files 'config1.yaml config2.yaml'
    
Example:
    # Run on Douban dataset
    python run_vug_bitgcf.py --config_files 'recbole_cdr/properties/dataset/Douban.yaml'
    
    # Run on Amazon dataset
    python run_vug_bitgcf.py --config_files 'recbole_cdr/properties/dataset/Amazon.yaml'
    
    # Custom configuration
    python run_vug_bitgcf.py --config_files 'recbole_cdr/properties/dataset/Douban.yaml' \
                            --n_layers 3 --gen_weight 0.15 --enhance_mode add
"""

import argparse
from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VUG_BiTGCF model for cross-domain recommendation')
    
    # Model name (fixed)
    parser.add_argument('--model', '-m', type=str, default='VUG_BiTGCF', 
                        help='Name of the model (default: VUG_BiTGCF)')
    
    # Configuration files
    parser.add_argument('--config_files', type=str, default=None,
                        help='Config files separated by space (e.g., "config1.yaml config2.yaml")')
    
    # BiTGCF parameters
    parser.add_argument('--embedding_size', type=int, default=None,
                        help='Embedding dimension (default: 64)')
    parser.add_argument('--n_layers', type=int, default=None,
                        help='Number of GCN layers (default: 2)')
    parser.add_argument('--reg_weight', type=float, default=None,
                        help='Regularization weight (default: 0.001)')
    parser.add_argument('--lambda_source', type=float, default=None,
                        help='Source domain transfer weight (default: 0.8)')
    parser.add_argument('--lambda_target', type=float, default=None,
                        help='Target domain transfer weight (default: 0.8)')
    parser.add_argument('--connect_way', type=str, default=None,
                        choices=['concat', 'mean'],
                        help='Layer aggregation method: concat/mean (default: concat)')
    parser.add_argument('--drop_rate', type=float, default=None,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--is_transfer', type=bool, default=None,
                        help='Enable transfer learning (default: True)')
    
    # VUG generator parameters
    parser.add_argument('--gen_weight', type=float, default=None,
                        help='Weight for generator loss (default: 0.1)')
    parser.add_argument('--user_weight_attn', type=float, default=None,
                        help='Balance between user-level and item-level attention (default: 0.5)')
    parser.add_argument('--enhance_mode', type=str, default=None,
                        choices=['add', 'replace', 'concat'],
                        help='How to use generated embeddings: add/replace/concat (default: add)')
    parser.add_argument('--enhance_weight', type=float, default=None,
                        help='Weight for virtual embedding enhancement (default: 0.5)')
    parser.add_argument('--use_virtual', type=bool, default=None,
                        help='Whether to use virtual user generation (default: True)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 300)')
    
    args, _ = parser.parse_known_args()

    # Parse config file list
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    # Build config_dict from command-line arguments
    config_dict = {}
    if args.embedding_size is not None:
        config_dict['embedding_size'] = args.embedding_size
    if args.n_layers is not None:
        config_dict['n_layers'] = args.n_layers
    if args.reg_weight is not None:
        config_dict['reg_weight'] = args.reg_weight
    if args.lambda_source is not None:
        config_dict['lambda_source'] = args.lambda_source
    if args.lambda_target is not None:
        config_dict['lambda_target'] = args.lambda_target
    if args.connect_way is not None:
        config_dict['connect_way'] = args.connect_way
    if args.drop_rate is not None:
        config_dict['drop_rate'] = args.drop_rate
    if args.is_transfer is not None:
        config_dict['is_transfer'] = args.is_transfer
    if args.gen_weight is not None:
        config_dict['gen_weight'] = args.gen_weight
    if args.user_weight_attn is not None:
        config_dict['user_weight_attn'] = args.user_weight_attn
    if args.enhance_mode is not None:
        config_dict['enhance_mode'] = args.enhance_mode
    if args.enhance_weight is not None:
        config_dict['enhance_weight'] = args.enhance_weight
    if args.use_virtual is not None:
        config_dict['use_virtual'] = args.use_virtual
    if args.learning_rate is not None:
        config_dict['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config_dict['train_epochs'] = [f"BOTH:{args.epochs}"]
    
    # Run the model
    print("=" * 80)
    print("Running VUG_BiTGCF Model")
    print("=" * 80)
    print(f"Config files: {config_file_list}")
    print(f"Config dict: {config_dict}")
    print("=" * 80)
    
    run_recbole_cdr(
        model=args.model,
        config_file_list=config_file_list,
        config_dict=config_dict if config_dict else None
    )