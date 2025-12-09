"""
Run VUG_CLFM Model
##################

This script trains and evaluates the VUG_CLFM model, which combines:
1. VUG (Virtual User Generation) - generates virtual source embeddings for non-overlapping users
2. CLFM (Cross-domain Latent Factor Mapping) - shared and domain-specific embeddings

Usage:
    python run_vug_clfm.py --config_files 'config1.yaml config2.yaml'
    
Example:
    # Run on Douban dataset
    python run_vug_clfm.py --config_files 'recbole_cdr/properties/dataset/Douban.yaml'
    
    # Run on Amazon dataset
    python run_vug_clfm.py --config_files 'recbole_cdr/properties/dataset/Amazon.yaml'
    
    # Custom configuration
    python run_vug_clfm.py --config_files 'recbole_cdr/properties/dataset/Douban.yaml' \
                          --alpha 0.3 --gen_weight 0.15 --enhance_mode add
"""

import argparse
from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VUG_CLFM model for cross-domain recommendation')
    
    # Model name (fixed)
    parser.add_argument('--model', '-m', type=str, default='VUG_CLFM', 
                        help='Name of the model (default: VUG_CLFM)')
    
    # Configuration files
    parser.add_argument('--config_files', type=str, default=None,
                        help='Config files separated by space (e.g., "config1.yaml config2.yaml")')
    
    # CLFM parameters
    parser.add_argument('--user_embedding_size', type=int, default=None,
                        help='User embedding dimension (default: 64)')
    parser.add_argument('--source_item_embedding_size', type=int, default=None,
                        help='Source item embedding dimension (default: 64)')
    parser.add_argument('--target_item_embedding_size', type=int, default=None,
                        help='Target item embedding dimension (default: 64)')
    parser.add_argument('--share_embedding_size', type=int, default=None,
                        help='Shared embedding dimension (default: 16)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Balance between source and target loss (default: 0.3)')
    parser.add_argument('--reg_weight', type=float, default=None,
                        help='Regularization weight (default: 0.0001)')
    
    # VUG generator parameters
    parser.add_argument('--gen_weight', type=float, default=None,
                        help='Weight for generator loss (default: 0.15)')
    parser.add_argument('--user_weight_attn', type=float, default=None,
                        help='Balance between user-level and item-level attention (default: 0.6)')
    parser.add_argument('--enhance_mode', type=str, default=None,
                        choices=['add', 'replace', 'concat'],
                        help='How to use generated embeddings: add/replace/concat (default: add)')
    parser.add_argument('--enhance_weight', type=float, default=None,
                        help='Weight for virtual embedding enhancement (default: 0.7)')
    parser.add_argument('--use_virtual', type=bool, default=None,
                        help='Whether to use virtual user generation (default: True)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 200+100)')
    
    args, _ = parser.parse_known_args()

    # Parse config file list
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    # Build config_dict from command-line arguments
    config_dict = {}
    if args.user_embedding_size is not None:
        config_dict['user_embedding_size'] = args.user_embedding_size
    if args.source_item_embedding_size is not None:
        config_dict['source_item_embedding_size'] = args.source_item_embedding_size
    if args.target_item_embedding_size is not None:
        config_dict['target_item_embedding_size'] = args.target_item_embedding_size
    if args.share_embedding_size is not None:
        config_dict['share_embedding_size'] = args.share_embedding_size
    if args.alpha is not None:
        config_dict['alpha'] = args.alpha
    if args.reg_weight is not None:
        config_dict['reg_weight'] = args.reg_weight
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
        config_dict['train_epochs'] = [f"BOTH:{args.epochs}", "TARGET:100"]
    
    # Run the model
    print("=" * 80)
    print("Running VUG_CLFM Model")
    print("=" * 80)
    print(f"Config files: {config_file_list}")
    print(f"Config dict: {config_dict}")
    print("=" * 80)
    
    run_recbole_cdr(
        model=args.model,
        config_file_list=config_file_list,
        config_dict=config_dict if config_dict else None
    )