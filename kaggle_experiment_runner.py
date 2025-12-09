"""
Kaggle Experiment Runner for VUG Models
=======================================

This script runs VUG experiments optimized for Kaggle environment.
Includes memory management and time optimization for Kaggle limits.
"""

import os
import sys
import json
import time
import pandas as pd
import gc
from pathlib import Path
from datetime import datetime

# Add current directory to path
current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


class KaggleExperimentRunner:
    """Run VUG experiments with Kaggle optimizations"""
    
    def __init__(self, output_dir="/kaggle/working/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Kaggle time limit awareness (9 hours for notebooks)
        self.start_time = time.time()
        self.max_runtime = 8.5 * 3600  # 8.5 hours to be safe
        
        print(f"🎯 Kaggle Experiment Runner initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏰ Max runtime: {self.max_runtime/3600:.1f} hours")
    
    def check_time_remaining(self):
        """Check if we have enough time to run another experiment"""
        elapsed = time.time() - self.start_time
        remaining = self.max_runtime - elapsed
        
        if remaining < 1800:  # Less than 30 minutes
            print(f"⚠️  Time warning: Only {remaining/60:.1f} minutes remaining")
            return False
        
        print(f"⏱️  Time remaining: {remaining/3600:.1f} hours")
        return True
    
    def optimize_config_for_kaggle(self, config_updates=None):
        """Apply Kaggle-specific config optimizations"""
        
        default_updates = {
            # Reduce epochs for faster training on Kaggle
            'train_epochs': ['BOTH:50', 'TARGET:25'],  # Reduced from 200/100
            'eval_step': 10,  # More frequent evaluation
            'save_step': 20,  # Less frequent saving to save space
            
            # Optimize batch size for memory
            'train_batch_size': 512,  # Smaller batch size
            'eval_batch_size': 1024,
            
            # Memory optimizations
            'embedding_size': 32,  # Reduced from 64 for faster training
            'n_layers': 1,  # Reduced layers for BiTGCF
            
            # Early stopping
            'stopping_step': 5,
            'valid_metric': 'NDCG@10',
        }
        
        if config_updates:
            default_updates.update(config_updates)
            
        return default_updates
    
    def run_single_experiment(self, model_name, dataset='Amazon', config_updates=None):
        """Run a single experiment with Kaggle optimizations"""
        
        if not self.check_time_remaining():
            print("❌ Insufficient time remaining for experiment")
            return None
        
        print(f"\n{'='*60}")
        print(f"🚀 Running {model_name} on {dataset}")
        print(f"{'='*60}")
        
        try:
            # Import RecBole CDR
            from recbole_cdr.quick_start import run_recbole_cdr
            
            # Setup configs
            dataset_config = f'./recbole_cdr/properties/dataset/{dataset}.yaml'
            model_config = f'./recbole_cdr/properties/model/{model_name}.yaml'
            
            # Apply Kaggle optimizations
            kaggle_config = self.optimize_config_for_kaggle(config_updates)
            
            print(f"📋 Config files: {dataset_config}, {model_config}")
            print(f"⚙️  Kaggle optimizations: {kaggle_config}")
            
            # Memory cleanup before experiment
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Run experiment with timeout protection
            experiment_start = time.time()
            
            result = run_recbole_cdr(
                model=model_name.replace('_', '') if model_name.startswith('VUG_') else model_name,
                config_file_list=[dataset_config, model_config],
                config_dict=kaggle_config
            )
            
            experiment_time = time.time() - experiment_start
            print(f"⏱️  Experiment completed in {experiment_time/60:.1f} minutes")
            
            # Extract and save results
            results = self.extract_results(result, model_name, dataset, experiment_time)
            self.save_results(results, model_name, dataset)
            
            # Cleanup after experiment
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            error_info = {
                'model': model_name,
                'dataset': dataset,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save error info
            error_file = self.output_dir / f"error_{model_name}_{dataset}.json"
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
            
            return None
    
    def extract_results(self, result, model_name, dataset, experiment_time):
        """Extract key metrics from RecBole result"""
        
        extracted = {
            'model': model_name,
            'dataset': dataset,
            'experiment_time_minutes': round(experiment_time / 60, 2),
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Extract test results
        if 'test_result' in result and 'rec' in result['test_result']:
            test_metrics = result['test_result']['rec']
            
            # Common metrics for cross-domain recommendation
            for metric in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20', 'MRR@10', 'Recall@10']:
                if metric in test_metrics:
                    extracted['metrics'][metric] = float(test_metrics[metric])
        
        # Extract config info
        if 'config' in result:
            extracted['config'] = {
                'embedding_size': result['config'].get('embedding_size', 'N/A'),
                'learning_rate': result['config'].get('learning_rate', 'N/A'),
                'train_epochs': result['config'].get('train_epochs', 'N/A'),
            }
        
        return extracted
    
    def save_results(self, results, model_name, dataset):
        """Save results to JSON and update CSV summary"""
        
        # Save detailed JSON
        json_file = self.output_dir / f"results_{model_name}_{dataset}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {json_file}")
        
        # Update CSV summary
        csv_file = self.output_dir / "experiment_summary.csv"
        
        # Create summary row
        summary_row = {
            'Model': model_name,
            'Dataset': dataset,
            'Timestamp': results['timestamp'],
            'Runtime_Minutes': results['experiment_time_minutes'],
            **{k: v for k, v in results['metrics'].items()}
        }
        
        # Append to CSV
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])
        
        df.to_csv(csv_file, index=False)
        print(f"📊 Summary updated in {csv_file}")
    
    def run_vug_models_experiment(self, dataset='Amazon'):
        """Run all VUG combination models"""
        
        models = ['VUG_CMF', 'VUG_CLFM', 'VUG_BiTGCF']
        results = {}
        
        print(f"\n🎯 Running VUG Models Experiment on {dataset}")
        print("="*60)
        
        for model in models:
            if not self.check_time_remaining():
                print(f"⚠️  Stopping experiments due to time limit")
                break
            
            result = self.run_single_experiment(model, dataset)
            if result:
                results[model] = result
                print(f"✅ {model} completed successfully")
            else:
                print(f"❌ {model} failed")
        
        return results
    
    def run_ablation_study(self, dataset='Amazon'):
        """Run ablation study experiments"""
        
        ablation_configs = {
            'VUG_wo_constrain': {'gen_weight': 0.0},
            'VUG_wo_super': {'gen_weight': 0.0, 'enhance_weight': 0.0}, 
            'VUG_wo_user_attn': {'user_weight_attn': 0.0},
            'VUG_wo_item_attn': {'user_weight_attn': 1.0},
            'VUG': {}  # Full model
        }
        
        results = {}
        
        print(f"\n🧪 Running Ablation Study on {dataset}")
        print("="*60)
        
        for config_name, config_updates in ablation_configs.items():
            if not self.check_time_remaining():
                print(f"⚠️  Stopping ablation study due to time limit")
                break
            
            # Use base VUG model with specific config updates
            result = self.run_single_experiment('VUG', dataset, config_updates)
            if result:
                results[config_name] = result
                print(f"✅ {config_name} completed successfully")
            else:
                print(f"❌ {config_name} failed")
        
        return results


def main():
    """Main execution function for Kaggle"""
    
    print("🎯 Starting Kaggle VUG Experiments")
    print("="*60)
    
    # Initialize runner
    runner = KaggleExperimentRunner()
    
    # Choose experiment type based on available time
    if runner.check_time_remaining():
        print("\n📋 Select experiment type:")
        print("1. VUG Models (VUG_CMF, VUG_CLFM, VUG_BiTGCF)")
        print("2. Ablation Study (VUG variants)")
        print("3. Both (if time permits)")
        
        # For Kaggle, let's default to VUG models first
        experiment_type = "vug_models"  # Can be modified in notebook
        
        if experiment_type == "vug_models":
            runner.run_vug_models_experiment('Amazon')
        elif experiment_type == "ablation":
            runner.run_ablation_study('Amazon')  
        else:
            # Run both if time permits
            runner.run_vug_models_experiment('Amazon')
            if runner.check_time_remaining():
                runner.run_ablation_study('Amazon')
    
    print("\n🎉 Kaggle experiments completed!")
    print(f"📁 Check results in: {runner.output_dir}")


if __name__ == "__main__":
    main()