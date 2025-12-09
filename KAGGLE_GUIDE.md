# VUG Experiments on Kaggle - Complete Guide

## Overview
This guide walks you through running VUG (Virtual User Generation) cross-domain recommendation experiments on Kaggle platform.

## 🎯 What You'll Get
- **VUG_CMF**: VUG + Collective Matrix Factorization results
- **VUG_CLFM**: VUG + Cross-domain Learning via Feature Mapping results  
- **VUG_BiTGCF**: VUG + BiTGCF Graph Convolution results
- **Ablation Studies**: Component analysis of VUG model
- **Visualizations**: Performance comparisons and analysis
- **Downloadable Results**: Complete results package

## 📋 Prerequisites
- Kaggle account (free)
- Basic understanding of Jupyter notebooks
- VUG source code (from this project)

## 🚀 Step-by-Step Instructions

### Step 1: Prepare VUG Dataset for Kaggle

1. **Zip your VUG project**:
   ```bash
   # In your VUG project directory
   zip -r VUG_Source_Code.zip . -x "*.git*" "*__pycache__*" "*.pyc"
   ```

2. **Upload to Kaggle**:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload `VUG_Source_Code.zip`
   - Title: "VUG Cross-Domain Recommendation"
   - Description: "VUG source code for cross-domain recommendation experiments"
   - Make public or keep private
   - Click "Create"

### Step 2: Create Kaggle Notebook

1. **Start new notebook**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code)
   - Click "New Notebook"
   - Choose "Notebook" (not Script)
   - **Enable GPU**: Settings → Accelerator → GPU T4 x2 (recommended)

2. **Add VUG dataset**:
   - Click "+ Add data" 
   - Search for your "VUG Cross-Domain Recommendation" dataset
   - Add it to your notebook

### Step 3: Run the Experiments

1. **Copy the notebook**:
   - Open `VUG_Kaggle_Experiments.ipynb` (created by this guide)
   - Copy all cells to your Kaggle notebook

2. **Execute sequentially**:
   - Run cells one by one from top to bottom
   - Monitor progress and time remaining
   - Each cell will show progress indicators

### Step 4: Monitor and Optimize

**Time Management**:
- Kaggle notebooks have 9-hour limit
- VUG models take ~30-60 minutes each
- Ablation studies take ~20-30 minutes each
- Built-in time monitoring will warn you

**Memory Management**:
- Automatic cleanup after each experiment
- GPU memory clearing when available
- Reduced batch sizes for Kaggle limits

**Optimization Applied**:
- Smaller embedding dimensions (32 vs 64)
- Reduced epochs (30/15 vs 200/100) 
- Frequent evaluation and early stopping
- Optimized batch sizes

### Step 5: Download Results

After experiments complete:

1. **Find results in output**:
   - Results saved to `/kaggle/working/results/`
   - Automatic zip package created: `VUG_Results_YYYYMMDD_HHMMSS.zip`

2. **Download**:
   - Click on the zip file in notebook output
   - Download contains:
     - Individual JSON result files
     - CSV summary table
     - Visualization plots
     - Comprehensive text report

## 📊 Expected Results Structure

### Individual Model Results
```json
{
  "model": "VUG_CMF",
  "dataset": "Amazon", 
  "runtime_minutes": 45.2,
  "metrics": {
    "HR@10": 0.1234,
    "HR@20": 0.1456, 
    "NDCG@10": 0.0789,
    "NDCG@20": 0.0912
  }
}
```

### Summary CSV
| Model | HR@10 | HR@20 | NDCG@10 | NDCG@20 | Runtime_min |
|-------|-------|-------|---------|---------|-------------|
| VUG_CMF | 0.1234 | 0.1456 | 0.0789 | 0.0912 | 45.2 |
| VUG_CLFM | 0.1189 | 0.1423 | 0.0756 | 0.0889 | 42.7 |
| VUG_BiTGCF | 0.1267 | 0.1489 | 0.0812 | 0.0934 | 58.1 |

## ⚠️ Troubleshooting

### Common Issues and Solutions

1. **"RecBole not found"**:
   - Make sure VUG dataset is added to notebook inputs
   - Check that setup cell completed successfully
   - Restart notebook and re-run setup

2. **"Out of memory"**:
   - Reduce batch_size in kaggle_config()
   - Reduce embedding_size to 16
   - Skip some experiments if needed

3. **"Time limit exceeded"**:
   - Focus on most important models first
   - Skip ablation studies if running late
   - Reduce epochs further

4. **"Dataset not found"**:
   - Check dataset path in notebook inputs
   - Ensure Amazon.yaml exists in properties/dataset/
   - Copy sample dataset if needed

### Performance Tips

1. **Use GPU**:
   - Always enable GPU in notebook settings
   - Verify GPU is detected in setup cell

2. **Optimize configurations**:
   - Reduce epochs for faster results
   - Use smaller embedding dimensions
   - Increase evaluation frequency

3. **Monitor resources**:
   - Watch memory usage in notebook
   - Check time remaining regularly
   - Save intermediate results

## 📈 Interpreting Results

### Key Metrics
- **HR@K**: Hit Rate at top-K (higher is better)
- **NDCG@K**: Normalized DCG at top-K (higher is better)  
- **Runtime**: Time taken for training and evaluation

### Model Comparison
- **VUG_CMF**: Best for shared embedding scenarios
- **VUG_CLFM**: Good balance of shared/specific features
- **VUG_BiTGCF**: Excellent with graph structure data

### Ablation Analysis
- **w/o L_constrain**: Impact of constraint loss
- **w/o L_super**: Impact of supervision loss
- **w/o α_u^user**: Impact of user-level attention
- **w/o α_u^item**: Impact of item-level attention

## 🎉 Success Checklist

- [ ] VUG dataset uploaded to Kaggle
- [ ] Kaggle notebook created with GPU enabled
- [ ] VUG source code successfully imported
- [ ] At least one VUG model completed
- [ ] Results visualized and downloaded
- [ ] Performance compared with baselines

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Kaggle notebook logs for error details
3. Try running with reduced configurations first
4. Ensure all dependencies are properly installed

## 📚 Additional Resources

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [RecBole Documentation](https://recbole.io/)
- [VUG Paper Reference](https://link-to-paper)

---

**Good luck with your experiments! 🚀**