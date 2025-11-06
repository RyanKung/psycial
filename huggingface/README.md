# Hugging Face Upload Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face

```bash
# Method 1: Use CLI login
huggingface-cli login

# Method 2: Set environment variable
export HF_TOKEN=your_huggingface_token
```

### 3. Train Model

```bash
cd /home/ryan/dev/farcaster/psycial

# Activate environment
conda activate psycial
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Train best model
./target/release/psycial hybrid train --multi-task

# Verify model files generated
ls -lh models/mlp_weights_multitask.pt
ls -lh models/tfidf_vectorizer_multitask.json
```

### 4. Upload to Hugging Face

```bash
cd huggingface

# Upload to lderRyan/polyjuice
python upload_to_hf.py --repo-id ElderRyan/psycial

# Specify token if needed
python upload_to_hf.py \
  --repo-id lderRyan/polyjuice \
  --token hf_xxxxxxxxxxxx
```

## Uploaded Files

The upload script will automatically upload:

1. **mlp_weights_multitask.pt** (~27MB) - Neural network weights
2. **tfidf_vectorizer_multitask.json** (~213KB) - TF-IDF vocabulary and weights
3. **README.md** - Model card (auto-generated from MODEL_CARD.md)
4. **config.json** - Model metadata

## Download from Hugging Face

```python
from huggingface_hub import hf_hub_download

# Download model files
mlp_weights = hf_hub_download(
    repo_id="ElderRyan/psycial",
    filename="mlp_weights_multitask.pt"
)

tfidf_vectorizer = hf_hub_download(
    repo_id="ElderRyan/psycial",
    filename="tfidf_vectorizer_multitask.json"
)
```

## Model Information

- **Framework**: Rust + tch-rs (PyTorch bindings)
- **BERT Model**: sentence-transformers/all-MiniLM-L6-v2
- **Accuracy**: 49.80% (overall)
- **Training Data**: 6940 samples
- **Test Data**: 1735 samples

## Notes

1. **Model Size**: Total ~27.2 MB
2. **Dependencies**: Requires PyTorch/libtorch for inference
3. **Language**: English text only
4. **Use Case**: Research and entertainment only, not for hiring or clinical decisions

## License

GPLv3 License
