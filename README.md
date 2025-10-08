# SAGS: Sparse Autoencoder-based Generation Steering

A research toolkit for controlling and analyzing Large Language Model (LLM) behavior using Sparse Autoencoders (SAEs). This project consists of two main components: **Feature Activation Steering (FAS)** for manipulating model outputs and **Semantic Feature Detection (SFD)** for identifying semantic concepts in prompts.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Components](#components)
  - [FAS: Feature Activation Steering](#fas-feature-activation-steering)
  - [SFD: Semantic Feature Detection](#sfd-semantic-feature-detection)
  - [Utils: Evaluation Tools](#utils-evaluation-tools)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Citation](#citation)

## Overview

SAGS provides a comprehensive framework for:
- **Steering**: Manipulating LLM behavior by modifying Sparse Autoencoder (SAE) activations
- **Detection**: Identifying semantic features and concepts in prompts using SAE representations
- **Evaluation**: Assessing instruction-following and refusal behavior in model outputs

The toolkit is built on top of [SAE Lens](https://github.com/jbloomAus/SAELens) and supports models from the Gemma-2 family with their corresponding SAE checkpoints.

## Project Structure

```
SAGS/
├── FAS/                          # Feature Activation Steering
│   ├── 1_extrat_sparse_activation.py
│   ├── 2_generate_steering_vector.py
│   ├── 3_steering.py
│   └── generated/                # Output directory
├── SFD/                          # Semantic Feature Detection
│   ├── 1_extract_sae_post_orginal.py
│   ├── 2_sae_same_index.py
│   ├── 3_classification_analysis.py
│   └── generated/                # Output directory
├── utils/                        # Evaluation utilities
│   ├── I-F_LLM_eval.py
│   ├── Refusal_LLM_eval.py
│   └── Refusal_Rate.py
├── data/                         # Data directory
│   ├── FAS_data/
│   ├── SFD_data/
│   └── prompt_data/
└── requirements.txt
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB GPU memory for Gemma-2-2B

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SAGS.git
cd SAGS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project requires the following main packages:
- `transformers==4.56.0`
- `sae-lens==5.10.5`
- `torch` (with CUDA support)
- `datasets==2.19.2`
- `safetensors==0.4.4`

## Components

### FAS: Feature Activation Steering

Feature Activation Steering manipulates model behavior by modifying SAE activations during generation. The example implementation demonstrates **capitalization steering** - forcing the model to generate outputs in all uppercase letters.

#### Pipeline

1. **Extract Sparse Activations** (`1_extrat_sparse_activation.py`)
   - Extracts SAE activation patterns for contrastive prompt pairs
   - Processes datasets with both target behavior and control prompts
   - Outputs: Sparse activation vectors in JSONL format

2. **Generate Steering Vectors** (`2_generate_steering_vector.py`)
   - Computes difference vectors between target and control activations
   - Applies threshold-based filtering to identify key dimensions
   - Default activation ratio: 0.6 (60% of samples must activate a dimension)
   - Outputs: Steering vectors in JSON format

3. **Apply Steering** (`3_steering.py`)
   - Modifies SAE activations during generation using steering vectors
   - Supports multiple scaling factors (e.g., 1.0, 2.5, 5.0)
   - Generates outputs with and without steering for comparison
   - Outputs: JSONL files with baseline and steered generations

#### Example Use Case: Capitalization Steering

```python
# Configure in 3_steering.py
modification_scales_dict = {
    "capital": [1, 2.5, 5],  # Scaling factors for steering strength
}
```

The capitalization steering example demonstrates how to:
- Extract activation patterns for uppercase vs. normal text
- Generate a steering vector that captures the "capitalization" feature
- Apply the vector to force the model to generate uppercase text

### SFD: Semantic Feature Detection

Semantic Feature Detection identifies and classifies semantic concepts in prompts using SAE activation patterns.

#### Pipeline

1. **Extract SAE Activations** (`1_extract_sae_post_orginal.py`)
   - Processes prompts with various semantic constraints
   - Extracts SAE activations from the last token
   - Applies threshold filtering (default: 0.01)
   - Outputs: Activation patterns in JSONL format

2. **Generate Detection Vectors** (`2_sae_same_index.py`)
   - Analyzes commonly activated dimensions across prompts of the same type
   - Computes mean activation values for frequently-occurring dimensions
   - Tests multiple occurrence ratios (1%-90%)
   - Outputs: Detection vectors for each semantic type

3. **Classification Analysis** (`3_classification_analysis.py`)
   - Trains a classifier using detection vectors as class prototypes
   - Uses sparse similarity metrics (Jaccard + harmonic mean)
   - Performs hyperparameter search on training set
   - Evaluates on test set with precision, recall, and F1 scores
   - Outputs: Classification results and performance metrics

#### Supported Semantic Types

- `capital`: All uppercase constraints
- `lowercase`: All lowercase constraints
- `no_comma`: No comma constraints
- `format_constrained`: Constrained response format
- `format_title`: Title format requirements
- `format_json`: JSON format requirements
- `quotation`: Quotation mark requirements
- `repeat`: Repetition requirements
- `two_responses`: Multiple response requirements

### Utils: Evaluation Tools

#### I-F_LLM_eval.py

Evaluates instruction-following behavior using LLM-as-a-judge:
- **Answer Relevance**: How well the response addresses the query
- **Instruction Following**: Adherence to specific constraints
- **Sentence Quality**: Overall linguistic quality
- Supports batch evaluation with multiple runs for reliability

#### Refusal_LLM_eval.py

Evaluates model refusal behavior:
- **Refusal Detection**: Degree of refusal (0.0 = full compliance, 1.0 = explicit refusal)
- **Language Quality**: Linguistic quality of the response
- Useful for safety research and jailbreak detection

#### Refusal_Rate.py

Computes aggregate statistics on refusal behavior across datasets.

## Usage

### FAS: Steering Model Behavior

```bash
cd FAS

# Step 1: Extract SAE activations from contrastive datasets
python 1_extrat_sparse_activation.py

# Step 2: Generate steering vectors
python 2_generate_steering_vector.py --ratio 0.6

# Step 3: Apply steering to generate outputs
python 3_steering.py
```

**Key Configuration** (in each script):
- `MODEL_NAME`: Model to use (default: `google/gemma-2-2b`)
- `SAE_RELEASE`: SAE checkpoint (default: `gemma-scope-2b-pt-res-canonical`)
- `SAE_ID`: Specific SAE layer (e.g., `layer_18/width_65k/canonical`)
- `ACTIVATION_RATIO`: Threshold for dimension selection (default: 0.6)

### SFD: Detecting Semantic Features

```bash
cd SFD

# Step 1: Extract SAE activations from semantic constraint datasets
python 1_extract_sae_post_orginal.py

# Step 2: Generate detection vectors
python 2_sae_same_index.py

# Step 3: Train and evaluate classifier
python 3_classification_analysis.py
```

### Evaluation

```bash
cd utils

# Evaluate instruction-following
python I-F_LLM_eval.py

# Evaluate refusal behavior
python Refusal_LLM_eval.py

# Compute refusal rates
python Refusal_Rate.py
```

**Note**: Evaluation tools require OpenAI API access. Configure `API_SECRET_KEY` and `BASE_URL` in the respective files.

## Configuration

### Model and SAE Configuration

Both FAS and SFD components share similar configuration patterns:

```python
# Model Configuration
MODEL_NAME = "gemma-2-2b"
model_name = "google/gemma-2-2b"  # Full HuggingFace model name

# SAE Configuration
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_18/width_65k/canonical"  # Format: layer_X/width_Yk/canonical
```

### Template Usage

The code supports optional prompt templates:

```python
USE_TEMPLATE = True  # Enable "Q: {prompt}\nA:" template
USE_TEMPLATE = False  # Use raw prompts
```

### Path Configuration

All scripts use project-relative paths:

```python
PROJECT_ROOT = Path(__file__).parent.parent  # Automatically finds SAGS root
data_path = PROJECT_ROOT / "data" / "FAS_data" / "I-F"
output_path = PROJECT_ROOT / "FAS" / "generated"
```

## Data Format

### FAS Input Format

Contrastive dataset in JSON:
```json
{
  "suffix_i": [
    "Write a story about adventure...",
    "Explain quantum mechanics..."
  ],
  "contr_suffix_v2": [
    "Write a story about adventure...",  // Control version
    "Explain quantum mechanics..."
  ]
}
```

### SFD Input Format

Semantic constraint dataset in JSON:
```json
{
  "if_test": [
    {
      "capital": "Write in ALL CAPS: Explain gravity",
      "lowercase": "write in lowercase: explain gravity",
      "no_comma": "Explain gravity (without commas)",
      ...
    }
  ],
  "train": [...]
}
```

### Output Format

SAE activations (JSONL):
```json
{
  "prompt_idx": 1,
  "prompt": "Example prompt text",
  "sae_activations": {
    "indices": [100, 523, 1847],
    "values": [2.34, 1.56, 0.89]
  },
  "total_dimensions": 65536,
  "non_zero_count": 3,
  "sparsity_ratio": 0.99995
}
```

Steering results (JSONL):
```json
{
  "prompt": "Example prompt",
  "baseline": "Normal generation output",
  "modification_1.0": "Weakly steered output",
  "modification_2.5": "Moderately steered output",
  "modification_5.0": "Strongly steered output"
}
```

## Example: Capitalization Steering

Here's a complete example of using FAS for capitalization steering:

1. **Prepare Data**: Create a JSON file with uppercase and normal text prompts in `data/FAS_data/I-F/contrastive_capital.json`

2. **Extract Activations**:
```python
# In 1_extrat_sparse_activation.py
CONTENT_TYPE_FILTER = "capital_only"  # Process only capital data
layers_to_process = range(18, 19)  # Process layer 18
```

3. **Generate Steering Vector**:
```python
# In 2_generate_steering_vector.py
ACTIVATION_RATIO = 0.6  # 60% occurrence threshold
```

4. **Apply Steering**:
```python
# In 3_steering.py
modification_scales_dict = {
    "capital": [1, 2.5, 5],  # Test different steering strengths
}
layers_to_process = [18]
```

5. **Evaluate**: Use `utils/I-F_LLM_eval.py` to assess how well the steered outputs follow the capitalization constraint.

## Technical Details

### Memory Management

The codebase includes several optimizations for GPU memory:
- BFloat16 precision for reduced memory usage
- Batch processing with configurable batch sizes
- Automatic cache clearing between batches
- Stop-at-layer optimization (only computes up to SAE layer)

### Reproducibility

For reproducible results:
- Temperature set to 0.0 for deterministic generation
- `do_sample=False` for greedy decoding
- Fixed random seeds (where applicable)

### Multi-Layer Support

Both FAS and SFD support processing multiple SAE layers:
```python
for layer_num in range(0, 26):  # Process all 26 layers
    current_sae_id = f"layer_{layer_num}/width_65k/canonical"
    # Process layer...
```

