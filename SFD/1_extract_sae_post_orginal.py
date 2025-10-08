import gc
import json
import os
from pathlib import Path
import torch
import torch as t

# SAE-related libraries
from sae_lens import SAE, HookedSAETransformer

# Utility libraries
from tabulate import tabulate

# =========================== Configuration Parameters ===========================

MODEL_NAME = "gemma-2-2b"  # Model name
SAE_ID = "layer_5/width_65k/canonical"  # SAE ID format (with slashes)

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"  # SAE release version
model_name = "google/gemma-2-2b" # Full model name for loading



# =========================== Path Configuration ===========================
# Generate safe file name (replace slashes with underscores)
SAE_ID_SAFE = SAE_ID.replace("/", "_")

# Get project root directory (current script is in SAGS/SFD/ directory)
SAGS_ROOT = Path(__file__).parent.parent
data_path = SAGS_ROOT / "data" / "SFD_data" / "condition_ifeval_v2.json"
output_base_path = SAGS_ROOT / "SFD" / "generated" / MODEL_NAME / SAE_ID_SAFE

# Device initialization
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Set CUDA memory allocation strategy to reduce fragmentation
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"Using device: {device}")

# Disable gradient computation to save memory (not needed during inference)
t.set_grad_enabled(False)

# Ensure model is on correct device (using BF16 to reduce GPU memory usage)
base_model = HookedSAETransformer.from_pretrained(model_name, device=device, torch_dtype=torch.bfloat16)
base_model.to(device)  # Explicitly move model to specified device

# Load SAE
sae_release = SAE_RELEASE
sae_id = SAE_ID  # Use original SAE ID (with slashes)

base_model_sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,  # Use correct format with slashes
    device=device
)

print(f"Using SAE ID: {SAE_ID_SAFE}")

print(tabulate(base_model_sae.cfg.__dict__.items(), headers=["name", "value"], tablefmt="simple_outline"))


# Generate response using model
def generate_response(model, prompt, max_tokens=2):
    # Convert input text to tokens
    tokens = model.tokenizer(prompt, return_tensors="pt").to(device)
    input_tokens = tokens['input_ids']
    
    with torch.no_grad():
        # Use HookedTransformer's generate method (temperature=0 for reproducibility)
        output = model.generate(
            input=input_tokens,
            max_new_tokens=max_tokens,  # Use max_new_tokens instead of max_tokens_generated
            temperature=0.0,  # Set to 0 for reproducibility
            do_sample=False   # Use greedy decoding for reproducibility
        )
    
    # Decode generated tokens
    response = model.tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def process_ifeval_data(data_split, model, sae, device, content_type='capital', batch_size=10):
  
    all_activations = []
    all_top_k = []
    
    # Store results for all prompts in jsonl format
    prompt_results = []
    
    # Count valid samples (filter empty fields)
    valid_samples = []
    for data_item in data_split:
        if content_type in data_item and data_item[content_type].strip():  # Check field exists and non-empty
            valid_samples.append(data_item)
    
    if not valid_samples:
        print(f"  Warning: No valid samples for {content_type} type")
        return [], {}, 0, []
    
    # Process data in batches to save GPU memory
    total_samples = len(valid_samples)
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_data = valid_samples[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} "
              f"(samples {batch_start+1}-{batch_end}/{total_samples})")
        
        # Process each data item in current batch
        for i, data_item in enumerate(batch_data):
            prompt = data_item[content_type]  # Select content based on content_type

            try:
                # Get SAE activation values
                tokens = model.tokenizer(prompt, return_tensors="pt")
                input_tokens = tokens['input_ids'].to(device)
                
                with torch.no_grad():  # Ensure no gradient computation
                    _, cache = model.run_with_cache_with_saes(
                        input_tokens,
                        saes=[sae],
                        stop_at_layer=sae.cfg.hook_layer + 1
                    )
                    sae_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :].to(device)
                    model.reset_saes()
                
                # Filter out dimensions with activation values less than 0.01
                threshold = 0.01
                non_zero_mask = sae_acts >= threshold
                non_zero_acts = sae_acts[non_zero_mask]
                
                # Get all activation values above threshold
                if len(non_zero_acts) > 0:  # Ensure there are activations above threshold
                    # Get all activation values and corresponding indices above threshold
                    original_indices = torch.where(non_zero_mask)[0]
                    topk_values = non_zero_acts
                else:
                    # Handle case when no activations above threshold
                    topk_values = torch.tensor([])
                    original_indices = torch.tensor([])
                
                # Move activations to CPU to save GPU memory
                all_activations.append(sae_acts.cpu())
                all_top_k.append((topk_values.cpu(), original_indices.cpu()))
                
                # Save current prompt's results
                prompt_result = {
                    "prompt": prompt,
                    "indices": original_indices.cpu().numpy().tolist(),
                    "values": topk_values.cpu().numpy().tolist()
                }
                prompt_results.append(prompt_result)
                
                # Clean up cache and temporary variables
                del cache, sae_acts, tokens, input_tokens, non_zero_mask, non_zero_acts
                if len(topk_values) > 0:
                    del topk_values, original_indices
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"  Warning: Out of GPU memory when processing sample {batch_start + i + 1}, skipping")
                print(f"  Error message: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing sample {batch_start + i + 1}, skipping")
                print(f"  Error message: {e}")
                continue
        
        # Clean GPU memory after each batch
        clear_gpu_memory()
        
        # Periodically report progress
        if (batch_start // batch_size + 1) % 10 == 0 or batch_end == total_samples:
            print(f"  Completed {batch_end}/{total_samples} samples")
    
    # Record total number of prompts
    num_prompts = len(all_activations)
    print(f"  Successfully processed {num_prompts} samples")
    
    if num_prompts == 0:
        print(f"  Warning: No samples processed successfully")
        return [], {}, 0, []
    
    # Skip common activation neuron calculation (will be handled in separate module)
    print(f"  Skipping common activation neuron calculation (will be handled in separate module)")
    
    return all_top_k, {}, num_prompts, prompt_results

# Data collection dictionary - dynamically built
all_datasets_results = {}

# Clear GPU memory before processing data
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

# Create output directories
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(os.path.join(output_base_path, "train"), exist_ok=True)
os.makedirs(os.path.join(output_base_path, "if_test"), exist_ok=True)

# Load data
with open(data_path, 'r', encoding='utf-8') as f:
    ifeval_data = json.load(f)

# Define content types to process
content_types = ['capital', 'no_comma', 'format_constrained', 'format_title', 'format_json', 'lowercase', 'quotation', 'repeat', 'two_responses']

# Process data splits
for split_name in ['if_test', 'train']:
    data_split = ifeval_data[split_name]
    
    # Skip empty data splits
    if not data_split:
        print(f"\nSkipping {split_name} dataset: Empty data")
        continue
        
    output_dir = Path(output_base_path) / split_name
    output_dir.mkdir(parents=True, exist_ok=True) 

    print(f"\nProcessing {split_name} dataset: {len(data_split)} samples")
    
    # Process all content types
    for content_type in content_types:
        print(f"\n  Processing {content_type} content...")
        
        # Process data split (using smaller batch size to save GPU memory)
        top_k_results, common_activations_results, num_prompts, prompt_results = process_ifeval_data(
            data_split, base_model, base_model_sae, device, content_type=content_type, batch_size=50  # Reduced batch size
        )
        
        # Save files only when samples processed successfully
        if prompt_results:
            # Save basic statistics
            stats_key = f"{split_name}_{content_type}"
            all_datasets_results[stats_key] = {
                "total_prompts": num_prompts,
                "processed_successfully": len(prompt_results)
            }
            
            # Save prompt results as jsonl file
            jsonl_output = output_dir / f"{content_type}.jsonl"
            with open(jsonl_output, 'w', encoding='utf-8') as f:
                for result in prompt_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"  {content_type} activation results saved to: {jsonl_output}")
        else:
            print(f"  {content_type} has no valid samples, skipping save")
        
        # Clean GPU memory
        clear_gpu_memory()

# Print basic statistics (not saved to file)
print(f"\n=== Basic Statistics ===")
print(json.dumps(all_datasets_results, indent=2, ensure_ascii=False))