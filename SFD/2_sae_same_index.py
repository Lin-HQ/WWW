import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

# =========================== Configuration Parameters ===========================
# Test dataset version configuration - modify this to switch between different test datasets
TEST_VERSION = "train"  # Usually "train", extract base vectors from train set

MODEL_NAME = "gemma-2-2b"  # Model name
SAE_ID = "layer_5/width_65k/canonical"   # SAE ID format (with slashes)

# Generate safe file name (replace slashes with underscores)
SAE_ID_SAFE = SAE_ID.replace("/", "_") 

# =========================== Extraction Ratios ===========================

def analyze_common_indices(jsonl_file_path, output_dir, content_type):
    """
    Analyze commonly occurring SAE activation dimensions in JSONL file
    
    Args:
        jsonl_file_path: Path to JSONL file
        output_dir: Output directory
        content_type: Content type (e.g., capital, comma, etc.)
    """
    # Read all JSON data
    all_data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                all_data.append(data)
    
    total_samples = len(all_data)
    print(f"Total samples read: {total_samples}")
    
    # Count occurrences of each index and corresponding values
    index_occurrences = defaultdict(list)  # {index: [value1, value2, ...]}
    index_counts = defaultdict(int)        # {index: count}
    
    # Traverse all data and count occurrence of each index
    for data in all_data:
        indices = data['indices']
        values = data['values']
        
        # Ensure indices and values have same length
        assert len(indices) == len(values), f"Length mismatch between indices and values: {len(indices)} vs {len(values)}"
        
        # Record each index and its corresponding value
        for idx, val in zip(indices, values):
            index_occurrences[idx].append(val)
            index_counts[idx] += 1
    
    print(f"Total unique activation dimensions found: {len(index_counts)}")
    
    # Define ratio thresholds to analyze
    # "mean" represents mean calculated based on all samples (unactivated samples have value 0)
    # 0 means no ratio threshold, calculate mean as long as dimension is activated (based on activated samples)
    ratios = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    results = {}
    
    # Analyze for each ratio
    for ratio in ratios:
        if ratio == "mean":
            # Special handling for ratio="mean": calculate mean based on all samples (unactivated samples = 0)
            min_occurrences = 1  # Count if appeared at least once
            print(f"\nAnalyzing ratio={ratio}, calculating mean based on all samples (unactivated samples value=0)")
        elif ratio == 0:
            # Special handling for ratio=0: no ratio threshold, all activated dimensions participate in calculation
            min_occurrences = 1  # Count if appeared at least once
            print(f"\nAnalyzing ratio={ratio}, no threshold, all activated dimensions participate in calculation")
        else:
            min_occurrences = int(total_samples * ratio)
            print(f"\nAnalyzing ratio={ratio}, minimum occurrences: {min_occurrences}")
        
        # Find indices that meet the condition
        qualifying_indices = []
        qualifying_values = []
        qualifying_counts = []
        
        for idx, count in index_counts.items():
            if count >= min_occurrences:
                if ratio == "mean":
                    # For "mean" parameter: calculate mean based on all samples (unactivated samples = 0)
                    total_value = sum(index_occurrences[idx])  # Sum of all activation values
                    mean_value = total_value / total_samples   # Divide by total samples
                else:
                    # For other parameters: calculate average activation value for this index (based on actual activation count, not total samples)
                    mean_value = np.mean(index_occurrences[idx])
                
                qualifying_indices.append(idx)
                qualifying_values.append(mean_value)
                qualifying_counts.append(count)
        
        # Sort by average activation value in descending order
        sorted_data = sorted(zip(qualifying_indices, qualifying_values, qualifying_counts), 
                           key=lambda x: x[1], reverse=True)
        
        # Use permille naming to avoid decimal point issues
        # Special handling for ratio="mean" and ratio=0
        if ratio == "mean":
            key_name = "mean_permille"  # Special naming for mean parameter
        elif ratio == 0:
            key_name = "0_permille"  # Special naming for 0 threshold
        else:
            key_name = f"{int(ratio*1000)}_permille"  # Permille naming
        
        if sorted_data:
            sorted_indices, sorted_values, sorted_counts = zip(*sorted_data)
            results[key_name] = {
                "ratio": ratio,
                "min_occurrences": min_occurrences,
                "total_samples": total_samples,
                "qualifying_count": len(sorted_indices),
                "indices": list(sorted_indices),
                "mean_values": list(sorted_values),
                "occurrence_counts": list(sorted_counts)
            }
        else:
            results[key_name] = {
                "ratio": ratio,
                "min_occurrences": min_occurrences,
                "total_samples": total_samples,
                "qualifying_count": 0,
                "indices": [],
                "mean_values": [],
                "occurrence_counts": []
            }
        
        print(f"  Found {len(qualifying_indices)} qualifying dimensions")
        if len(qualifying_indices) > 0:
            print(f"  Activation value range: {min(qualifying_values):.4f} - {max(qualifying_values):.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results - dynamically generate filename
    output_file = Path(output_dir) / f"last_token_{content_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary information
    print("\n=== Analysis Summary ===")
    for ratio in ratios:
        # Handle key naming consistency
        if ratio == "mean":
            key = "mean_permille"
        elif ratio == 0:
            key = "0_permille"
        else:
            key = f"{int(ratio*1000)}_permille"  # Use permille naming
        
        if key in results:
            result = results[key]
            if ratio == "mean":
                print(f"Ratio {ratio} (based on all samples): {result['qualifying_count']} dimensions "
                      f"(all activated dimensions)")
            elif ratio == 0:
                print(f"Ratio {ratio} (no threshold): {result['qualifying_count']} dimensions "
                      f"(all activated dimensions)")
            else:
                print(f"Ratio {ratio}: {result['qualifying_count']} dimensions "
                      f"(minimum occurrences: {result['min_occurrences']})")
            if result['qualifying_count'] > 0:
                top_indices = result['indices'][:5]  # Display top 5
                top_values = result['mean_values'][:5]
                print(f"  Top 5 dimensions: {list(zip(top_indices, [f'{v:.4f}' for v in top_values]))}")

def main():
    # Get project root directory (current script is in SAGS/SFD/ directory)
    SAGS_ROOT = Path(__file__).parent.parent
    
    # Define base paths
    base_path = SAGS_ROOT / "SFD" / "generated" / MODEL_NAME / SAE_ID_SAFE / TEST_VERSION
    output_dir = SAGS_ROOT / "SFD" / "generated" / MODEL_NAME / "detection_vector"

    # Define content types to process
    content_types =  ['capital', 'no_comma', 'format_constrained', 'format_title', 'format_json', 'lowercase', 'quotation', 'repeat', 'two_responses']
    
    # Process each content type
    for content_type in content_types:
        print(f"\n{'='*60}")
        print(f"Starting to process {content_type} type data")
        print(f"{'='*60}")
        
        # Construct file path
        jsonl_file_path = f"{base_path}/{content_type}.jsonl"

        # Check if input file exists
        if not os.path.exists(jsonl_file_path):
            print(f"Warning: Input file does not exist, skipping: {jsonl_file_path}")
            continue
        
        # Execute analysis
        try:
            analyze_common_indices(jsonl_file_path, output_dir, content_type)
            print(f"\n{content_type} type analysis completed!")
        except Exception as e:
            print(f"Error occurred during {content_type} type analysis: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All data types analysis completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()