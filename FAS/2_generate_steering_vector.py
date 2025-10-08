#!/usr/bin/env python3
"""
SAE Encoding Data Processing Script
Batch processing sae_encodings.jsonl data from multiple layer folders, generating difference vectors
"""

import json
import numpy as np
from pathlib import Path
import datetime
from tqdm import tqdm
import re
import argparse

# Get project root directory (SAGS folder)
def get_project_root():
    """Get the SAGS project root directory"""
    current_file = Path(__file__).resolve()
    # Navigate up from FAS/ to SAGS/
    project_root = current_file.parent.parent
    return project_root

PROJECT_ROOT = get_project_root()

# Add global variable to store threshold ratio
ACTIVATION_RATIO = 0.6 # ✅ Directly modify your desired default value here

def process_pair_difference(normal_file, contr_file, output_dir, silent=True):
    """Process the difference between normal and contr files in the same folder, generating sae_vector_new_{ratio}.json"""
    
    try:
        # 1. Read normal file data - new sparse format
        normal_data = []
        total_dimensions = None
        with open(normal_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if total_dimensions is None:
                    total_dimensions = obj["total_dimensions"]
                sae_activations = obj["sae_activations"]
                normal_data.append({
                    "indices": sae_activations["indices"],
                    "values": sae_activations["values"]
                })
        
        # 2. Read contr file data - new sparse format
        contr_data = []
        with open(contr_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sae_activations = obj["sae_activations"]
                contr_data.append({
                    "indices": sae_activations["indices"],
                    "values": sae_activations["values"]
                })
        
        # 3. Check if sample counts match
        num_normal_samples = len(normal_data)
        num_contr_samples = len(contr_data)
        
        if num_normal_samples != num_contr_samples:
            if not silent:
                print(f"Warning: Sample counts of normal and contr files don't match {num_normal_samples} vs {num_contr_samples}")
            # Take minimum sample count
            min_samples = min(num_normal_samples, num_contr_samples)
            normal_data = normal_data[:min_samples]
            contr_data = contr_data[:min_samples]
        
        num_samples = len(normal_data)
        
        # 4. Subtract pairwise in order: normal[i] - contr[i], handling sparse format
        temp = {}  # Store difference results {dim_idx: [diff_values]}
        
        for i in range(num_samples):
            normal_sparse = normal_data[i]
            contr_sparse = contr_data[i]
            
            # Convert to dictionary for easy lookup - allows quick lookup of values for specific indices
            normal_dict = dict(zip(normal_sparse["indices"], normal_sparse["values"]))
            contr_dict = dict(zip(contr_sparse["indices"], contr_sparse["values"]))
            
            # Find all potentially involved dimension indices (union of minuend and subtrahend)
            all_indices = set(normal_dict.keys()) | set(contr_dict.keys())
            
            # Calculate difference: normal[idx] - contr[idx]
            # For your example:
            # normal: indices=[1,2,3], values=[1.5,1.6,1.7] 
            # contr:  indices=[1,2,4], values=[1,4,5]
            # all_indices = {1,2,3,4}
            for idx in all_indices:
                normal_val = normal_dict.get(idx, 0.0)  # 0 if not exists
                contr_val = contr_dict.get(idx, 0.0)    # 0 if not exists
                diff_val = normal_val - contr_val
                
                # Calculate as in your example:
                # idx=1: 1.5 - 1.0 = 0.5
                # idx=2: 1.6 - 4.0 = -2.4  
                # idx=3: 1.7 - 0.0 = 1.7
                # idx=4: 0.0 - 5.0 = -5.0
                
                # Only record non-zero differences
                if diff_val != 0.0:
                    if idx not in temp:
                        temp[idx] = []
                    temp[idx].append(diff_val)
        
        # 5. Apply filtering using global threshold ratio
        threshold = int(np.ceil(num_samples * ACTIVATION_RATIO))
        
        # Count non-zero difference occurrences for each dimension
        selected_dims = []
        processing_stats = []
        
        for dim_idx, diff_values in temp.items():
            nonzero_count = len(diff_values)
            if nonzero_count >= threshold:
                selected_dims.append(dim_idx)
                
                # Calculate statistics for this dimension
                diff_array = np.array(diff_values)
                mean_val = diff_array.mean()
                std_val = diff_array.std()
                
                processing_stats.append({
                    "dim": int(dim_idx),
                    "nonzero_count": int(nonzero_count),
                    "mean_value": float(mean_val),
                    "std_value": float(std_val)
                })
        
        # 6. Restore temp to full vector form
        result_vector = np.zeros(total_dimensions, dtype=np.float32)
        
        for dim_idx in selected_dims:
            diff_values = temp[dim_idx]
            mean_val = np.array(diff_values).mean()
            result_vector[dim_idx] = mean_val
        
        # 7. Save result as sae_vector_new_{ratio}.json
        ratio_str = f"{ACTIVATION_RATIO:.1f}".replace(".", "_")  # 0.7 -> "0_7"
        new_vector_path = output_dir / f"sae_vector_new_{ratio_str}.json"
        with open(new_vector_path, "w", encoding="utf-8") as f:
            json.dump([float(x) for x in result_vector], f, ensure_ascii=False)
        
        return result_vector, selected_dims, processing_stats, {
            "total_samples": num_samples,
            "total_dims": total_dimensions,
            "selected_dims_count": len(selected_dims),
            "threshold_used": threshold,
            "activation_ratio": ACTIVATION_RATIO,
            "temp_dims_count": len(temp)  # Add dimension count statistics in temp
        }
        
    except Exception as e:
        if not silent:
            print(f"Error processing difference: {e}")
        return None, None, None, None

def process_single_layer(layer_path):
    """Process all SAE encoding data for a single layer"""
    
    summary_path = layer_path / "summary.txt"
    
    # Find all folders containing both sae_encodings.jsonl and sae_encodings_contr.jsonl
    folders_with_both = {}
    
    for folder in layer_path.iterdir():
        if folder.is_dir():
            # Check for both types of files
            contr_file = folder / "sae_encodings_contr.jsonl"
            normal_file = folder / "sae_encodings.jsonl"
            
            # If this folder contains both types of files, record it
            if contr_file.exists() and normal_file.exists():
                folders_with_both[folder.name] = {
                    "normal": normal_file,
                    "contr": contr_file
                }
    
    # Create log writing function
    log_messages = []
    def log_writer(message):
        log_messages.append(message)
    
    # Start processing
    log_writer(f"SAE difference vector processing started - {layer_path.name}")
    log_writer(f"Processing time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_writer(f"Activation threshold ratio: {ACTIVATION_RATIO}")
    log_writer(f"Found {len(folders_with_both)} folders containing both normal and contr files:")
    for folder_name in folders_with_both.keys():
        log_writer(f"  - {folder_name}")
    
    # Process difference vectors
    log_writer(f"\n{'='*60}")
    log_writer(f"Starting difference vector processing (normal - contr)")
    log_writer(f"Activation threshold ratio: {ACTIVATION_RATIO}")
    log_writer(f"{'='*60}")
    
    successful_count = 0
    failed_count = 0
    diff_results = []
    
    for folder_name, folder_files in folders_with_both.items():
        try:
            folder_path = layer_path / folder_name
            normal_file = folder_files["normal"]
            contr_file = folder_files["contr"]
            
            diff_vector, diff_selected_dims, diff_stats, diff_info = process_pair_difference(
                normal_file, contr_file, folder_path, silent=True
            )
            
            if diff_vector is not None and diff_info is not None:
                successful_count += 1
                ratio_str = f"{ACTIVATION_RATIO:.1f}".replace(".", "_")
                diff_results.append({
                    "folder": folder_name,
                    "total_samples": diff_info["total_samples"],
                    "total_dims": diff_info["total_dims"],
                    "selected_dims_count": diff_info["selected_dims_count"],
                    "threshold_used": diff_info["threshold_used"],
                    "activation_ratio": diff_info["activation_ratio"]
                })
                
                log_writer(f"✓ {folder_name}/sae_vector_new_{ratio_str}.json difference processing successful")
                log_writer(f"    Sample count: {diff_info['total_samples']}, Selected dimensions: {diff_info['selected_dims_count']}")
            else:
                failed_count += 1
                log_writer(f"✗ {folder_name} difference processing failed")
                
        except Exception as e:
            failed_count += 1
            log_writer(f"✗ {folder_name} difference processing failed: {e}")
    
    # Summary statistics
    log_writer(f"\n{'='*60}")
    log_writer(f"Difference vector processing summary - {layer_path.name}")
    log_writer(f"{'='*60}")
    log_writer(f"Activation threshold ratio: {ACTIVATION_RATIO}")
    log_writer(f"Processable folders: {len(folders_with_both)}")
    log_writer(f"Difference processing successful: {successful_count}")
    log_writer(f"Difference processing failed: {failed_count}")
    log_writer(f"Difference processing success rate: {successful_count/len(folders_with_both)*100:.1f}%" if folders_with_both else "N/A")
    
    if diff_results:
        log_writer(f"\nDifference processing details:")
        log_writer("Folder Name\t\tSamples\tTotal Dims\tSelected Dims\tThreshold\tRatio")
        log_writer("-" * 75)
        for result in diff_results:
            log_writer(f"{result['folder']:<20}\t{result['total_samples']:6d}\t{result['total_dims']:8d}\t{result['selected_dims_count']:8d}\t{result['threshold_used']:6d}\t{result['activation_ratio']:.1f}")
    
    # Write all logs to summary.txt
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_messages))
    except Exception as e:
        print(f"Error saving summary.txt: {e}")
        return False, 0, 0
    
    return True, successful_count, failed_count

def process_all_layers():
    """Batch process SAE encoding data from all layers"""
    
    # Base path - use relative path based on project root
    base_path = PROJECT_ROOT / "FAS" / "generated" / "gemma-2-2b"
    
    # Get all layer folders, sort by number
    layer_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith("layer_"):
            # Extract layer number for numerical sorting
            layer_num_match = re.search(r'layer_(\d+)', folder.name)
            if layer_num_match:
                layer_num = int(layer_num_match.group(1))
                layer_folders.append((layer_num, folder))
    
    # Sort by layer number
    layer_folders.sort(key=lambda x: x[0])
    
    print(f"Found {len(layer_folders)} layer folders")
    print(f"Using activation threshold ratio: {ACTIVATION_RATIO}")
    print(f"Processing start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Global statistics
    all_layers_success = True
    total_success_files = 0
    total_failed_files = 0
    layer_results = {}
    
    # Use tqdm to show progress
    for layer_num, layer_folder in tqdm(layer_folders, desc="Processing layers", unit="layer"):
        try:
            success, success_count, failed_count = process_single_layer(layer_folder)
            
            layer_results[layer_folder.name] = {
                "success": success,
                "success_count": success_count,
                "failed_count": failed_count
            }
            
            total_success_files += success_count
            total_failed_files += failed_count
            
            if not success:
                all_layers_success = False
                
        except Exception as e:
            print(f"Error processing layer {layer_folder.name}: {e}")
            all_layers_success = False
            layer_results[layer_folder.name] = {
                "success": False,
                "success_count": 0,
                "failed_count": 0,
                "error": str(e)
            }
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"All layer difference vector processing completed")
    print(f"{'='*80}")
    print(f"Completion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using activation threshold ratio: {ACTIVATION_RATIO}")
    print(f"Total layers: {len(layer_folders)}")
    
    successful_layers = sum(1 for result in layer_results.values() if result.get("success", False))
    failed_layers = len(layer_folders) - successful_layers
    
    print(f"Successfully processed layers: {successful_layers}")
    print(f"Failed processed layers: {failed_layers}")
    print(f"Layer processing success rate: {successful_layers/len(layer_folders)*100:.1f}%")
    
    print(f"\nTotal difference vector processing statistics:")
    print(f"Successfully processed difference vectors: {total_success_files}")
    print(f"Failed processed difference vectors: {total_failed_files}")
    print(f"Difference vector processing success rate: {total_success_files/(total_success_files+total_failed_files)*100:.1f}%" if (total_success_files+total_failed_files) > 0 else "N/A")
    
    # Show detailed results for each layer
    print(f"\nProcessing details for each layer:")
    print("Layer Name\t\tStatus\tSuccess Count\tFail Count")
    print("-" * 60)
    for layer_num, layer_folder in layer_folders:
        result = layer_results[layer_folder.name]
        status = "✓" if result.get("success", False) else "✗"
        success_count = result.get("success_count", 0)
        failed_count = result.get("failed_count", 0)
        print(f"{layer_folder.name:<15}\t{status}\t{success_count:8d}\t{failed_count:8d}")
    
    # Save global summary
    ratio_str = f"{ACTIVATION_RATIO:.1f}".replace(".", "_")
    global_summary_path = base_path / f"global_diff_vector_summary_{ratio_str}.json"
    try:
        global_summary = {
            "processing_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "activation_ratio": ACTIVATION_RATIO,
            "total_layers": len(layer_folders),
            "successful_layers": successful_layers,
            "failed_layers": failed_layers,
            "layer_success_rate": successful_layers/len(layer_folders),
            "total_success_vectors": total_success_files,
            "total_failed_vectors": total_failed_files,
            "vector_success_rate": total_success_files/(total_success_files+total_failed_files) if (total_success_files+total_failed_files) > 0 else 0,
            "layer_details": layer_results
        }
        
        with open(global_summary_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)
        print(f"\nGlobal difference vector processing summary saved to: {global_summary_path}")
    except Exception as e:
        print(f"Error saving global summary file: {e}")
    
    if all_layers_success:
        print(f"\n🎉 All layer difference vector processing successful!")
    else:
        print(f"\n⚠️  Some layers failed processing, please check detailed logs")
    
    return all_layers_success

def main():
    """Main function"""
    global ACTIVATION_RATIO
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='SAE difference vector processing script')
    parser.add_argument('--ratio', type=float, default=ACTIVATION_RATIO,
                       help=f'Activation threshold ratio (default: {ACTIVATION_RATIO})')
    
    args = parser.parse_args()
    ACTIVATION_RATIO = args.ratio
    
    print(f"Starting SAE difference vector processing, activation threshold ratio: {ACTIVATION_RATIO}")
    
    try:
        success = process_all_layers()
        if success:
            print(f"\n✅ All difference vector processing tasks completed")
        else:
            print(f"\n❌ Difference vector processing tasks have issues")
        
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()