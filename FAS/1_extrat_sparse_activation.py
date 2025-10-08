#!/usr/bin/env python3
"""
Simplified SAE Encoding Extraction Experiment - Batch Processing Contrastive Dataset
Only extracts SAE encodings, no text generation
"""

# Fix Intel oneMKL error
import os
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import json
import glob
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from sae_lens import SAE, HookedSAETransformer

# Get project root directory (SAGS folder)
def get_project_root():
    """Get the SAGS project root directory"""
    current_file = Path(__file__).resolve()
    # Navigate up from FAS/ to SAGS/
    project_root = current_file.parent.parent
    return project_root

PROJECT_ROOT = get_project_root()

# =========================== Configuration ===========================
# Template configuration
USE_TEMPLATE = True

if USE_TEMPLATE:
    use_template = "use_template_question_asking"
else:
    use_template = "no_template"

# Content type filter configuration
# Options: "capital_only" - only process capital type
#          "all" - process all types
CONTENT_TYPE_FILTER = "capital_only"  # Change to "all" to process all types
# =========================== End Configuration ===========================

class SimpleSAEExperiment:
    """Simplified SAE Experiment Class"""
    
    def __init__(self, 
                 model_name: str = "google/gemma-2-2b", 
                 sae_release: str = "gemma-scope-2b-pt-res-canonical",
                 sae_id: str = "layer_22/width_65k/canonical",
                 experiment_name: str = "sae_bf16",
                 base_output_dir: str = None):
        """Initialize parameters"""
        # Use relative path based on project root if not specified
        if base_output_dir is None:
            base_output_dir = str(PROJECT_ROOT / "FAS" / "generated")
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_id = sae_id
        
        # Detect if it's an instruction-tuned model
        self.is_instruct_model = "it" in model_name.lower() or "instruct" in model_name.lower()
        
        # Unified path configuration
        self.experiment_name = experiment_name
        self.base_output_dir = Path(base_output_dir)
        self.output_dir = self.base_output_dir / experiment_name
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SAE configuration info (set to None during init, will be set when loading SAE)
        self.sae_hook_layer = None
        self.sae_hook_name = None
        
        # Model and SAE instances (load once)
        self.model = None
        self.sae = None
        
        print(f"Using device: {self.device}")
        print(f"Model type: {'Instruction-tuned model' if self.is_instruct_model else 'Pre-trained model'}")
        print(f"Experiment name: {self.experiment_name}")
        print(f"Output directory: {self.output_dir}")
        
    def _load_model_and_sae(self, verbose=True):
        """Load model and SAE once"""
        if self.model is None or self.sae is None:
            if verbose:
                print(f"Loading model to {self.device}...")
            self.model = HookedSAETransformer.from_pretrained(
                self.model_name, 
                device=self.device,
                torch_dtype=torch.bfloat16
            )
            
            if verbose:
                print(f"Loading SAE to {self.device}...")
            self.sae, _, _ = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device
            )
            
            # Save SAE configuration info to instance variables
            self.sae_hook_layer = self.sae.cfg.hook_layer
            self.sae_hook_name = self.sae.cfg.hook_name
            
            if verbose:
                print("Model and SAE loaded successfully!")
        
        return self.model, self.sae
    
    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template for instruction-tuned model"""
        if self.is_instruct_model:
            # Use Gemma-2-2b-IT chat template
            # return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            return f"<start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n{prompt}"
        else:
            # For pre-trained model, decide whether to use QA template based on global template parameter
            if USE_TEMPLATE:
                return f'Q: {prompt}\nA:'
            else:
                return prompt
    
    def extract_sae_encoding(self, prompt: str):
        """Extract SAE encoding only, no text generation"""
        # Get loaded model and SAE (don't show verbose messages)
        model, sae = self._load_model_and_sae(verbose=False)
        
        # Apply appropriate template (instruction-tuned or pre-trained model)
        formatted_prompt = self.apply_chat_template(prompt)
    
        # Prepare input
        tokens = model.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_tokens = tokens['input_ids']
        
        # Extract SAE encoding, only run to specified layer
        stop_layer = sae.cfg.hook_layer + 1
        
        with torch.no_grad():
            # Run model and get SAE activations, only to specified layer
            _, cache = model.run_with_cache_with_saes(
                input_tokens,
                saes=[sae],
                stop_at_layer=stop_layer
            )
    
        # Extract SAE activations for the last token
        sae_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :].to(self.device)
        
        # Find non-zero activations
        non_zero_mask = sae_acts > 0
        non_zero_count = torch.sum(non_zero_mask).item()
        
        # Only clear cache, don't delete model
        model.reset_hooks()
        torch.cuda.empty_cache()
        
        return sae_acts, non_zero_count
    
    def save_sae_encoding(self, prompt: str, sae_acts, non_zero_count, prompt_idx: int, file_suffix: str = ""):
        """Save SAE encoding data to JSONL (sparse format)"""
        # Use unified output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract indices and values of non-zero activations
        non_zero_mask = sae_acts > 0
        non_zero_indices = torch.where(non_zero_mask)[0].cpu().numpy().tolist()
        non_zero_values = sae_acts[non_zero_mask].cpu().numpy().tolist()
        
        # Save sparse format SAE activations to JSONL
        jsonl_data = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "sae_activations": {
                "indices": non_zero_indices,
                "values": non_zero_values
            },
            "total_dimensions": len(sae_acts),
            "non_zero_count": non_zero_count,
            "sparsity_ratio": (len(sae_acts) - non_zero_count) / len(sae_acts)
        }
        
        # Determine filename based on file_suffix
        if file_suffix:
            jsonl_filename = f"sae_encodings_{file_suffix}.jsonl"
        else:
            jsonl_filename = "sae_encodings.jsonl"
        
        jsonl_file = self.output_dir / jsonl_filename
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(jsonl_data, ensure_ascii=False) + '\n')
    
    def run_experiment(self, prompts, dataset_name: str = "", field_name: str = "suffix_i", file_suffix: str = ""):
        """Run experiment"""
        print(f"\n=== Starting SAE Encoding Extraction Experiment ===")
        print(f"Dataset: {dataset_name}")
        print(f"Field: {field_name}")
        print(f"Mode: Extract SAE encodings only, no text generation")
        print(f"Total prompts: {len(prompts)}")
        print(f"Output directory: {self.output_dir}")
        
        # Pre-load model and SAE
        self._load_model_and_sae(verbose=True)
        
        # Clear previous JSONL file
        if file_suffix:
            jsonl_filename = f"sae_encodings_{file_suffix}.jsonl"
        else:
            jsonl_filename = "sae_encodings.jsonl"
        
        jsonl_file = self.output_dir / jsonl_filename
        if jsonl_file.exists():
            jsonl_file.unlink()
        
        # Use tqdm to show progress bar
        progress_bar = tqdm(enumerate(prompts), total=len(prompts), desc=f"Processing {field_name}")
        
        for i, prompt in progress_bar:
            try:
                # Extract SAE encoding only
                sae_acts, non_zero_count = self.extract_sae_encoding(prompt)
                
                # Update progress bar description
                sparsity = (len(sae_acts) - non_zero_count) / len(sae_acts) * 100
                progress_bar.set_postfix({
                    'shape': f"{sae_acts.shape[0]}",
                    'non_zero': f"{non_zero_count}",
                    'sparsity': f"{sparsity:.1f}%"
                })
                
                # Save SAE encoding data
                self.save_sae_encoding(prompt, sae_acts, non_zero_count, i+1, file_suffix)
                
            except Exception as e:
                progress_bar.write(f"Failed to process prompt {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        progress_bar.close()
        
        # Save summary information
        self.save_summary(dataset_name, field_name, file_suffix)
    
    def save_summary(self, dataset_name: str = "", field_name: str = "suffix_i", file_suffix: str = ""):
        """Save experiment summary information"""
        if file_suffix:
            summary_filename = f"experiment_summary_{file_suffix}.txt"
        else:
            summary_filename = "experiment_summary.txt"
        
        summary_file = self.output_dir / summary_filename
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== SAE Encoding Extraction Experiment Summary ===\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Processing field: {field_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"SAE release: {self.sae_release}\n")
            f.write(f"SAE ID: {self.sae_id}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Experiment name: {self.experiment_name}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Experiment time: {datetime.now()}\n\n")
            
            f.write("Experiment strategy: Extract SAE encodings only, no text generation\n")
            if self.is_instruct_model:
                f.write("Template strategy: Using Gemma-2-2b-IT chat template '<start_of_turn>user\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n'\n")
            else:
                if USE_TEMPLATE:
                    f.write("Template strategy: Using QA template 'Q: {prompt}\\nA:'\n")
                else:
                    f.write("Template strategy: No template used, input raw prompt directly\n")
            # Use instance variables to support SAE at any layer
            if self.sae_hook_layer is not None:
                f.write(f"SAE layer: layer_{self.sae_hook_layer}\n")
                f.write(f"Stop layer: layer_{self.sae_hook_layer + 1}\n")
                f.write(f"Hook name: {self.sae_hook_name}\n")
            else:
                f.write("SAE config: Unknown (possibly due to loading failure)\n")
            f.write("- Only run to SAE layer, no full inference\n\n")
            
            f.write("File description:\n")
            if file_suffix:
                f.write(f"- sae_encodings_{file_suffix}.jsonl: Complete SAE activation data ({field_name} field)\n")
                f.write(f"- experiment_summary_{file_suffix}.txt: Experiment summary information ({field_name} field)\n")
            else:
                f.write(f"- sae_encodings.jsonl: Complete SAE activation data ({field_name} field)\n")
                f.write(f"- experiment_summary.txt: Experiment summary information ({field_name} field)\n")
        
        print(f"Experiment summary saved: {summary_file}")
    
    def cleanup_sae_only(self):
        """Clean up SAE resources only, keep model"""
        if self.sae is not None:
            del self.sae
            self.sae = None
        torch.cuda.empty_cache()
        print("SAE resources cleared, model retained")
    
    def cleanup(self):
        """Manually clean up all resources (optional)"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.sae is not None:
            del self.sae
            self.sae = None
        torch.cuda.empty_cache()
        print("Model and SAE resources cleared")

def load_json_dataset(json_file_path: str, field_name: str = "suffix_i"):
    """Load JSON dataset and return content of specified field"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = data.get(field_name, [])
    print(f"Loaded {len(prompts)} prompts from {json_file_path} (field: {field_name})")
    return prompts

def main():
    """Main function"""
    # Dataset path - use relative path based on project root
    data_dir = str(PROJECT_ROOT / "data" / "FAS_data" / "I-F")
    
    # Model configuration - use instruction-tuned model
    model_name = "google/gemma-2-2b"  # Use instruction-tuned version
    model_short_name = model_name.split("/")[-1]  # Extract short name, e.g. "gemma-2-2b-it"
    
    # Get all JSON files
    all_json_files = glob.glob(os.path.join(data_dir, "*.json"))
    all_json_files.sort()  # Sort to ensure consistent processing order
    
    # Filter files based on configuration
    if CONTENT_TYPE_FILTER == "capital_only":
        json_files = [f for f in all_json_files if "capital" in os.path.basename(f).lower()]
        print(f"Configuration: Only processing capital type")
    else:  # "all"
        json_files = all_json_files
        print(f"Configuration: Processing all types")
    
    print(f"Found {len(all_json_files)} JSON dataset files, will process {len(json_files)}:")
    for json_file in json_files:
        print(f"  - {os.path.basename(json_file)}")
    
    if CONTENT_TYPE_FILTER == "capital_only" and len(json_files) < len(all_json_files):
        skipped_count = len(all_json_files) - len(json_files)
        print(f"  (Skipped {skipped_count} non-capital type files)")
    
    print(f"Model: {model_name}")
    print(f"Model short name: {model_short_name}")
    
    # Outer loop: iterate through layer_0 to layer_25
    for layer_num in range(18,19,1):  # Layers 0 to 25
        print(f"\n{'='*100}")
        print(f"Starting to process layer {layer_num} SAE ({model_short_name})")
        print(f"{'='*100}")
        
        # Set output directory for current layer, including model name
        output_base_dir = str(PROJECT_ROOT / "FAS" / "generated" / model_short_name / f"layer_{layer_num}")
        
        # Set SAE ID for current layer
        current_sae_id = f"layer_{layer_num}/width_65k/canonical"
        
        # Create experiment object for current layer
        global_experiment = SimpleSAEExperiment(
            model_name=model_name,
            sae_id=current_sae_id,
            experiment_name="temp",  # Temporary name, will be overridden
            base_output_dir=output_base_dir
        )
        
        # Pre-load model and current layer SAE
        print(f"\nLoading layer {layer_num} SAE...")
        try:
            global_experiment._load_model_and_sae()
        except Exception as e:
            print(f"Failed to load layer {layer_num} SAE: {e}")
            continue
        
        # Process each JSON dataset one by one
        for json_file in json_files:
            dataset_name = os.path.splitext(os.path.basename(json_file))[0]  # Remove .json suffix
            print(f"\n{'-'*60}")
            print(f"Processing dataset: {dataset_name} (layer {layer_num})")
            print(f"{'-'*60}")
            
            try:
                # Update experiment object's output path, but keep model unchanged
                global_experiment.experiment_name = dataset_name
                global_experiment.output_dir = global_experiment.base_output_dir / dataset_name
                
                # Step 1: Process suffix_i field
                print(f"\n--- Step 1: Processing suffix_i field ---")
                prompts_suffix = load_json_dataset(json_file, "suffix_i")
                
                if prompts_suffix:
                    # Run experiment (using already loaded model)
                    global_experiment.run_experiment(prompts_suffix, dataset_name, "suffix_i", "")
                    print(f"suffix_i field processing completed! Results saved as: sae_encodings.jsonl, experiment_summary.txt")
                else:
                    print(f"Warning: suffix_i field not found in dataset {dataset_name}, skipping...")
                
                # Step 2: Process contr_suffix_v2 field
                print(f"\n--- Step 2: Processing contr_suffix_v2 field ---")
                prompts_contr = load_json_dataset(json_file, "contr_suffix_v2")
                
                if prompts_contr:
                    # Run experiment (using already loaded model)
                    global_experiment.run_experiment(prompts_contr, dataset_name, "contr_suffix_v2", "contr")
                    print(f"contr_suffix_v2 field processing completed! Results saved as: sae_encodings_contr.jsonl, experiment_summary_contr.txt")
                else:
                    print(f"Warning: contr_suffix_v2 field not found in dataset {dataset_name}, skipping...")
                
                print(f"\n{dataset_name} dataset processing completed!")
                print(f"Results saved in: {global_experiment.output_dir}")
                
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up current layer's SAE resources, free VRAM
        print(f"\nLayer {layer_num} processing completed, cleaning up SAE resources...")
        if global_experiment.sae is not None:
            del global_experiment.sae
            global_experiment.sae = None
        torch.cuda.empty_cache()
        print(f"Layer {layer_num} SAE resources cleared")
        
        print(f"\nAll datasets for layer {layer_num} processing completed!")
        print(f"Results saved in: {output_base_dir}")
    
    # Finally clean up model resources
    if 'global_experiment' in locals() and global_experiment.model is not None:
        del global_experiment.model
        torch.cuda.empty_cache()
        print("Model resources cleared")
    
    print(f"\n{'='*100}")
    print(f"All layers and datasets processing completed!")
   
    print(f"{'='*100}")

if __name__ == "__main__":
    main()