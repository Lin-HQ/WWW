#!/usr/bin/env python3
"""
SAE Activation Value Modification Experiment
Add specified vector element-wise to SAE activation values of all tokens in each autoregressive round, observe effects on generation results
"""

import os
import json
import torch
import time
from typing import List, Dict, Tuple
from functools import partial
import glob
import warnings
from pathlib import Path

# Disable all warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# SAE-related libraries
from sae_lens import SAE, HookedSAETransformer
from tqdm.auto import tqdm

# Disable transformers and other library progress bars
import transformers
transformers.logging.set_verbosity_error()

# Get project root directory (SAGS folder)
def get_project_root():
    """Get the SAGS project root directory"""
    current_file = Path(__file__).resolve()
    # Navigate up from FAS/ to SAGS/
    project_root = current_file.parent.parent
    return project_root

PROJECT_ROOT = get_project_root()




# Keep old configuration as backup
modification_scales_dict = {
    "capital":[1,2.5,5],
    # "no_comma":[1,2.5,5],
    # "format_constrained":[1,2.5,5],
    # "format_title":[1,2.5,5],
    # "format_json":[1,2.5,5],
    # "lowercase":[1,2.5,5],
    # "quotation":[1,2.5,5],
    # "repeat":[1,2.5,5],
    # "two_responses":[1,2.5,5],
    # "harmful":[1.5, 3, 4.5]  
}


# Configuration parameters - use relative path based on project root
base_dir = str(PROJECT_ROOT / "FAS" / "generated" / "gemma-2-2b")  # Base directory
vector_filter_suffix = ""  # "filter" or ""
layers_to_process =  list(range(18, 19))   # [31,32,33,34,36] 

# Add global template control parameter
USE_TEMPLATE = True  # Set to True to enable QA template, False to disable

class SAEActivationModificationExperiment:
    """SAE Activation Value Modification Experiment Class"""
    
    def __init__(self, model_name: str = "google/gemma-2-2b", 
                 sae_release: str = "gemma-scope-2b-pt-res-canonical"):
        """Initialize parameters"""
        self.model_name = model_name
        self.sae_release = sae_release
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perturbation_vector = None
        self.vector_file = None
        self.model = None
        self.current_sae = None
        self.current_layer = None
        
    def _extract_perturbation_type(self, vector_file_path: str) -> str:
        """Extract perturbation type from vector file path"""
        if "contrastive_" in vector_file_path:
            start_idx = vector_file_path.find("contrastive_") + len("contrastive_")
            end_idx = vector_file_path.find("/", start_idx)
            if end_idx == -1:
                end_idx = len(vector_file_path)
            perturbation_type = vector_file_path[start_idx:end_idx]
            return perturbation_type
        else:
            return "capital"
    
    def _load_test_prompts_from_base_data(self, perturbation_type: str, 
                                         base_data_path: str = None) -> List[str]:
        """Load test prompts of corresponding type from base_data.json"""
        # Use relative path based on project root if not specified
        if base_data_path is None:
            base_data_path = str(PROJECT_ROOT / "data" / "prompt_data" / "base_data_formal.json")
        
        try:
            with open(base_data_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
            
            if perturbation_type in base_data:
                test_prompts = base_data[perturbation_type]
                return test_prompts
            else:
                return base_data.get("capital", [])
                
        except Exception as e:
            print(f"Failed to load base_data.json: {e}")
            return [
                "Write a song about regrets in the style of Taylor Swift. Please include explanations for the lyrics you write. Make sure your entire response is in English, and in all capital letters.",
                "Create a riddle about the name Sheldon using only 10 words. Make sure to only use capital letters in your entire response."
            ]
    
    def set_vector_file(self, vector_file: str):
        """Set vector file path"""
        self.vector_file = vector_file
        
    def _load_perturbation_vector(self):
        """Load perturbation vector from JSON file"""
        try:
            with open(self.vector_file, 'r') as f:
                vector_data = json.load(f)
            
            # Convert to torch tensor
            self.perturbation_vector = torch.tensor(vector_data, dtype=torch.float32, device=self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load perturbation vector: {e}")
    
    def _load_model(self):
        """Load model (load only once)"""
        if self.model is None:
            # Temporarily disable all progress bars
            from tqdm import tqdm as original_tqdm
            import tqdm
            
            # Replace tqdm with empty function
            def disabled_tqdm(*args, **kwargs):
                if args:
                    return args[0]
                return []
            
            # Save original tqdm
            original_tqdm_func = tqdm.tqdm
            original_auto_tqdm = tqdm.auto.tqdm
            
            try:
                # Disable all tqdm
                tqdm.tqdm = disabled_tqdm
                tqdm.auto.tqdm = disabled_tqdm
                
                self.model = HookedSAETransformer.from_pretrained(
                    self.model_name, 
                    device=self.device, 
                    torch_dtype=torch.bfloat16
                )
            finally:
                # Restore tqdm
                tqdm.tqdm = original_tqdm_func
                tqdm.auto.tqdm = original_auto_tqdm
    
    def _load_sae_for_layer(self, layer: int):
        """Load SAE for specified layer"""
        # If SAE for the same layer is already loaded, return directly
        if self.current_layer == layer and self.current_sae is not None:
            return self.current_sae
        
        # Clean up previous SAE
        if self.current_sae is not None:
            del self.current_sae
            torch.cuda.empty_cache()
        
        # Temporarily disable progress bar to load SAE
        from tqdm import tqdm as original_tqdm
        import tqdm
        
        def disabled_tqdm(*args, **kwargs):
            if args:
                return args[0]
            return []
        
        original_tqdm_func = tqdm.tqdm
        original_auto_tqdm = tqdm.auto.tqdm
        
        try:
            # Disable all tqdm
            tqdm.tqdm = disabled_tqdm
            tqdm.auto.tqdm = disabled_tqdm
            
            # Load new SAE
            sae_id = f"layer_{layer}/width_65k/canonical"
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=sae_id,
                device=self.device
            )
        finally:
            # Restore tqdm
            tqdm.tqdm = original_tqdm_func
            tqdm.auto.tqdm = original_auto_tqdm
        
        self.current_sae = sae
        self.current_layer = layer
        return sae
    
    def _sae_activation_modification_hook(self, sae_acts, hook, modification_scale=1.0, 
                                     zero_threshold=1e-6):
        """
        Hook function to modify SAE activation values
        Use pre-loaded perturbation vector to add element-wise to SAE activation values at all positions
        
        Args:
            sae_acts: SAE activation value tensor [batch_size, seq_len, sae_features]
            hook: hook point information
            modification_scale: scaling factor for perturbation vector
            zero_threshold: threshold for judging as zero (absolute value less than this is considered zero) (parameter retained but not used)
        """
        batch_size, seq_len, sae_features = sae_acts.shape
        
        # Verify perturbation vector dimensions
        if self.perturbation_vector is None:
            raise RuntimeError("Perturbation vector not loaded!")
        
        if len(self.perturbation_vector) != sae_features:
            raise RuntimeError(f"Perturbation vector length ({len(self.perturbation_vector)}) does not match SAE feature dimension ({sae_features})!")
        
        # Scale perturbation vector
        scaled_perturbation = self.perturbation_vector * modification_scale
        
        # Broadcast perturbation vector to all token positions: [sae_features] -> [batch_size, seq_len, sae_features]
        perturbation_broadcasted = scaled_perturbation.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Add perturbation to all positions
        modified_acts = sae_acts + perturbation_broadcasted
        
        return modified_acts
    
    def _generate_with_modified_sae(self, model, sae, prompt: str, 
                                   modification_scale: float = 1.0,
                                   max_new_tokens: int = 50) -> str:
        """
        Generate output using modified SAE activation values
        Modify SAE activation values of all tokens in each autoregressive round
        """
        # Decide whether to use QA template based on global template parameter
        if USE_TEMPLATE:
            formatted_prompt = f'Q: {prompt}\nA:'
        else:
            formatted_prompt = prompt

        # Temporarily disable progress bars
        from tqdm import tqdm as original_tqdm
        import tqdm
        
        def disabled_tqdm(*args, **kwargs):
            if args:
                return args[0]
            return []
        
        original_tqdm_func = tqdm.tqdm
        original_auto_tqdm = tqdm.auto.tqdm
        
        try:
            # Disable all tqdm
            tqdm.tqdm = disabled_tqdm
            tqdm.auto.tqdm = disabled_tqdm
            
            # Prepare input - use formatted prompt
            tokens = model.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            input_tokens = tokens['input_ids']
            
            # Reset and configure SAE
            model.reset_hooks()
            model.reset_saes()
            
            # Set SAE mode
            sae.use_error_term = True # Use SAE-reconstructed activation values
            model.add_sae(sae)
            
            # Create hook to modify activation values
            hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
            modification_hook = partial(
                self._sae_activation_modification_hook, 
                modification_scale=modification_scale
            )
            
            # Add hook to modify SAE activation values
            model.add_hook(hook_name, modification_hook)
            
            # Set deterministic generation parameters
            generation_kwargs = {
                'input': input_tokens,
                'max_new_tokens': max_new_tokens,
                'temperature': 0.0,
                'do_sample': False,
                'freq_penalty':1.0
            }
            
            with torch.no_grad():
                output_tokens = model.generate(**generation_kwargs)
            
            # Decode output, only return newly generated part (remove original prompt)
            generated_tokens = output_tokens[0][len(input_tokens[0]):]  # Only take newly generated tokens
            output = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Add output truncation detection
            output = self._truncate_output(output)
            
            # Reset all states
            model.reset_hooks()
            model.reset_saes()
            
            return output
        finally:
            # Restore tqdm
            tqdm.tqdm = original_tqdm_func
            tqdm.auto.tqdm = original_auto_tqdm
    

    def _generate_baseline(self, model, sae, prompt: str, 
                          max_new_tokens: int = 50) -> str:
        """
        Generate baseline output (use SAE but don't add perturbation vector)
        """
        # Decide whether to use QA template based on global template parameter
        if USE_TEMPLATE:
            formatted_prompt = f'Q: {prompt}\nA:'
        else:
            formatted_prompt = prompt

        # Temporarily disable progress bars
        from tqdm import tqdm as original_tqdm
        import tqdm
        
        def disabled_tqdm(*args, **kwargs):
            if args:
                return args[0]
            return []
        
        original_tqdm_func = tqdm.tqdm
        original_auto_tqdm = tqdm.auto.tqdm
        
        try:
            # Disable all tqdm
            tqdm.tqdm = disabled_tqdm
            tqdm.auto.tqdm = disabled_tqdm
            
            # Prepare input
            tokens = model.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            input_tokens = tokens['input_ids']
            
            # Reset and configure SAE
            model.reset_hooks()
            model.reset_saes()
            
            # Set SAE mode (use SAE but don't add perturbation)
            sae.use_error_term = True
            model.add_sae(sae)
            
            # Set deterministic generation parameters
            generation_kwargs = {
                'input': input_tokens,
                'max_new_tokens': max_new_tokens,
                'temperature': 0.0,
                'do_sample': False,
                'freq_penalty': 1.0
            }
            
            with torch.no_grad():
                output_tokens = model.generate(**generation_kwargs)
            
            # Decode output, only return newly generated part
            generated_tokens = output_tokens[0][len(input_tokens[0]):]
            output = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Add output truncation detection
            output = self._truncate_output(output)
            
            # Reset all states
            model.reset_hooks()
            model.reset_saes()
            
            return output
        finally:
            # Restore tqdm
            tqdm.tqdm = original_tqdm_func
            tqdm.auto.tqdm = original_auto_tqdm
    
    def _truncate_output(self, output: str) -> str:
        """
        Truncate output text, detect and handle multiple stopping conditions
    
        Args:
            output: Original output text
        
        Returns:
            Truncated output text
        """
        if not output:
            return output
        
        # 1. Detect pattern of new question starting (Q:)
        q_pattern_markers = ["Q:", "Question:", "Q :", "问:", "问题:"]
        for marker in q_pattern_markers:
            if marker in output:
                # Find position of first Q: and truncate
                q_index = output.find(marker)
                if q_index > 0:  # Make sure it's not at the beginning
                    output = output[:q_index].strip()
                    break
    
        # 2. Detect repetitive patterns or loop generation
        lines = output.split('\n')
        if len(lines) > 3:
            # Check if there are consecutive repeated lines
            for i in range(len(lines) - 2):
                if lines[i] == lines[i+1] == lines[i+2] and lines[i].strip():
                    # Found three consecutive identical lines, truncate to first line
                    output = '\n'.join(lines[:i+1])
                    break

        lines = output.split('\n')  # Re-get lines, as it may have been modified in step 2
        if len(lines) >= 4:  # Need at least 4 lines to detect double-line repetition
            for i in range(len(lines) - 3):
                # Check if line i and i+1 are the same as lines i+2 and i+3
                if (lines[i] == lines[i+2] and 
                    lines[i+1] == lines[i+3] and 
                    lines[i].strip() and lines[i+1].strip()):
                    # Found double-line repetition pattern, keep only first two lines
                    output = '\n'.join(lines[:i+2])
                    break
            # 3. Check for repetition pattern with blank lines (e.g.: content-blank-content-blank)
        lines = output.split('\n')  # Re-get lines
        if len(lines) >= 4:  # Only need 4 lines to detect this pattern
            for i in range(len(lines) - 3):
                # Check pattern: non-empty line -> empty line -> same non-empty line -> empty line
                if (lines[i].strip() and  # First line non-empty
                    not lines[i+1].strip() and  # Second line empty
                    lines[i+2].strip() and  # Third line non-empty
                    not lines[i+3].strip() and  # Fourth line empty
                    lines[i] == lines[i+2]):  # Lines 1 and 3 have same content
                    # Found repetition pattern with blank lines, keep only first two lines (content line + blank line)
                    output = '\n'.join(lines[:i+2])
                    break
        
        # 3. Detect abnormal length or formatting issues
        # If output is too long, may have loop generation
        if len(output) > 2000:  # Set reasonable length threshold
            # Try to truncate at period or newline
            for i in range(1500, len(output)):
                if output[i] in '.。\n':
                    output = output[:i+1]
                    break
    
        # 4. Remove incomplete sentences or words at the end
        output = output.strip()
    
        # 5. Detect special ending patterns
        end_markers = ["</s>", "<|endoftext|>", "[END]", "---"]
        for marker in end_markers:
            if marker in output:
                marker_index = output.find(marker)
                output = output[:marker_index].strip()
                break
    
        return output
    
    def generate_modified_outputs(self, prompts: List[str], layer: int,
                                 modification_scales: List[float] = [1.0, 1.5, 2]) -> Dict:
        """
        Generate baseline and outputs under different perturbation scaling factors
        
        Args:
            prompts: Input prompt list
            layer: Current processing layer number
            modification_scales: List of perturbation vector scaling factors to test
        """
        # Load perturbation vector
        self._load_perturbation_vector()
        
        # Load model and SAE for corresponding layer
        self._load_model()
        sae = self._load_sae_for_layer(layer)
        
        results = {}
        
        # Time statistics variables
        generation_times = []  # Record time for each generation
        prompt_processing_times = []  # Record complete processing time for each prompt
        
        # Simple progress display without using tqdm
        for i, prompt in enumerate(prompts):
            # Display simple progress
            print(f"{i+1}/{len(prompts)}", end=" ", flush=True)
            
            # Record start time for entire prompt processing
            prompt_start_time = time.time()
            
            prompt_results = {
                "prompt": prompt,
                "baseline_output": None,
                "modified_outputs": {}
            }
            
            try:
                # First generate baseline (use SAE but don't add perturbation)
                start_time = time.time()
                baseline_output = self._generate_baseline(
                    self.model, sae, prompt
                )
                baseline_time = time.time() - start_time
                generation_times.append(baseline_time)
                prompt_results["baseline_output"] = baseline_output
                
                # Generate outputs under various perturbation scaling factors (use_error_term=True, with hook)
                for scale in modification_scales:
                    start_time = time.time()
                    modified_output = self._generate_with_modified_sae(
                        self.model, sae, prompt, modification_scale=scale
                    )
                    scale_time = time.time() - start_time
                    generation_times.append(scale_time)
                    prompt_results["modified_outputs"][str(scale)] = modified_output
                
                results[f"prompt_{i+1}"] = prompt_results
                
                # Record total time for entire prompt processing
                prompt_total_time = time.time() - prompt_start_time
                prompt_processing_times.append(prompt_total_time)
                
                # Display average time statistics every 10 samples
                if (i + 1) % 10 == 0:
                    self._display_timing_statistics(i + 1, generation_times, prompt_processing_times)
                
            except Exception as e:
                prompt_results["error"] = str(e)
                results[f"prompt_{i+1}"] = prompt_results
        
        # Display final statistics
        if generation_times and prompt_processing_times:
            avg_generation_time = sum(generation_times) / len(generation_times)
            avg_prompt_time = sum(prompt_processing_times) / len(prompt_processing_times)
            print(f"\n⏱️  Final statistics: Total generations {len(generation_times)}, Average generation time {avg_generation_time:.2f}s/time")
            print(f"            Total prompts {len(prompt_processing_times)}, Average processing time {avg_prompt_time:.2f}s/prompt")
        
        return results
    
    def _display_timing_statistics(self, current_prompt: int, generation_times: List[float], 
                                  prompt_processing_times: List[float]):
        """Display time statistics information (display once every 10 samples)"""
        if not generation_times or not prompt_processing_times:
            return
        
        # Calculate statistical data
        avg_generation_time = sum(generation_times) / len(generation_times)
        total_generations = len(generation_times)
        avg_prompt_time = sum(prompt_processing_times) / len(prompt_processing_times)
        
        # Calculate statistics for recent 10 samples (if available)
        recent_generation_times = generation_times[-30:] if len(generation_times) >= 30 else generation_times  # Recent 10 samples × 3 generations
        recent_avg_generation_time = sum(recent_generation_times) / len(recent_generation_times) if recent_generation_times else 0
        
        recent_prompt_times = prompt_processing_times[-10:] if len(prompt_processing_times) >= 10 else prompt_processing_times  # Recent 10 samples
        recent_avg_prompt_time = sum(recent_prompt_times) / len(recent_prompt_times) if recent_prompt_times else 0
        
        print(f"\n      📊 Statistics (sample {current_prompt}): Average generation time {avg_generation_time:.2f}s/time, "
              f"Total generations {total_generations}")
        print(f"            Average processing time {avg_prompt_time:.2f}s/prompt, Recent average processing time {recent_avg_prompt_time:.2f}s/prompt")
    
    def save_modification_results(self, results: Dict, 
                                 save_dir: str = "/data1/lhq/test0/data_contr/out_filter"):
        """Save perturbation modification results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed modification results
        with open(f"{save_dir}/vector_modification_results.txt", "w", encoding="utf-8") as f:
            f.write("=== SAE Activation Value Vector Perturbation Experiment Results ===\n\n")
            f.write(f"Perturbation vector file: {self.vector_file}\n\n")
            
            for prompt_key, prompt_data in results.items():
                f.write(f"{'='*80}\n")
                f.write(f"{prompt_key.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Prompt: {prompt_data['prompt']}\n\n")
                
                if "error" in prompt_data:
                    f.write(f"Error: {prompt_data['error']}\n\n")
                    continue
                
                # Baseline output
                if prompt_data.get('baseline_output'):
                    f.write(f"Baseline output (no perturbation):\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{prompt_data['baseline_output']}\n\n")
                
                # Modified outputs
                for scale, modified_output in prompt_data['modified_outputs'].items():
                    f.write(f"Output with perturbation scaling factor={scale}:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{modified_output}\n\n")
                
                f.write("\n")
        
        # Save JSONL format results
        with open(f"{save_dir}/results.jsonl", "w", encoding="utf-8") as f:
            for prompt_key, prompt_data in results.items():
                if "error" in prompt_data:
                    continue
                
                # Build JSONL format data
                jsonl_entry = {
                    "prompt": prompt_data["prompt"]
                }
                
                # Add baseline output
                if prompt_data.get("baseline_output"):
                    jsonl_entry["baseline"] = prompt_data["baseline_output"]
                
                # Add outputs for each perturbation scaling factor
                for scale, modified_output in prompt_data["modified_outputs"].items():
                    jsonl_entry[f"modification_{scale}"] = modified_output
                
                # Write to JSONL file
                f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
        
        # Save summary statistics
        with open(f"{save_dir}/summary.txt", "w", encoding="utf-8") as f:
            total_prompts = len([k for k in results.keys() if not results[k].get('error')])
            error_count = len([k for k in results.keys() if results[k].get('error')])
            
            f.write("=== SAE Activation Value Vector Perturbation Experiment Summary ===\n\n")
            f.write(f"Perturbation vector file: {self.vector_file}\n")
            f.write(f"Total prompts: {len(results)}\n")
            f.write(f"Successfully processed: {total_prompts}\n")
            f.write(f"Processing failed: {error_count}\n\n")
            
            if total_prompts > 0:
                # Statistics on generation for each perturbation scaling factor
                f.write("Generation statistics for each perturbation scaling factor:\n")
                f.write("-" * 40 + "\n")
                
                # Count successful generations for each scaling factor
                scale_stats = {}
                for prompt_key, prompt_data in results.items():
                    if prompt_data.get('error'):
                        continue
                    
                    for scale in prompt_data['modified_outputs'].keys():
                        if scale not in scale_stats:
                            scale_stats[scale] = 0
                        scale_stats[scale] += 1
                
                for scale, count in sorted(scale_stats.items(), key=lambda x: float(x[0])):
                    f.write(f"  Scaling factor={scale}: {count} samples generated successfully\n")
        
        # Don't display save info to avoid cluttered output
        print(f"Perturbation results saved to: {save_dir}")

def get_contrastive_folders(base_path: str) -> List[str]:
    """Get all contrastive_* folders"""
    pattern = os.path.join(base_path, "contrastive_*")
    folders = glob.glob(pattern)
    # Filter out non-directory files
    folders = [f for f in folders if os.path.isdir(f)]
    # Sort by folder name
    folders.sort()
    return folders

def process_single_folder(experiment: SAEActivationModificationExperiment, 
                         folder_path: str, 
                         layer: int,
                         vector_filter_suffix: str = "filter") -> str:
    """Process single folder
    
    Returns:
        str: "success" - successful, "skipped" - skipped, "failed" - failed
    """
    folder_name = os.path.basename(folder_path)
    
    # Build vector file path
    if vector_filter_suffix == "filter":
        vector_file = os.path.join(folder_path, "sae_json_vector_filter.json")
        save_dir = os.path.join(folder_path, "out_filter")
    else:
        vector_file = os.path.join(folder_path, "sae_vector_new_0_6.json")
        save_dir = os.path.join(folder_path, "out_0_6")
    
    # Check if vector file exists
    if not os.path.exists(vector_file):
        return "failed"
    
    try:
        # Set vector file
        experiment.set_vector_file(vector_file)
        
        # Extract perturbation type
        perturbation_type = experiment._extract_perturbation_type(vector_file)
        
        # Load test data
        test_prompts = experiment._load_test_prompts_from_base_data(perturbation_type)
        
        # Check if perturbation type is in configuration dictionary
        if perturbation_type not in modification_scales_dict:
            return "skipped"
        
        # Get perturbation scaling factors corresponding to this type
        modification_scales = modification_scales_dict[perturbation_type]
        
        # Display number of prompts being processed for current folder
        print(f"        Processing {len(test_prompts)} prompts: ", end="", flush=True)
        print(f"Using {perturbation_type}-specific parameters {modification_scales}", end=" ", flush=True)
        
        # Run experiment
        results = experiment.generate_modified_outputs(test_prompts, layer, modification_scales)
        
        # Save results
        experiment.save_modification_results(results, save_dir)
        
        # Count results
        total_prompts = len([k for k in results.keys() if not results[k].get('error')])
        error_count = len([k for k in results.keys() if results[k].get('error')])
        
        print("Completed")  # Display completion on new line
        
        return "success" if error_count == 0 else "failed"
        
    except Exception as e:
        print(f"Failed - {str(e)}")
        return "failed"

def process_single_layer(experiment: SAEActivationModificationExperiment, 
                        layer: int, 
                        vector_filter_suffix: str = "filter") -> Tuple[int, int, int]:
    """
    Process all folders in a single layer
    
    Returns:
        Tuple[int, int, int]: (total folder count, successfully processed folder count, skipped folder count)
    """
    layer_path = os.path.join(base_dir, f"layer_{layer}")
    
    if not os.path.exists(layer_path):
        print(f"❌ Layer{layer}: Path does not exist - {layer_path}")
        return 0, 0, 0
    
    # Get all contrastive_* folders for this layer
    folders = get_contrastive_folders(layer_path)
    
    if not folders:
        print(f"⚠️  Layer{layer}: No contrastive_* folders found")
        return 0, 0, 0
    
    print(f"\n🔄 Processing Layer{layer} ({len(folders)} folders)")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Simplified folder processing without progress bar to avoid interference
    for i, folder in enumerate(folders):
        folder_name = os.path.basename(folder)
        print(f"   📁 [{i+1}/{len(folders)}] Processing {folder_name}...", end="", flush=True)
        
        result = process_single_folder(experiment, folder, layer, vector_filter_suffix)
        
        if result == "success":
            success_count += 1
            print(f" ✅ Completed")
        elif result == "skipped":
            skipped_count += 1
            print(f" ⏭️ Skipped (not in configuration)")
        else:  # failed
            failed_count += 1
            print(f" ❌ Failed")
    
    print(f"📊 Layer{layer}: {success_count} successful / {skipped_count} skipped / {failed_count} failed (total {len(folders)})")
    return len(folders), success_count, skipped_count

def main():
    """Main function"""
    print("🚀 Starting batch processing of multi-layer SAE activation value modification experiments")
    print(f"Base directory: {base_dir}")
    print(f"Processing layers: Layer{layers_to_process[0]} to Layer{layers_to_process[-1]}")
    print(f"Vector file suffix: {'filter' if vector_filter_suffix == 'filter' else 'normal'}")
    
    # Create experiment object (only need to create once, will reuse model)
    experiment = SAEActivationModificationExperiment()
    
    # Statistics variables
    total_layers = len(layers_to_process)
    successful_layers = 0
    total_folders = 0
    total_successful_folders = 0
    total_skipped_folders = 0
    
    # Simplified processing without progress bar to avoid conflict with internal progress bars
    print(f"\n📋 Starting to process {total_layers} layers...")
    
    for i, layer in enumerate(layers_to_process):
        print(f"\n🔢 [{i+1}/{total_layers}] Currently processing Layer{layer}")
        
        try:
            folder_count, success_count, skipped_count = process_single_layer(experiment, layer, vector_filter_suffix)
            total_folders += folder_count
            total_successful_folders += success_count
            total_skipped_folders += skipped_count
            
            processed_count = success_count + skipped_count
            
            if folder_count > 0 and processed_count == folder_count:
                successful_layers += 1
                print(f"✅ Layer{layer}: All processing completed ({success_count} successful, {skipped_count} skipped)")
            elif success_count > 0:
                print(f"⚠️  Layer{layer}: Partial success ({success_count} successful, {skipped_count} skipped, {folder_count - processed_count} failed)")
            else:
                print(f"❌ Layer{layer}: No successfully processed folders")
            
        except Exception as e:
            print(f"❌ Layer{layer}: Processing failed - {str(e)}")
            continue
    
    # Clean up resources
    if experiment.current_sae is not None:
        del experiment.current_sae
    if experiment.model is not None:
        del experiment.model
    torch.cuda.empty_cache()
    
    # Final statistics
    print(f"\n🎉 Batch processing completed!")
    print(f"   Selected layers: {total_layers}")
    print(f"   Successfully processed layers: {successful_layers}")
    print(f"   Total folders: {total_folders}")
    print(f"   Successfully processed folders: {total_successful_folders}")
    print(f"   Skipped folders: {total_skipped_folders}")
    print(f"   Failed folders: {total_folders - total_successful_folders - total_skipped_folders}")
    
    if successful_layers == total_layers:
        print("✅ All layers processing completed!")
    elif successful_layers > 0:
        print(f"⚠️  Some layers processing completed ({successful_layers}/{total_layers})")
    else:
        print("❌ No successful folders in all layers")

if __name__ == "__main__":
    main()