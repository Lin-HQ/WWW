import json
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_fscore_support

# =========================== Configuration Parameters ===========================
# Training set for finding best parameters
TRAIN_VERSION = "train"
# Test set for final evaluation
TEST_VERSION = "if_test"

MODEL_NAME = "gemma-2-2b"
SAE_ID = "layer_5/width_65k/canonical"

# Generate safe file name (replace slashes with underscores)
SAE_ID_SAFE = SAE_ID.replace("/", "_")

# Get project root directory (current script is in SAGS/SFD/ directory)
SAGS_ROOT = Path(__file__).parent.parent

# Define content types to process
content_types = ['capital', 'no_comma', 'format_constrained', 'format_title', 
                 'format_json', 'lowercase', 'quotation', 'repeat', 'two_responses']

# =========================== Sparse Similarity Algorithms ===========================

def _indices_values_to_dict(indices, values):
    """Convert indices and values lists to dict for O(1) lookup"""
    return dict(zip(indices, values))

def sparse_similarity_with_coverage_weight(A_indices, A_values, B_indices, B_values, alpha):
    """Use harmonic mean + coverage weight"""
    A_dict = _indices_values_to_dict(A_indices, A_values)
    B_dict = _indices_values_to_dict(B_indices, B_values)
    
    intersection_indices = set(A_indices) & set(B_indices)
    union_indices = set(A_indices) | set(B_indices)

    if not intersection_indices:
        return 0.0

    jaccard_similarity = (len(intersection_indices) / len(union_indices)) ** alpha

    similarity = 0.0

    if alpha != 0:
        beta = 1 / alpha
    else:
        beta = 1.0

    for idx in intersection_indices:
        a_i = A_dict[idx]
        b_i = B_dict[idx]
        similarity += ((2 * min(a_i, b_i)) / (a_i + b_i))**(beta)

    return similarity * jaccard_similarity

# Algorithm dictionary
SIMILARITY_ALGORITHMS = {
    "coverage_weight": sparse_similarity_with_coverage_weight,
}

# Define parameters - permille
permilles = [10,30,50,70,90,100,300,500,700,900]  # Can be adjusted as needed
alpha_values = [0.1,0.2,0.3,0.4,0.5]  # Can be adjusted as needed

# =========================== Data Loading Functions ===========================

def load_base_vectors():
    """Load base vectors data (detection vectors)"""
    base_vectors = {}
    base_path = SAGS_ROOT / "SFD" / "generated" / MODEL_NAME / "detection_vector"
    
    for content_type in content_types:
        file_path = base_path / f"last_token_{content_type}.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                base_vectors[content_type] = json.load(f)
            print(f"Successfully loaded {content_type} base vector")
        except FileNotFoundError:
            print(f"Warning: File not found {file_path}")
        except Exception as e:
            print(f"Error: Failed to load {content_type} base vector: {e}")
    
    return base_vectors

def load_test_data(data_version):
    """Load test data
    
    Args:
        data_version: Dataset version, e.g., "train" or "if_test"
    """
    test_data = {}
    base_path = SAGS_ROOT / "SFD" / "generated" / MODEL_NAME / SAE_ID_SAFE / data_version
    
    for content_type in content_types:
        file_path = base_path / f"{content_type}.jsonl"
        test_data[content_type] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        # Add label information to each sample
                        data['label'] = content_type
                        test_data[content_type].append(data)
            print(f"Successfully loaded {content_type} data: {len(test_data[content_type])} samples")
        except FileNotFoundError:
            print(f"Warning: File not found {file_path}")
        except Exception as e:
            print(f"Error: Failed to load {content_type} data: {e}")
    
    return test_data

# =========================== Accuracy and F1 Calculation ===========================

def calculate_f1_scores(y_true, y_pred, content_types):
    """Calculate F1 Score for each class and average F1 Score"""
    # Calculate precision, recall and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=content_types, average=None, zero_division=0
    )
    
    # Calculate macro average F1 score
    macro_f1 = f1_score(y_true, y_pred, labels=content_types, average='macro', zero_division=0)
    
    # Calculate weighted average F1 score
    weighted_f1 = f1_score(y_true, y_pred, labels=content_types, average='weighted', zero_division=0)
    
    # Build result dictionary
    class_f1_scores = {}
    for i, content_type in enumerate(content_types):
        class_f1_scores[content_type] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    return {
        'class_f1_scores': class_f1_scores,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1)
    }

def calculate_multiclass_accuracy(base_vectors, test_data, permille, alpha, similarity_func):
    """Calculate multi-class accuracy and F1 Score"""
    # Handle different permille types
    if permille == "mean":
        permille_key = "mean_permille"
    else:
        permille_key = f"{permille}_permille"
    
    # Get all base vectors
    base_vector_dict = {}
    
    for content_type in content_types:
        if content_type in base_vectors and permille_key in base_vectors[content_type]:
            base_vector_dict[content_type] = {
                "indices": base_vectors[content_type][permille_key]["indices"],
                "values": base_vectors[content_type][permille_key]["mean_values"]
            }
    
    # Return error if base vectors are insufficient
    if len(base_vector_dict) < len(content_types):
        missing_types = set(content_types) - set(base_vector_dict.keys())
        raise ValueError(f"Missing base vector data: {missing_types}")
    
    # Count classification results
    total_correct = 0
    total_samples = 0
    total_computation_time = 0.0
    
    # Lists of true labels and predicted labels for F1 calculation
    y_true = []
    y_pred = []
    
    # Classification statistics - count by true label and predicted label
    classification_stats = {}
    for content_type in content_types:
        classification_stats[content_type] = {
            "total": 0,
            "correct": 0,
            "predicted_as": {ct: 0 for ct in content_types}
        }
    
    # Traverse all test data
    for true_label, test_samples in test_data.items():
        if true_label not in content_types:
            continue
            
        classification_stats[true_label]["total"] = len(test_samples)
        
        for test_sample in test_samples:
            test_vector = {
                "indices": test_sample["indices"],
                "values": test_sample["values"]
            }
            
            # Calculate similarity with all base vectors
            similarities = {}
            start_time = time.time()
            
            for base_type, base_vector in base_vector_dict.items():
                similarity = similarity_func(
                    base_vector["indices"], base_vector["values"],
                    test_vector["indices"], test_vector["values"], alpha
                )
                similarities[base_type] = similarity
            
            end_time = time.time()
            total_computation_time += (end_time - start_time)
            
            # Find base vector with highest similarity as prediction
            predicted_label = max(similarities, key=similarities.get)
            
            # Count results
            classification_stats[true_label]["predicted_as"][predicted_label] += 1
            y_true.append(true_label)
            y_pred.append(predicted_label)
            
            if predicted_label == true_label:
                classification_stats[true_label]["correct"] += 1
                total_correct += 1
            
            total_samples += 1
    
    # Calculate overall accuracy and average computation time
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_computation_time = total_computation_time / total_samples if total_samples > 0 else 0.0
    
    # Calculate accuracy for each class
    class_accuracies = {}
    for content_type in content_types:
        if classification_stats[content_type]["total"] > 0:
            class_accuracies[content_type] = classification_stats[content_type]["correct"] / classification_stats[content_type]["total"]
        else:
            class_accuracies[content_type] = 0.0
    
    # Calculate F1 Score
    f1_results = calculate_f1_scores(y_true, y_pred, content_types)
    
    return {
        "overall_accuracy": float(overall_accuracy),
        "total_correct": int(total_correct),
        "total_samples": int(total_samples),
        "avg_computation_time": float(avg_computation_time),
        "class_accuracies": {k: float(v) for k, v in class_accuracies.items()},
        "classification_stats": classification_stats,
        "f1_scores": f1_results
    }

# =========================== Main Function ===========================

def main():
    # ============================================================================
    # Phase 1: Find best hyperparameters on training set
    # ============================================================================
    print("="*80)
    print("Phase 1: Finding best hyperparameters on training set")
    print("="*80)
    
    # Load base vector data
    print("\nLoading base vectors...")
    base_vectors = load_base_vectors()
    
    # Load training data
    print(f"\nLoading training data ({TRAIN_VERSION})...")
    train_data = load_test_data(TRAIN_VERSION)
    
    # Count training data
    total_train_samples = sum(len(samples) for samples in train_data.values())
    print(f"\nTraining set sample statistics:")
    print(f"  Total samples: {total_train_samples}")
    for content_type, samples in train_data.items():
        print(f"  {content_type}: {len(samples)} samples")
    
    # Store all results from training phase
    train_results = {}
    global_best_accuracy = 0
    global_best_params = None
    
    print(f"\n{'='*80}")
    print("Starting parameter search...")
    print(f"Number of algorithms: {len(SIMILARITY_ALGORITHMS)}")
    print(f"Number of classes: {len(content_types)}")
    print(f"Permille combinations: {len(permilles)}")
    print(f"Alpha values: {len(alpha_values)}")
    print(f"Total calculations: {len(SIMILARITY_ALGORITHMS) * len(permilles) * len(alpha_values)}")
    print(f"{'='*80}")
    
    # Traverse all similarity algorithms
    for algo_name, similarity_func in SIMILARITY_ALGORITHMS.items():
        print(f"\n🔍 Testing algorithm: {algo_name}")
        print(f"{'='*60}")
        
        algo_results = {}
        algo_best_accuracy = 0
        algo_best_params = None
        algo_total_time = 0.0
        algo_total_samples = 0
        
        # Traverse all permille and alpha combinations
        for permille in permilles:
            print(f"\n  Processing {permille}‰ base vectors...")
            
            permille_results = {}
            
            for alpha in alpha_values:
                try:
                    # Calculate accuracy on training set
                    accuracy_result = calculate_multiclass_accuracy(
                        base_vectors, train_data, permille, alpha, similarity_func
                    )
                    
                    # Accumulate algorithm time statistics
                    algo_total_time += accuracy_result["avg_computation_time"] * accuracy_result["total_samples"]
                    algo_total_samples += accuracy_result["total_samples"]
                    
                    # Save results
                    alpha_key = f"alpha_{alpha:.2f}"
                    permille_results[alpha_key] = accuracy_result
                    
                    # Update best parameters within algorithm
                    if accuracy_result["overall_accuracy"] > algo_best_accuracy:
                        algo_best_accuracy = accuracy_result["overall_accuracy"]
                        algo_best_params = (permille, alpha)
                    
                    # Update global best parameters
                    if accuracy_result["overall_accuracy"] > global_best_accuracy:
                        global_best_accuracy = accuracy_result["overall_accuracy"]
                        global_best_params = (algo_name, permille, alpha)
                    
                    # Print results
                    f1_macro = accuracy_result['f1_scores']['macro_f1']
                    print(f"    {permille}‰ + α={alpha}: "
                          f"Accuracy {accuracy_result['overall_accuracy']:.4f} "
                          f"F1(macro) {f1_macro:.4f} "
                          f"({accuracy_result['total_correct']}/{accuracy_result['total_samples']})")
                
                except Exception as e:
                    print(f"    Error {permille}‰ + α={alpha}: {str(e)}")
                    alpha_key = f"alpha_{alpha:.2f}"
                    permille_results[alpha_key] = {
                        "error": str(e),
                        "overall_accuracy": 0.0
                    }
            
            # Save current permille results
            if permille == "mean":
                permille_key = "mean_permille"
            else:
                permille_key = f"{permille}_permille"
            algo_results[permille_key] = permille_results
        
        # Calculate algorithm's average computation time
        algo_avg_time = algo_total_time / algo_total_samples if algo_total_samples > 0 else 0.0
        
        # Print algorithm summary
        if algo_best_params:
            best_permille, best_alpha = algo_best_params
            if best_permille == "mean":
                best_permille_key = "mean_permille"
            else:
                best_permille_key = f"{best_permille}_permille"
            best_result = algo_results[best_permille_key][f"alpha_{best_alpha:.2f}"]
            
            print(f"\n  📊 {algo_name} algorithm best results (training set):")
            print(f"     Permille: {best_permille}‰, Alpha: {best_alpha}")
            print(f"     Overall accuracy: {algo_best_accuracy:.4f} ({best_result['total_correct']}/{best_result['total_samples']})")
            if 'f1_scores' in best_result:
                print(f"     Macro F1: {best_result['f1_scores']['macro_f1']:.4f}")
                print(f"     Weighted F1: {best_result['f1_scores']['weighted_f1']:.4f}")
        
        # Save algorithm results
        train_results[algo_name] = {
            "results": algo_results,
            "best_accuracy": algo_best_accuracy,
            "best_params": algo_best_params,
            "avg_computation_time": algo_avg_time
        }
    
    # Save training phase results
    output_dir = SAGS_ROOT / "SFD" / "generated" / MODEL_NAME / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output_file = output_dir / f"classification_results_train.json"
    
    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("🎯 Training phase completed!")
    print(f"Training results saved to: {train_output_file}")
    print(f"{'='*80}")
    
    # Print best results on training set
    if global_best_params:
        best_algo, best_permille, best_alpha = global_best_params
        print(f"\n🏆 Best hyperparameters on training set:")
        print(f"   Algorithm: {best_algo}")
        print(f"   Permille: {best_permille}‰")
        print(f"   Alpha: {best_alpha}")
        print(f"   Training accuracy: {global_best_accuracy:.4f}")
    
    # ============================================================================
    # Phase 2: Evaluate on test set using best hyperparameters
    # ============================================================================
    if not global_best_params:
        print("\nError: No valid best parameters found, cannot evaluate on test set")
        return
    
    print(f"\n{'='*80}")
    print(f"Phase 2: Evaluating on test set ({TEST_VERSION}) using best hyperparameters")
    print(f"{'='*80}")
    
    best_algo, best_permille, best_alpha = global_best_params
    best_similarity_func = SIMILARITY_ALGORITHMS[best_algo]
    
    # Load test data
    print(f"\nLoading test data ({TEST_VERSION})...")
    test_data = load_test_data(TEST_VERSION)
    
    # Count test data
    total_test_samples = sum(len(samples) for samples in test_data.values())
    print(f"\nTest set sample statistics:")
    print(f"  Total samples: {total_test_samples}")
    for content_type, samples in test_data.items():
        print(f"  {content_type}: {len(samples)} samples")
    
    # Evaluate on test set using best parameters
    print(f"\nEvaluating using best parameters:")
    print(f"  Algorithm: {best_algo}")
    print(f"  Permille: {best_permille}‰")
    print(f"  Alpha: {best_alpha}")
    
    try:
        test_result = calculate_multiclass_accuracy(
            base_vectors, test_data, best_permille, best_alpha, best_similarity_func
        )
        
        # Save test results
        test_output = {
            "best_params": {
                "algorithm": best_algo,
                "permille": best_permille,
                "alpha": best_alpha
            },
            "train_accuracy": global_best_accuracy,
            "test_result": test_result
        }
        
        test_output_file = output_dir / f"classification_results_test_{TEST_VERSION}.json"
        with open(test_output_file, 'w', encoding='utf-8') as f:
            json.dump(test_output, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print("✅ Test set evaluation completed!")
        print(f"{'='*80}")
        
        # Print final test results
        print(f"\n📊 Final test set results:")
        print(f"   Overall accuracy: {test_result['overall_accuracy']:.4f} ({test_result['total_correct']}/{test_result['total_samples']})")
        print(f"   Macro F1: {test_result['f1_scores']['macro_f1']:.4f}")
        print(f"   Weighted F1: {test_result['f1_scores']['weighted_f1']:.4f}")
        print(f"   Average computation time: {test_result['avg_computation_time']:.6f}s/sample")
        
        # Print detailed results for each class
        print(f"\n   Per-class results:")
        for content_type in content_types:
            if content_type in test_result['class_accuracies']:
                class_acc = test_result['class_accuracies'][content_type]
                class_stats = test_result['classification_stats'][content_type]
                f1_info = ""
                precision_recall_info = ""
                if content_type in test_result['f1_scores']['class_f1_scores']:
                    class_f1_data = test_result['f1_scores']['class_f1_scores'][content_type]
                    f1_info = f" F1: {class_f1_data['f1_score']:.4f}"
                    precision_recall_info = f" (P: {class_f1_data['precision']:.4f}, R: {class_f1_data['recall']:.4f})"
                print(f"     {content_type}: Accuracy {class_acc:.4f}{f1_info}{precision_recall_info} ({class_stats['correct']}/{class_stats['total']})")
        
        print(f"\nTest results saved to: {test_output_file}")
        
    except Exception as e:
        print(f"\nError: Failed to evaluate on test set: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()