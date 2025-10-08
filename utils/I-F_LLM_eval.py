import os
from openai import OpenAI
import openai
import requests
import time
import json
from tqdm import tqdm
import re
import statistics

# --- Configuration ---
API_SECRET_KEY = ""
BASE_URL = ""



instruction_list = {
    "capital": "Candidate must be all uppercase letters, no lowercase allowed. (Uppercase only)",
    "lowercase": "Candidate must be all lowercase letters, no uppercase allowed. (Lowercase only)",
    "format_json": "Candidate must be in JSON format; any content outside JSON is invalid. (Strict JSON format)",
    "format_constrained": "Candidate must contain one of: 'My answer is yes.', 'My answer is no.', 'My answer is maybe.'. (Constrained output)",
    "format_title": "Candidate must include a title in double angular brackets, e.g., <<title>>. (Title in << >>)",
    "repeat": "Candidate must first repeat the query ({query}), then provide the response. (Repeat the query)",
    "two_responses": "Candidate responses must be separated by ****** (six asterisks). (Separated by ******)",
    "no_comma": "Candidate must not contain any commas. (No commas)",
    "quotation": "Candidate must be wrapped in double quotes. (Wrapped in double quotes)",
}
# Create mapping relationship from instruction_id to instruction_list
instruction_id_mapping = {
    "change_case:english_capital": "capital",
    "change_case:english_lowercase": "lowercase", 
    "combination:repeat_prompt": "repeat",
    "combination:two_responses": "two_responses",
    "detectable_format:constrained_response": "format_constrained",
    "detectable_format:json_format": "format_json",
    "detectable_format:title": "format_title",
    "punctuation:no_comma": "no_comma",
    "startend:quotation": "quotation",
}




# Initialize the OpenAI client
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

def evaluate_response(query_a, instruction_b, candidate_answer):
    """
    Calls the OpenAI API to evaluate a candidate answer based on a query and instructions.

    Args:
        query_a (str): The main user request.
        instruction_b (str): An additional constraint or requirement.
        candidate_answer (str): The text to be evaluated.

    Returns:
        dict: A dictionary with the evaluation scores, or None if an error occurs.
    """
    
    # If candidate_answer is empty, directly return 0 score
    if not candidate_answer or candidate_answer.strip() == "":
        return {
            "answer_relevance": 0.0,
            "instruction_following": 0.0,
            "sentence_quality": 0.0
        }
    
    # Prompt formatted as a Python f-string to easily insert variables
    prompt_template = f"""
You are acting as a fair and balanced evaluator (LLM-as-a-Judge).
You will receive three inputs:
- Query: The user's main request.
- Instruction: An additional requirement or constraint.
- Candidate: The response to be evaluated.

Assign three separate scores, each strictly one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
Return **only valid JSON**, no extra text or formatting.

IMPORTANT:
- Always evaluate the Candidate based on the actual Instruction provided.
- Avoid giving 0.0 unless the response truly fails to meet the minimum expectations.
- Avoid giving 1.0 unless the response perfectly meets all criteria.
- Partial credit is allowed, but do not inflate scores for clearly inadequate responses.
- Ignore examples in the guidelines; use them only as neutral references for scoring.

1. **answer_relevance** – How well the Candidate addresses the Query:
   - 0.0: Addresses almost none of the Query, completely off-topic
   - 0.2: Barely addresses the Query, mostly off-topic
   - 0.4: Partially relevant, touches on some aspects but misses key points
   - 0.6: Addresses main aspects, but incomplete or lacking depth
   - 0.8: Addresses most aspects with minor gaps
   - 1.0: Fully and comprehensively addresses the Query

2. **instruction_following** – How well Candidate follows the Instruction (content + format):
   - 0.0: Fails to follow most or all of the Instruction
   - 0.2: Follows very little, with major violations
   - 0.4: Partially follows, significant issues remain
   - 0.6: Mostly follows, some notable issues
   - 0.8: Follows the instruction well, minor deviations
   - 1.0: Perfectly complies with all Instruction requirements
   - Note: For formatting or structural requirements, approximate compliance counts:
       * Mostly correct → high score (0.8–1.0)
       * Partially correct → medium score (0.4–0.6)
       * Mostly incorrect → low score (0.0–0.2)
   - Always evaluate based strictly on the current Instruction; partial credit should be given for attempts that demonstrate understanding, even if execution is not perfect.

3. **sentence_quality** – Evaluate clarity, grammar, and coherence:
   - 0.0: Incomprehensible, repetitive gibberish, or completely broken
   - 0.2: Very poor quality, repetitive content, mostly unclear
   - 0.4: Poor grammar and clarity, difficult to understand
   - 0.6: Understandable but with notable grammar or coherence issues
   - 0.8: Generally clear and well-structured with minor errors
   - 1.0: Excellent language quality, clear and well-structured
   - Note: Do not penalize sentence_quality for format symbols, braces, quotes, or other formatting required by the Instruction.

JSON format:
{{
  "answer_relevance": <score>,
  "instruction_following": <score>,
  "sentence_quality": <score>
}}

---
**INPUTS FOR EVALUATION**

Query: {query_a}  
Instruction: {instruction_b}  
Candidate: {candidate_answer}
"""

    # print("Sending request to the API...")  # Commented out to avoid conflict with tqdm
    try:
        # Generate content using OpenAI chat completions
        response = client.chat.completions.create(
            model="gpt-4o",
            messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a fair and balanced evaluator (LLM-as-a-Judge). "
                                "Evaluate Candidate answers strictly based on the provided Query and Instruction. "
                                "Do not consider any external examples; always prioritize the current Instruction. "
                                "Return only valid JSON with no extra text or explanation. "
                                "Give partial credit for responses that show understanding or attempt to follow the Instruction. "
                                "For formatting or structural requirements, approximate compliance should guide the instruction_following score: "
                                "mostly correct → high score (0.8–1.0), partially correct → medium score (0.4–0.6), mostly incorrect → low score (0.0–0.2)."
                            )
                        },
                        {"role": "user", "content": prompt_template}
                    ],
            temperature=0
        )
        
        # Get the response content
        response_content = response.choices[0].message.content
        
        # Clean the response to ensure it's valid JSON
        # Models sometimes wrap JSON in markdown backticks (```json ... ```)
        cleaned_response = response_content.strip().replace("```json", "").replace("```", "")
        
        # Parse the JSON string into a Python dictionary
        scores = json.loads(cleaned_response)
        return scores

    except json.JSONDecodeError:
        print("❌ Error: Failed to decode JSON from the API response.")
        print(f"Raw Response: {response_content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def extract_model_name(input_file_path):
    """
    Extract model name from input file path
    
    Args:
        input_file_path (str): Input file path
        
    Returns:
        str: Model name, e.g. 'gemma-2-2b'
    """
    # Extract model name from filename
    filename = os.path.basename(input_file_path)
    
    # Match input_response_data_<model_name>_orginal.jsonl format
    match = re.search(r'input_response_data_(.+?)_orginal\.jsonl', filename)
    if match:
        return match.group(1)
    
    # If not matched, try other possible formats
    match = re.search(r'input_response_data_(.+?)\.jsonl', filename)
    if match:
        return match.group(1)
    
    # If still not matched, use filename with extension removed as model name
    model_name = filename.replace('.jsonl', '').replace('input_response_data_', '').replace('_orginal', '')
    
    return model_name if model_name else 'unknown_model'


def get_processed_keys_and_stats(output_file_path, num_evaluations=3):
    """
    Get list of processed keys and statistics data for resumption
    
    Args:
        output_file_path (str): Output file path
        num_evaluations (int): Number of evaluations
        
    Returns:
        tuple: (processed_keys_set, running_stats_dict, processed_count)
    """
    processed_keys = set()
    running_stats = {
        'answer_relevance': [],
        'instruction_following': [],
        'sentence_quality': []
    }
    processed_count = 0
    
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        if 'key' in data:
                            processed_keys.add(data['key'])
                            processed_count += 1
                            
                            # Restore statistics data
                            for i in range(1, num_evaluations + 1):
                                score_key = f'score_{i}'
                                if score_key in data and data[score_key] is not None:
                                    score_data = data[score_key]
                                    for metric in ['answer_relevance', 'instruction_following', 'sentence_quality']:
                                        if metric in score_data and score_data[metric] is not None:
                                            running_stats[metric].append(float(score_data[metric]))
        except Exception as e:
            print(f"⚠️ Error reading existing output file: {e}")
    
    return processed_keys, running_stats, processed_count


def get_processed_keys(output_file_path):
    """
    Get list of processed keys for resumption (compatibility function)
    
    Args:
        output_file_path (str): Output file path
        
    Returns:
        set: Set of processed keys
    """
    processed_keys, _, _ = get_processed_keys_and_stats(output_file_path)
    return processed_keys


def evaluate_single_item_multiple_times(query_a, instruction_b, candidate_answer, num_evaluations=3):
    """
    Perform multiple evaluations on a single data item
    
    Args:
        query_a (str): Main user request
        instruction_b (str): Additional constraint or requirement
        candidate_answer (str): Text to be evaluated
        num_evaluations (int): Number of evaluations, default 3
        
    Returns:
        dict: Dictionary containing multiple evaluation results, format: {"score_1": {...}, "score_2": {...}, "score_3": {...}}
    """
    scores = {}
    
    for i in range(1, num_evaluations + 1):
        tqdm.write(f"      🔄 Evaluation {i}...")
        evaluation_score = evaluate_response(query_a, instruction_b, candidate_answer)
        
        if evaluation_score:
            scores[f"score_{i}"] = evaluation_score
            tqdm.write(f"      ✅ Evaluation {i} completed: {json.dumps(evaluation_score, ensure_ascii=False)}")
        else:
            tqdm.write(f"      ❌ Evaluation {i} failed")
            scores[f"score_{i}"] = None
        
        # Evaluation interval to avoid API pressure
        if i < num_evaluations:
            time.sleep(1)  # Brief interval
    
    return scores


def update_running_stats(running_stats, evaluation_results, num_evaluations):
    """
    Update real-time statistics data
    
    Args:
        running_stats (dict): Current statistics data
        evaluation_results (dict): New evaluation results, containing score_1, score_2, score_3, etc.
        num_evaluations (int): Number of evaluations
    """
    for i in range(1, num_evaluations + 1):
        score_key = f'score_{i}'
        if score_key in evaluation_results and evaluation_results[score_key] is not None:
            score_data = evaluation_results[score_key]
            for metric in ['answer_relevance', 'instruction_following', 'sentence_quality']:
                if metric in score_data and score_data[metric] is not None:
                    running_stats[metric].append(float(score_data[metric]))


def get_current_averages(running_stats):
    """
    Calculate current average scores
    
    Args:
        running_stats (dict): Current statistics data
        
    Returns:
        dict: Current average scores
    """
    averages = {}
    for metric in ['answer_relevance', 'instruction_following', 'sentence_quality']:
        if running_stats[metric]:
            averages[metric] = round(statistics.mean(running_stats[metric]), 3)
        else:
            averages[metric] = 0.0
    return averages


def format_stats_for_tqdm(averages, processed_count):
    """
    Format statistics information for tqdm display
    
    Args:
        averages (dict): Current average scores
        processed_count (int): Number processed
        
    Returns:
        str: Formatted statistics information
    """
    if processed_count == 0:
        return "Waiting to process..."
    
    ar = averages['answer_relevance']
    if_score = averages['instruction_following'] 
    sq = averages['sentence_quality']
    
    return f"Avg - AR:{ar:.3f} IF:{if_score:.3f} SQ:{sq:.3f} (n={processed_count})"


def process_input_file(input_file_path, output_file_path, num_evaluations=3):
    """
    Process input JSONL file, perform multiple API evaluations for each data item, and save results one by one
    Supports resumption, skips already processed data
    
    Args:
        input_file_path (str): Input JSONL file path
        output_file_path (str): Output JSONL file path
        num_evaluations (int): Number of evaluations per data item, default 3
    """
    # Get processed key list and restore statistics data
    processed_keys, running_stats, initial_processed_count = get_processed_keys_and_stats(output_file_path, num_evaluations)
    if processed_keys:
        print(f"🔄 Detected {len(processed_keys)} processed data items, will skip these and continue processing")
        print(f"📊 Statistics data restored, current sample count: AR={len(running_stats['answer_relevance'])}, "
              f"IF={len(running_stats['instruction_following'])}, SQ={len(running_stats['sentence_quality'])}")
    
    # Read all lines first to calculate total count
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Calculate number of data items to process
    total_lines = len(lines)
    skipped_count = 0
    processed_count = initial_processed_count  # Start counting from already processed data
    error_count = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Open output file in append mode
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        # Calculate initial average scores for tqdm display
        if initial_processed_count > 0:
            initial_averages = get_current_averages(running_stats)
            initial_stats_display = format_stats_for_tqdm(initial_averages, initial_processed_count)
        else:
            initial_stats_display = "Waiting to process..."
        
        # Create tqdm progress bar with initial statistics
        pbar = tqdm(lines, desc=f"Processing data | {initial_stats_display}")
        
        # Use tqdm to display progress bar
        for line_num, line in enumerate(pbar, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract necessary fields
                key = data.get('key')
                query_a = data.get('query')
                candidate_answer = data.get('response')
                instruction_id_list = data.get('instruction_id_list', [])
                
                # Check if already processed
                if key and key in processed_keys:
                    skipped_count += 1
                    tqdm.write(f"⏭️ Line {line_num} (key: {key}) already processed, skipping")
                    continue
                
                if not query_a or not instruction_id_list:
                    error_count += 1
                    tqdm.write(f"❌ Line {line_num} data missing required fields, skipping")
                    continue
                
                # Get instruction_b corresponding to the first instruction_id
                instruction_id = instruction_id_list[0]  # Take the first one
                
                if instruction_id not in instruction_id_mapping:
                    error_count += 1
                    tqdm.write(f"❌ Line {line_num}: instruction_id '{instruction_id}' mapping not found, skipping")
                    continue
                
                instruction_key = instruction_id_mapping[instruction_id]
                instruction_b = instruction_list.get(instruction_key)
                
                if not instruction_b:
                    error_count += 1
                    tqdm.write(f"❌ Line {line_num}: instruction_key '{instruction_key}' instruction not found, skipping")
                    continue
                
                tqdm.write(f"📊 Processing line {line_num} (key: {key})")
                tqdm.write(f"   instruction_id: {instruction_id}")
                tqdm.write(f"   instruction_b: {instruction_b}")
                
                # Check if response is empty, if so directly give 0 score and skip API call
                if not candidate_answer or candidate_answer.strip() == "":
                    tqdm.write(f"   ⚠️ Response is empty, directly giving 0 score")
                    # Create 0 score result
                    zero_score = {
                        "answer_relevance": 0.0,
                        "instruction_following": 0.0,
                        "sentence_quality": 0.0
                    }
                    multiple_scores = {}
                    for i in range(1, num_evaluations + 1):
                        multiple_scores[f"score_{i}"] = zero_score
                else:
                    # Perform multiple evaluations
                    multiple_scores = evaluate_single_item_multiple_times(query_a, instruction_b, candidate_answer, num_evaluations)
                
                # Check if at least one evaluation succeeded
                successful_evaluations = sum(1 for score in multiple_scores.values() if score is not None)
                
                if successful_evaluations > 0:
                    # Build output result containing multiple evaluation scores
                    result = {
                        "key": key,
                        "prompt": data.get('prompt', ''),
                        "response": candidate_answer,
                        "instruction_id_list": instruction_id_list,
                        **multiple_scores  # Expand multiple evaluation results: score_1, score_2, score_3
                    }
                    
                    # Write to file immediately
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_file.flush()  # Ensure immediate write to disk
                    
                    # Update real-time statistics
                    update_running_stats(running_stats, multiple_scores, num_evaluations)
                    processed_count += 1
                    
                    # Calculate current average scores
                    current_averages = get_current_averages(running_stats)
                    stats_display = format_stats_for_tqdm(current_averages, processed_count)
                    
                    # Update tqdm description to display real-time average
                    pbar.set_description(f"Processing data | {stats_display}")
                    
                    tqdm.write(f"✅ Line {line_num} evaluation completed and saved!")
                    tqdm.write(f"   Successful evaluations: {successful_evaluations}/{num_evaluations}")
                    tqdm.write(f"   Current average: AR={current_averages['answer_relevance']:.3f} "
                              f"IF={current_averages['instruction_following']:.3f} "
                              f"SQ={current_averages['sentence_quality']:.3f}")
                else:
                    error_count += 1
                    tqdm.write(f"❌ Line {line_num} all evaluations failed")
                
                # Add delay to avoid API limits
                time.sleep(0.5)
                
            except json.JSONDecodeError as e:
                error_count += 1
                tqdm.write(f"❌ Line {line_num} JSON parsing error: {e}")
                continue
            except Exception as e:
                error_count += 1
                tqdm.write(f"❌ Line {line_num} processing error: {e}")
                continue
    
    # Output processing statistics
    print(f"\n📊 Processing completion statistics:")
    print(f"   Total lines: {total_lines}")
    print(f"   Skipped (already processed): {skipped_count}")
    print(f"   Newly processed successfully: {processed_count}")
    print(f"   Errors/Skipped: {error_count}")
    print(f"✅ Results saved to: {output_file_path}")


def calculate_average_scores(input_file_path, output_file_path, num_evaluations=3):
    """
    Calculate average scores of evaluation results and save as JSON file
    
    Args:
        input_file_path (str): Input JSONL file path (containing evaluation results)
        output_file_path (str): Output JSON file path
        num_evaluations (int): Number of evaluations, default 3
    """
    # Store all scores
    all_scores = {
        'answer_relevance': [],
        'instruction_following': [],
        'sentence_quality': []
    }
    
    # Store scores grouped by score
    score_groups = {}
    for i in range(1, num_evaluations + 1):
        score_groups[f'score_{i}'] = {
            'answer_relevance': [],
            'instruction_following': [],
            'sentence_quality': []
        }
    
    print(f"📊 Starting to calculate average scores...")
    
    # Read evaluation results file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Process each score_x field
                for i in range(1, num_evaluations + 1):
                    score_key = f'score_{i}'
                    if score_key in data and data[score_key] is not None:
                        score_data = data[score_key]
                        
                        # Extract scores of three metrics
                        for metric in ['answer_relevance', 'instruction_following', 'sentence_quality']:
                            if metric in score_data and score_data[metric] is not None:
                                score_value = float(score_data[metric])
                                # Add to overall scores
                                all_scores[metric].append(score_value)
                                # Add to corresponding score group
                                score_groups[score_key][metric].append(score_value)
                                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"⚠️ Line {line_num} data parsing error: {e}")
                continue
            except Exception as e:
                print(f"❌ Line {line_num} processing error: {e}")
                continue
    
    # Calculate average scores
    result = {}
    
    # Calculate overall average scores
    result["mean of all"] = {}
    for metric in ['answer_relevance', 'instruction_following', 'sentence_quality']:
        if all_scores[metric]:
            result["mean of all"][metric] = round(statistics.mean(all_scores[metric]), 4)
        else:
            result["mean of all"][metric] = 0.0
    
    # Calculate average score for each score
    for i in range(1, num_evaluations + 1):
        score_key = f'score_{i}'
        result[f"mean of {score_key}"] = {}
        for metric in ['answer_relevance', 'instruction_following', 'sentence_quality']:
            if score_groups[score_key][metric]:
                result[f"mean of {score_key}"][metric] = round(statistics.mean(score_groups[score_key][metric]), 4)
            else:
                result[f"mean of {score_key}"][metric] = 0.0
    
    # Save results to JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"📈 Average score calculation completed!")
    print(f"📊 Statistics:")
    print(f"   Overall sample count: answer_relevance={len(all_scores['answer_relevance'])}, "
          f"instruction_following={len(all_scores['instruction_following'])}, "
          f"sentence_quality={len(all_scores['sentence_quality'])}")
    
    for i in range(1, num_evaluations + 1):
        score_key = f'score_{i}'
        print(f"   {score_key} sample count: answer_relevance={len(score_groups[score_key]['answer_relevance'])}, "
              f"instruction_following={len(score_groups[score_key]['instruction_following'])}, "
              f"sentence_quality={len(score_groups[score_key]['sentence_quality'])}")
    
    print(f"💾 Average score results saved to: {output_file_path}")
    
    # Display results preview
    print(f"\n📋 Results preview:")
    for key, value in result.items():
        print(f"   {key}:")
        for metric, score in value.items():
            print(f"     {metric}: {score}")
    
    return result


def run_evaluation_with_multiple_scores(input_file_path, base_output_dir, num_evaluations=3):
    """
    Run evaluation, perform multiple evaluations for each data item and save in same file, then calculate average scores
    
    Args:
        input_file_path (str): Input file path
        base_output_dir (str): Base output directory
        num_evaluations (int): Number of evaluations per data item, default 3
    """
    # Extract model name
    model_name = extract_model_name(input_file_path)
    print(f"🤖 Detected model: {model_name}")
    
    # Create model-specific directory
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Build output file paths
    output_file = os.path.join(model_output_dir, "multiple_eval_results.jsonl")
    average_file = os.path.join(model_output_dir, "multiple_eval_results_average_analysis.json")
    
    print(f"📁 Output directory: {model_output_dir}")
    print(f"🔄 Each data item will undergo {num_evaluations} independent evaluations")
    print(f"📝 Input file: {input_file_path}")
    print(f"💾 Output file: {output_file}")
    print(f"📊 Average score file: {average_file}")
    print(f"📋 Output format: Each record contains score_1, score_2, score_3 fields")
    
    print(f"\n" + "="*60)
    print(f"🚀 Starting multiple evaluation processing")
    print("="*60)
    
    try:
        # Run evaluation
        process_input_file(input_file_path, output_file, num_evaluations)
        print(f"✅ Multiple evaluations completed!")
        
        # Calculate average scores
        print(f"\n" + "="*60)
        print(f"📊 Starting to calculate average scores")
        print("="*60)
        calculate_average_scores(output_file, average_file, num_evaluations)
        
    except Exception as e:
        print(f"❌ Evaluation process error: {e}")
        return
    
    print(f"\n" + "="*60)
    print(f"🎉 All evaluations completed!")
    print(f"📊 Results saved in: {output_file}")
    print(f"📈 Average scores saved in: {average_file}")
    print(f"📋 Each data item contains {num_evaluations} independent evaluation results")
    print("="*60)


# --- Main Execution ---
if __name__ == "__main__":
    # Set input/output file paths
    input_file = ""  # TODO: Set your input file path here
    base_output_dir = ""  # TODO: Set your output directory path here
    
    if input_file and base_output_dir:
        # Run multiple evaluations (evaluate each data item 3 times, save in same file)
        run_evaluation_with_multiple_scores(input_file, base_output_dir, num_evaluations=3)
    else:
        print("⚠️ Warning: Please set input_file and base_output_dir paths before running.")
