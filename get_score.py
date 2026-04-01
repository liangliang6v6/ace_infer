import json
import re
import string
from collections import Counter
import argparse

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculates token-level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    """Checks if the normalized strings match exactly."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def accuracy_score(prediction, ground_truth):
    """Checks if the normalized ground truth is contained within the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediction)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Evaluates against all possible valid answers and takes the highest score."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(file_path):
    f1 = exact_match = accuracy = total = 0
    print(f"Evaluating: {file_path}\n" + "-"*40)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 1. Extract the prediction using the <answer> tags
            final_answer_raw = data.get("final_answer", "")
            match = re.search(r"<answer>(.*?)</answer>", final_answer_raw, re.IGNORECASE | re.DOTALL)
            
            if match:
                prediction = match.group(1).strip()
            else:
                # Fallback if the model forgot the tags
                prediction = final_answer_raw.strip()

            # 2. Get the ground truth list
            ground_truths = data.get("answer", [])
            if not isinstance(ground_truths, list):
                ground_truths = [ground_truths]
                
            # 3. Calculate max metrics for this specific question
            exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
            accuracy += metric_max_over_ground_truths(accuracy_score, prediction, ground_truths)
            total += 1

    # 4. Aggregate final percentages
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    accuracy = 100.0 * accuracy / total
    
    print(f"Total Questions Evaluated: {total}")
    print(f"Accuracy (Contains)      : {accuracy:.2f}%")
    print(f"Exact Match (EM)         : {exact_match:.2f}%")
    print(f"F1 Score                 : {f1:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate AceSearcher output JSONL against ground truth.")
    parser.add_argument(
        '--file',
        type=str,
        default="eval_datasets/test/hotpotqa/prompts_decompose_test_qwen3_baseline_test/test_e5-large_k10_passage1.jsonl",
        help='Path to the generated JSONL file'
    )
    args = parser.parse_args()
    evaluate(args.file)