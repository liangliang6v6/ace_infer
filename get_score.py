import json
import re
import string
from collections import Counter
import argparse
from pathlib import Path


def normalize_answer(s):
    if s is None:
        return ""

    s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def prf_score(prediction, ground_truth):
    """Return precision, recall, f1."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def accuracy_score(prediction, ground_truth):
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return float(gt_norm in pred_norm)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def prf_max_over_ground_truths(prediction, ground_truths):
    best = (0.0, 0.0, 0.0)
    for gt in ground_truths:
        p, r, f1 = prf_score(prediction, gt)
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def extract_prediction(final_answer_raw):
    if final_answer_raw is None:
        return ""

    final_answer_raw = str(final_answer_raw)
    match = re.search(r"<answer>(.*?)</answer>", final_answer_raw, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()
    return final_answer_raw.strip()


def build_id_fields(data, idx):
    question_id = data.get("question_id", data.get("id", idx))
    source_index = data.get("source_index", data.get("index"))
    result_id = data.get("id", idx)

    id_fields = {
        "id": result_id,
        "question_id": question_id,
    }

    if source_index is not None:
        id_fields["source_index"] = source_index
        id_fields["pair_id"] = f"{question_id}::{source_index}"
    else:
        id_fields["pair_id"] = str(question_id)

    if "decompose_id" in data:
        id_fields["decompose_id"] = data["decompose_id"]
    if "index" in data:
        id_fields["index"] = data["index"]

    return id_fields


def evaluate(file_path, output_path):
    total_em = 0.0
    total_f1 = 0.0
    total_acc = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total = 0

    print(f"Evaluating: {file_path}")
    print("-" * 60)

    with open(file_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            question = data.get("question", "")
            final_answer_raw = data.get("final_answer", "")
            prediction = extract_prediction(final_answer_raw)

            ground_truths = data.get("answer", [])
            if not isinstance(ground_truths, list):
                ground_truths = [ground_truths]
            if len(ground_truths) == 0:
                ground_truths = [""]

            # scores
            em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            acc = metric_max_over_ground_truths(accuracy_score, prediction, ground_truths)
            precision, recall, f1 = prf_max_over_ground_truths(prediction, ground_truths)

            result = {
                **build_id_fields(data, idx),
                "question": question,
                "answer": ground_truths,
                "prediction": prediction,
                "accuracy": acc,
                "exact_match": em,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            total_em += em
            total_f1 += f1
            total_acc += acc
            total_precision += precision
            total_recall += recall
            total += 1

    if total == 0:
        print("No valid examples found.")
        return

    print(f"Output JSONL: {output_path}")
    print(f"Total: {total}")
    print(f"Accuracy: {total_acc / total}")
    print(f"Exact Match: {total_em / total}")
    print(f"Precision: {total_precision / total}")
    print(f"Recall: {total_recall / total}")
    print(f"F1: {total_f1 / total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="eval_datasets/test/2wiki_origin/prompts_decompose_test_multi/test_e5-large-v2_k10_passage0.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    output_path = args.output if args.output else str(input_path.with_name(input_path.stem + "_scored.jsonl"))

    evaluate(args.file, output_path)
