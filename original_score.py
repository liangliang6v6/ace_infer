import argparse
import json
import re
import string
from pathlib import Path


def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""

    if s is None:
        s = ""

    s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = set(gold_tokens) & set(pred_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_acc(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).lower()
    pred_tokens = normalize_answer(a_pred).lower()

    if not gold_tokens and not pred_tokens:
        return 1
    # gt includes pre or pre includes gt, the acc will be 1
    if pred_tokens in gold_tokens or gold_tokens in pred_tokens:
        return 1
    return 0


def extract_answer(text):
    if text is None:
        return ""

    text = str(text)
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return ""


def get_prediction(record):
    if "prediction" in record:
        prediction = record.get("prediction", "")
        return "" if prediction is None else str(prediction).strip()

    return extract_answer(record.get("final_answer", ""))


def get_ground_truths(record):
    answers = record.get("answer", [])
    if isinstance(answers, list):
        return answers if answers else [""]
    return [answers]


def evaluate(file_path, output_path):
    total = 0
    total_acc = 0.0
    total_em = 0.0
    total_f1 = 0.0

    print(f"Evaluating with original scoring: {file_path}")
    print("-" * 60)

    with open(file_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            prediction = get_prediction(data)
            ground_truths = get_ground_truths(data)

            em = max(compute_exact(gt, prediction) for gt in ground_truths)
            f1 = max(compute_f1(gt, prediction) for gt in ground_truths)
            acc = max(compute_acc(gt, prediction) for gt in ground_truths)

            result = {
                "id": data.get("id", data.get("question_id", idx)),
                "question": data.get("question", ""),
                "answer": ground_truths,
                "prediction": prediction,
                "accuracy": float(acc),
                "exact_match": float(em),
                "f1": float(f1),
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            total += 1
            total_acc += acc
            total_em += em
            total_f1 += f1

    if total == 0:
        print("No valid examples found.")
        return

    print(f"Output JSONL: {output_path}")
    print(f"Total: {total}")
    print(f"Accuracy: {total_acc / total}")
    print(f"Exact Match: {total_em / total}")
    print(f"F1: {total_f1 / total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="eval_datasets/test/musique/prompts_decompose_test_acesearcher/test_e5-large-v2_k10_passage1.jsonl",
        help="Path to a raw result JSONL or a scored JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path. Defaults to <input>_original_scored.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    output_path = args.output if args.output else str(input_path.with_name(input_path.stem + "_original_scored.jsonl"))
    evaluate(args.file, output_path)
