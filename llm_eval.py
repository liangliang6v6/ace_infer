import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def build_semantic_judge_prompt(question, prediction, ground_truths):
    gold_answers = json.dumps(ground_truths, ensure_ascii=False)

    return f"""
You are evaluating question answering outputs.

Task:
Decide whether the prediction is semantically correct for the question.

Rules:
1. Use the question to interpret the answer.
2. Accept aliases, abbreviations, alternate spellings, title/name variants, and common shortened forms if they refer to the same entity.
3. Accept slightly more specific or slightly less specific answers only if they still unambiguously refer to the same final answer.
4. Reject partial answers, overly broad answers, wrong entities, wrong dates, wrong places, wrong numbers, or answers that only overlap lexically.
5. For yes/no answers, require the same meaning.
6. For numeric/date answers, require semantic equivalence.
7. If multiple gold answers are given, prediction is correct if it semantically matches any one of them.

Return ONLY one JSON object in exactly this format:
{{
  "semantic_match": 0 or 1,
  "matched_answer": "best matched gold answer or empty string",
  "reason": "short explanation"
}}

Question:
{question}

Prediction:
{prediction}

Gold Answers:
{gold_answers}
""".strip()


def call_semantic_judge(client, question, prediction, ground_truths, max_retries=3, sleep_seconds=1.5):
    prompt = build_semantic_judge_prompt(question, prediction, ground_truths)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
            )

            text = response.output_text.strip()

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if not match:
                    raise ValueError(f"Model output is not valid JSON: {text}")
                obj = json.loads(match.group(0))

            semantic_match = int(obj.get("semantic_match", 0))
            semantic_match = 1 if semantic_match == 1 else 0

            matched_answer = str(obj.get("matched_answer", ""))
            reason = str(obj.get("reason", ""))

            return {
                "semantic_match": float(semantic_match),
                "matched_answer": matched_answer,
                "reason": reason,
            }

        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)

    return {
        "semantic_match": 0.0,
        "matched_answer": "",
        "reason": f"judge_error: {last_error}",
    }


def load_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY not found in .env")
    return OpenAI(api_key=api_key)


def upgraded_scores(original_scores, semantic_match):
    if float(semantic_match) == 1.0:
        return {
            "accuracy": 1.0,
            "exact_match": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    return {
        "accuracy": float(original_scores.get("accuracy", 0.0)),
        "exact_match": float(original_scores.get("exact_match", 0.0)),
        "precision": float(original_scores.get("precision", 0.0)),
        "recall": float(original_scores.get("recall", 0.0)),
        "f1": float(original_scores.get("f1", 0.0)),
    }


def count_examples(file_path):
    total = 0
    failed = 0
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            data = json.loads(line)
            if float(data.get("exact_match", 0.0)) == 0.0:
                failed += 1
    return total, failed


def iter_with_progress(iterable, total, desc):
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc)
        return

    processed = 0
    for item in iterable:
        processed += 1
        if processed == 1 or processed % 50 == 0 or processed == total:
            print(f"{desc}: {processed}/{total}")
        yield item


def evaluate_with_llm(file_path, output_path):
    client = load_client()
    total_examples, failed_examples = count_examples(file_path)

    total_acc = 0.0
    total_em = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    llm_calls = 0

    print(f"Reading scored file: {file_path}")
    print("-" * 60)
    print(f"Total examples: {total_examples}")
    print(f"Examples requiring LLM check: {failed_examples}")
    print("Writing output JSONL line by line.")

    with open(file_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(iter_with_progress(fin, total_examples, "LLM eval")):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            question = data.get("question", "")
            prediction = data.get("prediction", "")
            ground_truths = data.get("answer", [])
            if not isinstance(ground_truths, list):
                ground_truths = [ground_truths]
            if not ground_truths:
                ground_truths = [""]

            original_exact_match = float(data.get("exact_match", 0.0))
            original_scores = {
                "accuracy": float(data.get("accuracy", 0.0)),
                "exact_match": original_exact_match,
                "precision": float(data.get("precision", 0.0)),
                "recall": float(data.get("recall", 0.0)),
                "f1": float(data.get("f1", 0.0)),
            }

            if original_exact_match == 1.0:
                result = dict(data)
                result["explanation"] = ""
            else:
                llm_result = call_semantic_judge(
                    client=client,
                    question=question,
                    prediction=prediction,
                    ground_truths=ground_truths,
                )
                llm_calls += 1
                final_scores = upgraded_scores(original_scores, llm_result["semantic_match"])
                result = {
                    "id": data.get("id", idx),
                    "question": question,
                    "answer": ground_truths,
                    "prediction": prediction,
                    "accuracy": final_scores["accuracy"],
                    "exact_match": final_scores["exact_match"],
                    "precision": final_scores["precision"],
                    "recall": final_scores["recall"],
                    "f1": final_scores["f1"],
                    "explanation": llm_result["reason"],
                }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            total_acc += float(result.get("accuracy", 0.0))
            total_em += float(result.get("exact_match", 0.0))
            total_precision += float(result.get("precision", 0.0))
            total_recall += float(result.get("recall", 0.0))
            total_f1 += float(result.get("f1", 0.0))

    if total_examples == 0:
        print("No valid examples found.")
        return

    print(f"Output JSONL: {output_path}")
    print(f"Total examples written: {total_examples}")
    print(f"LLM calls made: {llm_calls}")
    print("Final overall scores after LLM evaluation:")
    print(f"Accuracy: {total_acc / total_examples}")
    print(f"Exact Match: {total_em / total_examples}")
    print(f"Precision: {total_precision / total_examples}")
    print(f"Recall: {total_recall / total_examples}")
    print(f"F1: {total_f1 / total_examples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="eval_datasets/test/hotpotqa/prompts_decompose_test_acesearcher/test_e5-large-v2_k10_passage1_scored.jsonl",
        help="Path to the scored JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output LLM-evaluated JSONL",
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    output_path = args.output if args.output else str(
        input_path.with_name(input_path.stem + "_llm_checked.jsonl")
    )

    evaluate_with_llm(args.file, output_path)
