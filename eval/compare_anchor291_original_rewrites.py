import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from get_score import (  # noqa: E402
    accuracy_score,
    exact_match_score,
    extract_prediction,
    prf_max_over_ground_truths,
)


ANCHOR_FILE = REPO_ROOT.parent / "ours/dataset/2wikimultihopqa/test_subsampled_anchor.jsonl"
ORIGINAL_RAW = REPO_ROOT / "eval_datasets/test/2wikimultihopqa/prompts_decompose_test_acesearcher/test_e5-large-v2_k10_passage0.jsonl"
ORIGINAL_SCORED = REPO_ROOT / "eval_datasets/test/2wikimultihopqa/prompts_decompose_test_acesearcher/test_e5-large-v2_k10_passage0_scored.jsonl"
REWRITE_RAW = REPO_ROOT / "eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage0.jsonl"
REWRITE_SCORED = REPO_ROOT / "eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage0_scored.jsonl"
OUTPUT_DIR = REPO_ROOT / "eval_datasets/test/2wiki/prompts_decompose_test_ace/eval_results/anchor291_clean"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(row, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(row, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def answers(row):
    values = row.get("answer", [])
    if not isinstance(values, list):
        values = [values]
    return values or [""]


def score_prediction(prediction, gold_answers):
    exact_match = max(exact_match_score(prediction, gold) for gold in gold_answers)
    accuracy = max(accuracy_score(prediction, gold) for gold in gold_answers)
    precision, recall, f1 = prf_max_over_ground_truths(prediction, gold_answers)
    return {
        "accuracy": accuracy,
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": exact_match >= 1.0,
    }


def format_decomposition(row):
    decomposed = row.get("decomposed", [])
    if not isinstance(decomposed, list):
        return []

    formatted = []
    for index, item in enumerate(decomposed, 1):
        if isinstance(item, dict):
            formatted.append(
                {
                    "label": item.get("label", f"Q{index}"),
                    "text": item.get("text", ""),
                }
            )
        else:
            formatted.append({"label": f"Q{index}", "text": str(item)})
    return formatted


def build_original_rows():
    anchor_rows = read_jsonl(ANCHOR_FILE)
    original_raw_rows = read_jsonl(ORIGINAL_RAW)
    original_scored_rows = read_jsonl(ORIGINAL_SCORED)

    anchor_by_question = {row["question"]: row for row in anchor_rows}
    scored_by_question = {row.get("question", ""): row for row in original_scored_rows}
    original_by_id = {}
    missing = []
    stored_score_disagreements = []

    for raw_row in original_raw_rows:
        question = raw_row.get("question", "")
        anchor = anchor_by_question.get(question)
        if anchor is None:
            continue

        gold_answers = answers(raw_row)
        prediction = extract_prediction(raw_row.get("final_answer", ""))
        scores = score_prediction(prediction, gold_answers)
        scored_row = scored_by_question.get(question)
        if scored_row and float(scored_row.get("exact_match", 0.0)) != scores["exact_match"]:
            stored_score_disagreements.append(anchor["id"])

        original_by_id[anchor["id"]] = {
            "question_id": anchor["id"],
            "original_q": question,
            "answer": gold_answers,
            "original_pre": prediction,
            "original_correct": scores["correct"],
            "original_scores": {
                key: scores[key]
                for key in ("accuracy", "exact_match", "precision", "recall", "f1")
            },
            "original_decomp": format_decomposition(raw_row),
            "original_inter": raw_row.get("intermediate_answers", {}),
            "reasoning_steps": anchor.get("reasoning_steps", []),
            "reasoning_path": anchor.get("reasoning_path", []),
            "anchor_entities": anchor.get("anchor_entities", []),
        }

    for anchor in anchor_rows:
        if anchor["id"] not in original_by_id:
            missing.append(anchor["id"])

    return anchor_rows, original_by_id, missing, stored_score_disagreements


def build_rewrite_rows():
    rewrite_raw_rows = read_jsonl(REWRITE_RAW)
    rewrite_scored_rows = read_jsonl(REWRITE_SCORED)
    rewrite_rows = []
    stored_score_disagreements = []

    for index, raw_row in enumerate(rewrite_raw_rows):
        scored_row = rewrite_scored_rows[index] if index < len(rewrite_scored_rows) else {}
        gold_answers = answers(raw_row)
        prediction = extract_prediction(raw_row.get("final_answer", ""))
        scores = score_prediction(prediction, gold_answers)
        if scored_row and float(scored_row.get("exact_match", 0.0)) != scores["exact_match"]:
            stored_score_disagreements.append(index)

        rewrite_id = raw_row.get("source_index", raw_row.get("index", index))
        question_id = raw_row.get("question_id", scored_row.get("question_id"))
        rewrite_rows.append(
            {
                "question_id": question_id,
                "pair_id": f"{question_id}::{rewrite_id}",
                "rewrite_id": rewrite_id,
                "rewrite_q": raw_row.get("question", ""),
                "answer": gold_answers,
                "rewrite_pre": prediction,
                "rewrite_correct": scores["correct"],
                "rewrite_scores": {
                    key: scores[key]
                    for key in ("accuracy", "exact_match", "precision", "recall", "f1")
                },
                "rewrite_decomp": format_decomposition(raw_row),
                "rewrite_inter": raw_row.get("intermediate_answers", {}),
            }
        )

    return rewrite_rows, stored_score_disagreements


def pct(numerator, denominator):
    return round(numerator * 100.0 / denominator, 2) if denominator else 0.0


def avg(rows, score_key):
    if not rows:
        return 0.0
    return sum(row[score_key] for row in rows) / len(rows)


def main():
    anchor_rows, original_by_id, missing_original_ids, original_score_disagreements = build_original_rows()
    rewrite_rows, rewrite_score_disagreements = build_rewrite_rows()

    pairs = []
    induced_failures = []
    groups = {}
    counts = {"CC": 0, "CW": 0, "WC": 0, "WW": 0}

    for rewrite in rewrite_rows:
        original = original_by_id.get(rewrite["question_id"])
        if original is None:
            continue

        original_correct = original["original_correct"]
        rewrite_correct = rewrite["rewrite_correct"]
        if original_correct and rewrite_correct:
            bucket = "CC"
        elif original_correct and not rewrite_correct:
            bucket = "CW"
        elif not original_correct and rewrite_correct:
            bucket = "WC"
        else:
            bucket = "WW"
        counts[bucket] += 1

        pair = {
            "bucket": bucket,
            "question_id": rewrite["question_id"],
            "pair_id": rewrite["pair_id"],
            "rewrite_id": rewrite["rewrite_id"],
            "original_q": original["original_q"],
            "answer": original["answer"],
            "original_pre": original["original_pre"],
            "original_correct": original_correct,
            "original_scores": original["original_scores"],
            "original_decomp": original["original_decomp"],
            "original_inter": original["original_inter"],
            "rewrite_q": rewrite["rewrite_q"],
            "rewrite_pre": rewrite["rewrite_pre"],
            "rewrite_correct": rewrite_correct,
            "rewrite_scores": rewrite["rewrite_scores"],
            "rewrite_decomp": rewrite["rewrite_decomp"],
            "rewrite_inter": rewrite["rewrite_inter"],
        }
        pairs.append(pair)
        if bucket == "CW":
            induced_failures.append(pair)

        group = groups.setdefault(
            rewrite["question_id"],
            {
                "question_id": rewrite["question_id"],
                "original_q": original["original_q"],
                "answer": original["answer"],
                "original_pre": original["original_pre"],
                "original_correct": original_correct,
                "rewrite_count": 0,
                "rewrite_correct_count": 0,
                "rewrite_wrong_count": 0,
                "buckets": {"CC": 0, "CW": 0, "WC": 0, "WW": 0},
                "rewrites": [],
            },
        )
        group["rewrite_count"] += 1
        group["rewrite_correct_count"] += int(rewrite_correct)
        group["rewrite_wrong_count"] += int(not rewrite_correct)
        group["buckets"][bucket] += 1
        group["rewrites"].append(
            {
                "pair_id": rewrite["pair_id"],
                "rewrite_id": rewrite["rewrite_id"],
                "bucket": bucket,
                "rewrite_q": rewrite["rewrite_q"],
                "rewrite_pre": rewrite["rewrite_pre"],
                "rewrite_correct": rewrite_correct,
            }
        )

    original_score_rows = [row["original_scores"] for row in original_by_id.values()]
    rewrite_score_rows = [row["rewrite_scores"] for row in rewrite_rows]
    original_correct_count = sum(row["original_correct"] for row in original_by_id.values())
    rewrite_correct_count = sum(row["rewrite_correct"] for row in rewrite_rows)
    rewrite_ids = {row["question_id"] for row in rewrite_rows}
    anchor_ids = {row["id"] for row in anchor_rows}
    rewrite_covered_original_rows = [
        original_by_id[question_id]
        for question_id in sorted(rewrite_ids)
        if question_id in original_by_id
    ]
    rewrite_covered_original_score_rows = [
        row["original_scores"]
        for row in rewrite_covered_original_rows
    ]
    rewrite_covered_original_correct_count = sum(
        row["original_correct"]
        for row in rewrite_covered_original_rows
    )

    summary = {
        "definition": "For each original anchor question q, recompute the model score from prediction vs answer. Then compare each rewrite q'_j with the same question_id as an independent pair.",
        "files": {
            "anchor": str(ANCHOR_FILE),
            "original_raw": str(ORIGINAL_RAW),
            "original_scored_checked_only": str(ORIGINAL_SCORED),
            "rewrite_raw": str(REWRITE_RAW),
            "rewrite_scored_checked_only": str(REWRITE_SCORED),
        },
        "coverage": {
            "anchor_questions": len(anchor_rows),
            "original_predictions_matched_to_anchor": len(original_by_id),
            "anchor_questions_missing_original_prediction": len(missing_original_ids),
            "rewrite_pairs": len(rewrite_rows),
            "rewrite_covered_original_questions": len(rewrite_ids),
            "anchor_questions_without_rewrites": len(anchor_ids - rewrite_ids),
            "rewrite_questions_not_in_anchor": len(rewrite_ids - anchor_ids),
        },
        "score_validation": {
            "original_stored_exact_match_disagreements": len(original_score_disagreements),
            "rewrite_stored_exact_match_disagreements": len(rewrite_score_disagreements),
        },
        "original_metrics": {
            "exact_match": {
                "correct": original_correct_count,
                "total": len(original_by_id),
                "percent": pct(original_correct_count, len(original_by_id)),
            },
            "accuracy_avg": avg(original_score_rows, "accuracy"),
            "precision_avg": avg(original_score_rows, "precision"),
            "recall_avg": avg(original_score_rows, "recall"),
            "f1_avg": avg(original_score_rows, "f1"),
        },
        "rewrite_covered_original_metrics": {
            "exact_match": {
                "correct": rewrite_covered_original_correct_count,
                "total": len(rewrite_covered_original_rows),
                "percent": pct(
                    rewrite_covered_original_correct_count,
                    len(rewrite_covered_original_rows),
                ),
            },
            "accuracy_avg": avg(rewrite_covered_original_score_rows, "accuracy"),
            "precision_avg": avg(rewrite_covered_original_score_rows, "precision"),
            "recall_avg": avg(rewrite_covered_original_score_rows, "recall"),
            "f1_avg": avg(rewrite_covered_original_score_rows, "f1"),
        },
        "rewrite_metrics": {
            "exact_match": {
                "correct": rewrite_correct_count,
                "total": len(rewrite_rows),
                "percent": pct(rewrite_correct_count, len(rewrite_rows)),
            },
            "accuracy_avg": avg(rewrite_score_rows, "accuracy"),
            "precision_avg": avg(rewrite_score_rows, "precision"),
            "recall_avg": avg(rewrite_score_rows, "recall"),
            "f1_avg": avg(rewrite_score_rows, "f1"),
        },
        "pair_level_counts": {
            "CC_original_correct_rewrite_correct": counts["CC"],
            "CW_original_correct_rewrite_wrong": counts["CW"],
            "WC_original_wrong_rewrite_correct": counts["WC"],
            "WW_original_wrong_rewrite_wrong": counts["WW"],
        },
        "pair_metrics": {
            "rewrite_induced_failure_rate": {
                "numerator": counts["CW"],
                "denominator": counts["CC"] + counts["CW"],
                "percent": pct(counts["CW"], counts["CC"] + counts["CW"]),
            },
            "useful_candidate_yield": {
                "numerator": counts["CW"],
                "denominator": len(rewrite_rows),
                "percent": pct(counts["CW"], len(rewrite_rows)),
            },
        },
        "outputs": {
            "original_291_scored": str(OUTPUT_DIR / "original_291_scored.jsonl"),
            "original_rewrite_ids_scored": str(OUTPUT_DIR / "original_rewrite_ids_scored.jsonl"),
            "pairs": str(OUTPUT_DIR / "pairs.jsonl"),
            "groups": str(OUTPUT_DIR / "groups.jsonl"),
            "rewrite_induced_failures": str(OUTPUT_DIR / "rewrite_induced_failures.jsonl"),
            "summary_json": str(OUTPUT_DIR / "summary.json"),
            "summary_txt": str(OUTPUT_DIR / "summary.txt"),
        },
    }

    original_rows = list(original_by_id.values())
    group_rows = sorted(groups.values(), key=lambda row: row["question_id"])

    write_jsonl(original_rows, OUTPUT_DIR / "original_291_scored.jsonl")
    write_jsonl(rewrite_covered_original_rows, OUTPUT_DIR / "original_rewrite_ids_scored.jsonl")
    write_jsonl(pairs, OUTPUT_DIR / "pairs.jsonl")
    write_jsonl(group_rows, OUTPUT_DIR / "groups.jsonl")
    write_jsonl(induced_failures, OUTPUT_DIR / "rewrite_induced_failures.jsonl")
    write_json(summary, OUTPUT_DIR / "summary.json")

    lines = [
        "Anchor291 Original vs Rewrite Evaluation",
        "",
        "Coverage",
        f"- Anchor original questions: {len(anchor_rows)}",
        f"- Original model predictions matched: {len(original_by_id)}",
        f"- Rewrite pairs: {len(rewrite_rows)}",
        f"- Rewrite-covered original questions: {len(rewrite_ids)}",
        f"- Anchor questions without rewrites in this result: {len(anchor_ids - rewrite_ids)}",
        "",
        "Direct Scores",
        (
            "- Original EM on rewrite-covered IDs: "
            f"{rewrite_covered_original_correct_count} / {len(rewrite_covered_original_rows)} = "
            f"{pct(rewrite_covered_original_correct_count, len(rewrite_covered_original_rows))}%"
        ),
        f"- Original EM on all anchor IDs: {original_correct_count} / {len(original_by_id)} = {pct(original_correct_count, len(original_by_id))}%",
        f"- Rewrite EM: {rewrite_correct_count} / {len(rewrite_rows)} = {pct(rewrite_correct_count, len(rewrite_rows))}%",
        (
            "- Original F1 on rewrite-covered IDs: "
            f"{avg(rewrite_covered_original_score_rows, 'f1') * 100.0:.2f}%"
        ),
        f"- Rewrite F1: {avg(rewrite_score_rows, 'f1') * 100.0:.2f}%",
        "",
        "Pair Counts",
        f"- CC = original correct, rewrite correct: {counts['CC']}",
        f"- CW = original correct, rewrite wrong: {counts['CW']}",
        f"- WC = original wrong, rewrite correct: {counts['WC']}",
        f"- WW = original wrong, rewrite wrong: {counts['WW']}",
        "",
        "Key Metrics",
        f"- Rewrite-Induced Failure Rate = CW / (CC + CW) = {counts['CW']} / {counts['CC'] + counts['CW']} = {pct(counts['CW'], counts['CC'] + counts['CW'])}%",
        f"- Useful Candidate Yield = CW / all rewrite pairs = {counts['CW']} / {len(rewrite_rows)} = {pct(counts['CW'], len(rewrite_rows))}%",
        "",
        "Score Validation",
        f"- Original stored-score disagreements: {len(original_score_disagreements)}",
        f"- Rewrite stored-score disagreements: {len(rewrite_score_disagreements)}",
        "",
        "Outputs",
        f"- {OUTPUT_DIR / 'summary.json'}",
        f"- {OUTPUT_DIR / 'original_rewrite_ids_scored.jsonl'}",
        f"- {OUTPUT_DIR / 'pairs.jsonl'}",
        f"- {OUTPUT_DIR / 'groups.jsonl'}",
        f"- {OUTPUT_DIR / 'rewrite_induced_failures.jsonl'}",
    ]
    summary_txt = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)


if __name__ == "__main__":
    main()
