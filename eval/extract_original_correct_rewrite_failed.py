import json
from pathlib import Path


ORIGINAL_SOURCE = Path("eval_datasets/2wiki_origin/test_subsampled.jsonl")
REWRITE_ORIGINAL_SOURCE = Path("../ours/dataset/2wikimultihopqa/test_subsampled_anchor.jsonl")
ORIGINAL_RAW = Path("eval_datasets/test/2wiki_origin/prompts_decompose_test_ace/test_e5-large-v2_k10_passage1.jsonl")
ORIGINAL_SCORED = Path("eval_datasets/test/2wiki_origin/prompts_decompose_test_ace/test_e5-large-v2_k10_passage1_scored.jsonl")
REWRITE_RAW = Path("eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage0.jsonl")
REWRITE_SCORED = Path("eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage0_scored.jsonl")
OUTPUT_DIR = Path("eval_datasets/test/2wiki/prompts_decompose_test_ace/eval_results")
OUTPUT_JSONL = OUTPUT_DIR / "original_correct_rewrite_failed.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "original_correct_rewrite_failed_summary.json"
SUMMARY_TXT = OUTPUT_DIR / "original_correct_rewrite_failed_summary.txt"


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


def format_decomposition(row):
    decomposed = row.get("decomposed", [])
    if not isinstance(decomposed, list):
        return [str(decomposed)] if decomposed else []

    formatted = []
    for index, item in enumerate(decomposed, 1):
        if isinstance(item, dict):
            label = item.get("label", f"Q{index}")
            text = item.get("text", "")
        else:
            label = f"Q{index}"
            text = str(item)
        formatted.append({"label": label, "text": text})
    return formatted


def correct(scored_row):
    return float(scored_row.get("exact_match", 0.0)) >= 1.0


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round(numerator * 100.0 / denominator, 2)


def get_source_id_by_original_index():
    source_rows = read_jsonl(ORIGINAL_SOURCE)
    return {
        index: row.get("question_id", row.get("id", index))
        for index, row in enumerate(source_rows)
    }


def row_id(row, fallback):
    return row.get("question_id", row.get("id", fallback))


def get_rewrite_original_rows():
    if REWRITE_ORIGINAL_SOURCE.exists():
        return read_jsonl(REWRITE_ORIGINAL_SOURCE)
    return read_jsonl(ORIGINAL_SOURCE)


def build_original_status():
    rewrite_original_rows = get_rewrite_original_rows()
    rewrite_original_by_question = {
        row.get("question", ""): row_id(row, index)
        for index, row in enumerate(rewrite_original_rows)
    }
    source_id_by_index = get_source_id_by_original_index()
    raw_rows = read_jsonl(ORIGINAL_RAW)
    scored_rows = read_jsonl(ORIGINAL_SCORED)
    original_by_id = {}
    scored_ids = set()

    for index, (raw_row, scored_row) in enumerate(zip(raw_rows, scored_rows)):
        original_id = rewrite_original_by_question.get(raw_row.get("question", ""))
        if original_id is None:
            original_id = source_id_by_index.get(index, raw_row.get("question_id", index))

        scored_ids.add(original_id)
        is_correct = correct(scored_row)
        previous = original_by_id.get(original_id)
        if previous is not None and previous["correct"] and not is_correct:
            continue

        original_by_id[original_id] = {
            "correct": is_correct,
            "original_q": raw_row.get("question", scored_row.get("question", "")),
            "original_pre": scored_row.get("prediction", ""),
            "original_decomp": format_decomposition(raw_row),
            "original_inter": raw_row.get("intermediate_answers", {}),
        }

    rewrite_original_ids = {
        row_id(row, index)
        for index, row in enumerate(rewrite_original_rows)
    }
    return rewrite_original_rows, rewrite_original_ids, scored_ids, original_by_id


def build_original_correct_map():
    _, _, _, original_by_id = build_original_status()
    return {
        original_id: row
        for original_id, row in original_by_id.items()
        if row["correct"]
    }


def build_original_correct_ids():
    return set(build_original_correct_map())


def build_outputs():
    original_raw_rows = read_jsonl(ORIGINAL_RAW)
    original_scored_rows = read_jsonl(ORIGINAL_SCORED)
    rewrite_original_rows, rewrite_original_ids, original_scored_ids, original_by_id = build_original_status()
    original_correct = {
        original_id: row
        for original_id, row in original_by_id.items()
        if row["correct"]
    }
    original_correct_ids = set(original_correct)
    rewrite_raw_rows = read_jsonl(REWRITE_RAW)
    rewrite_scored_rows = read_jsonl(REWRITE_SCORED)

    original_reference_total = len(original_scored_rows)
    original_total = len(rewrite_original_ids)
    original_scored_in_pool = len(rewrite_original_ids & original_scored_ids)
    original_unscored_count = original_total - original_scored_in_pool
    original_correct_count = len(original_correct_ids & rewrite_original_ids)
    original_failed_count = original_total - original_correct_count
    rewrite_total = len(rewrite_scored_rows)
    rewrite_correct_count = sum(1 for row in rewrite_scored_rows if correct(row))
    rewrite_failed_count = rewrite_total - rewrite_correct_count
    rewrite_question_ids = {
        raw_row.get("question_id", scored_row.get("question_id"))
        for raw_row, scored_row in zip(rewrite_raw_rows, rewrite_scored_rows)
    }

    outputs = []
    rewrite_failed = 0
    rewrite_failed_with_original_correct = 0
    cc = 0
    cw = 0
    wc = 0
    ww = 0
    rewrite_pairs_missing_original_score = 0

    for index, (raw_row, scored_row) in enumerate(zip(rewrite_raw_rows, rewrite_scored_rows)):
        rewrite_correct = correct(scored_row)
        question_id = raw_row.get("question_id", scored_row.get("question_id"))
        original_is_correct = question_id in original_correct_ids
        if question_id not in original_by_id:
            rewrite_pairs_missing_original_score += 1

        if original_is_correct and rewrite_correct:
            cc += 1
        elif original_is_correct and not rewrite_correct:
            cw += 1
        elif not original_is_correct and rewrite_correct:
            wc += 1
        else:
            ww += 1

        if rewrite_correct:
            continue

        rewrite_failed += 1
        original = original_correct.get(question_id)
        if original is None:
            continue

        rewrite_failed_with_original_correct += 1
        rewrite_id = raw_row.get("source_index", raw_row.get("index", index))
        outputs.append(
            {
                "question_id": question_id,
                "pair_id": f"{question_id}::{rewrite_id}",
                "rewrite_id": rewrite_id,
                "original_q": original["original_q"],
                "answer": scored_row.get("answer", raw_row.get("answer", [])),
                "original_pre": original["original_pre"],
                "original_decomp": original["original_decomp"],
                "original_inter": original["original_inter"],
                "rewrite_q": raw_row.get("question", scored_row.get("question", "")),
                "rewrite_pre": scored_row.get("prediction", ""),
                "rewrite_decomp": format_decomposition(raw_row),
                "rewrite_inter": raw_row.get("intermediate_answers", {}),
            }
        )

    summary = {
        "original_reference_rows": len(original_raw_rows),
        "original_reference_scored_rows": original_reference_total,
        "rewrite_original_source": str(REWRITE_ORIGINAL_SOURCE),
        "rewrite_original_questions": original_total,
        "rewrite_original_scored_questions": original_scored_in_pool,
        "rewrite_original_unscored_questions": original_unscored_count,
        "rewrite_original_correct_questions": original_correct_count,
        "rewrite_original_failed_or_unscored_questions": original_failed_count,
        "rewrite_result_questions": len(rewrite_question_ids),
        "rewrite_result_questions_missing_from_original_scores": len(rewrite_question_ids - set(original_by_id)),
        "rewrite_pairs_missing_original_score": rewrite_pairs_missing_original_score,
        "rewrite_rows": len(rewrite_raw_rows),
        "rewrite_scored_rows": rewrite_total,
        "rewrite_correct_rows": rewrite_correct_count,
        "rewrite_failed_rows": rewrite_failed_count,
        "pair_level_counts": {
            "CC_original_correct_rewrite_correct": cc,
            "CW_original_correct_rewrite_wrong": cw,
            "WC_original_wrong_rewrite_correct": wc,
            "WW_original_wrong_rewrite_wrong": ww,
        },
        "main_metrics": {
            "original_accuracy": {
                "numerator": original_correct_count,
                "denominator": original_total,
                "percent": pct(original_correct_count, original_total),
            },
            "rewrite_accuracy": {
                "numerator": rewrite_correct_count,
                "denominator": rewrite_total,
                "percent": pct(rewrite_correct_count, rewrite_total),
            },
            "rewrite_induced_failure_rate": {
                "numerator": cw,
                "denominator": cc + cw,
                "percent": pct(cw, cc + cw),
            },
            "useful_candidate_yield": {
                "numerator": cw,
                "denominator": rewrite_total,
                "percent": pct(cw, rewrite_total),
            },
        },
        "original_correct_rewrite_failed_pairs": len(outputs),
        "rewrite_failed_with_original_correct": rewrite_failed_with_original_correct,
        "selection_definition": "Keep rewrite rows where the same original question was answered correctly, but the rewritten question was answered incorrectly.",
        "output_jsonl": str(OUTPUT_JSONL),
    }
    return outputs, summary


def write_summary_text(summary):
    lines = [
        "Original-correct but rewrite-failed pairs",
        "",
        "Pair-level counts",
        f"- CC = original correct, rewrite correct: {summary['pair_level_counts']['CC_original_correct_rewrite_correct']}",
        f"- CW = original correct, rewrite wrong: {summary['pair_level_counts']['CW_original_correct_rewrite_wrong']}",
        f"- WC = original wrong, rewrite correct: {summary['pair_level_counts']['WC_original_wrong_rewrite_correct']}",
        f"- WW = original wrong, rewrite wrong: {summary['pair_level_counts']['WW_original_wrong_rewrite_wrong']}",
        "",
        "Main metrics",
        (
            "- Original Accuracy = "
            f"{summary['main_metrics']['original_accuracy']['numerator']} / "
            f"{summary['main_metrics']['original_accuracy']['denominator']} = "
            f"{summary['main_metrics']['original_accuracy']['percent']}%"
        ),
        (
            "- Rewrite Accuracy = "
            f"{summary['main_metrics']['rewrite_accuracy']['numerator']} / "
            f"{summary['main_metrics']['rewrite_accuracy']['denominator']} = "
            f"{summary['main_metrics']['rewrite_accuracy']['percent']}%"
        ),
        (
            "- Rewrite-Induced Failure Rate = CW / (CC + CW) = "
            f"{summary['main_metrics']['rewrite_induced_failure_rate']['numerator']} / "
            f"{summary['main_metrics']['rewrite_induced_failure_rate']['denominator']} = "
            f"{summary['main_metrics']['rewrite_induced_failure_rate']['percent']}%"
        ),
        (
            "- Useful Candidate Yield = CW / all_rewrites = "
            f"{summary['main_metrics']['useful_candidate_yield']['numerator']} / "
            f"{summary['main_metrics']['useful_candidate_yield']['denominator']} = "
            f"{summary['main_metrics']['useful_candidate_yield']['percent']}%"
        ),
        "",
        "Original side",
        f"- Rewrite original source: {summary['rewrite_original_source']}",
        f"- Rewrite original questions: {summary['rewrite_original_questions']}",
        f"- Rewrite original scored questions: {summary['rewrite_original_scored_questions']}",
        f"- Rewrite original unscored questions: {summary['rewrite_original_unscored_questions']}",
        f"- Rewrite original correct questions: {summary['rewrite_original_correct_questions']}",
        f"- Rewrite original failed or unscored questions: {summary['rewrite_original_failed_or_unscored_questions']}",
        f"- Original ACE reference rows: {summary['original_reference_rows']}",
        f"- Original ACE reference scored rows: {summary['original_reference_scored_rows']}",
        "",
        "Rewrite side",
        f"- Rewrite result questions: {summary['rewrite_result_questions']}",
        f"- Rewrite result questions missing from original scores: {summary['rewrite_result_questions_missing_from_original_scores']}",
        f"- Rewrite pairs missing original score: {summary['rewrite_pairs_missing_original_score']}",
        f"- Rewrite rows: {summary['rewrite_rows']}",
        f"- Rewrite scored rows: {summary['rewrite_scored_rows']}",
        f"- Rewrite correct rows: {summary['rewrite_correct_rows']}",
        f"- Rewrite failed rows: {summary['rewrite_failed_rows']}",
        "",
        "Selected output",
        (
            "- Pairs where original was correct but rewrite failed: "
            f"{summary['original_correct_rewrite_failed_pairs']}"
        ),
        f"- Definition: {summary['selection_definition']}",
        "",
        f"JSONL: {summary['output_jsonl']}",
    ]
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))
        fout.write("\n")


def main():
    rows, summary = build_outputs()
    write_jsonl(rows, OUTPUT_JSONL)
    write_json(summary, SUMMARY_JSON)
    write_summary_text(summary)
    print(f"Rewrite original questions: {summary['rewrite_original_questions']}")
    print(f"Rewrite original correct questions: {summary['rewrite_original_correct_questions']}")
    print(f"Rewrite failed rows: {summary['rewrite_failed_rows']}")
    print(f"Original correct + rewrite failed pairs: {summary['original_correct_rewrite_failed_pairs']}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
