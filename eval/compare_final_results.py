import json
from collections import Counter, defaultdict
from pathlib import Path


FINAL_RAW = Path("eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage0.jsonl")
FINAL_SCORED = Path("eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage0_scored.jsonl")
ORIGINAL_SOURCE = Path("eval_datasets/2wiki_origin/test_subsampled.jsonl")
BAD_DECOMPOSITIONS = Path("eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/eval_results/bad_decompositions.jsonl")
OUTPUT_DIR = Path("eval_datasets/test/2wiki/prompts_decompose_test_ace/eval_results")
PAIR_OUTPUT = OUTPUT_DIR / "final_pair_results.jsonl"
GROUP_OUTPUT = OUTPUT_DIR / "final_group_results.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "final_summary.json"
SUMMARY_TEXT = OUTPUT_DIR / "final_summary.txt"


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


def pct(count, total):
    if total == 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def get_original_question_map():
    if not ORIGINAL_SOURCE.exists():
        return {}
    rows = read_jsonl(ORIGINAL_SOURCE)
    return {
        row.get("question_id"): row.get("question_text", row.get("question", ""))
        for row in rows
    }


def get_bad_decomposition_ids():
    if not BAD_DECOMPOSITIONS.exists():
        return set()
    return {row["pair_id"] for row in read_jsonl(BAD_DECOMPOSITIONS)}


def pair_id(raw_row, fallback_index):
    original_id = raw_row.get("question_id")
    rewrite_id = raw_row.get("source_index", raw_row.get("index", fallback_index))
    return f"{original_id}::{rewrite_id}"


def is_correct(scored_row, metric="exact_match", min_score=1.0):
    return float(scored_row.get(metric, 0.0)) >= min_score


def build_pair_rows(raw_rows, scored_rows):
    original_questions = get_original_question_map()
    bad_decomposition_ids = get_bad_decomposition_ids()
    pair_rows = []

    for index, (raw_row, scored_row) in enumerate(zip(raw_rows, scored_rows)):
        current_pair_id = pair_id(raw_row, index)
        original_id = raw_row.get("question_id")
        rewrite_id = raw_row.get("source_index", raw_row.get("index", index))
        correct = is_correct(scored_row)
        bad_decomposition = current_pair_id in bad_decomposition_ids
        pair_rows.append(
            {
                "pair_id": current_pair_id,
                "original_question_id": original_id,
                "rewrite_question_id": rewrite_id,
                "original_question": original_questions.get(original_id, ""),
                "rewritten_question": raw_row.get("question", scored_row.get("question", "")),
                "answer": scored_row.get("answer", raw_row.get("answer", [])),
                "prediction": scored_row.get("prediction", ""),
                "exact_match": float(scored_row.get("exact_match", 0.0)),
                "accuracy": float(scored_row.get("accuracy", 0.0)),
                "f1": float(scored_row.get("f1", 0.0)),
                "final_correct": correct,
                "final_failed": not correct,
                "bad_decomposition": bad_decomposition,
                "decomposition_and_final_failed": bad_decomposition and not correct,
                "decomposition_bad_but_final_correct": bad_decomposition and correct,
                "decomposition_good_but_final_failed": (not bad_decomposition) and (not correct),
            }
        )

    return pair_rows


def build_group_rows(pair_rows):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[row["original_question_id"]].append(row)

    group_rows = []
    for original_id, rows in sorted(grouped.items()):
        total = len(rows)
        correct = sum(row["final_correct"] for row in rows)
        failed = total - correct
        bad_decomp = sum(row["bad_decomposition"] for row in rows)
        bad_decomp_and_failed = sum(row["decomposition_and_final_failed"] for row in rows)
        good_decomp_but_failed = sum(row["decomposition_good_but_final_failed"] for row in rows)
        bad_decomp_but_correct = sum(row["decomposition_bad_but_final_correct"] for row in rows)
        group_rows.append(
            {
                "original_question_id": original_id,
                "original_question": rows[0].get("original_question", ""),
                "num_rewrites": total,
                "num_final_correct": correct,
                "num_final_failed": failed,
                "final_accuracy": correct / total if total else 0.0,
                "num_bad_decomposition": bad_decomp,
                "num_bad_decomposition_and_final_failed": bad_decomp_and_failed,
                "num_good_decomposition_but_final_failed": good_decomp_but_failed,
                "num_bad_decomposition_but_final_correct": bad_decomp_but_correct,
                "failed_pair_ids": [row["pair_id"] for row in rows if row["final_failed"]],
                "correct_pair_ids": [row["pair_id"] for row in rows if row["final_correct"]],
            }
        )

    return group_rows


def build_summary(raw_rows, scored_rows, pair_rows, group_rows):
    total = len(pair_rows)
    correct = sum(row["final_correct"] for row in pair_rows)
    failed = total - correct
    bad_decomp = sum(row["bad_decomposition"] for row in pair_rows)
    both_bad_and_failed = sum(row["decomposition_and_final_failed"] for row in pair_rows)
    bad_decomp_but_correct = sum(row["decomposition_bad_but_final_correct"] for row in pair_rows)
    good_decomp_but_failed = sum(row["decomposition_good_but_final_failed"] for row in pair_rows)
    groups_with_any_failure = sum(row["num_final_failed"] > 0 for row in group_rows)
    groups_all_failed = sum(row["num_final_failed"] == row["num_rewrites"] for row in group_rows)
    groups_all_correct = sum(row["num_final_correct"] == row["num_rewrites"] for row in group_rows)

    failure_by_group_size = Counter(row["num_final_failed"] for row in group_rows)
    worst_groups = sorted(
        group_rows,
        key=lambda row: (row["num_final_failed"], row["num_rewrites"]),
        reverse=True,
    )[:10]

    return {
        "files": {
            "raw_final_results": str(FINAL_RAW),
            "scored_final_results": str(FINAL_SCORED),
            "bad_decompositions": str(BAD_DECOMPOSITIONS),
        },
        "counts": {
            "raw_rows": len(raw_rows),
            "scored_rows": len(scored_rows),
            "paired_rows": total,
            "original_question_groups": len(group_rows),
            "final_correct_pairs": correct,
            "final_failed_pairs": failed,
            "bad_decomposition_pairs": bad_decomp,
            "bad_decomposition_and_final_failed": both_bad_and_failed,
            "bad_decomposition_but_final_correct": bad_decomp_but_correct,
            "good_decomposition_but_final_failed": good_decomp_but_failed,
            "groups_with_any_final_failure": groups_with_any_failure,
            "groups_all_final_failed": groups_all_failed,
            "groups_all_final_correct": groups_all_correct,
        },
        "rates_percent": {
            "final_exact_match": pct(correct, total),
            "final_failure": pct(failed, total),
            "bad_decomposition_given_final_failure": pct(both_bad_and_failed, failed),
            "final_failure_given_bad_decomposition": pct(both_bad_and_failed, bad_decomp),
            "final_success_despite_bad_decomposition": pct(bad_decomp_but_correct, bad_decomp),
            "failure_with_good_decomposition": pct(good_decomp_but_failed, failed),
            "groups_with_any_final_failure": pct(groups_with_any_failure, len(group_rows)),
            "groups_all_final_failed": pct(groups_all_failed, len(group_rows)),
            "groups_all_final_correct": pct(groups_all_correct, len(group_rows)),
        },
        "failure_count_by_group": dict(sorted(failure_by_group_size.items())),
        "worst_groups": [
            {
                "original_question_id": row["original_question_id"],
                "original_question": row["original_question"],
                "num_rewrites": row["num_rewrites"],
                "num_final_failed": row["num_final_failed"],
                "num_bad_decomposition_and_final_failed": row["num_bad_decomposition_and_final_failed"],
            }
            for row in worst_groups
        ],
        "outputs": {
            "pair_results": str(PAIR_OUTPUT),
            "group_results": str(GROUP_OUTPUT),
            "summary_json": str(SUMMARY_JSON),
            "summary_text": str(SUMMARY_TEXT),
        },
    }


def write_text_summary(summary):
    counts = summary["counts"]
    rates = summary["rates_percent"]
    lines = [
        "Final result comparison by original question id and pair",
        "",
        "Inputs",
        f"- Raw final results: {summary['files']['raw_final_results']}",
        f"- Scored final results: {summary['files']['scored_final_results']}",
        "",
        "Pair Counts",
        f"- Paired rows: {counts['paired_rows']}",
        f"- Original question groups: {counts['original_question_groups']}",
        f"- Final correct pairs: {counts['final_correct_pairs']} ({rates['final_exact_match']}%)",
        f"- Final failed pairs: {counts['final_failed_pairs']} ({rates['final_failure']}%)",
        "",
        "Final Failure vs Bad Decomposition",
        f"- Bad decomposition pairs: {counts['bad_decomposition_pairs']}",
        f"- Bad decomposition and final failed: {counts['bad_decomposition_and_final_failed']}",
        f"- Bad decomposition but final correct: {counts['bad_decomposition_but_final_correct']} ({rates['final_success_despite_bad_decomposition']}% of bad decompositions)",
        f"- Good decomposition but final failed: {counts['good_decomposition_but_final_failed']} ({rates['failure_with_good_decomposition']}% of final failures)",
        f"- Among final failures, bad decomposition rate: {rates['bad_decomposition_given_final_failure']}%",
        f"- Among bad decompositions, final failure rate: {rates['final_failure_given_bad_decomposition']}%",
        "",
        "Group Counts",
        f"- Groups with any final failure: {counts['groups_with_any_final_failure']} ({rates['groups_with_any_final_failure']}%)",
        f"- Groups all final failed: {counts['groups_all_final_failed']} ({rates['groups_all_final_failed']}%)",
        f"- Groups all final correct: {counts['groups_all_final_correct']} ({rates['groups_all_final_correct']}%)",
        "",
        "Worst Groups",
    ]
    for row in summary["worst_groups"][:10]:
        lines.append(
            f"- {row['num_final_failed']}/{row['num_rewrites']} failed, "
            f"{row['num_bad_decomposition_and_final_failed']} also bad decomp: "
            f"{row['original_question']}"
        )
    lines.append("")
    lines.append(f"Pair output: {summary['outputs']['pair_results']}")
    lines.append(f"Group output: {summary['outputs']['group_results']}")

    SUMMARY_TEXT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TEXT, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))
        fout.write("\n")


def main():
    raw_rows = read_jsonl(FINAL_RAW)
    scored_rows = read_jsonl(FINAL_SCORED)
    pair_rows = build_pair_rows(raw_rows, scored_rows)
    group_rows = build_group_rows(pair_rows)
    summary = build_summary(raw_rows, scored_rows, pair_rows, group_rows)

    write_jsonl(pair_rows, PAIR_OUTPUT)
    write_jsonl(group_rows, GROUP_OUTPUT)
    write_json(summary, SUMMARY_JSON)
    write_text_summary(summary)

    counts = summary["counts"]
    rates = summary["rates_percent"]
    print(f"Paired rows: {counts['paired_rows']}")
    print(f"Final correct: {counts['final_correct_pairs']} ({rates['final_exact_match']}%)")
    print(f"Final failed: {counts['final_failed_pairs']} ({rates['final_failure']}%)")
    print(f"Bad decomposition and final failed: {counts['bad_decomposition_and_final_failed']}")
    print(f"Summary: {SUMMARY_TEXT}")


if __name__ == "__main__":
    main()
