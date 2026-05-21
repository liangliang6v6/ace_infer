import json
from collections import Counter, defaultdict
from pathlib import Path


RESULT_DIR = Path("eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/eval_results")
ORIGINAL_PLAN_FILE = Path("eval_datasets/test/2wiki_origin/prompts_decompose_test_t0.0_ace/generate.jsonl")
REWRITE_FILE = Path("eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/generate.jsonl")
PLAN_FILE = RESULT_DIR / "plan_pair_eval.jsonl"
JSON_OUTPUT = RESULT_DIR / "analysis_summary.json"
TEXT_OUTPUT = RESULT_DIR / "analysis_summary.txt"
BAD_OUTPUT = RESULT_DIR / "bad_decompositions.jsonl"
EXAMPLES_OUTPUT = RESULT_DIR / "bad_decomposition_examples.txt"


FAILURE_MODE_INFO = {
    "first_hop_relation_changed": (
        "Q1 changes the relation used by the original decomposition.",
        "The model follows a different reasoning path from the start.",
    ),
    "first_hop_anchor_missing": (
        "Q1 drops the original first-hop anchor/entity.",
        "The model may retrieve a different entity before later hops begin.",
    ),
    "rewrite_constraint_frontloaded": (
        "Rewrite-only constraint is pulled into Q1 while the first hop drifts.",
        "A distractor/detail from the rewrite hijacks the first retrieval step.",
    ),
    "foreign_relation_substitution": (
        "Q1 introduces a new relation while drifting from the original plan.",
        "The model substitutes another relation for the intended first hop.",
    ),
    "low_first_hop_overlap": (
        "Q1 has very low content overlap with original Q1.",
        "The first subquestion is not preserving the original decomposition step.",
    ),
}


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


def pct(count, total):
    if total == 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def get_question(row):
    return row.get("question", row.get("question_text", ""))


def format_decomposition(row):
    decomposed = row.get("decomposed", []) if row else []
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
        formatted.append(f"{label}: {text}")
    return formatted


def rewrite_pair_id(row, fallback_index):
    rewrite_id = row.get("source_index", row.get("index", fallback_index))
    return f"{row.get('question_id')}::{rewrite_id}"


def build_plan_maps():
    original_plans = read_jsonl(ORIGINAL_PLAN_FILE)
    rewrite_plans = read_jsonl(REWRITE_FILE)
    original_by_question = {get_question(row): row for row in original_plans}
    rewrite_by_pair_id = {
        rewrite_pair_id(row, index): row
        for index, row in enumerate(rewrite_plans)
    }
    return original_by_question, rewrite_by_pair_id


def failure_modes(row):
    modes = []
    token_drift = row.get("original_first_token_coverage", 1.0) < 0.35
    relation_changed = bool(row.get("relation_changed"))
    anchor_missing = bool(row.get("anchor_missing"))
    frontloaded = bool(row.get("rewrite_constraint_moved_to_first"))
    foreign_relation = bool(row.get("foreign_relation_introduced"))

    if relation_changed:
        modes.append("first_hop_relation_changed")
    if anchor_missing:
        modes.append("first_hop_anchor_missing")
    if frontloaded and (relation_changed or anchor_missing or token_drift):
        modes.append("rewrite_constraint_frontloaded")
    if foreign_relation and (relation_changed or anchor_missing):
        modes.append("foreign_relation_substitution")
    if token_drift:
        modes.append("low_first_hop_overlap")
    return modes


def bad_reason(modes):
    return " ".join(FAILURE_MODE_INFO[mode][0] for mode in modes)


def compact_bad_row(row, original_by_question, rewrite_by_pair_id):
    modes = failure_modes(row)
    original_plan = original_by_question.get(row["original_question"])
    rewrite_plan = rewrite_by_pair_id.get(row["pair_id"])
    return {
        "pair_id": row["pair_id"],
        "original_question_id": row["original_question_id"],
        "rewrite_question_id": row["rewrite_question_id"],
        "failure_modes": modes,
        "why_bad": bad_reason(modes),
        "original_question": row["original_question"],
        "rewritten_question": row["rewritten_question"],
        "original_decomposition": format_decomposition(original_plan),
        "rewritten_decomposition": format_decomposition(rewrite_plan),
        "original_first_subquestion": row["original_first_subquestion"],
        "rewritten_first_subquestion": row["rewrite_first_subquestion"],
    }


def choose_examples(bad_rows, limit=8):
    examples = []
    used = set()
    by_mode = defaultdict(list)
    for row in bad_rows:
        for mode in row["failure_modes"]:
            by_mode[mode].append(row)

    for mode in FAILURE_MODE_INFO:
        for row in by_mode.get(mode, []):
            if row["pair_id"] in used:
                continue
            examples.append(row)
            used.add(row["pair_id"])
            break

    for row in bad_rows:
        if len(examples) >= limit:
            break
        if row["pair_id"] not in used:
            examples.append(row)
            used.add(row["pair_id"])
    return examples


def build_summary():
    plan_rows = read_jsonl(PLAN_FILE)
    original_by_question, rewrite_by_pair_id = build_plan_maps()

    bad_rows = [
        compact_bad_row(row, original_by_question, rewrite_by_pair_id)
        for row in plan_rows
        if failure_modes(row)
    ]

    mode_counts = Counter()
    for row in bad_rows:
        mode_counts.update(row["failure_modes"])

    original_ids = {row["original_question_id"] for row in plan_rows}
    bad_original_ids = {row["original_question_id"] for row in bad_rows}
    total_by_original = Counter(row["original_question_id"] for row in plan_rows)
    bad_by_original = Counter(row["original_question_id"] for row in bad_rows)
    all_bad_groups = sum(
        1
        for original_id, total in total_by_original.items()
        if bad_by_original.get(original_id, 0) == total
    )

    examples = choose_examples(bad_rows)
    summary = {
        "definition": (
            "A bad decomposition is a rewritten-question decomposition whose first hop "
            "drifts from the original decomposition's first hop. This identifies model "
            "decomposition failure, not rewritten-question invalidity."
        ),
        "counts": {
            "evaluated_pairs": len(plan_rows),
            "bad_decompositions": len(bad_rows),
            "good_decompositions": len(plan_rows) - len(bad_rows),
            "original_question_groups": len(original_ids),
            "groups_with_bad_decomposition": len(bad_original_ids),
            "groups_without_bad_decomposition": len(original_ids - bad_original_ids),
            "groups_all_rewrites_bad": all_bad_groups,
        },
        "rates_percent": {
            "bad_decomposition_rate": pct(len(bad_rows), len(plan_rows)),
            "group_bad_decomposition_rate": pct(len(bad_original_ids), len(original_ids)),
            "all_rewrites_bad_group_rate": pct(all_bad_groups, len(original_ids)),
        },
        "failure_modes": [
            {
                "mode": mode,
                "count": mode_counts.get(mode, 0),
                "percent_of_bad": pct(mode_counts.get(mode, 0), len(bad_rows)),
                "what_it_means": FAILURE_MODE_INFO[mode][0],
                "likely_cause": FAILURE_MODE_INFO[mode][1],
            }
            for mode in FAILURE_MODE_INFO
            if mode_counts.get(mode, 0)
        ],
        "examples": examples,
        "outputs": {
            "bad_decompositions": str(BAD_OUTPUT),
            "summary_txt": str(TEXT_OUTPUT),
            "examples_txt": str(EXAMPLES_OUTPUT),
        },
    }
    return summary, bad_rows, examples


def write_text_summary(summary, examples):
    counts = summary["counts"]
    rates = summary["rates_percent"]
    lines = [
        "Bad decomposition analysis",
        "",
        "Goal",
        "- Identify model decomposition failures by comparing each rewritten decomposition against the original decomposition.",
        "- This does not judge whether the rewritten question itself is bad.",
        "",
        "Counts",
        f"- Evaluated pairs: {counts['evaluated_pairs']}",
        f"- Bad decompositions: {counts['bad_decompositions']} ({rates['bad_decomposition_rate']}%)",
        f"- Good decompositions: {counts['good_decompositions']}",
        f"- Original question groups: {counts['original_question_groups']}",
        f"- Groups with at least one bad decomposition: {counts['groups_with_bad_decomposition']} ({rates['group_bad_decomposition_rate']}%)",
        f"- Groups with no bad decomposition: {counts['groups_without_bad_decomposition']}",
        f"- Groups where all rewrites decomposed badly: {counts['groups_all_rewrites_bad']} ({rates['all_rewrites_bad_group_rate']}%)",
        "",
        "Failure modes found in the results",
    ]

    for item in summary["failure_modes"]:
        lines.append(
            f"- {item['mode']}: {item['count']} ({item['percent_of_bad']}% of bad)"
        )
        lines.append(f"  Cause: {item['likely_cause']}")

    lines.extend(["", "Examples"])
    for index, example in enumerate(examples, 1):
        lines.append(f"Example {index}: {example['pair_id']}")
        lines.append(f"Failure modes: {', '.join(example['failure_modes'])}")
        lines.append(f"Why bad: {example['why_bad']}")
        lines.append(f"Original: {example['original_question']}")
        lines.append(f"Rewritten: {example['rewritten_question']}")
        lines.append("Original decomposition:")
        for item in example["original_decomposition"]:
            lines.append(f"  {item}")
        lines.append("Rewritten decomposition:")
        for item in example["rewritten_decomposition"]:
            lines.append(f"  {item}")
        lines.append("")

    with open(TEXT_OUTPUT, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))
        fout.write("\n")

    with open(EXAMPLES_OUTPUT, "w", encoding="utf-8") as fout:
        for index, example in enumerate(examples, 1):
            fout.write(f"Example {index}: {example['pair_id']}\n")
            fout.write(f"Failure modes: {', '.join(example['failure_modes'])}\n")
            fout.write(f"Why bad: {example['why_bad']}\n")
            fout.write(f"Original: {example['original_question']}\n")
            fout.write(f"Rewritten: {example['rewritten_question']}\n")
            fout.write("Original decomposition:\n")
            for item in example["original_decomposition"]:
                fout.write(f"  {item}\n")
            fout.write("Rewritten decomposition:\n")
            for item in example["rewritten_decomposition"]:
                fout.write(f"  {item}\n")
            fout.write("\n")


def main():
    summary, bad_rows, examples = build_summary()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")
    write_jsonl(bad_rows, BAD_OUTPUT)
    write_text_summary(summary, examples)

    counts = summary["counts"]
    rates = summary["rates_percent"]
    print(f"Bad decompositions: {counts['bad_decompositions']}/{counts['evaluated_pairs']} ({rates['bad_decomposition_rate']}%)")
    print(f"Groups with bad decomposition: {counts['groups_with_bad_decomposition']}/{counts['original_question_groups']} ({rates['group_bad_decomposition_rate']}%)")
    print(f"Summary: {TEXT_OUTPUT}")
    print(f"Bad rows: {BAD_OUTPUT}")
    print(f"Examples: {EXAMPLES_OUTPUT}")


if __name__ == "__main__":
    main()
