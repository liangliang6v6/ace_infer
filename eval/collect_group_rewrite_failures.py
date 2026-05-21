import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ACE_ROOT = Path(__file__).resolve().parents[1]
if str(ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_ROOT))

from eval.score_rewrite_pairs import (  # noqa: E402
    ORIGINAL_ID_KEYS,
    build_source_index,
    get_first,
    get_question,
    get_rewrite_instance_id,
    make_pair_id,
    read_jsonl,
    resolve_source_id,
)
from eval.evaluate_decomposition_pairs import (  # noqa: E402
    evaluate_decomposition_pairs,
    summarize,
    write_jsonl as write_plan_jsonl,
)


DEFAULT_ORIGINAL_SOURCE = "eval_datasets/2wiki_origin/test_subsampled.jsonl"
DEFAULT_REWRITE_SOURCE = "eval_datasets/2wiki/test_subsampled.jsonl"
DEFAULT_ORIGINAL_PLANS = "eval_datasets/test/2wiki_origin/prompts_decompose_test_t0.0_ace/generate.jsonl"
DEFAULT_REWRITE_RECORDS = "eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/generate.jsonl"
DEFAULT_OUTPUT_DIR = "eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/eval_results"
DEFAULT_PLAN_EVAL = f"{DEFAULT_OUTPUT_DIR}/plan_pair_eval.jsonl"
DEFAULT_FAILED_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/failed_rewrites.jsonl"
DEFAULT_GROUP_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/failed_rewrite_groups.jsonl"
DEFAULT_SUMMARY_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/summary.json"


def write_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(record, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(record, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def record_pair_id(record):
    pair_id = record.get("pair_id")
    if pair_id not in (None, ""):
        return str(pair_id)

    original_id = get_first(record, ("original_question_id",) + ORIGINAL_ID_KEYS)
    rewrite_id = get_first(
        record,
        ("rewrite_question_id", "rewritten_question_id", "rewrite_id", "rewritten_id"),
    )
    if original_id not in (None, "") and rewrite_id not in (None, ""):
        return make_pair_id(original_id, rewrite_id)

    value = record.get("id")
    return None if value in (None, "") else str(value)


def index_by_pair_id(records):
    indexed = {}
    duplicates = 0
    missing = 0
    for record in records:
        pair_id = record_pair_id(record)
        if pair_id is None:
            missing += 1
            continue
        if pair_id in indexed:
            duplicates += 1
            continue
        indexed[pair_id] = record
    return indexed, {"duplicates": duplicates, "missing_pair_id": missing}


def resolve_rewrite_source_id(record, rewrite_source_index, original_source_index):
    question = get_question(record)
    if question in rewrite_source_index["by_question"]:
        return rewrite_source_index["by_question"][question]

    source_id = resolve_source_id(record, rewrite_source_index, ORIGINAL_ID_KEYS)
    if source_id in original_source_index["by_id"]:
        return source_id

    return resolve_source_id(record, original_source_index, ORIGINAL_ID_KEYS)


def build_pair_rows(rewrite_records, original_source, rewrite_source):
    original_source_index = build_source_index(original_source)
    rewrite_source_index = build_source_index(rewrite_source)
    rows = []
    skipped = defaultdict(int)

    for fallback_index, rewrite_record in enumerate(rewrite_records):
        original_id = resolve_rewrite_source_id(
            rewrite_record,
            rewrite_source_index,
            original_source_index,
        )
        if original_id not in original_source_index["by_id"]:
            skipped["rewrite_without_original_source"] += 1
            continue

        rewrite_id = get_rewrite_instance_id(rewrite_record, fallback_index)
        original_record = original_source_index["by_id"][original_id]
        pair_id = make_pair_id(original_id, rewrite_id)
        rows.append(
            {
                "pair_id": pair_id,
                "original_question_id": original_id,
                "rewrite_question_id": rewrite_id,
                "original_question": get_question(original_record),
                "rewritten_question": get_question(rewrite_record),
                "answer": rewrite_record.get("answer", original_record.get("answer")),
            }
        )

    return rows, dict(skipped)


def answer_failed(score_record, metric, min_score):
    if score_record is None:
        return False
    value = score_record.get(metric)
    if value is None:
        return False
    return float(value) < min_score


def plan_failed(plan_record, failed_risks):
    if plan_record is None:
        return False
    return plan_record.get("first_hop_risk") in failed_risks


def failure_reasons(plan_record, score_record, failed_risks, metric, min_score):
    reasons = []
    if plan_failed(plan_record, failed_risks):
        reasons.append(f"plan_risk={plan_record.get('first_hop_risk')}")
    if answer_failed(score_record, metric, min_score):
        reasons.append(f"{metric}<{min_score}")
    return reasons


def merge_pair_context(pair_row, plan_record, rewrite_score_record, original_score_record):
    merged = dict(pair_row)
    if plan_record:
        plan_fields = (
            "original_first_subquestion",
            "rewrite_first_subquestion",
            "first_hop_risk",
            "relation_changed",
            "anchor_missing",
            "original_key_missing",
            "original_relation_missing",
            "foreign_relation_introduced",
            "rewrite_constraint_moved_to_first",
            "missing_original_question_relations",
            "introduced_foreign_relations",
            "original_question_relations",
            "rewrite_first_relations",
        )
        for field in plan_fields:
            if field in plan_record:
                merged[field] = plan_record[field]

    if rewrite_score_record:
        for field in ("prediction", "accuracy", "exact_match", "precision", "recall", "f1"):
            if field in rewrite_score_record:
                merged[f"rewrite_{field}"] = rewrite_score_record[field]

    if original_score_record:
        for field in ("prediction", "accuracy", "exact_match", "precision", "recall", "f1"):
            if field in original_score_record:
                merged[f"original_{field}"] = original_score_record[field]

    return merged


def summarize_groups(failed_rows, all_pair_rows):
    grouped_failed = defaultdict(list)
    total_by_original = defaultdict(int)
    original_question = {}

    for row in all_pair_rows:
        original_id = row["original_question_id"]
        total_by_original[original_id] += 1
        original_question[original_id] = row.get("original_question", "")

    for row in failed_rows:
        grouped_failed[row["original_question_id"]].append(row)

    summaries = []
    for original_id in sorted(grouped_failed):
        failures = grouped_failed[original_id]
        summaries.append(
            {
                "id": original_id,
                "original_question_id": original_id,
                "original_question": original_question.get(original_id, ""),
                "num_rewrites": total_by_original[original_id],
                "num_failed_rewrites": len(failures),
                "failed_rewrite_question_ids": [
                    row["rewrite_question_id"] for row in failures
                ],
                "failed_rewrites": failures,
            }
        )

    return summaries


def default_output_path(input_path, suffix):
    input_path = Path(input_path)
    return input_path.with_name(input_path.stem + suffix + input_path.suffix)


def should_use_plan(failure_source):
    return failure_source in {"plan", "either", "both"}


def load_or_build_plan_eval(args, original_source, rewrite_source):
    if not args.plan_eval or not should_use_plan(args.failure_source):
        return {}, {}

    plan_path = Path(args.plan_eval)
    if plan_path.exists():
        return index_by_pair_id(read_jsonl(plan_path))

    original_plans = read_jsonl(args.original_plans)
    rewrite_plans = read_jsonl(args.rewrite_records)
    rows, skipped = evaluate_decomposition_pairs(
        original_plans,
        rewrite_plans,
        original_source,
        rewrite_source,
    )
    write_plan_jsonl(rows, plan_path)
    summary = summarize(rows)
    print(f"Built missing plan eval: {plan_path}")
    print(f"Plan eval pairs: {summary['total']}")
    if skipped:
        print("Plan eval skipped:", json.dumps(skipped, sort_keys=True))
    return index_by_pair_id(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Group multiple rewritten questions under each original question and collect "
            "the failed rewrite variants for final analysis."
        )
    )
    parser.add_argument(
        "--original-source",
        default=DEFAULT_ORIGINAL_SOURCE,
        help="Original source JSONL.",
    )
    parser.add_argument(
        "--rewrite-source",
        default=DEFAULT_REWRITE_SOURCE,
        help="Rewrite source JSONL with several rows per original question ID.",
    )
    parser.add_argument(
        "--rewrite-records",
        default=DEFAULT_REWRITE_RECORDS,
        help=(
            "Rewrite plan/result JSONL. Defaults to --rewrite-source. Used to define "
            "the full set of (q, q'_j) pairs."
        ),
    )
    parser.add_argument(
        "--original-plans",
        default=DEFAULT_ORIGINAL_PLANS,
        help="Original decomposition JSONL. Used to auto-build --plan-eval when needed.",
    )
    parser.add_argument(
        "--plan-eval",
        default=DEFAULT_PLAN_EVAL,
        help="Optional output from evaluate_decomposition_pairs.py.",
    )
    parser.add_argument(
        "--rewrite-scored",
        default=None,
        help="Optional scored rewrite output from score_rewrite_pairs.py or get_score.py.",
    )
    parser.add_argument(
        "--original-scored",
        default=None,
        help="Optional paired original scored output from score_rewrite_pairs.py.",
    )
    parser.add_argument(
        "--failure-source",
        choices=("answer", "plan", "either", "both"),
        default="plan",
        help=(
            "Which signal marks a rewrite as failed. answer uses --rewrite-scored; "
            "plan uses --plan-eval; either/both combine them."
        ),
    )
    parser.add_argument(
        "--metric",
        default="exact_match",
        help="Rewrite score metric used when --rewrite-scored is provided.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="A rewrite answer is failed when metric < min-score.",
    )
    parser.add_argument(
        "--failed-risks",
        default="high",
        help="Comma-separated first_hop_risk labels treated as plan failures.",
    )
    parser.add_argument(
        "--failed-output",
        default=DEFAULT_FAILED_OUTPUT,
        help="Flat JSONL of failed rewrite pairs.",
    )
    parser.add_argument(
        "--group-output",
        default=DEFAULT_GROUP_OUTPUT,
        help="Grouped JSONL, one row per original question with failed rewrites.",
    )
    parser.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Compact JSON summary of the grouped failure results.",
    )
    args = parser.parse_args()

    original_source = read_jsonl(args.original_source)
    rewrite_source = read_jsonl(args.rewrite_source)
    rewrite_records = read_jsonl(args.rewrite_records or args.rewrite_source)
    pair_rows, skipped_pairs = build_pair_rows(
        rewrite_records,
        original_source,
        rewrite_source,
    )

    plan_index = {}
    plan_index_stats = {}
    plan_index, plan_index_stats = load_or_build_plan_eval(
        args,
        original_source,
        rewrite_source,
    )

    rewrite_score_index = {}
    rewrite_score_stats = {}
    if args.rewrite_scored:
        rewrite_score_index, rewrite_score_stats = index_by_pair_id(read_jsonl(args.rewrite_scored))

    original_score_index = {}
    original_score_stats = {}
    if args.original_scored:
        original_score_index, original_score_stats = index_by_pair_id(read_jsonl(args.original_scored))

    failed_risks = {risk.strip() for risk in args.failed_risks.split(",") if risk.strip()}
    failed_rows = []
    for pair_row in pair_rows:
        pair_id = pair_row["pair_id"]
        plan_record = plan_index.get(pair_id)
        rewrite_score_record = rewrite_score_index.get(pair_id)
        original_score_record = original_score_index.get(pair_id)

        plan_is_failed = plan_failed(plan_record, failed_risks)
        answer_is_failed = answer_failed(rewrite_score_record, args.metric, args.min_score)

        if args.failure_source == "plan":
            is_failed = plan_is_failed
        elif args.failure_source == "answer":
            is_failed = answer_is_failed
        elif args.failure_source == "both":
            is_failed = plan_is_failed and answer_is_failed
        else:
            is_failed = plan_is_failed or answer_is_failed

        if not is_failed:
            continue

        merged = merge_pair_context(
            pair_row,
            plan_record,
            rewrite_score_record,
            original_score_record,
        )
        merged["failure_reasons"] = failure_reasons(
            plan_record,
            rewrite_score_record,
            failed_risks,
            args.metric,
            args.min_score,
        )
        failed_rows.append(merged)

    grouped_rows = summarize_groups(failed_rows, pair_rows)

    base_path = args.plan_eval or args.rewrite_scored or args.rewrite_records or args.rewrite_source
    failed_output = args.failed_output or default_output_path(base_path, "_failed_rewrites")
    group_output = args.group_output or default_output_path(base_path, "_failed_rewrite_groups")
    summary_output = args.summary_output
    write_jsonl(failed_rows, failed_output)
    write_jsonl(grouped_rows, group_output)

    summary_record = {
        "pairs": len(pair_rows),
        "original_groups": len({row["original_question_id"] for row in pair_rows}),
        "failed_rewrite_pairs": len(failed_rows),
        "groups_with_failures": len(grouped_rows),
        "failure_source": args.failure_source,
        "failed_risks": sorted(failed_risks),
        "metric": args.metric,
        "min_score": args.min_score,
        "skipped_pairs": skipped_pairs,
        "plan_index": plan_index_stats,
        "rewrite_score_index": rewrite_score_stats,
        "original_score_index": original_score_stats,
        "outputs": {
            "plan_eval": args.plan_eval,
            "failed_rewrites": str(failed_output),
            "failed_rewrite_groups": str(group_output),
            "summary": str(summary_output),
        },
    }
    write_json(summary_record, summary_output)

    print(f"Pairs: {len(pair_rows)}")
    print(f"Original groups: {len({row['original_question_id'] for row in pair_rows})}")
    print(f"Failed rewrite pairs: {len(failed_rows)}")
    print(f"Groups with failures: {len(grouped_rows)}")
    if skipped_pairs:
        print("Skipped pairs:", json.dumps(skipped_pairs, sort_keys=True))
    if plan_index_stats:
        print("Plan index:", json.dumps(plan_index_stats, sort_keys=True))
    if rewrite_score_stats:
        print("Rewrite score index:", json.dumps(rewrite_score_stats, sort_keys=True))
    if original_score_stats:
        print("Original score index:", json.dumps(original_score_stats, sort_keys=True))
    print(f"Failed output: {failed_output}")
    print(f"Group output: {group_output}")
    print(f"Summary output: {summary_output}")


if __name__ == "__main__":
    main()
