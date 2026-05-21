import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ACE_ROOT = Path(__file__).resolve().parents[1]
if str(ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_ROOT))

from get_score import (  # noqa: E402
    accuracy_score,
    exact_match_score,
    extract_prediction,
    metric_max_over_ground_truths,
    prf_max_over_ground_truths,
)


ID_KEYS = (
    "question_id",
    "id",
    "_id",
    "qid",
    "uid",
)
ORIGINAL_ID_KEYS = (
    "original_question_id",
    "original_id",
    "source_question_id",
    "source_id",
)
REWRITE_ID_KEYS = (
    "rewrite_question_id",
    "rewritten_question_id",
    "rewrite_id",
    "rewritten_id",
)
QUESTION_KEYS = ("question", "question_text", "claim")
DEFAULT_ORIGINAL_SOURCE = "eval_datasets/2wiki_origin/test_subsampled.jsonl"
DEFAULT_REWRITE_SOURCE = "eval_datasets/2wiki/test_subsampled.jsonl"
DEFAULT_ORIGINAL_RESULTS = "eval_datasets/test/2wiki_origin/prompts_decompose_test_ace/test_e5-large-v2_k10_passage1.jsonl"
DEFAULT_REWRITE_RESULTS = "eval_datasets/test/2wiki/prompts_decompose_test_ace/test_e5-large-v2_k10_passage1.jsonl"
DEFAULT_OUTPUT_DIR = "eval_datasets/test/2wiki/prompts_decompose_test_ace/eval_results"
DEFAULT_ORIGINAL_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/original_pair_scored.jsonl"
DEFAULT_REWRITE_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/rewrite_pair_scored.jsonl"


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_index"] = line_idx
            records.append(record)
    return records


def get_first(record, keys, default=None):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def as_key(value):
    if value is None:
        return None
    return str(value)


def get_question(record):
    return get_first(record, QUESTION_KEYS, "")


def get_ground_truths(record):
    answers = record.get("answer")
    if answers is None and "answers_objects" in record:
        answers = []
        for answer_obj in record.get("answers_objects", []):
            answers.extend(answer_obj.get("spans", []))

    if isinstance(answers, list):
        return answers if answers else [""]
    if answers is None:
        return [""]
    return [answers]


def build_source_index(records):
    by_id = {}
    by_index = {}
    by_question = {}

    for index, record in enumerate(records):
        source_id = as_key(get_first(record, ID_KEYS, index))
        by_id[source_id] = record
        by_index[index] = source_id
        question = get_question(record)
        if question:
            by_question[question] = source_id

    return {
        "by_id": by_id,
        "by_index": by_index,
        "by_question": by_question,
    }


def index_to_source_id(value, source_index):
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return source_index["by_index"].get(index)


def resolve_source_id(record, source_index, preferred_keys=()):
    for keys in (preferred_keys, ORIGINAL_ID_KEYS, ID_KEYS):
        value = get_first(record, keys)
        key = as_key(value)
        if key in source_index["by_id"]:
            return key

        indexed_key = index_to_source_id(value, source_index)
        if indexed_key is not None:
            return indexed_key

    for key in ("source_index", "index"):
        indexed_key = index_to_source_id(record.get(key), source_index)
        if indexed_key is not None:
            return indexed_key

    question = get_question(record)
    if question in source_index["by_question"]:
        return source_index["by_question"][question]

    value = get_first(record, preferred_keys + ID_KEYS)
    return as_key(value)


def get_rewrite_instance_id(record, fallback_index):
    value = get_first(record, REWRITE_ID_KEYS)
    if value not in (None, ""):
        return value
    return record.get("source_index", record.get("index", fallback_index))


def make_pair_id(original_question_id, rewrite_question_id):
    return f"{original_question_id}::{rewrite_question_id}"


def score_record(record, source_record, output_id, extra_fields=None):
    prediction = record.get("prediction")
    if prediction is None:
        prediction = extract_prediction(record.get("final_answer", ""))
    else:
        prediction = str(prediction).strip()

    ground_truths = get_ground_truths(record)
    if ground_truths == [""] and source_record is not None:
        ground_truths = get_ground_truths(source_record)

    em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    acc = metric_max_over_ground_truths(accuracy_score, prediction, ground_truths)
    precision, recall, f1 = prf_max_over_ground_truths(prediction, ground_truths)

    result = {
        "id": output_id,
        "question": get_question(record),
        "answer": ground_truths,
        "prediction": prediction,
        "accuracy": acc,
        "exact_match": em,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if extra_fields:
        result.update(extra_fields)
    return result


def write_scored(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(float)
    with open(output_path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            for key in ("accuracy", "exact_match", "precision", "recall", "f1"):
                totals[key] += float(record[key])

    total = len(records)
    if total == 0:
        return {"total": 0}

    summary = {"total": total}
    for key in ("accuracy", "exact_match", "precision", "recall", "f1"):
        summary[key] = totals[key] / total
    return summary


def build_result_index(records, source_index):
    by_source_id = {}
    duplicates = 0
    for record in records:
        source_id = resolve_source_id(record, source_index, ORIGINAL_ID_KEYS)
        if source_id is None:
            continue
        if source_id in by_source_id:
            duplicates += 1
            continue
        by_source_id[source_id] = record
    return by_source_id, duplicates


def evaluate_pairs(original_results, rewrite_results, original_source, rewrite_source):
    original_source_index = build_source_index(original_source)
    rewrite_index = build_source_index(rewrite_source)
    original_result_index, duplicate_original_results = build_result_index(
        original_results,
        original_source_index,
    )
    original_scored = []
    rewrite_scored = []
    skipped_rewrites = 0
    missing_original_results = 0

    for fallback_index, record in enumerate(rewrite_results):
        source_id = resolve_source_id(record, rewrite_index, ORIGINAL_ID_KEYS)
        if source_id not in original_source_index["by_id"]:
            source_id = resolve_source_id(record, original_source_index, ORIGINAL_ID_KEYS)

        if source_id not in original_source_index["by_id"]:
            skipped_rewrites += 1
            continue

        original_result = original_result_index.get(source_id)
        if original_result is None:
            missing_original_results += 1
            continue

        rewrite_instance_id = get_rewrite_instance_id(record, fallback_index)
        pair_id = make_pair_id(source_id, rewrite_instance_id)
        original_record = original_source_index["by_id"][source_id]
        original_extra = {
            "pair_id": pair_id,
            "original_question_id": source_id,
            "rewrite_question_id": rewrite_instance_id,
            "original_question": get_question(original_record),
            "rewritten_question": get_question(record),
        }
        rewrite_extra = dict(original_extra)

        original_scored.append(
            score_record(
                original_result,
                original_record,
                output_id=pair_id,
                extra_fields=original_extra,
            )
        )
        rewrite_scored.append(
            score_record(
                record,
                original_record,
                output_id=pair_id,
                extra_fields=rewrite_extra,
            )
        )

    skipped = {
        "rewrite_without_source": skipped_rewrites,
        "rewrite_without_original_result": missing_original_results,
        "duplicate_original_results": duplicate_original_results,
    }
    return original_scored, rewrite_scored, skipped


def print_summary(name, path, summary, skipped=None):
    print(f"{name}: {path}")
    print(f"Total: {summary['total']}")
    if isinstance(skipped, int) and skipped:
        print(f"Skipped: {skipped}")
    if summary["total"] == 0:
        return
    print(f"Accuracy: {summary['accuracy']}")
    print(f"Exact Match: {summary['exact_match']}")
    print(f"Precision: {summary['precision']}")
    print(f"Recall: {summary['recall']}")
    print(f"F1: {summary['f1']}")


def default_output_path(input_path, suffix):
    input_path = Path(input_path)
    return input_path.with_name(input_path.stem + suffix + input_path.suffix)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score original and rewritten 2Wiki outputs as independent (q, q'_j) pairs. "
            "The output rows keep the same metric fields as get_score.py, with extra "
            "original/rewrite ID fields for paired analysis."
        )
    )
    parser.add_argument(
        "--original-source",
        default=DEFAULT_ORIGINAL_SOURCE,
        help="Original 2Wiki source JSONL.",
    )
    parser.add_argument(
        "--rewrite-source",
        default=DEFAULT_REWRITE_SOURCE,
        help="Filtered/expanded rewritten 2Wiki source JSONL.",
    )
    parser.add_argument(
        "--original-results",
        default=DEFAULT_ORIGINAL_RESULTS,
        help="Raw model output JSONL for original questions.",
    )
    parser.add_argument(
        "--rewrite-results",
        default=DEFAULT_REWRITE_RESULTS,
        help="Raw model output JSONL for rewritten questions.",
    )
    parser.add_argument(
        "--original-output",
        default=None,
        help=(
            "Scored output for original questions, repeated once per rewrite pair. "
            "Defaults to <original-results>_paired_original_scored.jsonl."
        ),
    )
    parser.add_argument(
        "--rewrite-output",
        default=None,
        help="Scored output for rewritten questions. Defaults to <rewrite-results>_paired_rewrite_scored.jsonl.",
    )
    args = parser.parse_args()

    original_source = read_jsonl(args.original_source)
    rewrite_source = read_jsonl(args.rewrite_source)
    original_results = read_jsonl(args.original_results)
    rewrite_results = read_jsonl(args.rewrite_results)

    original_scored, rewrite_scored, skipped = evaluate_pairs(
        original_results,
        rewrite_results,
        original_source,
        rewrite_source,
    )

    if args.original_output:
        original_output = args.original_output
    elif args.original_results == DEFAULT_ORIGINAL_RESULTS:
        original_output = DEFAULT_ORIGINAL_OUTPUT
    else:
        original_output = default_output_path(args.original_results, "_paired_original_scored")

    if args.rewrite_output:
        rewrite_output = args.rewrite_output
    elif args.rewrite_results == DEFAULT_REWRITE_RESULTS:
        rewrite_output = DEFAULT_REWRITE_OUTPUT
    else:
        rewrite_output = default_output_path(args.rewrite_results, "_paired_rewrite_scored")

    original_summary = write_scored(original_scored, original_output)
    rewrite_summary = write_scored(rewrite_scored, rewrite_output)

    print("Pairs:", len(rewrite_scored))
    if skipped["rewrite_without_source"]:
        print("Rewrite rows without source ID:", skipped["rewrite_without_source"])
    if skipped["rewrite_without_original_result"]:
        print("Rewrite rows without original result:", skipped["rewrite_without_original_result"])
    if skipped["duplicate_original_results"]:
        print("Duplicate original result rows ignored:", skipped["duplicate_original_results"])
    print("-" * 60)
    print_summary("Original scored output", original_output, original_summary)
    print("-" * 60)
    print_summary("Rewrite scored output", rewrite_output, rewrite_summary)


if __name__ == "__main__":
    main()
