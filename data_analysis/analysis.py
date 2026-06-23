import argparse
import json
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "2wiki"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "original_acc1_rewrite_acc0.jsonl"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_decomposition(row):
    decomposed = row.get("decomposed", [])
    if not isinstance(decomposed, list):
        return [str(decomposed)] if decomposed else []

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


def record_key(row, fallback_index):
    question_id = row.get("question_id", row.get("id", fallback_index))
    source_index = row.get("source_index", row.get("index", fallback_index))
    decompose_id = row.get("decompose_id", 0)
    index = row.get("index", fallback_index)
    return question_id, source_index, decompose_id, index


def index_rows(rows):
    indexed = {}
    duplicates = []
    for index, row in enumerate(rows):
        key = record_key(row, index)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    return indexed, duplicates


def is_score(row, field, value):
    try:
        return float(row.get(field, 0.0)) == float(value)
    except (TypeError, ValueError):
        return False


def score_value(row, field):
    try:
        return float(row.get(field, 0.0))
    except (TypeError, ValueError):
        return 0.0


def correct(row, field):
    return score_value(row, field) == 1.0


def pair_id_from_score(row, key):
    return row.get("pair_id", f"{key[0]}::{key[1]}")


def build_original_line(original_full, original_scored, pair_id, answer):
    return {
        "id": pair_id,
        "original_q": original_full.get(
            "question", original_scored.get("question", "")
        ),
        "answer": answer,
        "original_pre": original_scored.get("prediction", ""),
        "original_decomp": format_decomposition(original_full),
        "original_inter": original_full.get("intermediate_answers", {}),
    }


def build_rewrite_line(rewrite_full, rewrite_scored, pair_id, answer):
    return {
        "id": pair_id,
        "rewrite_q": rewrite_full.get(
            "question", rewrite_scored.get("question", "")
        ),
        "answer": rewrite_scored.get("answer", answer),
        "rewrite_pre": rewrite_scored.get("prediction", ""),
        "rewrite_decomp": format_decomposition(rewrite_full),
        "rewrite_inter": rewrite_full.get("intermediate_answers", {}),
    }


def build_pair_example(
    original_full,
    original_scored,
    rewrite_full,
    rewrite_scored,
    pair_id,
    answer,
    score_field,
):
    return {
        "id": pair_id,
        "answer": answer,
        "original": {
            "question": original_full.get(
                "question", original_scored.get("question", "")
            ),
            "prediction": original_scored.get("prediction", ""),
            "score": score_value(original_scored, score_field),
            "decomp": format_decomposition(original_full),
            "intermediate_answers": original_full.get("intermediate_answers", {}),
        },
        "rewrite": {
            "question": rewrite_full.get(
                "question", rewrite_scored.get("question", "")
            ),
            "prediction": rewrite_scored.get("prediction", ""),
            "score": score_value(rewrite_scored, score_field),
            "decomp": format_decomposition(rewrite_full),
            "intermediate_answers": rewrite_full.get("intermediate_answers", {}),
        },
    }


def add_example(summary, section_name, example, limit):
    section = summary["sections"][section_name]
    section["count"] += 1
    if section_name != "original_succeed_rewrite_failed":
        return
    if limit is None or len(section["examples"]) < limit:
        section["examples"].append(example)


def build_lines(
    original_raw,
    original_score,
    rewrite_raw,
    rewrite_score,
    score_field,
    summary_example_limit,
):
    original_by_key, original_duplicates = index_rows(original_raw)
    rewrite_by_key, rewrite_duplicates = index_rows(rewrite_raw)

    output_lines = []
    summary = {
        "original_raw_rows": len(original_raw),
        "original_score_rows": len(original_score),
        "rewrite_raw_rows": len(rewrite_raw),
        "rewrite_score_rows": len(rewrite_score),
        "score_field": score_field,
        "matched_pairs_checked": 0,
        "original_score_1_rewrite_score_0_pairs": 0,
        "missing_original_full_rows": 0,
        "missing_rewrite_full_rows": 0,
        "duplicate_original_full_keys": len(original_duplicates),
        "duplicate_rewrite_full_keys": len(rewrite_duplicates),
        "overview": {},
        "sections": {
            "both_succeed": {"count": 0, "examples": []},
            "both_failed": {"count": 0, "examples": []},
            "original_failed": {"count": 0, "examples": []},
            "rewrite_failed": {"count": 0, "examples": []},
            "original_failed_rewrite_succeed": {"count": 0, "examples": []},
            "original_succeed_rewrite_failed": {"count": 0, "examples": []},
        },
    }

    for index, (original_scored, rewrite_scored) in enumerate(
        zip(original_score, rewrite_score)
    ):
        original_key = record_key(original_scored, index)
        rewrite_key = record_key(rewrite_scored, index)
        original_full = original_by_key.get(original_key)
        rewrite_full = rewrite_by_key.get(rewrite_key)

        if original_full is None:
            summary["missing_original_full_rows"] += 1
            continue
        if rewrite_full is None:
            summary["missing_rewrite_full_rows"] += 1
            continue

        summary["matched_pairs_checked"] += 1
        original_correct = correct(original_scored, score_field)
        rewrite_correct = correct(rewrite_scored, score_field)
        pair_id = pair_id_from_score(original_scored, original_key)
        answer = original_scored.get("answer", original_full.get("answer", []))
        example = build_pair_example(
            original_full,
            original_scored,
            rewrite_full,
            rewrite_scored,
            pair_id,
            answer,
            score_field,
        )

        if original_correct and rewrite_correct:
            add_example(summary, "both_succeed", example, summary_example_limit)
        elif not original_correct and not rewrite_correct:
            add_example(summary, "both_failed", example, summary_example_limit)

        if not original_correct:
            add_example(summary, "original_failed", example, summary_example_limit)
        if not rewrite_correct:
            add_example(summary, "rewrite_failed", example, summary_example_limit)
        if not original_correct and rewrite_correct:
            add_example(
                summary,
                "original_failed_rewrite_succeed",
                example,
                summary_example_limit,
            )
        if original_correct and not rewrite_correct:
            add_example(
                summary,
                "original_succeed_rewrite_failed",
                example,
                summary_example_limit,
            )

        if not (original_correct and not rewrite_correct):
            continue

        summary["original_score_1_rewrite_score_0_pairs"] += 1

        output_lines.append(
            build_original_line(original_full, original_scored, pair_id, answer)
        )
        output_lines.append(
            build_rewrite_line(rewrite_full, rewrite_scored, pair_id, answer)
        )

    sections = summary["sections"]
    summary["overview"] = {
        "both_succeed": sections["both_succeed"]["count"],
        "both_failed": sections["both_failed"]["count"],
        "original_failed": sections["original_failed"]["count"],
        "rewrite_failed": sections["rewrite_failed"]["count"],
        "original_failed_rewrite_succeed": sections[
            "original_failed_rewrite_succeed"
        ]["count"],
        "original_succeed_rewrite_failed": sections[
            "original_succeed_rewrite_failed"
        ]["count"],
    }

    return output_lines, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find cases where the original question scored 1 but the rewritten "
            "question scored 0, and write original/rewrite records as JSONL."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--original-score", type=Path)
    parser.add_argument("--rewrite", type=Path)
    parser.add_argument("--rewrite-score", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--score-field", default="accuracy")
    parser.add_argument(
        "--summary-example-limit",
        type=int,
        default=-1,
        help="Number of examples to keep for each summary section. Default -1 keeps all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    original_path = args.original or args.data_dir / "original.jsonl"
    original_score_path = args.original_score or args.data_dir / "original_score.jsonl"
    rewrite_path = args.rewrite or args.data_dir / "rewrite.jsonl"
    rewrite_score_path = args.rewrite_score or args.data_dir / "rewrite_score.jsonl"
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")

    output_lines, summary = build_lines(
        read_jsonl(original_path),
        read_jsonl(original_score_path),
        read_jsonl(rewrite_path),
        read_jsonl(rewrite_score_path),
        args.score_field,
        None if args.summary_example_limit < 0 else args.summary_example_limit,
    )
    write_jsonl(output_lines, args.output)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    print(f"Wrote {len(output_lines)} lines to {args.output}")
    print(
        "Pairs:",
        summary["original_score_1_rewrite_score_0_pairs"],
        f"({args.score_field}: original=1, rewrite=0)",
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
