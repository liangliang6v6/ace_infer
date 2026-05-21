import json
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path} line {line_no}: {exc}") from exc
    return records


def record_key(record, fallback_index):
    for field in ("id", "question_id", "index"):
        if field in record:
            return record[field]
    return fallback_index


def build_record_map(records):
    mapping = {}
    for idx, record in enumerate(records):
        mapping[record_key(record, idx)] = record
    return mapping


def is_success(score_record):
    return float(score_record.get("accuracy", 0.0)) == 1.0


def is_failure(score_record):
    return float(score_record.get("accuracy", 0.0)) == 0.0


def make_output_record(original_data, original_score, rewrite_data, rewrite_score):
    return {
        "original_q": original_data["question"],
        "answer": original_data["answer"],
        "original_pre": original_score.get("prediction"),
        "original_decomposed": original_data.get("decomposed"),
        "original_intermediate": original_data.get("intermediate_answers"),
        "rewritten_q": rewrite_data["question"],
        "rewritten_pre": rewrite_score.get("prediction"),
        "rewritten_decomposed": rewrite_data.get("decomposed"),
        "rewritten_intermediate": rewrite_data.get("intermediate_answers"),
    }


def compare(base_dir: Path):
    original_records = build_record_map(load_jsonl(base_dir / "original.jsonl"))
    original_score_records = build_record_map(load_jsonl(base_dir / "original_score.jsonl"))
    rewrite_records = build_record_map(load_jsonl(base_dir / "rewrite.jsonl"))
    rewrite_score_records = build_record_map(load_jsonl(base_dir / "rewrite_score.jsonl"))

    shared_keys = sorted(
        set(original_records)
        & set(original_score_records)
        & set(rewrite_records)
        & set(rewrite_score_records)
    )

    output_path = base_dir / "original_success_rewritten_failure.jsonl"
    matched = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for key in shared_keys:
            original_score = original_score_records[key]
            rewrite_score = rewrite_score_records[key]
            if not (is_success(original_score) and is_failure(rewrite_score)):
                continue

            output_record = make_output_record(
                original_records[key],
                original_score,
                rewrite_records[key],
                rewrite_score,
            )
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            matched += 1

    return output_path, matched, len(shared_keys)


def main():
    base_dir = Path(__file__).resolve().parent
    output_path, matched, total = compare(base_dir)
    print(f"Wrote {matched} instances to {output_path} from {total} aligned questions.")


if __name__ == "__main__":
    main()
