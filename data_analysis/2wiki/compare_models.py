import json
from pathlib import Path


ACE_NAME = "ace"
QWEN_NAME = "qwen"


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


def make_output_record(original_data, ace_data, ace_score, qwen_data, qwen_score):
    return {
        "original_q": original_data["question"],
        "rewritten_q": ace_data["question"],
        "answer": original_data["answer"],
        "ace_pre": ace_score.get("prediction"),
        "ace_decomposed": ace_data.get("decomposed"),
        "ace_intermediate": ace_data.get("intermediate_answers"),
        "qwen_pre": qwen_score.get("prediction"),
        "qwen_decomposed": qwen_data.get("decomposed"),
        "qwen_intermediate": qwen_data.get("intermediate_answers"),
    }


def write_directional_output(output_path: Path, original_records, ace_records, ace_scores, qwen_records, qwen_scores, mode: str):
    shared_keys = sorted(
        set(original_records)
        & set(ace_records)
        & set(ace_scores)
        & set(qwen_records)
        & set(qwen_scores)
    )

    matched = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for key in shared_keys:
            ace_score = ace_scores[key]
            qwen_score = qwen_scores[key]

            if mode == "ace_success_qwen_failure":
                keep = is_success(ace_score) and is_failure(qwen_score)
            elif mode == "qwen_success_ace_failure":
                keep = is_success(qwen_score) and is_failure(ace_score)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            if not keep:
                continue

            output_record = make_output_record(
                original_records[key],
                ace_records[key],
                ace_score,
                qwen_records[key],
                qwen_score,
            )
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            matched += 1

    return matched, len(shared_keys)


def main():
    base_dir = Path(__file__).resolve().parent

    original_records = build_record_map(load_jsonl(base_dir / "original.jsonl"))
    ace_records = build_record_map(load_jsonl(base_dir / "rewrite.jsonl"))
    ace_scores = build_record_map(load_jsonl(base_dir / "rewrite_score.jsonl"))
    qwen_records = build_record_map(load_jsonl(base_dir / "qwen_rewrite.jsonl"))
    qwen_scores = build_record_map(load_jsonl(base_dir / "qwen_rewrite_score.jsonl"))

    ace_output = base_dir / "ace_success_qwen_failure.jsonl"
    qwen_output = base_dir / "qwen_success_ace_failure.jsonl"

    ace_matched, total = write_directional_output(
        ace_output,
        original_records,
        ace_records,
        ace_scores,
        qwen_records,
        qwen_scores,
        mode="ace_success_qwen_failure",
    )
    qwen_matched, _ = write_directional_output(
        qwen_output,
        original_records,
        ace_records,
        ace_scores,
        qwen_records,
        qwen_scores,
        mode="qwen_success_ace_failure",
    )

    print(f"Wrote {ace_matched} instances to {ace_output} where {ACE_NAME} succeeds and {QWEN_NAME} fails.")
    print(f"Wrote {qwen_matched} instances to {qwen_output} where {QWEN_NAME} succeeds and {ACE_NAME} fails.")
    print(f"Compared {total} aligned rewritten questions.")


if __name__ == "__main__":
    main()
