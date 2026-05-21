import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ACE_ROOT = Path(__file__).resolve().parents[1]
if str(ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_ROOT))

from eval.score_rewrite_pairs import (  # noqa: E402
    ORIGINAL_ID_KEYS,
    build_source_index,
    get_question,
    get_rewrite_instance_id,
    make_pair_id,
    read_jsonl,
    resolve_source_id,
)


DEFAULT_ORIGINAL_SOURCE = "eval_datasets/2wiki_origin/test_subsampled.jsonl"
DEFAULT_REWRITE_SOURCE = "eval_datasets/2wiki/test_subsampled.jsonl"
DEFAULT_ORIGINAL_PLANS = "eval_datasets/test/2wiki_origin/prompts_decompose_test_t0.0_ace/generate.jsonl"
DEFAULT_REWRITE_PLANS = "eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/generate.jsonl"
DEFAULT_OUTPUT_DIR = "eval_datasets/test/2wiki/prompts_decompose_test_t0.0_ace/eval_results"
DEFAULT_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/plan_pair_eval.jsonl"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}

RELATION_PHRASES = (
    "association football player",
    "award received",
    "birth name",
    "capital of",
    "cause of death",
    "child of",
    "country of citizenship",
    "date of birth",
    "date of death",
    "date of release",
    "director of",
    "educated at",
    "established first",
    "father in law",
    "father of",
    "film director",
    "founded by",
    "founded in",
    "headquartered in",
    "located in",
    "member of",
    "mother of",
    "place of birth",
    "place of death",
    "played for",
    "producer of",
    "publication date",
    "release date",
    "released earlier",
    "screenwriter",
    "sibling of",
    "spouse of",
    "written by",
)

RELATION_WORDS = {
    word
    for phrase in RELATION_PHRASES
    for word in phrase.split()
    if word not in STOPWORDS
}


def normalize_text(text):
    text = "" if text is None else str(text)
    text = re.sub(r"(?i)'s\b", "", text)
    text = re.sub(r"[^a-zA-Z0-9#]+", " ", text.lower())
    return " ".join(text.split())


def content_tokens(text):
    return {
        token
        for token in normalize_text(text).split()
        if token not in STOPWORDS and len(token) > 1 and not token.startswith("#")
    }


def extract_entities(text):
    text = "" if text is None else str(text)
    spans = re.findall(
        r"(?:[A-Z][A-Za-z0-9'&.-]+|[A-Z]{2,})(?:\s+(?:of|the|and|for|in|[A-Z][A-Za-z0-9'&.-]+|[A-Z]{2,}))*",
        text,
    )
    clean_spans = []
    for span in spans:
        span = span.strip(" ,.?;:()[]{}")
        if not span or span.lower() in STOPWORDS:
            continue
        tokens = content_tokens(span)
        if tokens:
            clean_spans.append(span)
    return sorted(set(clean_spans))


def extract_relation_terms(text):
    norm = normalize_text(text)
    tokens = set(norm.split())
    terms = {phrase for phrase in RELATION_PHRASES if phrase in norm}
    terms.update(token for token in tokens if token in RELATION_WORDS)
    return terms


def jaccard(left, right):
    left = set(left)
    right = set(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def coverage(required, observed):
    required = set(required)
    observed = set(observed)
    if not required:
        return 1.0
    return len(required & observed) / len(required)


def get_first_subquestion(record):
    decomposed = record.get("decomposed", [])
    if isinstance(decomposed, list) and decomposed:
        first = decomposed[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return str(first)
    if isinstance(decomposed, str):
        return decomposed
    return ""


def build_plan_index(records, source_index):
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


def resolve_rewrite_source_id(record, rewrite_source_index, original_source_index):
    question = get_question(record)
    if question in rewrite_source_index["by_question"]:
        return rewrite_source_index["by_question"][question]

    source_id = resolve_source_id(record, rewrite_source_index, ORIGINAL_ID_KEYS)
    if source_id in original_source_index["by_id"]:
        return source_id

    return resolve_source_id(record, original_source_index, ORIGINAL_ID_KEYS)


def analyze_pair(original_record, rewrite_record, source_id, rewrite_instance_id):
    original_question = get_question(original_record)
    rewrite_question = get_question(rewrite_record)
    original_first = get_first_subquestion(original_record)
    rewrite_first = get_first_subquestion(rewrite_record)

    original_first_tokens = content_tokens(original_first)
    rewrite_first_tokens = content_tokens(rewrite_first)
    original_first_entities = extract_entities(original_first)
    rewrite_first_entities = extract_entities(rewrite_first)
    original_first_relations = extract_relation_terms(original_first)
    rewrite_first_relations = extract_relation_terms(rewrite_first)

    original_question_tokens = content_tokens(original_question)
    rewrite_question_tokens = content_tokens(rewrite_question)
    original_question_entities = extract_entities(original_question)
    original_question_relations = extract_relation_terms(original_question)
    rewrite_added_tokens = rewrite_question_tokens - original_question_tokens

    first_relation_coverage = coverage(original_first_relations, rewrite_first_relations)
    first_entity_coverage = coverage(
        {normalize_text(entity) for entity in original_first_entities},
        {normalize_text(entity) for entity in rewrite_first_entities},
    )
    first_token_coverage = coverage(original_first_tokens, rewrite_first_tokens)
    rewrite_added_coverage = coverage(rewrite_added_tokens, rewrite_first_tokens)
    original_question_relation_coverage = coverage(
        original_question_relations,
        rewrite_first_relations,
    )
    original_question_entity_coverage = coverage(
        {normalize_text(entity) for entity in original_question_entities},
        {normalize_text(entity) for entity in rewrite_first_entities},
    )
    original_question_key_terms = (
        set(original_question_relations)
        | {normalize_text(entity) for entity in original_question_entities}
    )
    rewrite_first_key_terms = (
        set(rewrite_first_relations)
        | {normalize_text(entity) for entity in rewrite_first_entities}
    )
    original_question_key_coverage = coverage(
        original_question_key_terms,
        rewrite_first_key_terms,
    )

    missing_original_relations = sorted(original_first_relations - rewrite_first_relations)
    missing_original_question_relations = sorted(
        original_question_relations - rewrite_first_relations
    )
    introduced_foreign_relations = sorted(
        rewrite_first_relations - original_question_relations
    )
    introduced_first_relations = sorted(rewrite_first_relations - original_first_relations)
    missing_original_entities = sorted(
        set(original_first_entities)
        - {
            entity
            for entity in original_first_entities
            if normalize_text(entity)
            in {normalize_text(rewrite_entity) for rewrite_entity in rewrite_first_entities}
        }
    )

    relation_changed = bool(original_first_relations) and first_relation_coverage < 0.5
    anchor_missing = bool(original_first_entities) and first_entity_coverage < 0.5
    original_key_missing = bool(original_question_key_terms) and original_question_key_coverage < 0.35
    original_relation_missing = (
        bool(original_question_relations) and original_question_relation_coverage < 0.5
    )
    foreign_relation_introduced = bool(introduced_foreign_relations)
    low_token_overlap = first_token_coverage < 0.35
    rewrite_constraint_first = bool(rewrite_added_tokens) and rewrite_added_coverage >= 0.35

    if (original_relation_missing and foreign_relation_introduced) or (
        original_key_missing and rewrite_constraint_first
    ):
        risk = "high"
    elif relation_changed or anchor_missing or original_key_missing or low_token_overlap:
        risk = "medium"
    else:
        risk = "low"

    return {
        "id": make_pair_id(source_id, rewrite_instance_id),
        "pair_id": make_pair_id(source_id, rewrite_instance_id),
        "original_question_id": source_id,
        "rewrite_question_id": rewrite_instance_id,
        "original_question": original_question,
        "rewritten_question": rewrite_question,
        "original_first_subquestion": original_first,
        "rewrite_first_subquestion": rewrite_first,
        "first_subquestion_token_jaccard": jaccard(original_first_tokens, rewrite_first_tokens),
        "first_subquestion_relation_jaccard": jaccard(original_first_relations, rewrite_first_relations),
        "first_subquestion_entity_jaccard": jaccard(
            {normalize_text(entity) for entity in original_first_entities},
            {normalize_text(entity) for entity in rewrite_first_entities},
        ),
        "original_first_relation_coverage": first_relation_coverage,
        "original_first_entity_coverage": first_entity_coverage,
        "original_first_token_coverage": first_token_coverage,
        "original_question_relation_coverage_in_first": original_question_relation_coverage,
        "original_question_entity_coverage_in_first": original_question_entity_coverage,
        "original_question_key_coverage_in_first": original_question_key_coverage,
        "rewrite_added_constraint_coverage_in_first": rewrite_added_coverage,
        "relation_changed": relation_changed,
        "anchor_missing": anchor_missing,
        "original_key_missing": original_key_missing,
        "original_relation_missing": original_relation_missing,
        "foreign_relation_introduced": foreign_relation_introduced,
        "rewrite_constraint_moved_to_first": rewrite_constraint_first,
        "first_hop_risk": risk,
        "missing_original_relations": missing_original_relations,
        "missing_original_question_relations": missing_original_question_relations,
        "introduced_foreign_relations": introduced_foreign_relations,
        "introduced_first_relations": introduced_first_relations,
        "missing_original_entities": missing_original_entities,
        "original_question_relations": sorted(original_question_relations),
        "original_question_entities": original_question_entities,
        "original_first_relations": sorted(original_first_relations),
        "rewrite_first_relations": sorted(rewrite_first_relations),
        "original_first_entities": original_first_entities,
        "rewrite_first_entities": rewrite_first_entities,
        "rewrite_added_tokens": sorted(rewrite_added_tokens),
    }


def evaluate_decomposition_pairs(original_plans, rewrite_plans, original_source, rewrite_source):
    original_source_index = build_source_index(original_source)
    rewrite_source_index = build_source_index(rewrite_source)
    original_plan_index, duplicate_original_plans = build_plan_index(
        original_plans,
        original_source_index,
    )

    rows = []
    skipped = defaultdict(int)
    for fallback_index, rewrite_record in enumerate(rewrite_plans):
        source_id = resolve_rewrite_source_id(
            rewrite_record,
            rewrite_source_index,
            original_source_index,
        )

        if source_id not in original_source_index["by_id"]:
            skipped["rewrite_without_source"] += 1
            continue

        original_record = original_plan_index.get(source_id)
        if original_record is None:
            skipped["rewrite_without_original_plan"] += 1
            continue

        rewrite_instance_id = get_rewrite_instance_id(rewrite_record, fallback_index)
        rows.append(analyze_pair(original_record, rewrite_record, source_id, rewrite_instance_id))

    skipped["duplicate_original_plans"] = duplicate_original_plans
    return rows, dict(skipped)


def write_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(rows):
    summary = {
        "total": len(rows),
        "risk_counts": defaultdict(int),
    }
    numeric_keys = (
        "first_subquestion_token_jaccard",
        "first_subquestion_relation_jaccard",
        "first_subquestion_entity_jaccard",
        "original_first_relation_coverage",
        "original_first_entity_coverage",
        "original_first_token_coverage",
        "original_question_relation_coverage_in_first",
        "original_question_entity_coverage_in_first",
        "original_question_key_coverage_in_first",
        "rewrite_added_constraint_coverage_in_first",
    )
    totals = defaultdict(float)
    flags = defaultdict(int)
    for row in rows:
        summary["risk_counts"][row["first_hop_risk"]] += 1
        for key in numeric_keys:
            totals[key] += float(row[key])
        for key in (
            "relation_changed",
            "anchor_missing",
            "original_key_missing",
            "original_relation_missing",
            "foreign_relation_introduced",
            "rewrite_constraint_moved_to_first",
        ):
            flags[key] += int(row[key])

    if rows:
        for key in numeric_keys:
            summary[key] = totals[key] / len(rows)
        for key, value in flags.items():
            summary[key] = value / len(rows)

    summary["risk_counts"] = dict(summary["risk_counts"])
    return summary


def print_summary(output_path, summary, skipped):
    print(f"Output JSONL: {output_path}")
    print(f"Total pairs: {summary['total']}")
    if skipped:
        print("Skipped:", json.dumps(skipped, sort_keys=True))
    print("Risk counts:", json.dumps(summary["risk_counts"], sort_keys=True))
    if summary["total"] == 0:
        return
    print(f"Avg first Q token Jaccard: {summary['first_subquestion_token_jaccard']}")
    print(f"Avg first Q relation Jaccard: {summary['first_subquestion_relation_jaccard']}")
    print(f"Avg first Q entity Jaccard: {summary['first_subquestion_entity_jaccard']}")
    print(
        "Avg original-question relation coverage in first Q: "
        f"{summary['original_question_relation_coverage_in_first']}"
    )
    print(
        "Avg original-question key coverage in first Q: "
        f"{summary['original_question_key_coverage_in_first']}"
    )
    print(f"Relation changed rate: {summary.get('relation_changed', 0.0)}")
    print(f"Anchor missing rate: {summary.get('anchor_missing', 0.0)}")
    print(f"Original relation missing rate: {summary.get('original_relation_missing', 0.0)}")
    print(f"Foreign relation introduced rate: {summary.get('foreign_relation_introduced', 0.0)}")
    print(
        "Rewrite-added constraint in first-hop rate: "
        f"{summary.get('rewrite_constraint_moved_to_first', 0.0)}"
    )


def default_output_path(input_path):
    input_path = Path(input_path)
    return input_path.with_name(input_path.stem + "_plan_pair_eval" + input_path.suffix)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate decomposition-plan differences for independent (q, q'_j) pairs. "
            "The main diagnostic checks whether the rewritten first subquestion keeps "
            "the original first-hop relation/anchor or starts from a different relation."
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
        "--original-plans",
        default=DEFAULT_ORIGINAL_PLANS,
        help="Original decomposition JSONL or solver output JSONL containing decomposed.",
    )
    parser.add_argument(
        "--rewrite-plans",
        default=DEFAULT_REWRITE_PLANS,
        help="Rewritten decomposition JSONL or solver output JSONL containing decomposed.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Pair-level diagnostic output JSONL. Defaults to <rewrite-plans>_plan_pair_eval.jsonl.",
    )
    args = parser.parse_args()

    original_source = read_jsonl(args.original_source)
    rewrite_source = read_jsonl(args.rewrite_source)
    original_plans = read_jsonl(args.original_plans)
    rewrite_plans = read_jsonl(args.rewrite_plans)

    rows, skipped = evaluate_decomposition_pairs(
        original_plans,
        rewrite_plans,
        original_source,
        rewrite_source,
    )
    output_path = args.output or default_output_path(args.rewrite_plans)
    write_jsonl(rows, output_path)
    print_summary(output_path, summarize(rows), skipped)


if __name__ == "__main__":
    main()
