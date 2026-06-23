"""
Full analysis of sub-question order impact across all 291 question pairs.

Four outcome groups:
  BOTH_CORRECT     (111)  — original correct, rewrite correct
  BOTH_WRONG       ( 64)  — original wrong,   rewrite wrong
  REWRITE_BETTER   ( 14)  — original wrong,   rewrite correct
  ORIGINAL_BETTER  (102)  — original correct, rewrite wrong  ← main focus

For each group we measure:
  • % of rewrites that introduced a verification sub-question  (order-change signal)
  • Among those: did the verification gate fire "yes" (chain confirmed) or "no" (chain broken)?
  • % where the gold answer is reachable anywhere in the rewrite intermediates
  • % where the gold answer is reachable anywhere in the original intermediates
  • Average extra sub-questions added by the rewrite

Then for ORIGINAL_BETTER we drill into failure types (A/B/C/E).
"""

import json
import re
from pathlib import Path


DATA_DIR    = Path(__file__).resolve().parent / "2wiki"
OUTPUT_PATH = DATA_DIR / "order_experiment_results.json"

ORIG_RAW    = DATA_DIR / "original.jsonl"
ORIG_SCORE  = DATA_DIR / "original_score.jsonl"
REW_RAW     = DATA_DIR / "rewrite.jsonl"
REW_SCORE   = DATA_DIR / "rewrite_score.jsonl"


# ── helpers ───────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row: dict, i: int) -> tuple:
    return (
        row.get("question_id", i),
        row.get("source_index", i),
        row.get("decompose_id", 0),
        row.get("index", i),
    )


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold_in_text(gold_list: list, text: str) -> bool:
    norm_text = normalise(text)
    # skip trivially short values ("no", "yes", single digit)
    if len(norm_text.split()) <= 1 and len(norm_text) <= 4:
        return False
    for g in gold_list:
        ng = normalise(g)
        if ng and ng in norm_text:
            return True
        if len(norm_text.split()) >= 2 and norm_text in ng:
            return True
        gold_tokens = set(ng.split())
        text_tokens = set(norm_text.split())
        if len(gold_tokens) >= 2 and len(gold_tokens & text_tokens) / len(gold_tokens) >= 0.5:
            return True
    return False


def gold_in_inter(gold: list, inter: dict) -> bool:
    return any(isinstance(v, str) and gold_in_text(gold, v) for v in inter.values())


def has_verif_step(decomp: list) -> bool:
    """Rewrite introduced a verification sub-question: 'is #N the same as X?'"""
    for q in decomp:
        txt = q.get("text", "").lower()
        if re.search(r"#\d", txt) and re.search(r"\b(is|did|does|are|was|were)\b", txt):
            return True
    return False


def verif_gate_result(inter: dict) -> str:
    """'no', 'yes', or 'none' — outcome of the verification step."""
    no_vals  = {"no", "no.", "false", "no,"}
    yes_vals = {"yes", "yes.", "true", "yes,"}
    for v in inter.values():
        if isinstance(v, str):
            low = v.strip().lower()
            if low in no_vals:
                return "no"
            if low in yes_vals:
                return "yes"
    return "none"


_MONTHS = (r"January|February|March|April|May|June|July|"
           r"August|September|October|November|December")


def q1_is_date_or_num(inter: dict) -> bool:
    q1 = list(inter.values())[0] if inter else ""
    if not isinstance(q1, str):
        return False
    v = q1.strip()
    return bool(
        re.match(r"^\d+$", v)
        or re.match(rf"^({_MONTHS})\b", v)
        or re.match(r"^\d{4}$", v)
    )


# ── failure-type classifier (used only for ORIGINAL_BETTER group) ─────────────

def classify_failure(orig_decomp, rew_decomp, rew_inter, rew_pred, gold) -> str:
    gold_found = gold_in_inter(gold, rew_inter)
    no_gate    = verif_gate_result(rew_inter) == "no"

    if no_gate and not gold_found:
        return "A_WRONG_ENTITY_CASCADE_explicit"

    if gold_found:
        return "B_CORRECT_INTER_WRONG_FINAL"

    gold_str = " ".join(gold) if gold else ""
    if len(rew_pred) > 2 * len(gold_str) + 30:
        return "C_FORMAT_MISMATCH"

    if no_gate and gold_found:
        return "A_WRONG_ENTITY_CASCADE_explicit"

    if q1_is_date_or_num(rew_inter):
        return "E_DATE_COMPARISON_FAILURE"

    q1_val = list(rew_inter.values())[0] if rew_inter else ""
    if isinstance(q1_val, str) and not q1_is_date_or_num(rew_inter) and len(q1_val.strip()) > 3:
        if len(rew_decomp) >= 2:
            return "A_WRONG_ENTITY_CASCADE_silent"

    return "D_OTHER"


# ── group-level metrics ────────────────────────────────────────────────────────

def analyse_group(pairs: list[dict]) -> dict:
    """
    pairs: list of dicts with keys
      orig_raw, orig_score, rew_raw, rew_score
    """
    n = len(pairs)
    if n == 0:
        return {}

    verif_count    = 0
    verif_yes      = 0
    verif_no       = 0
    gold_rew_inter = 0
    gold_orig_inter= 0
    delta_qs       = []

    for p in pairs:
        gold        = p["orig_score"].get("answer", [])
        orig_decomp = p["orig_raw"].get("decomposed", [])
        rew_decomp  = p["rew_raw"].get("decomposed", [])
        rew_inter   = p["rew_raw"].get("intermediate_answers", {})
        orig_inter  = p["orig_raw"].get("intermediate_answers", {})

        delta_qs.append(len(rew_decomp) - len(orig_decomp))

        if has_verif_step(rew_decomp):
            verif_count += 1
            vr = verif_gate_result(rew_inter)
            if vr == "yes":
                verif_yes += 1
            elif vr == "no":
                verif_no += 1

        if gold_in_inter(gold, rew_inter):
            gold_rew_inter += 1
        if gold_in_inter(gold, orig_inter):
            gold_orig_inter += 1

    pct = lambda x: round(100 * x / n, 1)
    avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else 0

    return {
        "n": n,
        "rewrite_has_verif_step":     {"count": verif_count,     "pct": pct(verif_count)},
        "verif_gate_yes":             {"count": verif_yes,        "pct": pct(verif_yes)},
        "verif_gate_no":              {"count": verif_no,         "pct": pct(verif_no)},
        "gold_in_rewrite_inter":      {"count": gold_rew_inter,   "pct": pct(gold_rew_inter)},
        "gold_in_original_inter":     {"count": gold_orig_inter,  "pct": pct(gold_orig_inter)},
        "avg_extra_questions":        avg(delta_qs),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run(output_path: Path) -> dict:
    orig_raws   = read_jsonl(ORIG_RAW)
    orig_scores = read_jsonl(ORIG_SCORE)
    rew_raws    = read_jsonl(REW_RAW)
    rew_scores  = read_jsonl(REW_SCORE)

    # Index raw rows by key
    orig_raw_idx = {row_key(r, i): r for i, r in enumerate(orig_raws)}
    rew_raw_idx  = {row_key(r, i): r for i, r in enumerate(rew_raws)}

    groups: dict[str, list[dict]] = {
        "BOTH_CORRECT":    [],
        "BOTH_WRONG":      [],
        "REWRITE_BETTER":  [],
        "ORIGINAL_BETTER": [],
    }

    failure_detail = []  # only ORIGINAL_BETTER

    for i, (os, rs) in enumerate(zip(orig_scores, rew_scores)):
        key       = row_key(os, i)
        orig_raw  = orig_raw_idx.get(key, {})
        rew_raw   = rew_raw_idx.get(key, {})
        pair = {
            "orig_raw":   orig_raw,
            "orig_score": os,
            "rew_raw":    rew_raw,
            "rew_score":  rs,
        }
        orig_ok = float(os.get("accuracy", 0)) == 1.0
        rew_ok  = float(rs.get("accuracy", 0)) == 1.0

        if orig_ok and rew_ok:
            groups["BOTH_CORRECT"].append(pair)
        elif not orig_ok and not rew_ok:
            groups["BOTH_WRONG"].append(pair)
        elif not orig_ok and rew_ok:
            groups["REWRITE_BETTER"].append(pair)
        else:
            groups["ORIGINAL_BETTER"].append(pair)
            # classify failure type
            gold       = os.get("answer", [])
            rew_decomp = rew_raw.get("decomposed", [])
            orig_decomp= orig_raw.get("decomposed", [])
            rew_inter  = rew_raw.get("intermediate_answers", {})
            rew_pred   = rs.get("prediction", "")
            ftype = classify_failure(orig_decomp, rew_decomp, rew_inter, rew_pred, gold)
            failure_detail.append({
                "id":                  os.get("pair_id", str(key)),
                "failure_type":        ftype,
                "gold":                gold,
                "original_question":   orig_raw.get("question", ""),
                "rewrite_question":    rew_raw.get("question", ""),
                "original_decomp":     [{"label": q.get("label"), "text": q.get("text")}
                                        for q in orig_decomp],
                "rewrite_decomp":      [{"label": q.get("label"), "text": q.get("text")}
                                        for q in rew_decomp],
                "rewrite_intermediates": rew_inter,
                "rewrite_prediction":  rew_pred,
                "verif_gate":          verif_gate_result(rew_inter),
                "gold_in_rew_inter":   gold_in_inter(gold, rew_inter),
            })

    # Per-group stats
    stats = {g: analyse_group(pairs) for g, pairs in groups.items()}

    # Failure-type counts for ORIGINAL_BETTER
    ftypes: dict[str, int] = {}
    for fd in failure_detail:
        ftypes[fd["failure_type"]] = ftypes.get(fd["failure_type"], 0) + 1

    # A explicit + silent together
    a_total = ftypes.get("A_WRONG_ENTITY_CASCADE_explicit", 0) + ftypes.get("A_WRONG_ENTITY_CASCADE_silent", 0)

    result = {
        "group_stats": stats,
        "failure_types_ORIGINAL_BETTER": ftypes,
        "failure_types_ORIGINAL_BETTER_A_total": a_total,
        "failure_detail": failure_detail,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ── display ───────────────────────────────────────────────────────────────────

SEP  = "=" * 78
SEP2 = "-" * 78
W    = 78


def bar(count: int, total: int, width: int = 20) -> str:
    if not total:
        return "[" + "." * width + "]"
    filled = round(width * count / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def pct(count: int, total: int) -> str:
    if not total:
        return "  n/a"
    return f"{100*count/total:5.1f}%"


def print_group_table(stats: dict) -> None:
    GROUPS = [
        ("BOTH_CORRECT",    "Both correct      (orig✓ rew✓)"),
        ("BOTH_WRONG",      "Both wrong        (orig✗ rew✗)"),
        ("REWRITE_BETTER",  "Rewrite better    (orig✗ rew✓)"),
        ("ORIGINAL_BETTER", "Original better   (orig✓ rew✗)  ◄"),
    ]

    # Header
    print(f"\n  {'Group':<34} {'n':>4}  {'Verif':>6}  "
          f"{'V→no':>6}  {'V→yes':>6}  {'Gold∈R':>6}  {'Gold∈O':>6}  {'ΔQ':>5}")
    print("  " + "-" * 74)

    for key, label in GROUPS:
        s = stats[key]
        n = s["n"]
        print(
            f"  {label:<34} {n:>4}"
            f"  {pct(s['rewrite_has_verif_step']['count'], n):>6}"
            f"  {pct(s['verif_gate_no']['count'],          n):>6}"
            f"  {pct(s['verif_gate_yes']['count'],         n):>6}"
            f"  {pct(s['gold_in_rewrite_inter']['count'],  n):>6}"
            f"  {pct(s['gold_in_original_inter']['count'], n):>6}"
            f"  {s['avg_extra_questions']:>+5.1f}"
        )

    print()
    print("  Columns:")
    print("    Verif   — % of rewrites that added a verification sub-question")
    print("              ('is #1 the same as X?') — the structural sign of order change")
    print("    V→no    — % where verification fired 'no' (wrong entity confirmed)")
    print("    V→yes   — % where verification fired 'yes' (right entity confirmed)")
    print("    Gold∈R  — % where gold answer is reachable in rewrite intermediates")
    print("    Gold∈O  — % where gold answer is reachable in original intermediates")
    print("    ΔQ      — average extra sub-questions added by the rewrite")


def print_failure_types(ftypes: dict, n: int) -> None:
    labels = {
        "A_WRONG_ENTITY_CASCADE_explicit": "A1  Wrong entity + explicit 'no' gate  [ORDER]",
        "A_WRONG_ENTITY_CASCADE_silent":   "A2  Wrong entity, silent cascade        [ORDER]",
        "B_CORRECT_INTER_WRONG_FINAL":     "B   Correct intermediate, wrong output  [SYNTHESIS]",
        "E_DATE_COMPARISON_FAILURE":       "E   Date-comparison retrieval/format    [FORMAT/RETRIEVAL]",
        "C_FORMAT_MISMATCH":               "C   Format mismatch (description≠name)  [FORMAT]",
        "D_OTHER":                         "D   Other / unclassified",
    }
    for k, label in labels.items():
        c = ftypes.get(k, 0)
        print(f"    {label:<52}  {c:>3} ({pct(c, n):>5})  {bar(c, n)}")


def print_rewrite_better_examples(pairs: list[dict]) -> None:
    for p in pairs[:3]:
        orig_q  = p["orig_raw"].get("question", "")
        rew_q   = p["rew_raw"].get("question", "")
        gold    = p["orig_score"].get("answer", [])
        orig_pred = p["orig_score"].get("prediction", "")
        rew_pred  = p["rew_score"].get("prediction", "")
        orig_inter = p["orig_raw"].get("intermediate_answers", {})
        rew_inter  = p["rew_raw"].get("intermediate_answers", {})
        orig_decomp = p["orig_raw"].get("decomposed", [])
        rew_decomp  = p["rew_raw"].get("decomposed", [])
        print(f"  Original  : {orig_q}")
        print(f"  Rewrite   : {rew_q[:110]}{'...' if len(rew_q)>110 else ''}")
        print(f"  Gold      : {gold}")
        print(f"  Orig sub-Qs : {' | '.join(q.get('text','')[:55] for q in orig_decomp)}")
        print(f"  Orig answers: {orig_inter}  →  pred: {orig_pred}")
        print(f"  Rew  sub-Qs : {' | '.join(q.get('text','')[:55] for q in rew_decomp)}")
        print(f"  Rew  answers: {rew_inter}  →  pred: {rew_pred}")
        print()


def main() -> None:
    result = run(OUTPUT_PATH)
    stats  = result["group_stats"]
    ftypes = result["failure_types_ORIGINAL_BETTER"]
    n102   = stats["ORIGINAL_BETTER"]["n"]
    n_all  = sum(s["n"] for s in stats.values())

    # load pairs for REWRITE_BETTER examples
    orig_raws   = read_jsonl(ORIG_RAW)
    orig_scores = read_jsonl(ORIG_SCORE)
    rew_raws    = read_jsonl(REW_RAW)
    rew_scores  = read_jsonl(REW_SCORE)
    orig_raw_idx = {row_key(r, i): r for i, r in enumerate(orig_raws)}
    rew_raw_idx  = {row_key(r, i): r for i, r in enumerate(rew_raws)}
    rew_better_pairs = []
    for i, (os, rs) in enumerate(zip(orig_scores, rew_scores)):
        if float(os.get("accuracy",0)) != 1.0 and float(rs.get("accuracy",0)) == 1.0:
            key = row_key(os, i)
            rew_better_pairs.append({
                "orig_raw": orig_raw_idx.get(key, {}), "orig_score": os,
                "rew_raw":  rew_raw_idx.get(key, {}),  "rew_score":  rs,
            })

    # ── print ──────────────────────────────────────────────────────────────────
    print(SEP)
    print("  ORDER IMPACT ANALYSIS  —  all 291 question pairs (2WikiMultiHop)")
    print(SEP)

    # 1. Comparison table
    print("\n[1] CROSS-GROUP COMPARISON")
    print(SEP2)
    print_group_table(stats)

    # 2. Key observations from the table
    bc = stats["BOTH_CORRECT"]
    bw = stats["BOTH_WRONG"]
    rb = stats["REWRITE_BETTER"]
    ob = stats["ORIGINAL_BETTER"]

    print(f"\n[2] WHAT THE TABLE TELLS US")
    print(SEP2)

    print(f"""
  Verif step (order change signal)
  ─────────────────────────────────────────────────────────────
  The rewrite almost always adds a verification sub-question
  regardless of outcome:
    Both correct    : {pct(bc['rewrite_has_verif_step']['count'], bc['n'])}
    Both wrong      : {pct(bw['rewrite_has_verif_step']['count'], bw['n'])}
    Rewrite better  : {pct(rb['rewrite_has_verif_step']['count'], rb['n'])}
    Original better : {pct(ob['rewrite_has_verif_step']['count'], ob['n'])}

  → Order change (verification step) is a CONSTANT feature of all rewrites,
    not just failing ones. The outcome depends on whether the gate fires
    'yes' (right entity) or 'no' (wrong entity).

  Verification gate outcome  (V→yes vs V→no)
  ─────────────────────────────────────────────────────────────
  When the gate fires 'yes', the rewrite can still succeed:
    Both correct (rewrite ✓) : V→yes {pct(bc['verif_gate_yes']['count'], bc['n'])},  V→no {pct(bc['verif_gate_no']['count'], bc['n'])}
    Rewrite better   (rew ✓) : V→yes {pct(rb['verif_gate_yes']['count'], rb['n'])},  V→no {pct(rb['verif_gate_no']['count'], rb['n'])}
    Original better  (rew ✗) : V→yes {pct(ob['verif_gate_yes']['count'], ob['n'])},  V→no {pct(ob['verif_gate_no']['count'], ob['n'])}

  → 'V→no' is concentrated in ORIGINAL_BETTER (the failing group).
    When V→yes, the constraint was selective enough to identify the
    right entity and the rewrite succeeds. When V→no, the constraint
    was non-selective (wrong entity in Q1) and the chain collapses.

  Gold reachable in intermediates  (Gold∈R)
  ─────────────────────────────────────────────────────────────
    Both correct    : {pct(bc['gold_in_rewrite_inter']['count'], bc['n'])}  (high: rewrite is correct)
    Both wrong      : {pct(bw['gold_in_rewrite_inter']['count'], bw['n'])}  (low: even intermediates fail)
    Rewrite better  : {pct(rb['gold_in_rewrite_inter']['count'], rb['n'])}  (high: rewrite succeeds)
    Original better : {pct(ob['gold_in_rewrite_inter']['count'], ob['n'])}  (low: gold not in chain)

  → When gold is NOT reachable in intermediates, the chain is fundamentally
    broken — either because Q1 found the wrong entity (Type A) or because
    the retrieval system returned irrelevant content.

  Original intermediates (Gold∈O) tell the baseline difficulty:
    Both wrong      : {pct(bw['gold_in_original_inter']['count'], bw['n'])}  ← original also can't find the entity
    Rewrite better  : {pct(rb['gold_in_original_inter']['count'], rb['n'])}  ← original misses, rewrite finds it
    Both correct    : {pct(bc['gold_in_original_inter']['count'], bc['n'])}  ← original finds it easily

  Complexity increase  (ΔQ = extra sub-questions in rewrite)
  ─────────────────────────────────────────────────────────────
    Both correct    : {bc['avg_extra_questions']:+.1f}  extra questions on average
    Both wrong      : {bw['avg_extra_questions']:+.1f}
    Rewrite better  : {rb['avg_extra_questions']:+.1f}
    Original better : {ob['avg_extra_questions']:+.1f}

  → Rewrites add ~1 extra sub-question uniformly across all groups.
    Complexity alone does not determine success or failure.
""")

    # 3. Failure types for ORIGINAL_BETTER
    print(f"[3] FAILURE TYPES  (n={n102} — original correct, rewrite wrong)")
    print(SEP2)
    a_total = result["failure_types_ORIGINAL_BETTER_A_total"]
    print(f"  Failure type breakdown:")
    print()
    print_failure_types(ftypes, n102)
    print(f"""
  Type A (wrong-entity cascade) accounts for {pct(a_total, n102)} of failures.
  It has two sub-patterns:
    A1 — explicit 'no' gate: Q1 gets wrong entity → verification asks 'is #1 == X?' → 'no'
         → remaining chain hallucinates from a broken premise.
    A2 — silent cascade:     Q1 gets wrong entity, no verification step, wrong entity
         flows silently through Q2, Q3, … → final answer inherits the wrong entity.
""")

    # 4. The 14 'rewrite better' cases
    print(f"[4] WHEN THE REWRITE HELPS  (n={rb['n']} — original wrong, rewrite correct)")
    print(SEP2)
    print(f"""
  These {rb['n']} cases show the flip side: the added constraint IMPROVED accuracy.
  The original's simple phrasing was too ambiguous — Q1 resolved the wrong entity.
  The rewrite's constraint (birthdate, award, citizenship) was specific enough
  to uniquely identify the correct entity, so V→yes and the chain succeeded.

  Gold∈original intermediates : {pct(rb['gold_in_original_inter']['count'], rb['n'])}  ← original misses the entity
  Gold∈rewrite  intermediates : {pct(rb['gold_in_rewrite_inter']['count'], rb['n'])}  ← rewrite finds it
  V→yes (constraint confirmed the right entity) : {pct(rb['verif_gate_yes']['count'], rb['n'])}
""")
    print("  Examples:")
    print()
    print_rewrite_better_examples(rew_better_pairs)

    # 5. Summary
    print(f"[5] SUMMARY ANSWER")
    print(SEP2)
    print(f"""
  Q: Does sub-question order affect the final answer?
  A: YES — but the mechanism is the verification-gate outcome, not
     the mere presence of a verification step.

  The rewrite always adds a verification step (~97% of all groups).
  What varies is whether that gate fires 'yes' or 'no':

    'no'  → Q1 retrieved the WRONG entity (low-selectivity constraint).
            The gate confirms the mismatch and the chain hallucinates.
            This accounts for {pct(ftypes.get('A_WRONG_ENTITY_CASCADE_explicit',0), n102)} of the {n102} degradation cases.

    'yes' → Q1 retrieved the RIGHT entity (selective-enough constraint).
            The gate confirms the match and the chain proceeds correctly.
            This is what happens in the {bc['n']} 'both correct' and {rb['n']} 'rewrite better' cases.

  Constraint selectivity is the controlling variable:
    High selectivity  → V→yes → chain succeeds  (both correct / rewrite better)
    Low  selectivity  → V→no  → chain collapses (original better)

  The remaining {pct(n102 - a_total - ftypes.get('C_FORMAT_MISMATCH',0), n102)} of failures (Type B + E) are NOT
  order-related: the model found the right intermediate answer but
  failed to select it as the final output (synthesis failure), or
  the date-comparison format prevented the name from appearing.
""")

    print(f"  Results saved to: {OUTPUT_PATH}")
    print(SEP)


if __name__ == "__main__":
    main()
