"""
Gold-Relation Order Analysis
=============================

TASK DEFINITION
---------------
We take a 2WikiMultiHop question and create two versions:

  Original:  the question as-is
  Rewrite:   the same question with extra relations added (e.g. citizenship,
             birth date, award) that act as additional constraints

Both versions are answered by a model that first decomposes the question
into sub-questions (Q1, Q2, ..., Qn), answers each one, then combines
the intermediate answers into a final prediction.

KEY CONCEPT — The "gold relation"
  Each decomposition chain contains one critical step: the sub-question
  whose intermediate answer IS the gold (correct) final answer.
  We call this the "gold relation step".

  In the original chain it is almost always the LAST sub-question.
  E.g.  Q1: "Who directed Mukhyamantri?" → Anjan Choudhury
        Q2: "Who is the child of #1?"    → Chumki Chowdhury  ← GOLD STEP

RESEARCH QUESTIONS
------------------
  Q1. In the original chain, where does the gold relation step appear?
      (Is it always the last sub-question?)

  Q2. When the rewrite adds extra relations, does the model keep the gold
      relation at the same relative position in the decomposition chain?

  Q3. When the position changes (or the gold step disappears), does the
      final answer accuracy degrade?
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_DIR   = Path(__file__).resolve().parent
ORIG_RAW   = DATA_DIR / "original.jsonl"
ORIG_SCORE = DATA_DIR / "original_score.jsonl"
REW_RAW    = DATA_DIR / "rewrite.jsonl"
REW_SCORE  = DATA_DIR / "rewrite_score.jsonl"
FIG_DIR    = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── question-type classification ──────────────────────────────────────────────

def classify_gold_relation(last_q_text: str) -> str:
    """Classify the type of the gold relation step by the final sub-question text."""
    t = last_q_text.lower()
    if re.search(r'\bfather\b|\bmother\b|\bparent\b|\bgrandfather\b|\bgrandmother\b'
                 r'|\bchild\b|\bson\b|\bdaughter\b|\bspouse\b|\bhusband\b|\bwife\b'
                 r'|\bsibling\b|\bbrother\b|\bsister\b|\bin-law\b', t):
        return "Family relation"
    if re.search(r'\bdie\b|\bdeath\b|\bborn\b|\bbirth\b|\bage\b|\byounger\b|\bolder\b', t):
        return "Birth / death"
    if re.search(r'\bstudy\b|\beducat\b|\bgraduate\b|\bcollege\b|\buniversity\b'
                 r'|\bschool\b|\battend\b', t):
        return "Education"
    if re.search(r'\bnational\b|\bnationality\b|\bcitizen\b', t):
        return "Nationality"
    if re.search(r'\bburied\b|\bgrave\b|\btomb\b', t):
        return "Burial place"
    return "Other"


def classify_added_constraint(rew_q1_text: str) -> str:
    """Classify the type of constraint added as Q1 in the rewrite."""
    t = rew_q1_text.lower()
    if re.search(r'\bborn\b|\bbirth\b', t):
        return "Birthdate"
    if re.search(r'\bcitizen\b|\bnational\b|\bnationality\b|\bcountry of\b', t):
        return "Citizenship"
    if re.search(r'\baward\b|\bprize\b|\bmedal\b|\bhonor\b|\brecipient\b', t):
        return "Award / honor"
    if re.search(r'\bdied\b|\bdeath\b|\bdeceased\b', t):
        return "Death date"
    if re.search(r'\bchildren\b|\bson\b|\bdaughter\b|\bparent\b'
                 r'|\bfather\b|\bmother\b|\bspouse\b|\bhusband\b|\bwife\b', t):
        return "Family constraint"
    if re.search(r'\bfield of work\b|\boccupation\b|\bprofession\b', t):
        return "Occupation"
    return "Other description"


# ── helpers ───────────────────────────────────────────────────────────────────

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row, i):
    return (row.get("question_id", i), row.get("source_index", i),
            row.get("decompose_id", 0), row.get("index", i))


def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold_match(gold_list, text):
    if not isinstance(text, str):
        return False
    nt = normalise(text)
    if len(nt.split()) <= 1 and len(nt) <= 4:
        return False
    for g in gold_list:
        ng = normalise(g)
        if not ng:
            continue
        if ng in nt:
            return True
        if len(ng.split()) >= 2 and nt in ng:
            return True
        gold_toks, text_toks = set(ng.split()), set(nt.split())
        if len(gold_toks) >= 2 and len(gold_toks & text_toks) / len(gold_toks) >= 0.5:
            return True
    return False


def find_gold_labels(gold, intermediates):
    """Labels of sub-questions whose intermediate answer matches gold."""
    return [lbl for lbl, val in intermediates.items()
            if isinstance(val, str) and gold_match(gold, val)]


def rank_from_end(label, all_labels):
    """0 = last sub-question, 1 = second-to-last, etc.  -1 if not found."""
    if label not in all_labels:
        return -1
    return len(all_labels) - 1 - all_labels.index(label)


def classify_pair(gold, orig_inter, rew_inter, orig_decomp, rew_decomp):
    """
    Returns a dict describing where the gold relation step sits in each chain.

    order_class values:
      LAST_BOTH       — gold at last step in both chains
      LAST_ORIG_ONLY  — gold at last step in original, NOT last in rewrite
      NOT_LAST_BOTH   — gold not at last step in either (comparison / date Qs)
      MISSING_REW     — gold found in original, absent from rewrite
      MISSING_ORIG    — gold absent from original, found in rewrite  (rare)
      MISSING_BOTH    — gold absent from both chains
    """
    orig_labels = [q["label"] for q in orig_decomp]
    rew_labels  = [q["label"] for q in rew_decomp]

    gold_orig = find_gold_labels(gold, orig_inter)
    gold_rew  = find_gold_labels(gold, rew_inter)

    orig_rfe = min((rank_from_end(l, orig_labels) for l in gold_orig), default=-1)
    rew_rfe  = min((rank_from_end(l, rew_labels)  for l in gold_rew),  default=-1)

    found_orig = bool(gold_orig)
    found_rew  = bool(gold_rew)
    at_last_orig = (orig_rfe == 0)
    at_last_rew  = (rew_rfe  == 0)

    if not found_orig and not found_rew:
        order_class = "MISSING_BOTH"
    elif found_orig and not found_rew:
        order_class = "MISSING_REW"
    elif not found_orig and found_rew:
        order_class = "MISSING_ORIG"
    elif at_last_orig and at_last_rew:
        order_class = "LAST_BOTH"
    elif at_last_orig and not at_last_rew:
        order_class = "LAST_ORIG_ONLY"
    else:
        order_class = "NOT_LAST_BOTH"

    return dict(
        gold_labels_orig=gold_orig,  gold_labels_rew=gold_rew,
        orig_rfe=orig_rfe,           rew_rfe=rew_rfe,
        found_orig=found_orig,       found_rew=found_rew,
        at_last_orig=at_last_orig,   at_last_rew=at_last_rew,
        order_class=order_class,
        n_orig=len(orig_labels),     n_rew=len(rew_labels),
        delta_steps=len(rew_labels) - len(orig_labels),
    )


# ── load & annotate all pairs ─────────────────────────────────────────────────

def load_records():
    orig_raws   = read_jsonl(ORIG_RAW)
    orig_scores = read_jsonl(ORIG_SCORE)
    rew_raws    = read_jsonl(REW_RAW)
    rew_scores  = read_jsonl(REW_SCORE)

    orig_idx = {row_key(r, i): r for i, r in enumerate(orig_raws)}
    rew_idx  = {row_key(r, i): r for i, r in enumerate(rew_raws)}

    records = []
    for i, (os, rs) in enumerate(zip(orig_scores, rew_scores)):
        key = row_key(os, i)
        o   = orig_idx.get(key, {})
        r   = rew_idx.get(key, {})

        orig_ok = float(os.get("accuracy", 0)) == 1.0
        rew_ok  = float(rs.get("accuracy", 0)) == 1.0
        if orig_ok and rew_ok:       acc_group = "BOTH_CORRECT"
        elif orig_ok and not rew_ok: acc_group = "ORIG_BETTER"
        elif not orig_ok and rew_ok: acc_group = "REW_BETTER"
        else:                        acc_group = "BOTH_WRONG"

        gold      = os.get("answer", [])
        orig_decomp = o.get("decomposed", [])
        rew_decomp  = r.get("decomposed", [])
        info = classify_pair(
            gold,
            o.get("intermediate_answers", {}),
            r.get("intermediate_answers", {}),
            orig_decomp,
            rew_decomp,
        )

        # question-type labels
        last_q_text = orig_decomp[-1].get("text", "") if orig_decomp else ""
        rew_q1_text = rew_decomp[0].get("text", "")  if rew_decomp  else ""
        gold_rel_type  = classify_gold_relation(last_q_text)
        added_con_type = classify_added_constraint(rew_q1_text)

        records.append({
            "index": i,
            "orig_ok": orig_ok, "rew_ok": rew_ok,
            "acc_group": acc_group,
            "gold": gold,
            "orig_question": o.get("question", ""),
            "rew_question":  r.get("question", ""),
            "orig_decomp": orig_decomp,
            "rew_decomp":  rew_decomp,
            "orig_inter":  o.get("intermediate_answers", {}),
            "rew_inter":   r.get("intermediate_answers", {}),
            "orig_pred":   os.get("prediction", ""),
            "rew_pred":    rs.get("prediction", ""),
            "gold_rel_type":  gold_rel_type,
            "added_con_type": added_con_type,
            **info,
        })
    return records


# ── figure helpers ────────────────────────────────────────────────────────────

COLORS = {
    "BOTH_CORRECT": "#2196F3",   # blue
    "ORIG_BETTER":  "#F44336",   # red
    "REW_BETTER":   "#4CAF50",   # green
    "BOTH_WRONG":   "#9E9E9E",   # grey
}
ORDER_COLOR = {
    "LAST_BOTH":       "#1565C0",
    "LAST_ORIG_ONLY":  "#FB8C00",
    "NOT_LAST_BOTH":   "#AB47BC",
    "MISSING_REW":     "#E53935",
    "MISSING_ORIG":    "#43A047",
    "MISSING_BOTH":    "#78909C",
}
ACC_LABELS = ["BOTH_CORRECT", "ORIG_BETTER", "REW_BETTER", "BOTH_WRONG"]
ORDER_CLASSES = [
    "LAST_BOTH", "LAST_ORIG_ONLY", "NOT_LAST_BOTH",
    "MISSING_REW", "MISSING_ORIG", "MISSING_BOTH",
]
ORDER_LABELS = {
    "LAST_BOTH":      "Gold at last step\nin BOTH chains",
    "LAST_ORIG_ONLY": "Gold shifts to\nnon-last in rewrite",
    "NOT_LAST_BOTH":  "Gold not last\nin either chain",
    "MISSING_REW":    "Gold VANISHES\nfrom rewrite chain",
    "MISSING_ORIG":   "Gold found in\nrewrite only",
    "MISSING_BOTH":   "Gold absent\nfrom both chains",
}


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Q1: In the original chain, where is the gold relation step?
# ══════════════════════════════════════════════════════════════════════════════

def fig_q1_original_gold_position(records):
    """
    Show where the gold step sits in the ORIGINAL chain.
    Rank-from-end: 0 = last sub-question, 1 = second-to-last, etc.
    """
    rfe_vals = [r["orig_rfe"] for r in records if r["found_orig"]]
    counter  = Counter(rfe_vals)
    n_found  = len(rfe_vals)
    n_total  = len(records)

    ranks = sorted(counter.keys())
    counts = [counter[k] for k in ranks]
    labels = ["last\n(rfe=0)" if k == 0 else f"−{k} from last\n(rfe={k})" for k in ranks]
    bar_colors = ["#1565C0" if k == 0 else "#90CAF9" for k in ranks]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(ranks)), counts, color=bar_colors, edgecolor="white", linewidth=0.8)

    for bar, c, k in zip(bars, counts, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{c}\n({100*c/n_found:.0f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Position of gold relation step (rank from end of chain)", fontsize=10)
    ax.set_ylabel("Number of questions", fontsize=10)
    ax.set_title(
        "Q1: Where is the gold relation step in the ORIGINAL chain?\n"
        f"(n={n_found} questions where gold step is detectable out of {n_total} total)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.25)

    n_last = counter.get(0, 0)
    ax.text(0.98, 0.97,
            f"{100*n_last/n_found:.0f}% of detectable cases\nhave gold at the last step",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1565C0", lw=1))

    fig.tight_layout()
    save(fig, "fig1_q1_original_gold_position.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Q2: Does the rewrite keep the gold relation at the same position?
# ══════════════════════════════════════════════════════════════════════════════

def fig_q2_order_preservation_overview(records):
    """
    Donut chart of order-class distribution across all 291 pairs.
    """
    n_total = len(records)
    counts  = {oc: sum(1 for r in records if r["order_class"] == oc)
               for oc in ORDER_CLASSES}

    sizes  = [counts[oc] for oc in ORDER_CLASSES]
    colors = [ORDER_COLOR[oc] for oc in ORDER_CLASSES]
    explode= [0.04] * len(ORDER_CLASSES)

    nice_labels = [
        f"Gold at last step\nin BOTH ({counts['LAST_BOTH']})",
        f"Gold shifts to\nnon-last in rewrite ({counts['LAST_ORIG_ONLY']})",
        f"Gold not last\nin either ({counts['NOT_LAST_BOTH']})",
        f"Gold VANISHES\nfrom rewrite ({counts['MISSING_REW']})",
        f"Gold only in\nrewrite ({counts['MISSING_ORIG']})",
        f"Gold absent\nfrom both ({counts['MISSING_BOTH']})",
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=nice_labels, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        pctdistance=0.78, startangle=140,
        textprops=dict(fontsize=8.5),
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
        at.set_color("white")

    # draw hole
    circle = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(circle)
    ax.text(0, 0, f"n={n_total}", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#333")

    ax.set_title(
        "Q2: Does the rewrite preserve the gold relation position?\n"
        "Distribution of order-preservation class across 291 pairs",
        fontsize=11, fontweight="bold", pad=16,
    )
    fig.tight_layout()
    save(fig, "fig2_q2_order_preservation_donut.png")


def fig_q2_position_shift_detail(records):
    """
    For cases where gold IS found in both chains, compare
    original vs rewrite rank-from-end side-by-side.
    """
    both_found = [r for r in records if r["found_orig"] and r["found_rew"]]

    # bin by rfe in original and rewrite
    orig_rfes = Counter(r["orig_rfe"] for r in both_found)
    rew_rfes  = Counter(r["rew_rfe"]  for r in both_found)

    all_rfe = sorted(set(orig_rfes) | set(rew_rfes))
    x = np.arange(len(all_rfe))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, [orig_rfes.get(k, 0) for k in all_rfe],
                w, label="Original chain", color="#1565C0", alpha=0.85)
    b2 = ax.bar(x + w/2, [rew_rfes.get(k, 0)  for k in all_rfe],
                w, label="Rewrite chain",  color="#FB8C00", alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    xlabels = ["last\n(rfe=0)" if k == 0 else f"−{k} from last\n(rfe={k})" for k in all_rfe]
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Position of gold relation step (rank from end)", fontsize=10)
    ax.set_ylabel("Number of questions", fontsize=10)
    ax.set_title(
        "Q2 (detail): How does the gold step position shift from original → rewrite?\n"
        f"(n={len(both_found)} pairs where gold step is detectable in BOTH chains)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(max(orig_rfes.values()), max(rew_rfes.values())) * 1.3)
    fig.tight_layout()
    save(fig, "fig2b_q2_position_shift.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Q3: When position changes, does accuracy drop?
# ══════════════════════════════════════════════════════════════════════════════

def fig_q3_degradation_by_order_class(records):
    """
    Bar chart: degradation rate (orig✓ rew✗) per order class,
    with stacked bar showing the full 4-group breakdown.
    """
    acc_group_order = ["BOTH_CORRECT", "ORIG_BETTER", "REW_BETTER", "BOTH_WRONG"]
    acc_colors      = [COLORS[g] for g in acc_group_order]
    acc_nice        = ["Both correct\n(orig✓ rew✓)", "Degraded\n(orig✓ rew✗)",
                       "Improved\n(orig✗ rew✓)", "Both wrong\n(orig✗ rew✗)"]

    oc_list   = ["LAST_BOTH", "LAST_ORIG_ONLY", "MISSING_REW", "MISSING_BOTH"]
    oc_labels = [ORDER_LABELS[oc] for oc in oc_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: stacked bar (absolute counts by acc group) ───────────────────
    ax = axes[0]
    bottoms = np.zeros(len(oc_list))
    bars_by_group = {}
    for ag, color in zip(acc_group_order, acc_colors):
        vals = np.array([
            sum(1 for r in records if r["order_class"] == oc and r["acc_group"] == ag)
            for oc in oc_list
        ])
        b = ax.bar(range(len(oc_list)), vals, bottom=bottoms, color=color,
                   label=ag, edgecolor="white", linewidth=0.8)
        bars_by_group[ag] = (b, vals, bottoms.copy())
        bottoms += vals

    # label each bar segment if large enough
    for ag, (b, vals, bots) in bars_by_group.items():
        for i, (bar, v, bot) in enumerate(zip(b, vals, bots)):
            if v >= 4:
                ax.text(bar.get_x() + bar.get_width()/2, bot + v/2,
                        str(v), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")

    ax.set_xticks(range(len(oc_list)))
    ax.set_xticklabels(oc_labels, fontsize=8.5)
    ax.set_ylabel("Number of question pairs", fontsize=10)
    ax.set_title("Accuracy outcome breakdown\nper order-preservation class", fontsize=10, fontweight="bold")
    ax.legend(labels=acc_nice, fontsize=7.5, loc="upper right",
              framealpha=0.9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: degradation rate (orig✓ rew✗ / all orig✓ in class) ─────────
    ax2 = axes[1]
    degrade_rates = []
    degrade_ns    = []
    bar_colors2   = []
    for oc in oc_list:
        subset   = [r for r in records if r["order_class"] == oc]
        orig_ok  = [r for r in subset if r["orig_ok"]]
        degraded = [r for r in orig_ok if not r["rew_ok"]]
        n_orig_ok = len(orig_ok)
        n_deg     = len(degraded)
        degrade_rates.append(100 * n_deg / n_orig_ok if n_orig_ok else 0)
        degrade_ns.append((n_deg, n_orig_ok))
        bar_colors2.append(ORDER_COLOR[oc])

    bars2 = ax2.bar(range(len(oc_list)), degrade_rates,
                    color=bar_colors2, edgecolor="white", linewidth=0.8, alpha=0.9)
    for bar, rate, (nd, no) in zip(bars2, degrade_rates, degrade_ns):
        if no > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{rate:.0f}%\n({nd}/{no})",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(range(len(oc_list)))
    ax2.set_xticklabels(oc_labels, fontsize=8.5)
    ax2.set_ylabel("Degradation rate  (orig✓ → rew✗) %", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.axhline(20, color="#ccc", lw=1, ls="--")
    ax2.axhline(50, color="#ccc", lw=1, ls="--")
    ax2.axhline(80, color="#ccc", lw=1, ls="--")
    ax2.set_title("Q3: How often does accuracy degrade\nper order-preservation class?",
                  fontsize=10, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Effect of Gold-Relation Position Change on Final Answer Accuracy",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, "fig3_q3_degradation_by_order_class.png")


def fig_q3_heatmap(records):
    """
    Heatmap: order class (rows) × accuracy group (cols), counts + % of row.
    """
    oc_list  = ["LAST_BOTH", "LAST_ORIG_ONLY", "MISSING_REW", "MISSING_ORIG", "MISSING_BOTH"]
    ag_list  = ["BOTH_CORRECT", "ORIG_BETTER", "REW_BETTER", "BOTH_WRONG"]
    ag_nice  = ["Both correct\n(orig✓ rew✓)", "Degraded\n(orig✓ rew✗)",
                "Improved\n(orig✗ rew✓)", "Both wrong\n(orig✗ rew✗)"]
    oc_nice  = [ORDER_LABELS[oc] for oc in oc_list]

    matrix = np.zeros((len(oc_list), len(ag_list)))
    for i, oc in enumerate(oc_list):
        for j, ag in enumerate(ag_list):
            matrix[i, j] = sum(1 for r in records
                               if r["order_class"] == oc and r["acc_group"] == ag)

    row_sums = matrix.sum(axis=1, keepdims=True)
    pct_matrix = np.where(row_sums > 0, 100 * matrix / row_sums, 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pct_matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(ag_list)))
    ax.set_xticklabels(ag_nice, fontsize=9)
    ax.set_yticks(range(len(oc_list)))
    ax.set_yticklabels(oc_nice, fontsize=9)

    for i in range(len(oc_list)):
        for j in range(len(ag_list)):
            n   = int(matrix[i, j])
            pct = pct_matrix[i, j]
            color = "white" if pct > 55 else "#222"
            ax.text(j, i, f"{n}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=8.5,
                    color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("% of row total", fontsize=9)

    ax.set_title(
        "Heatmap: order-preservation class × accuracy outcome\n"
        "Each cell shows count and % of that order class",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    fig.tight_layout()
    save(fig, "fig4_heatmap_order_vs_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Summary: the "funnel" from order change to accuracy drop
# ══════════════════════════════════════════════════════════════════════════════

def fig_summary_funnel(records):
    """
    Three-row visual summary:
      row 1: how many pairs had gold at last step in original
      row 2: of those, how many preserved that position in the rewrite
      row 3: of each class, how many degraded
    """
    n = len(records)
    n_orig_last      = sum(1 for r in records if r["at_last_orig"])
    n_orig_not_last  = sum(1 for r in records if r["found_orig"] and not r["at_last_orig"])
    n_orig_miss      = sum(1 for r in records if not r["found_orig"])

    # From n_orig_last: how many preserved (LAST_BOTH) vs shifted (LAST_ORIG_ONLY) vs other
    last_both     = [r for r in records if r["order_class"] == "LAST_BOTH"]
    last_orig_only= [r for r in records if r["order_class"] == "LAST_ORIG_ONLY"]
    miss_rew      = [r for r in records if r["order_class"] == "MISSING_REW"]

    def deg_rate(lst):
        orig_ok = [r for r in lst if r["orig_ok"]]
        if not orig_ok: return 0, 0, 0
        deg = [r for r in orig_ok if not r["rew_ok"]]
        return len(deg), len(orig_ok), 100*len(deg)/len(orig_ok)

    d_lb,  o_lb,  r_lb  = deg_rate(last_both)
    d_lo,  o_lo,  r_lo  = deg_rate(last_orig_only)
    d_mr,  o_mr,  r_mr  = deg_rate(miss_rew)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={"wspace": 0.45})

    # ── Panel A: original gold position ────────────────────────────────────
    ax = axes[0]
    vals   = [n_orig_last, n_orig_not_last, n_orig_miss]
    clrs   = ["#1565C0", "#90CAF9", "#CFD8DC"]
    lbls   = [f"Last step\n({n_orig_last})", f"Not last step\n({n_orig_not_last})",
              f"Not detectable\n({n_orig_miss})"]
    ax.pie(vals, labels=lbls, colors=clrs, startangle=90,
           autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
           pctdistance=0.75, textprops=dict(fontsize=9),
           wedgeprops=dict(linewidth=1.2, edgecolor="white"))
    circle = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(circle)
    ax.set_title("A: Where is the gold\nstep in ORIGINAL?", fontsize=10, fontweight="bold")

    # ── Panel B: preservation for last-step originals ───────────────────────
    ax = axes[1]
    vals2  = [len(last_both), len(last_orig_only), len(miss_rew),
              n - len(last_both) - len(last_orig_only) - len(miss_rew)]
    clrs2  = [ORDER_COLOR["LAST_BOTH"], ORDER_COLOR["LAST_ORIG_ONLY"],
              ORDER_COLOR["MISSING_REW"], "#CFD8DC"]
    lbls2  = [f"Preserved\nat last step\n({len(last_both)})",
              f"Shifted to\nnon-last\n({len(last_orig_only)})",
              f"Gold vanished\nfrom rewrite\n({len(miss_rew)})",
              f"Other\n({vals2[3]})"]
    ax.pie(vals2, labels=lbls2, colors=clrs2, startangle=90,
           autopct=lambda p: f"{p:.0f}%" if p > 2 else "",
           pctdistance=0.75, textprops=dict(fontsize=9),
           wedgeprops=dict(linewidth=1.2, edgecolor="white"))
    circle2 = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(circle2)
    ax.set_title("B: Does the rewrite preserve\ngold position?", fontsize=10, fontweight="bold")

    # ── Panel C: degradation rates ─────────────────────────────────────────
    ax = axes[2]
    oc_names   = ["Preserved\n(LAST_BOTH)", "Shifted\n(LAST_ORIG_ONLY)", "Vanished\n(MISSING_REW)"]
    rates      = [r_lb, r_lo, r_mr]
    ns_label   = [f"{d}/{o}" for d, o, _ in [(d_lb, o_lb, r_lb), (d_lo, o_lo, r_lo), (d_mr, o_mr, r_mr)]]
    bar_clrs   = [ORDER_COLOR["LAST_BOTH"], ORDER_COLOR["LAST_ORIG_ONLY"], ORDER_COLOR["MISSING_REW"]]

    bars = ax.bar(range(3), rates, color=bar_clrs, edgecolor="white",
                  linewidth=0.8, alpha=0.9, width=0.5)
    for bar, rate, ns in zip(bars, rates, ns_label):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{rate:.0f}%\n({ns} orig✓ cases)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(oc_names, fontsize=9)
    ax.set_ylabel("Degradation rate  (orig✓ → rew✗) %", fontsize=9)
    ax.set_ylim(0, 108)
    ax.axhline(50, color="#ccc", lw=1, ls="--")
    ax.set_title("C: How often does accuracy drop\nfor each preservation class?",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Summary: From Gold-Relation Position Change to Accuracy Degradation",
        fontsize=12, fontweight="bold", y=1.01,
    )
    save(fig, "fig5_summary_funnel.png")


# ══════════════════════════════════════════════════════════════════════════════
# PRINTED REPORT
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "=" * 76
SEP2 = "-" * 76
W    = 76

def pct_str(n, total):
    return f"{100*n/total:.1f}%" if total else "n/a"


def print_task_definition():
    print(SEP)
    print("  GOLD-RELATION ORDER ANALYSIS")
    print(SEP)
    print("""
  TASK DEFINITION
  ───────────────
  Dataset : 291 paired questions from 2WikiMultiHop
  Model   : Answers by decomposing into sub-questions Q1→Q2→…→Qn,
            answering each with a retriever, then combining into a
            final prediction.

  Two versions of each question:
    ORIGINAL  — the question as written
    REWRITE   — the same question with extra relations added
                (e.g. citizenship, birth date, award) as constraints

  Example:
    Original:  Who is the child of the director of Mukhyamantri (1996)?
      Q1: "Who directed Mukhyamantri?"  → Anjan Choudhury
      Q2: "Who is the child of #1?"     → Chumki Chowdhury  ← GOLD

    Rewrite:   Who is the child of the person born Nov 25 1944 with
               citizenship in British Raj who directed Mukhyamantri?
      Q1: "Who has citizenship in British Raj?" → Jawaharlal Nehru (WRONG)
      Q2: "Who directed Mukhyamantri?"          → Anjan Choudhury
      Q3: "Is #2 the same as #1?"               → no
      Q4: "Who is the child of #3?"             → Frances Bean Cobain (WRONG)
      Prediction: Frances Bean Cobain  ✗  Gold: Chumki Chowdhury

  GOLD RELATION STEP
    The sub-question whose intermediate answer = gold final answer.
    In the example above:  Q2 in original, (not found) in rewrite.

  RESEARCH QUESTIONS
    Q1. In the original chain, where does the gold relation step appear?
    Q2. When the rewrite adds extra relations, does the model keep the
        gold step at the same position in the chain?
    Q3. When the position changes (or the step disappears), does the
        final answer accuracy degrade?
""")


def print_q1(records):
    print(SEP)
    print("  Q1: In the original chain, where is the gold relation step?")
    print(SEP2)

    found  = [r for r in records if r["found_orig"]]
    missed = [r for r in records if not r["found_orig"]]
    n_total = len(records)

    rfe_counter = Counter(r["orig_rfe"] for r in found)
    n_last    = rfe_counter.get(0, 0)
    n_nonlast = len(found) - n_last

    print(f"""
  Of {n_total} total pairs:
    ✓ Gold step detectable in original intermediates : {len(found):>3}  ({pct_str(len(found), n_total)})
    ✗ Gold step NOT detectable (hard / format issue) : {len(missed):>3}  ({pct_str(len(missed), n_total)})

  Among detectable cases ({len(found)} questions):
    At the LAST sub-question (rfe=0) : {n_last:>3}  ({pct_str(n_last, len(found))})
    Not at last sub-question          : {n_nonlast:>3}  ({pct_str(n_nonlast, len(found))})

  Position breakdown (rank from end, 0 = last):""")

    for rfe in sorted(rfe_counter):
        c = rfe_counter[rfe]
        label = "last step   ← standard multi-hop endpoint" if rfe == 0 else f"{rfe} step(s) before last"
        bar = "█" * int(20 * c / len(found))
        print(f"    rfe={rfe}  {label:<40}  {c:>3}  ({pct_str(c, len(found)):>6})  {bar}")

    print(f"""
  ANSWER: The gold relation step is at the last sub-question in
  {pct_str(n_last, len(found))} of detectable cases.  The decomposition model
  consistently places the final answer-producing hop at the end of
  the chain in original questions.
""")


def print_q2(records):
    print(SEP)
    print("  Q2: Does the rewrite keep the gold relation at the same position?")
    print(SEP2)

    n = len(records)
    counts = Counter(r["order_class"] for r in records)

    CLASS_DESCS = {
        "LAST_BOTH":      "Gold at last step in BOTH original and rewrite       (order fully preserved)",
        "LAST_ORIG_ONLY": "Gold at last step in original, NOT last in rewrite   (order shifted)",
        "NOT_LAST_BOTH":  "Gold not at last step in either chain                (comparison / date Qs)",
        "MISSING_REW":    "Gold found in original, VANISHES from rewrite chain  (chain broken)",
        "MISSING_ORIG":   "Gold found in rewrite only                           (rewrite improves coverage)",
        "MISSING_BOTH":   "Gold absent from both chains                         (intrinsically hard Qs)",
    }

    print()
    for oc, desc in CLASS_DESCS.items():
        c = counts.get(oc, 0)
        bar = "█" * int(20 * c / n)
        print(f"  {oc:<17}  {desc}")
        print(f"  {'':17}  {c:>3} pairs  ({pct_str(c, n):>6})  {bar}")
        print()

    last_both    = counts.get("LAST_BOTH", 0)
    last_orig    = counts.get("LAST_ORIG_ONLY", 0)
    miss_rew     = counts.get("MISSING_REW", 0)
    miss_both    = counts.get("MISSING_BOTH", 0)

    print(f"""  ANSWER:
    • In {pct_str(last_both, n)} of pairs the gold step stays at the last position
      in both chains (order fully preserved).
    • In {pct_str(last_orig, n)} the gold step shifts to a non-last position in the
      rewrite (the model places the critical hop earlier in the chain).
    • In {pct_str(miss_rew, n)} the gold answer completely VANISHES from the
      rewrite's intermediate chain — the most alarming outcome.
    • The remaining {pct_str(miss_both, n)} are intrinsically hard questions where
      the gold step is undetectable in either chain.
""")


def print_q3(records):
    print(SEP)
    print("  Q3: When position changes, does accuracy degrade?")
    print(SEP2)

    oc_focus = [
        ("LAST_BOTH",      "Gold preserved at last step"),
        ("LAST_ORIG_ONLY", "Gold shifted to non-last in rewrite"),
        ("MISSING_REW",    "Gold vanished from rewrite chain"),
        ("MISSING_BOTH",   "Gold absent from both chains"),
    ]

    print(f"\n  {'Order class':<26}  {'n':>4}  {'orig✓':>6}  {'rew✓':>6}  "
          f"{'both✓':>6}  {'DEGRADED':>10}  {'Degrade%':>9}")
    print("  " + "-" * 75)

    for oc, label in oc_focus:
        subset   = [r for r in records if r["order_class"] == oc]
        n_sub    = len(subset)
        n_orig_ok= sum(1 for r in subset if r["orig_ok"])
        n_rew_ok = sum(1 for r in subset if r["rew_ok"])
        n_both   = sum(1 for r in subset if r["orig_ok"] and r["rew_ok"])
        n_deg    = sum(1 for r in subset if r["orig_ok"] and not r["rew_ok"])
        rate     = 100 * n_deg / n_orig_ok if n_orig_ok else 0
        bar      = "█" * int(rate / 5)
        print(f"  {label:<26}  {n_sub:>4}  {n_orig_ok:>6}  {n_rew_ok:>6}  "
              f"{n_both:>6}  {n_deg:>5}/{n_orig_ok:<5}  {rate:>6.1f}%  {bar}")

    print(f"""
  ANSWER:
    Order class        Degradation rate      Interpretation
    ─────────────────────────────────────────────────────────────────────
    LAST_BOTH          ~10%    Baseline. Even with order fully preserved,
                               ~10% fail (synthesis / format errors).
    LAST_ORIG_ONLY     ~24%    Shifting gold to a non-last position doubles
                               the degradation rate — a mild but real signal.
    MISSING_REW        ~84%    When the gold answer VANISHES from the rewrite
                               chain, the model almost always gives the wrong
                               final answer.  This is the dominant failure mode.
    MISSING_BOTH       ~36%    Reflects the base difficulty of hard questions
                               (both versions fail at similar rates).

  CONCLUSION:
    Gold-step position change alone (LAST_ORIG_ONLY) is a WEAK failure
    signal — it doubles the degradation rate but still leaves 76% correct.

    The decisive predictor is whether the gold answer is REACHABLE
    anywhere in the rewrite chain.  When it is:  rew accuracy ≈ orig.
    When it is NOT (MISSING_REW, 84% degrade):  the wrong-entity
    cascade has already broken the reasoning chain upstream.

    In practice this maps directly to the "V→no" gate from the earlier
    order_experiment analysis: Q1 retrieved the wrong entity, so the
    gold answer never appears in any intermediate step.
""")


def print_examples_section(records):
    print(SEP)
    print("  ILLUSTRATIVE EXAMPLES")
    print(SEP2)

    def show(r, title):
        print(f"\n  [{title}]")
        print(f"  Original Q : {r['orig_question']}")
        print(f"  Rewrite  Q : {r['rew_question'][:100]}{'...' if len(r['rew_question'])>100 else ''}")
        print(f"  Gold       : {r['gold']}")

        orig_labels = [q["label"] for q in r["orig_decomp"]]
        print(f"  Original chain ({r['n_orig']} steps):")
        for j, q in enumerate(r["orig_decomp"]):
            lbl  = q["label"]
            txt  = q.get("text", "")[:65]
            ans  = r["orig_inter"].get(lbl, "?")
            mark = "  ← GOLD STEP" if lbl in r["gold_labels_orig"] else ""
            print(f"    {lbl}: {txt}")
            print(f"         → {str(ans)[:60]}{mark}")

        rew_labels = [q["label"] for q in r["rew_decomp"]]
        print(f"  Rewrite chain ({r['n_rew']} steps):")
        for j, q in enumerate(r["rew_decomp"]):
            lbl  = q["label"]
            txt  = q.get("text", "")[:65]
            ans  = r["rew_inter"].get(lbl, "?")
            mark = "  ← GOLD STEP" if lbl in r["gold_labels_rew"] else ""
            print(f"    {lbl}: {txt}")
            print(f"         → {str(ans)[:60]}{mark}")

        print(f"  Outcome: orig={'✓' if r['orig_ok'] else '✗'}  rew={'✓' if r['rew_ok'] else '✗'}"
              f"  |  order_class: {r['order_class']}")

    # Example 1: LAST_BOTH, ORIG_BETTER — gold preserved but still failed
    ex = next((r for r in records
               if r["order_class"] == "LAST_BOTH" and r["acc_group"] == "ORIG_BETTER"), None)
    if ex:
        show(ex, "LAST_BOTH × ORIG_BETTER — gold preserved at last step, but rewrite still failed")

    # Example 2: LAST_ORIG_ONLY, ORIG_BETTER — gold shifted, answer wrong
    ex = next((r for r in records
               if r["order_class"] == "LAST_ORIG_ONLY" and r["acc_group"] == "ORIG_BETTER"), None)
    if ex:
        show(ex, "LAST_ORIG_ONLY × ORIG_BETTER — gold shifted to earlier step, answer wrong")

    # Example 3: MISSING_REW, ORIG_BETTER — gold vanished, chain broken
    ex = next((r for r in records
               if r["order_class"] == "MISSING_REW" and r["acc_group"] == "ORIG_BETTER"), None)
    if ex:
        show(ex, "MISSING_REW × ORIG_BETTER — gold vanished from rewrite chain (wrong-entity cascade)")

    # Example 4: LAST_ORIG_ONLY, BOTH_CORRECT — gold shifted but still correct
    ex = next((r for r in records
               if r["order_class"] == "LAST_ORIG_ONLY" and r["acc_group"] == "BOTH_CORRECT"), None)
    if ex:
        show(ex, "LAST_ORIG_ONLY × BOTH_CORRECT — gold shifted but rewrite still succeeded")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 0 — Overall performance + question-type breakdown
# ══════════════════════════════════════════════════════════════════════════════

def fig_overall_performance(records):
    """
    Three-panel figure:
      Left  — overall accuracy: original vs rewrite (4 outcome groups stacked)
      Middle — accuracy by gold relation type (what the question ultimately asks)
      Right  — degradation rate by type of constraint added in the rewrite
    """
    n = len(records)
    acc_groups = ["BOTH_CORRECT", "ORIG_BETTER", "REW_BETTER", "BOTH_WRONG"]
    acc_colors = [COLORS[g] for g in acc_groups]
    acc_nice   = ["Both correct\n(orig✓ rew✓)", "Degraded\n(orig✓ rew✗)",
                  "Improved\n(orig✗ rew✓)", "Both wrong\n(orig✗ rew✗)"]

    # ── build counts ──────────────────────────────────────────────────────────
    group_counts = Counter(r["acc_group"] for r in records)

    rel_types = ["Birth / death", "Family relation", "Nationality", "Education", "Other"]
    con_types = ["Birthdate", "Family constraint", "Other description",
                 "Award / honor", "Citizenship", "Death date"]

    # accuracy by gold relation type
    rel_acc = {}
    for rt in rel_types:
        sub = [r for r in records if r["gold_rel_type"] == rt]
        rel_acc[rt] = {
            "n":       len(sub),
            "orig_ok": sum(1 for r in sub if r["orig_ok"]),
            "rew_ok":  sum(1 for r in sub if r["rew_ok"]),
            "degrade": sum(1 for r in sub if r["orig_ok"] and not r["rew_ok"]),
            "orig_ok_n": sum(1 for r in sub if r["orig_ok"]),
        }

    # degradation rate by constraint type
    con_deg = {}
    for ct in con_types:
        sub      = [r for r in records if r["added_con_type"] == ct]
        orig_ok  = [r for r in sub if r["orig_ok"]]
        degraded = [r for r in orig_ok if not r["rew_ok"]]
        con_deg[ct] = (len(degraded), len(orig_ok), len(sub))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"wspace": 0.45})

    # ── Panel A: overall stacked bar (orig / rewrite) ─────────────────────
    ax = axes[0]
    n_orig_ok = sum(1 for r in records if r["orig_ok"])
    n_rew_ok  = sum(1 for r in records if r["rew_ok"])

    # two bars: original score, rewrite score — broken into 4 groups
    # (simplified: just show overall accuracy + the 4-group breakdown)
    group_order = ["BOTH_CORRECT", "REW_BETTER", "ORIG_BETTER", "BOTH_WRONG"]
    g_colors    = [COLORS[g] for g in group_order]
    g_nice      = ["Both correct", "Improved\n(rew only)", "Degraded\n(orig only)", "Both wrong"]

    vals = [group_counts[g] for g in group_order]
    bottoms = np.zeros(2)
    x_pos = [0, 1]

    # For each group, compute its contribution to each bar
    # orig bar: BOTH_CORRECT + ORIG_BETTER are correct
    # rew  bar: BOTH_CORRECT + REW_BETTER are correct
    stacks = {
        "BOTH_CORRECT": (group_counts["BOTH_CORRECT"], group_counts["BOTH_CORRECT"]),
        "REW_BETTER":   (0,                            group_counts["REW_BETTER"]),
        "ORIG_BETTER":  (group_counts["ORIG_BETTER"],  0),
        "BOTH_WRONG":   (group_counts["BOTH_WRONG"],   group_counts["BOTH_WRONG"]),
    }
    stack_colors = {
        "BOTH_CORRECT": "#2196F3",
        "REW_BETTER":   "#4CAF50",
        "ORIG_BETTER":  "#F44336",
        "BOTH_WRONG":   "#9E9E9E",
    }
    stack_labels = {
        "BOTH_CORRECT": f"Both correct ({group_counts['BOTH_CORRECT']})",
        "REW_BETTER":   f"Rewrite only ({group_counts['REW_BETTER']})",
        "ORIG_BETTER":  f"Original only ({group_counts['ORIG_BETTER']})",
        "BOTH_WRONG":   f"Both wrong ({group_counts['BOTH_WRONG']})",
    }

    bottoms = np.zeros(2)
    for g in ["BOTH_WRONG", "ORIG_BETTER", "REW_BETTER", "BOTH_CORRECT"]:
        orig_v, rew_v = stacks[g]
        bar_vals = [orig_v, rew_v]
        bars = ax.bar(x_pos, bar_vals, bottom=bottoms, color=stack_colors[g],
                      label=stack_labels[g], edgecolor="white", linewidth=0.8, width=0.5)
        for bar, v, bot in zip(bars, bar_vals, bottoms):
            if v >= 8:
                ax.text(bar.get_x() + bar.get_width()/2, bot + v/2,
                        str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += np.array(bar_vals)

    # accuracy % labels on top
    for xi, acc_n in zip(x_pos, [n_orig_ok, n_rew_ok]):
        ax.text(xi, n + 4, f"{100*acc_n/n:.1f}%\naccurate",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Original", "Rewrite"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of question pairs", fontsize=10)
    ax.set_ylim(0, n * 1.18)
    ax.set_title("A: Overall accuracy\n(n=291 pairs)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel B: orig & rewrite accuracy by gold relation type ────────────
    ax = axes[1]
    y = np.arange(len(rel_types))
    h = 0.32

    orig_rates = [100 * rel_acc[rt]["orig_ok"] / rel_acc[rt]["n"]
                  if rel_acc[rt]["n"] else 0 for rt in rel_types]
    rew_rates  = [100 * rel_acc[rt]["rew_ok"]  / rel_acc[rt]["n"]
                  if rel_acc[rt]["n"] else 0 for rt in rel_types]
    ns         = [rel_acc[rt]["n"] for rt in rel_types]

    b1 = ax.barh(y + h/2, orig_rates, h, label="Original", color="#1565C0", alpha=0.85)
    b2 = ax.barh(y - h/2, rew_rates,  h, label="Rewrite",  color="#FB8C00", alpha=0.85)

    for bar, rate, n_rt in zip(b1, orig_rates, ns):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{rate:.0f}%", va="center", fontsize=8)
    for bar, rate in zip(b2, rew_rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{rate:.0f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{rt}\n(n={rel_acc[rt]['n']})" for rt in rel_types], fontsize=8.5)
    ax.set_xlabel("Accuracy %", fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_title("B: Accuracy by gold relation type\n(what the question ultimately asks)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(50, color="#ddd", lw=1, ls="--")

    # ── Panel C: degradation rate by constraint type ────────────────────
    ax = axes[2]
    deg_rates = []
    deg_labels = []
    bar_clrs3  = []
    ns_labels3 = []

    palette = ["#C62828", "#E64A19", "#F57F17", "#558B2F", "#1565C0", "#6A1B9A"]
    for ci, ct in enumerate(con_types):
        nd, no, ntot = con_deg[ct]
        rate = 100 * nd / no if no else 0
        deg_rates.append(rate)
        deg_labels.append(f"{ct}\n(n={ntot})")
        bar_clrs3.append(palette[ci % len(palette)])
        ns_labels3.append(f"{nd}/{no}")

    # sort by degradation rate descending
    order = sorted(range(len(deg_rates)), key=lambda i: deg_rates[i], reverse=True)
    deg_rates  = [deg_rates[i]  for i in order]
    deg_labels = [deg_labels[i] for i in order]
    bar_clrs3  = [bar_clrs3[i]  for i in order]
    ns_labels3 = [ns_labels3[i] for i in order]

    bars3 = ax.barh(range(len(con_types)), deg_rates,
                    color=bar_clrs3, edgecolor="white", linewidth=0.8, alpha=0.9, height=0.55)
    for bar, rate, ns_lbl in zip(bars3, deg_rates, ns_labels3):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{rate:.0f}%  ({ns_lbl} orig✓)", va="center", fontsize=8)

    ax.set_yticks(range(len(con_types)))
    ax.set_yticklabels(deg_labels, fontsize=8.5)
    ax.set_xlabel("Degradation rate (orig✓ → rew✗) %", fontsize=10)
    ax.set_xlim(0, 115)
    ax.axvline(50, color="#ddd", lw=1, ls="--")
    ax.set_title("C: Degradation rate by added constraint type\n"
                 "(what the rewrite added as Q1)",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Overall Performance and Question-Type Breakdown  (n=291 pairs)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, "fig0_overall_performance.png")


def fig_order_class_by_question_type(records):
    """
    Stacked bar: for each gold relation type, show the distribution of order classes.
    Helps understand which question types are most vulnerable to chain breakage.
    """
    rel_types   = ["Birth / death", "Family relation", "Nationality", "Education", "Other"]
    oc_focus    = ["LAST_BOTH", "LAST_ORIG_ONLY", "MISSING_REW", "MISSING_BOTH", "OTHER_OC"]
    oc_colors_f = [ORDER_COLOR["LAST_BOTH"], ORDER_COLOR["LAST_ORIG_ONLY"],
                   ORDER_COLOR["MISSING_REW"], ORDER_COLOR["MISSING_BOTH"], "#B0BEC5"]
    oc_labels_f = ["Gold preserved\n(LAST_BOTH)", "Gold shifted\n(LAST_ORIG_ONLY)",
                   "Gold vanished\n(MISSING_REW)", "Gold absent both\n(MISSING_BOTH)", "Other"]

    # build matrix: rel_type × order_class
    matrix = {}
    for rt in rel_types:
        sub = [r for r in records if r["gold_rel_type"] == rt]
        row = {}
        for oc in oc_focus[:-1]:
            row[oc] = sum(1 for r in sub if r["order_class"] == oc)
        row["OTHER_OC"] = len(sub) - sum(row.values())
        matrix[rt] = row

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"wspace": 0.45})

    # ── Left: stacked bar (order class × rel type) ────────────────────────
    ax = axes[0]
    x  = np.arange(len(rel_types))
    bottoms = np.zeros(len(rel_types))
    for oc, color, label in zip(oc_focus, oc_colors_f, oc_labels_f):
        vals = np.array([matrix[rt][oc] for rt in rel_types])
        bars = ax.bar(x, vals, bottom=bottoms, color=color, label=label,
                      edgecolor="white", linewidth=0.8)
        for bar, v, bot in zip(bars, vals, bottoms):
            if v >= 5:
                ax.text(bar.get_x() + bar.get_width()/2, bot + v/2,
                        str(v), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottoms += vals

    totals = [sum(matrix[rt].values()) for rt in rel_types]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 1, f"n={tot}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(rel_types, fontsize=9)
    ax.set_ylabel("Number of pairs", fontsize=10)
    ax.set_title("Order-class breakdown\nby gold relation type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: MISSING_REW rate (chain-breakage rate) by rel type ────────
    ax2 = axes[1]
    miss_rates = [
        100 * matrix[rt]["MISSING_REW"] / sum(matrix[rt].values())
        if sum(matrix[rt].values()) else 0
        for rt in rel_types
    ]
    degrade_rates = []
    for rt in rel_types:
        sub = [r for r in records if r["gold_rel_type"] == rt]
        orig_ok  = [r for r in sub if r["orig_ok"]]
        degraded = [r for r in orig_ok if not r["rew_ok"]]
        degrade_rates.append(100 * len(degraded) / len(orig_ok) if orig_ok else 0)

    y = np.arange(len(rel_types))
    h = 0.32
    b1 = ax2.barh(y + h/2, miss_rates,    h, label="Chain breakage rate\n(MISSING_REW %)",
                  color=ORDER_COLOR["MISSING_REW"], alpha=0.85)
    b2 = ax2.barh(y - h/2, degrade_rates, h, label="Degradation rate\n(orig✓→rew✗ %)",
                  color="#F44336", alpha=0.7)

    for bar, v in zip(list(b1) + list(b2), miss_rates + degrade_rates):
        ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                 f"{v:.0f}%", va="center", fontsize=8)

    ax2.set_yticks(y)
    ax2.set_yticklabels(rel_types, fontsize=9)
    ax2.set_xlabel("%", fontsize=10)
    ax2.set_xlim(0, 100)
    ax2.axvline(30, color="#ddd", lw=1, ls="--")
    ax2.set_title("Chain breakage vs degradation rate\nby gold relation type",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Which Question Types Are Most Vulnerable to Rewrite-Induced Failures?",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, "fig6_order_class_by_question_type.png")


# ── print: overall performance ────────────────────────────────────────────────

def print_overall_performance(records):
    n = len(records)
    n_orig_ok = sum(1 for r in records if r["orig_ok"])
    n_rew_ok  = sum(1 for r in records if r["rew_ok"])
    group_counts = Counter(r["acc_group"] for r in records)

    print(SEP)
    print("  OVERALL PERFORMANCE")
    print(SEP2)
    print(f"""
  Dataset: {n} question pairs (2WikiMultiHop)

  Accuracy
  ─────────────────────────────────────────────
  Original:  {n_orig_ok}/{n}  ({100*n_orig_ok/n:.1f}%)
  Rewrite:   {n_rew_ok}/{n}  ({100*n_rew_ok/n:.1f}%)
  Δ accuracy: {100*(n_rew_ok - n_orig_ok)/n:+.1f} pp

  Outcome breakdown:
    Both correct   (orig✓ rew✓): {group_counts['BOTH_CORRECT']:>4}  ({pct_str(group_counts['BOTH_CORRECT'], n)})
    Degraded       (orig✓ rew✗): {group_counts['ORIG_BETTER']:>4}  ({pct_str(group_counts['ORIG_BETTER'], n)})
    Improved       (orig✗ rew✓): {group_counts['REW_BETTER']:>4}  ({pct_str(group_counts['REW_BETTER'], n)})
    Both wrong     (orig✗ rew✗): {group_counts['BOTH_WRONG']:>4}  ({pct_str(group_counts['BOTH_WRONG'], n)})
""")

    # by gold relation type
    rel_types = ["Birth / death", "Family relation", "Nationality", "Education", "Other"]
    print("  Accuracy by question type  (what the gold step ultimately asks)")
    print(f"  {'Question type':<22}  {'n':>4}  {'orig acc':>9}  {'rew acc':>9}  "
          f"{'Δ acc':>7}  {'degrade rate':>13}")
    print("  " + "-" * 72)
    for rt in rel_types:
        sub      = [r for r in records if r["gold_rel_type"] == rt]
        n_rt     = len(sub)
        orig_ok  = sum(1 for r in sub if r["orig_ok"])
        rew_ok   = sum(1 for r in sub if r["rew_ok"])
        n_orig_ok_rt = orig_ok
        degraded = sum(1 for r in sub if r["orig_ok"] and not r["rew_ok"])
        delta    = 100 * (rew_ok - orig_ok) / n_rt if n_rt else 0
        deg_r    = 100 * degraded / orig_ok if orig_ok else 0
        print(f"  {rt:<22}  {n_rt:>4}  "
              f"{100*orig_ok/n_rt:>7.1f}%  {100*rew_ok/n_rt:>7.1f}%  "
              f"{delta:>+6.1f}pp  {degraded:>3}/{orig_ok:<3} ({deg_r:.0f}%)")

    print()
    # by added constraint type
    con_types = ["Birthdate", "Family constraint", "Other description",
                 "Award / honor", "Citizenship", "Death date"]
    print("  Degradation rate by constraint type  (what was added to the rewrite Q1)")
    print(f"  {'Constraint type':<22}  {'n':>4}  {'orig acc':>9}  {'rew acc':>9}  "
          f"{'degrade rate':>13}")
    print("  " + "-" * 64)
    for ct in sorted(con_types, key=lambda c: -sum(
            1 for r in records if r["added_con_type"] == c and r["orig_ok"] and not r["rew_ok"]
        ) / max(sum(1 for r in records if r["added_con_type"] == c and r["orig_ok"]), 1)):
        sub      = [r for r in records if r["added_con_type"] == ct]
        n_ct     = len(sub)
        orig_ok  = sum(1 for r in sub if r["orig_ok"])
        rew_ok   = sum(1 for r in sub if r["rew_ok"])
        degraded = sum(1 for r in sub if r["orig_ok"] and not r["rew_ok"])
        deg_r    = 100 * degraded / orig_ok if orig_ok else 0
        print(f"  {ct:<22}  {n_ct:>4}  "
              f"{100*orig_ok/n_ct:>7.1f}%  {100*rew_ok/n_ct:>7.1f}%  "
              f"{degraded:>3}/{orig_ok:<3} ({deg_r:.0f}%)")

    print()
    # by n_steps in original chain
    print("  Accuracy by original chain length  (number of sub-questions)")
    print(f"  {'Orig chain length':<22}  {'n':>4}  {'orig acc':>9}  {'rew acc':>9}  "
          f"{'degrade rate':>13}")
    print("  " + "-" * 64)
    for nsteps in sorted(set(r["n_orig"] for r in records)):
        sub      = [r for r in records if r["n_orig"] == nsteps]
        n_ns     = len(sub)
        orig_ok  = sum(1 for r in sub if r["orig_ok"])
        rew_ok   = sum(1 for r in sub if r["rew_ok"])
        degraded = sum(1 for r in sub if r["orig_ok"] and not r["rew_ok"])
        deg_r    = 100 * degraded / orig_ok if orig_ok else 0
        print(f"  {nsteps} sub-questions{'':<9}  {n_ns:>4}  "
              f"{100*orig_ok/n_ns:>7.1f}%  {100*rew_ok/n_ns:>7.1f}%  "
              f"{degraded:>3}/{orig_ok:<3} ({deg_r:.0f}%)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", end=" ", flush=True)
    records = load_records()
    print(f"{len(records)} pairs loaded.\n")

    print("Generating figures...", flush=True)
    fig_overall_performance(records)
    fig_order_class_by_question_type(records)
    fig_q1_original_gold_position(records)
    fig_q2_order_preservation_overview(records)
    fig_q2_position_shift_detail(records)
    fig_q3_degradation_by_order_class(records)
    fig_q3_heatmap(records)
    fig_summary_funnel(records)
    print()

    print_task_definition()
    print_overall_performance(records)
    print_q1(records)
    print_q2(records)
    print_q3(records)
    print_examples_section(records)

    print(SEP)
    print(f"  Figures saved to: {FIG_DIR}/")
    print("  fig0_overall_performance.png            — Overall accuracy + question-type breakdown")
    print("  fig1_q1_original_gold_position.png      — Q1: position in original chain")
    print("  fig2_q2_order_preservation_donut.png    — Q2: overall preservation breakdown")
    print("  fig2b_q2_position_shift.png              — Q2: orig vs rewrite rank-from-end")
    print("  fig3_q3_degradation_by_order_class.png  — Q3: degradation rate per class")
    print("  fig4_heatmap_order_vs_accuracy.png       — Q3: full cross-tabulation heatmap")
    print("  fig5_summary_funnel.png                  — Summary: A→B→C flow")
    print("  fig6_order_class_by_question_type.png   — Which question types break most")
    print(SEP)


if __name__ == "__main__":
    main()
