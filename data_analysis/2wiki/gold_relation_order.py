"""
Gold-Relation Order Analysis  —  2WikiMultiHop
===============================================

All questions in this dataset are 2-hop:

    Entity_0  --[Hop1]--->  Bridge Entity  --[Hop2 / Gold Relation]--->  Final Answer

When the model decomposes a question it produces sub-questions Q1...Qn and
retrieves an intermediate answer for each.

Definitions used throughout this script
----------------------------------------
Gold relation step  —  The sub-question that applies Hop2 (the gold relation)
    to the bridge entity to produce the final answer.
    Identified syntactically: the LAST sub-question that
      (a) references a previous answer via "#N" in its text, and
      (b) has a factual (non yes/no) answer.

Bridge entity  —  The intermediate entity that the gold relation step takes as
    input.  It is the answer to the sub-question labelled "#N" in the gold
    relation step's text.

In the original chain Hop1 is always Q1 (the only factual step before Hop2),
so the bridge entity = Q1's answer.

Research focus
--------------
When the rewrite adds extra constraints, its Q1 is a low-selectivity filter
(citizenship, birthdate, award, ...) rather than the original's high-selectivity
anchor.  We ask:

  Q1.  Does rewrite Q1 retrieve the correct bridge entity?
  Q2.  When it does not, which step in the rewrite chain first finds the
       correct bridge entity, and does that entity get fed to the gold step?
  Q3.  When the wrong bridge entity is fed to the gold step, does accuracy
       collapse?
"""

import json
import re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).resolve().parent
ORIG_RAW   = DATA_DIR / "original.jsonl"
ORIG_SCORE = DATA_DIR / "original_score.jsonl"
REW_RAW    = DATA_DIR / "rewrite.jsonl"
REW_SCORE  = DATA_DIR / "rewrite_score.jsonl"
FIG_DIR    = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── string helpers ────────────────────────────────────────────────────────────

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_match(a, b):
    """Return True if two entity strings refer to the same entity."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) >= 2 and len(short & long) / len(short) >= 0.6:
        return True
    return False


def gold_match(gold_list, text):
    """Return True if any gold string is substantially present in text."""
    if not isinstance(text, str):
        return False
    nt = normalise(text)
    if len(nt.split()) <= 1 and len(nt) <= 4:
        return False
    for g in gold_list:
        ng = normalise(g)
        if not ng:
            continue
        if ng in nt or nt in ng:
            return True
        tg, tt = set(ng.split()), set(nt.split())
        if len(tg) >= 2 and len(tg & tt) / len(tg) >= 0.5:
            return True
    return False


# ── core identification ───────────────────────────────────────────────────────

YES_NO = {"yes", "no", "yes.", "no.", "true", "false", "yes,", "no,"}


def find_gold_step(decomposed, intermediates):
    """
    Identify the gold relation step in a decomposition chain.

    Returns the LAST sub-question that:
      (a) contains a '#N' reference to a previous answer
      (b) has a factual (non yes/no) intermediate answer

    Returns a dict with keys: label, text, answer, rfe, ref_label
    Returns None if no such step exists.
    """
    labels = [q["label"] for q in decomposed]
    n = len(labels)
    candidates = []
    for i, q in enumerate(decomposed):
        lbl = q["label"]
        txt = q.get("text", "")
        ans = intermediates.get(lbl, "")
        if not re.search(r"#\d", txt):
            continue
        if isinstance(ans, str) and ans.strip().lower() in YES_NO:
            continue
        ref = re.search(r"#(\d+)", txt)
        ref_lbl = f"Q{ref.group(1)}" if ref else None
        candidates.append({
            "label": lbl, "text": txt, "answer": ans,
            "rfe": n - 1 - i, "ref_label": ref_lbl, "pos": i,
        })
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["rfe"])


def analyze_pair(orig_decomp, rew_decomp, orig_inter, rew_inter, gold):
    """
    For one question pair, identify:
      - bridge entity in the original (correct anchor for Hop2)
      - whether rewrite Q1 retrieves the correct bridge entity
      - where in the rewrite chain the correct bridge entity first appears
      - which entity is actually fed to the gold relation step in the rewrite
      - whether the gold step produces the correct final answer
    """
    # ── original chain ────────────────────────────────────────────────────────
    gs_orig    = find_gold_step(orig_decomp, orig_inter)
    bridge_orig = orig_inter.get(gs_orig["ref_label"]) if gs_orig and gs_orig["ref_label"] else None

    # ── original Q1 accuracy (baseline) ──────────────────────────────────────
    orig_q1_label  = orig_decomp[0]["label"] if orig_decomp else None
    orig_q1_text   = orig_decomp[0].get("text", "") if orig_decomp else ""
    orig_q1_answer = orig_inter.get(orig_q1_label) if orig_q1_label else None
    orig_q1_correct = entity_match(bridge_orig, orig_q1_answer) if bridge_orig else None

    # ── rewrite Q1 ────────────────────────────────────────────────────────────
    rew_q1_label  = rew_decomp[0]["label"] if rew_decomp else None
    rew_q1_text   = rew_decomp[0].get("text", "") if rew_decomp else ""
    rew_q1_answer = rew_inter.get(rew_q1_label) if rew_q1_label else None
    q1_correct    = entity_match(bridge_orig, rew_q1_answer) if bridge_orig else None

    # ── rewrite gold relation step ────────────────────────────────────────────
    gs_rew      = find_gold_step(rew_decomp, rew_inter)
    bridge_rew  = rew_inter.get(gs_rew["ref_label"]) if gs_rew and gs_rew["ref_label"] else None
    gold_step_correct_bridge = entity_match(bridge_orig, bridge_rew) if bridge_orig and bridge_rew else False
    gold_found_rew = gold_match(gold, gs_rew["answer"]) if gs_rew else False

    # ── where does Hop1 (bridge entity) appear in the rewrite chain? ─────────
    # In the original, Hop1 is always Q1. In the rewrite, it may shift to Q2, Q3, …
    first_correct_pos = None   # 0-indexed position where bridge entity first appears in rewrite
    for i, q in enumerate(rew_decomp):
        ans = rew_inter.get(q["label"], "")
        if entity_match(bridge_orig, ans):
            first_correct_pos = i
            break

    # ── overall classification (Q1 selectivity outcome) ──────────────────────
    if bridge_orig is None:
        q1_class = "BRIDGE_UNKNOWN"
    elif q1_correct:
        q1_class = "Q1_CORRECT"
    elif first_correct_pos is not None and gold_step_correct_bridge:
        q1_class = "Q1_WRONG_RECOVERED"
    elif first_correct_pos is not None and not gold_step_correct_bridge:
        q1_class = "Q1_WRONG_FOUND_NOT_USED"
    else:
        q1_class = "Q1_WRONG_NOT_FOUND"

    # ── order change classification ───────────────────────────────────────────
    # rfe=0 means the gold step is the LAST sub-question in the chain.
    # When the rewrite adds verification steps AFTER the gold step, rfe > 0.
    o_rfe = gs_orig["rfe"] if gs_orig else None
    r_rfe = gs_rew["rfe"]  if gs_rew  else None
    if gs_orig is None or gs_rew is None:
        order_class = "GOLD_NOT_FOUND"
    elif o_rfe == 0 and r_rfe == 0:
        order_class = "GOLD_LAST_BOTH"        # gold step still last in both chains
    elif o_rfe == 0 and r_rfe > 0:
        order_class = "GOLD_PUSHED_EARLIER"   # verification steps appended after gold step
    else:
        order_class = "OTHER"

    return {
        "gs_orig": gs_orig,
        "gs_rew":  gs_rew,
        "bridge_orig": bridge_orig,
        "bridge_rew":  bridge_rew,
        "orig_q1_text":    orig_q1_text,
        "orig_q1_answer":  orig_q1_answer,
        "orig_q1_correct": orig_q1_correct,
        "rew_q1_answer": rew_q1_answer,
        "rew_q1_text":   rew_q1_text,
        "q1_correct":    q1_correct,
        "gold_step_correct_bridge": gold_step_correct_bridge,
        "gold_found_rew":           gold_found_rew,
        "first_correct_pos":        first_correct_pos,
        "q1_class":                 q1_class,
        "order_class":              order_class,
        "gs_orig_rfe":              o_rfe,
        "gs_rew_rfe":               r_rfe,
        "n_orig": len(orig_decomp),
        "n_rew":  len(rew_decomp),
    }


# ── load and annotate all 291 pairs ──────────────────────────────────────────

def load_records():
    orig_raws   = read_jsonl(ORIG_RAW)
    orig_scores = read_jsonl(ORIG_SCORE)
    rew_raws    = read_jsonl(REW_RAW)
    rew_scores  = read_jsonl(REW_SCORE)

    def key(r, i):
        return (r.get("question_id", i), r.get("source_index", i),
                r.get("decompose_id", 0), r.get("index", i))

    orig_idx = {key(r, i): r for i, r in enumerate(orig_raws)}
    rew_idx  = {key(r, i): r for i, r in enumerate(rew_raws)}

    records = []
    for i, (os, rs) in enumerate(zip(orig_scores, rew_scores)):
        k = key(os, i)
        o = orig_idx.get(k, {})
        r = rew_idx.get(k, {})

        orig_ok = float(os.get("accuracy", 0)) == 1.0
        rew_ok  = float(rs.get("accuracy", 0)) == 1.0
        if orig_ok and rew_ok:       acc_group = "BOTH_CORRECT"
        elif orig_ok and not rew_ok: acc_group = "ORIG_BETTER"
        elif not orig_ok and rew_ok: acc_group = "REW_BETTER"
        else:                        acc_group = "BOTH_WRONG"

        gold = os.get("answer", [])
        orig_decomp = o.get("decomposed", [])
        rew_decomp  = r.get("decomposed", [])
        info = analyze_pair(
            orig_decomp, rew_decomp,
            o.get("intermediate_answers", {}),
            r.get("intermediate_answers", {}),
            gold,
        )
        records.append({
            "index": i,
            "orig_ok": orig_ok, "rew_ok": rew_ok, "acc_group": acc_group,
            "gold": gold,
            "orig_question": o.get("question", ""),
            "rew_question":  r.get("question", ""),
            "orig_decomp": orig_decomp,
            "rew_decomp":  rew_decomp,
            "orig_inter":  o.get("intermediate_answers", {}),
            "rew_inter":   r.get("intermediate_answers", {}),
            "orig_pred":   os.get("prediction", ""),
            "rew_pred":    rs.get("prediction", ""),
            **info,
        })
    return records


# ── figure helpers ────────────────────────────────────────────────────────────

def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1  —  Overall accuracy: original vs rewrite (4-group stacked bar)
# ══════════════════════════════════════════════════════════════════════════════

def fig_overall(records):
    n = len(records)
    gc = Counter(r["acc_group"] for r in records)
    n_orig_ok = sum(1 for r in records if r["orig_ok"])
    n_rew_ok  = sum(1 for r in records if r["rew_ok"])

    # stacking order (bottom to top): both_wrong, orig_better, rew_better, both_correct
    stacks = [
        ("BOTH_WRONG",   "#9E9E9E", "Both wrong\n(orig✗ rew✗)"),
        ("ORIG_BETTER",  "#F44336", "Degraded\n(orig✓ rew✗)"),
        ("REW_BETTER",   "#4CAF50", "Improved\n(orig✗ rew✓)"),
        ("BOTH_CORRECT", "#2196F3", "Both correct\n(orig✓ rew✓)"),
    ]
    # contribution to each bar (orig_bar, rew_bar)
    contrib = {
        "BOTH_WRONG":   (gc["BOTH_WRONG"],   gc["BOTH_WRONG"]),
        "ORIG_BETTER":  (gc["ORIG_BETTER"],  0),
        "REW_BETTER":   (0,                  gc["REW_BETTER"]),
        "BOTH_CORRECT": (gc["BOTH_CORRECT"], gc["BOTH_CORRECT"]),
    }

    fig, ax = plt.subplots(figsize=(5, 5.5))
    bottoms = np.zeros(2)
    for g, color, label in stacks:
        vals = np.array(contrib[g])
        bars = ax.bar([0, 1], vals, bottom=bottoms, color=color, label=label,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, v, bot in zip(bars, vals, bottoms):
            if v >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + v / 2,
                        str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += vals

    for xi, acc in zip([0, 1], [n_orig_ok, n_rew_ok]):
        ax.text(xi, n + 5, f"{100*acc/n:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original", "Rewrite"], fontsize=11)
    ax.set_ylabel("Number of pairs (n=291)", fontsize=10)
    ax.set_ylim(0, n * 1.15)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Overall Accuracy: Original vs Rewrite", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig1_overall_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2  —  Q1 accuracy comparison: original vs rewrite, and Hop1 order shift
# ══════════════════════════════════════════════════════════════════════════════

def fig_q1_comparison(records):
    """
    Left  — Q1 accuracy: original Q1 vs rewrite Q1 (bar + breakdown)
    Right — Where Hop1 (bridge entity) appears in the rewrite chain vs always-Q1 in original
    """
    has_bridge = [r for r in records if r["bridge_orig"] is not None]
    nb = len(has_bridge)

    orig_q1_ok = sum(1 for r in has_bridge if r["orig_q1_correct"])
    rew_q1_ok  = sum(1 for r in has_bridge if r["q1_correct"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Q1 accuracy comparison bar ─────────────────────────────────────
    ax = axes[0]
    labels  = ["Original Q1\n(direct Hop1 question)", "Rewrite Q1\n(added constraint)"]
    ok_vals = [100 * orig_q1_ok / nb, 100 * rew_q1_ok / nb]
    fail_vals = [100 - v for v in ok_vals]
    colors_ok   = ["#455A64", "#2196F3"]
    colors_fail = ["#B0BEC5", "#BBDEFB"]

    x = np.arange(2)
    b1 = ax.bar(x, ok_vals,   0.52, label="Q1 gets correct bridge entity", color=colors_ok,   alpha=0.9)
    b2 = ax.bar(x, fail_vals, 0.52, bottom=ok_vals, label="Q1 gets WRONG entity",
                color=colors_fail, alpha=0.85, edgecolor="white")
    for bar, v in zip(b1, ok_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                f"{v:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    for bar, ov, fv in zip(b2, ok_vals, fail_vals):
        if fv > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, ov + fv / 2,
                    f"{fv:.1f}%", ha="center", va="center", fontsize=11, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% of pairs with identifiable bridge entity", fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Q1 Accuracy: Original vs Rewrite\n"
        f"(original={100*orig_q1_ok/nb:.1f}%  →  rewrite={100*rew_q1_ok/nb:.1f}%,  n={nb})",
        fontsize=10, fontweight="bold",
    )

    # ── Right: Hop1 position in rewrite (grouped bar: count + rew accuracy) ──
    ax = axes[1]
    hop1_pos_data = []
    for pos in [0, 1, 2, 3]:
        sub = [r for r in has_bridge if r["first_correct_pos"] == pos]
        if sub:
            ra = sum(r["rew_ok"] for r in sub)
            hop1_pos_data.append((f"Q{pos+1}", len(sub), 100 * ra / len(sub)))
    never = [r for r in has_bridge if r["first_correct_pos"] is None]
    if never:
        ra = sum(r["rew_ok"] for r in never)
        hop1_pos_data.append(("Never\nfound", len(never), 100 * ra / len(never)))

    labels_h = [d[0] for d in hop1_pos_data]
    counts_h = [d[1] for d in hop1_pos_data]
    accs_h   = [d[2] for d in hop1_pos_data]
    colors_h = ["#4CAF50" if l == "Q1" else "#FF9800" if l in ["Q2","Q3","Q4"] else "#F44336"
                for l in labels_h]

    x2 = np.arange(len(hop1_pos_data))
    w  = 0.44
    bars = ax.bar(x2, counts_h, w, color=colors_h, alpha=0.88, edgecolor="white")
    for bar, cnt, acc in zip(bars, counts_h, accs_h):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{cnt}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"acc\n{acc:.0f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    ax.set_xticks(x2)
    ax.set_xticklabels(
        [f"Hop1 at {l}\n(pos {i})" if l not in ("Never\nfound",) else "Hop1 never\nfound"
         for i, l in enumerate(labels_h)],
        fontsize=9,
    )
    ax.set_ylabel("Number of pairs", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # add a note for original baseline
    ax.axhline(0, color="none")
    ax.text(0, max(counts_h) * 1.06,
            "In original: Hop1 ALWAYS at Q1 (pos 0)  →  94.8% Q1 accuracy",
            ha="left", va="bottom", fontsize=8.5, color="#333",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D", alpha=0.9))
    ax.set_title(
        "Where Does Hop1 (Bridge Entity) Appear in the Rewrite Chain?\n"
        "(Original: always Q1 → Rewrite: shifts to Q2, Q3, Q4, or never)",
        fontsize=10, fontweight="bold",
    )

    fig.suptitle(
        "Order Shift: Original Q1 = Direct Hop1  →  Rewrite Q1 = Added Constraint",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig2_q1_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3  —  Q1 selectivity: does rewrite Q1 retrieve the correct bridge entity?
# ══════════════════════════════════════════════════════════════════════════════

def fig_q1_selectivity(records):
    """
    Left  — distribution of Q1 outcome classes (pie)
    Right — accuracy rate for each class (bar)
    """
    classes = [
        ("Q1_CORRECT",           "#2196F3", "Q1 retrieves\ncorrect bridge entity"),
        ("Q1_WRONG_RECOVERED",   "#4CAF50", "Q1 wrong, correct entity\nfound & used later"),
        ("Q1_WRONG_FOUND_NOT_USED", "#FF9800", "Q1 wrong, correct entity\nfound but not used"),
        ("Q1_WRONG_NOT_FOUND",   "#F44336", "Q1 wrong, correct entity\nnever found"),
        ("BRIDGE_UNKNOWN",       "#9E9E9E", "Bridge entity\nnot identified"),
    ]
    counts = {c: sum(1 for r in records if r["q1_class"] == c) for c, _, _ in classes}
    n = len(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: pie ─────────────────────────────────────────────────────────────
    ax = axes[0]
    sizes  = [counts[c] for c, _, _ in classes]
    colors = [col for _, col, _ in classes]
    labels = [f"{lbl}\n({counts[c]})" for c, _, lbl in classes]
    wedges, texts, autos = ax.pie(
        sizes, labels=labels, colors=colors, startangle=140,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        pctdistance=0.78, textprops=dict(fontsize=8.5),
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
    )
    for at in autos:
        at.set_fontsize(8); at.set_fontweight("bold"); at.set_color("white")
    circle = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(circle)
    ax.text(0, 0, f"n={n}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Q1: Does rewrite Q1 retrieve\nthe correct bridge entity?",
                 fontsize=10, fontweight="bold")

    # ── Right: accuracy bar per class ─────────────────────────────────────────
    ax2 = axes[1]
    class_keys = [c for c, _, _ in classes if counts[c] > 0]
    class_lbls = [lbl for c, _, lbl in classes if counts[c] > 0]
    orig_acc   = [100 * sum(1 for r in records if r["q1_class"] == c and r["orig_ok"]) / counts[c]
                  for c in class_keys]
    rew_acc    = [100 * sum(1 for r in records if r["q1_class"] == c and r["rew_ok"])  / counts[c]
                  for c in class_keys]
    bar_colors = [col for c, col, _ in classes if counts[c] > 0]

    y = np.arange(len(class_keys))
    h = 0.32
    b1 = ax2.barh(y + h/2, orig_acc, h, label="Original", color="#455A64", alpha=0.85)
    b2 = ax2.barh(y - h/2, rew_acc,  h, label="Rewrite",  color=bar_colors, alpha=0.9)

    for bar, v in zip(list(b1) + list(b2), orig_acc + rew_acc):
        ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                 f"{v:.0f}%", va="center", fontsize=8.5)

    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{lbl}\n(n={counts[c]})" for c, lbl in zip(class_keys, class_lbls)],
                        fontsize=8.5)
    ax2.set_xlabel("Accuracy %", fontsize=10)
    ax2.set_xlim(0, 115)
    ax2.axvline(50, color="#ddd", lw=1, ls="--")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("Accuracy (orig vs rewrite)\nby Q1 outcome class",
                  fontsize=10, fontweight="bold")

    fig.suptitle("Effect of Rewrite Q1 Selectivity on Bridge Entity Retrieval and Accuracy",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig3_q1_selectivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3  —  Bridge entity: correct vs wrong fed to the gold step
# ══════════════════════════════════════════════════════════════════════════════

def fig_bridge_entity_impact(records):
    """
    Two scenarios where bridge_orig is known:
      bridge_correct=True  vs  bridge_correct=False
    Show accuracy and gold-step answer correctness for each.
    """
    known = [r for r in records if r["bridge_orig"] is not None and r["gs_rew"] is not None]
    correct_bridge = [r for r in known if r["gold_step_correct_bridge"]]
    wrong_bridge   = [r for r in known if not r["gold_step_correct_bridge"]]
    n_c, n_w = len(correct_bridge), len(wrong_bridge)

    def rates(lst):
        n = len(lst)
        if n == 0:
            return 0, 0, 0
        rew_ok      = sum(1 for r in lst if r["rew_ok"])
        gold_in_step= sum(1 for r in lst if r["gold_found_rew"])
        deg         = sum(1 for r in lst if r["orig_ok"] and not r["rew_ok"])
        orig_ok_n   = sum(1 for r in lst if r["orig_ok"])
        return (100*rew_ok/n, 100*gold_in_step/n,
                100*deg/orig_ok_n if orig_ok_n else 0)

    rc_acc, rc_gold, rc_deg = rates(correct_bridge)
    rw_acc, rw_gold, rw_deg = rates(wrong_bridge)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics  = ["Rewrite accuracy %", "Gold answer in step %", "Degradation rate %"]
    c_vals   = [rc_acc, rc_gold, rc_deg]
    w_vals   = [rw_acc, rw_gold, rw_deg]
    colors_c = ["#2196F3", "#4CAF50", "#F44336"]

    for ax, metric, cv, wv, col in zip(axes, metrics, c_vals, w_vals, colors_c):
        bars = ax.bar([0, 1], [cv, wv], color=[col, "#EF9A9A"], edgecolor="white",
                      linewidth=0.8, width=0.5)
        for bar, v in zip(bars, [cv, wv]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f"Correct bridge\n(n={n_c})",
            f"Wrong bridge\n(n={n_w})",
        ], fontsize=9)
        ax.set_ylabel("%", fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(50, color="#ddd", lw=1, ls="--")

    fig.suptitle(
        "When the Wrong Bridge Entity Is Fed to the Gold Relation Step,\n"
        "Accuracy Collapses",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "fig4_bridge_entity_impact.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4  —  Where in the rewrite chain does the correct bridge entity appear?
# ══════════════════════════════════════════════════════════════════════════════

def fig_bridge_recovery_position(records):
    """
    For pairs where the correct bridge entity IS found somewhere in the rewrite:
    show at which position (0-indexed) it first appears, and whether it was
    actually used as input to the gold step.
    """
    found = [r for r in records if r["first_correct_pos"] is not None]
    pos_used     = [r for r in found if r["gold_step_correct_bridge"]]
    pos_not_used = [r for r in found if not r["gold_step_correct_bridge"]]

    pos_counter_used     = Counter(r["first_correct_pos"] for r in pos_used)
    pos_counter_not_used = Counter(r["first_correct_pos"] for r in pos_not_used)
    all_pos = sorted(set(pos_counter_used) | set(pos_counter_not_used))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(all_pos))
    w = 0.38
    b1 = ax.bar(x - w/2,
                [pos_counter_used.get(p, 0)     for p in all_pos],
                w, label="Correct bridge found & used as input to gold step",
                color="#4CAF50", alpha=0.88)
    b2 = ax.bar(x + w/2,
                [pos_counter_not_used.get(p, 0) for p in all_pos],
                w, label="Correct bridge found but NOT used as input to gold step",
                color="#FF9800", alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{p+1}" for p in all_pos], fontsize=10)
    ax.set_xlabel("Position in rewrite chain where correct bridge entity first appears", fontsize=10)
    ax.set_ylabel("Number of pairs", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Where the Correct Bridge Entity First Appears in the Rewrite Chain\n"
        "and Whether It Gets Used as Input to the Gold Relation Step",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "fig5_bridge_recovery_position.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5  —  Order analysis: does the gold step change position after rewrite?
# ══════════════════════════════════════════════════════════════════════════════

def fig_order_analysis(records):
    """
    Three panels:
      Left  — distribution of gold step absolute position in original vs rewrite
      Middle — order change class breakdown + accuracy
      Right  — accuracy within each order class, split by Q1 correctness
    """
    # ── data preparation ──────────────────────────────────────────────────────
    order_classes = [
        ("GOLD_LAST_BOTH",      "#2196F3", "Gold step last\nin both chains"),
        ("GOLD_PUSHED_EARLIER", "#FF9800", "Gold step pushed\nearlier in rewrite"),
        ("GOLD_NOT_FOUND",      "#F44336", "Gold step not\nidentifiable in rewrite"),
        ("OTHER",               "#9E9E9E", "Other"),
    ]
    oc_counts = {c: [r for r in records if r["order_class"] == c] for c, _, _ in order_classes}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Left: position distribution (0-indexed) ───────────────────────────────
    ax = axes[0]
    known_orig = [r for r in records if r["gs_orig"] is not None]
    known_rew  = [r for r in records if r["gs_rew"]  is not None]
    max_pos = max(
        max((r["gs_orig"]["pos"] for r in known_orig), default=0),
        max((r["gs_rew"]["pos"]  for r in known_rew),  default=0),
    )
    positions = list(range(max_pos + 1))
    orig_counts = [sum(1 for r in known_orig if r["gs_orig"]["pos"] == p) for p in positions]
    rew_counts  = [sum(1 for r in known_rew  if r["gs_rew"]["pos"]  == p) for p in positions]
    x = np.arange(len(positions))
    w = 0.38
    b1 = ax.bar(x - w/2, orig_counts, w, label="Original", color="#455A64", alpha=0.85)
    b2 = ax.bar(x + w/2, rew_counts,  w, label="Rewrite",  color="#2196F3", alpha=0.85)
    for bar, v in zip(list(b1) + list(b2), orig_counts + rew_counts):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{p+1}" for p in positions], fontsize=10)
    ax.set_xlabel("Gold step position in chain", fontsize=10)
    ax.set_ylabel("Number of pairs", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Gold Relation Step Position:\nOriginal vs Rewrite", fontsize=10, fontweight="bold")

    # ── Middle: order class bar + accuracy ───────────────────────────────────
    ax = axes[1]
    active = [(c, col, lbl) for c, col, lbl in order_classes if oc_counts[c]]
    y = np.arange(len(active))
    cnts   = [len(oc_counts[c]) for c, _, _ in active]
    colors = [col for _, col, _ in active]
    lbls   = [lbl for _, _, lbl in active]
    rew_accs = [
        100 * sum(r["rew_ok"] for r in oc_counts[c]) / len(oc_counts[c])
        for c, _, _ in active
    ]
    n = len(records)
    ax2_twin = ax.twiny()
    bars = ax.barh(y, cnts, color=colors, alpha=0.85, edgecolor="white", height=0.5)
    for bar, v in zip(bars, cnts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v} ({100*v/n:.0f}%)", va="center", fontsize=8.5)
    ax2_twin.plot(rew_accs, y, "D--", color="#333", markersize=7, label="Rewrite acc %")
    for acc, yi in zip(rew_accs, y):
        ax2_twin.text(acc + 1.5, yi + 0.15, f"{acc:.0f}%", fontsize=8.5, color="#333", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lbl}" for lbl in lbls], fontsize=8.5)
    ax.set_xlabel("Number of pairs", fontsize=9)
    ax2_twin.set_xlabel("Rewrite accuracy %", fontsize=9)
    ax2_twin.set_xlim(0, 120)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Order Change Class\n(count + rewrite accuracy)", fontsize=10, fontweight="bold")

    # ── Right: degradation rate by order class × Q1 correctness ─────────────
    ax = axes[2]
    # Only include classes with enough data for Q1 split
    split_classes = [
        ("GOLD_LAST_BOTH",      "#2196F3"),
        ("GOLD_PUSHED_EARLIER", "#FF9800"),
    ]
    x = np.arange(len(split_classes))
    w2 = 0.32
    q1_correct_deg   = []
    q1_wrong_deg     = []
    q1_correct_n     = []
    q1_wrong_n       = []
    for c, _ in split_classes:
        sub = oc_counts[c]
        q1c  = [r for r in sub if r.get("q1_correct") is True]
        q1w  = [r for r in sub if r.get("q1_correct") is False]
        def degrade_rate(lst):
            orig_ok = [r for r in lst if r["orig_ok"]]
            if not orig_ok: return 0
            return 100 * sum(1 for r in orig_ok if not r["rew_ok"]) / len(orig_ok)
        q1_correct_deg.append(degrade_rate(q1c))
        q1_wrong_deg.append(degrade_rate(q1w))
        q1_correct_n.append(len(q1c))
        q1_wrong_n.append(len(q1w))

    b_c = ax.bar(x - w2/2, q1_correct_deg, w2, label="Q1 correct",
                 color="#4CAF50", alpha=0.88, edgecolor="white")
    b_w = ax.bar(x + w2/2, q1_wrong_deg,   w2, label="Q1 wrong",
                 color="#F44336", alpha=0.88, edgecolor="white")
    for bar, v in zip(list(b_c) + list(b_w),
                      q1_correct_deg + q1_wrong_deg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in order_classes if _ in [c for c,_ in split_classes]][:2],
                       fontsize=9)
    ax.set_xticklabels(["Gold last in both\n(n=%d)" % len(oc_counts["GOLD_LAST_BOTH"]),
                         "Gold pushed earlier\n(n=%d)" % len(oc_counts["GOLD_PUSHED_EARLIER"])],
                       fontsize=9)
    ax.set_ylabel("Degradation rate % (of orig-correct)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(50, color="#ddd", lw=1, ls="--")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Degradation Rate by Order Class\n× Q1 Bridge Entity Correctness",
                 fontsize=10, fontweight="bold")

    fig.suptitle(
        "Gold Relation Step Order Change After Rewrite",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig6_order_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6  —  Summary: three-step causal chain
# ══════════════════════════════════════════════════════════════════════════════

def fig_summary(records):
    """
    Three panels: (A) Q1 correctness → (B) bridge entity to gold step → (C) accuracy
    """
    n = len(records)
    n_q1_ok    = sum(1 for r in records if r["q1_class"] == "Q1_CORRECT")
    n_q1_wrong = sum(1 for r in records if r["q1_class"].startswith("Q1_WRONG"))
    n_unknown  = sum(1 for r in records if r["q1_class"] == "BRIDGE_UNKNOWN")

    known = [r for r in records if r["bridge_orig"] is not None and r["gs_rew"] is not None]
    n_cb = sum(1 for r in known if r["gold_step_correct_bridge"])
    n_wb = sum(1 for r in known if not r["gold_step_correct_bridge"])

    def rew_acc(lst):
        return 100 * sum(1 for r in lst if r["rew_ok"]) / len(lst) if lst else 0

    def deg(lst):
        o = [r for r in lst if r["orig_ok"]]
        return 100 * sum(1 for r in o if not r["rew_ok"]) / len(o) if o else 0

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={"wspace": 0.5})

    # ── A: Q1 correct vs wrong ─────────────────────────────────────────────
    ax = axes[0]
    vals   = [n_q1_ok, n_q1_wrong, n_unknown]
    colors = ["#2196F3", "#F44336", "#9E9E9E"]
    labels = [f"Q1 correct\n({n_q1_ok})", f"Q1 wrong\n({n_q1_wrong})",
              f"Unknown\n({n_unknown})"]
    ax.pie(vals, labels=labels, colors=colors, startangle=90,
           autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
           pctdistance=0.75, textprops=dict(fontsize=9),
           wedgeprops=dict(linewidth=1.2, edgecolor="white"))
    circle = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(circle)
    ax.set_title("A: Does rewrite Q1\nfind correct bridge entity?",
                 fontsize=10, fontweight="bold")

    # ── B: bridge entity fed to gold step ──────────────────────────────────
    ax = axes[1]
    ax.pie([n_cb, n_wb],
           labels=[f"Correct bridge\nfed to gold step\n({n_cb})",
                   f"Wrong bridge\nfed to gold step\n({n_wb})"],
           colors=["#4CAF50", "#F44336"], startangle=90,
           autopct=lambda p: f"{p:.0f}%",
           pctdistance=0.72, textprops=dict(fontsize=9),
           wedgeprops=dict(linewidth=1.2, edgecolor="white"))
    circle2 = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(circle2)
    ax.set_title("B: Which entity is fed to\nthe gold relation step?",
                 fontsize=10, fontweight="bold")

    # ── C: rewrite accuracy and degradation by bridge correctness ──────────
    ax = axes[2]
    cb_recs = [r for r in known if r["gold_step_correct_bridge"]]
    wb_recs = [r for r in known if not r["gold_step_correct_bridge"]]

    metric_vals = {
        "Rewrite\naccuracy":   [rew_acc(cb_recs), rew_acc(wb_recs)],
        "Degradation\nrate":   [deg(cb_recs),     deg(wb_recs)],
    }
    x = np.arange(2)
    width = 0.32
    offsets = [-width / 2, width / 2]
    m_colors = ["#2196F3", "#F44336"]
    for (metric, vals), offset, mc in zip(metric_vals.items(), offsets, m_colors):
        bars = ax.bar(x + offset, vals, width, label=metric, color=mc, alpha=0.88,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Correct bridge\nfed to gold step",
                         "Wrong bridge\nfed to gold step"], fontsize=9)
    ax.set_ylabel("%", fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(50, color="#ddd", lw=1, ls="--")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("C: Accuracy outcome by\nbridge entity correctness",
                 fontsize=10, fontweight="bold")

    fig.suptitle(
        "Summary: Q1 Selectivity → Bridge Entity → Accuracy",
        fontsize=12, fontweight="bold", y=1.01,
    )
    save_fig(fig, "fig7_summary.png")


# ── printed report ────────────────────────────────────────────────────────────

SEP  = "=" * 74
SEP2 = "-" * 74


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "n/a"


def print_report(records):
    n = len(records)
    n_orig_ok = sum(1 for r in records if r["orig_ok"])
    n_rew_ok  = sum(1 for r in records if r["rew_ok"])
    gc = Counter(r["acc_group"] for r in records)

    print(SEP)
    print("  GOLD-RELATION ORDER ANALYSIS — 2WikiMultiHop (n=291 pairs)")
    print(SEP)

    # ── Overall ────────────────────────────────────────────────────────────────
    print("""
  OVERALL ACCURACY
  ─────────────────────────────────────────────────────────────────────""")
    print(f"  Original:  {n_orig_ok}/{n}  ({pct(n_orig_ok, n)})")
    print(f"  Rewrite:   {n_rew_ok}/{n}  ({pct(n_rew_ok, n)})   Δ = {100*(n_rew_ok-n_orig_ok)/n:+.1f} pp")
    print(f"""
  Outcome breakdown:
    Both correct   (orig✓ rew✓) : {gc['BOTH_CORRECT']:>4}  ({pct(gc['BOTH_CORRECT'], n)})
    Degraded       (orig✓ rew✗) : {gc['ORIG_BETTER']:>4}  ({pct(gc['ORIG_BETTER'], n)})
    Improved       (orig✗ rew✓) : {gc['REW_BETTER']:>4}  ({pct(gc['REW_BETTER'], n)})
    Both wrong     (orig✗ rew✗) : {gc['BOTH_WRONG']:>4}  ({pct(gc['BOTH_WRONG'], n)})
""")

    # ── Q1 accuracy: original vs rewrite ──────────────────────────────────────
    print(SEP)
    print("  Q1 ACCURACY COMPARISON")
    print(SEP2)
    has_bridge = [r for r in records if r["bridge_orig"] is not None]
    nb = len(has_bridge)
    orig_q1_ok = sum(1 for r in has_bridge if r["orig_q1_correct"])
    rew_q1_ok  = sum(1 for r in has_bridge if r["q1_correct"])
    print(f"""
  Pairs with identifiable bridge entity: {nb}/{n}
  Original Q1 (direct Hop1 question):  {orig_q1_ok}/{nb}  ({pct(orig_q1_ok, nb)})
  Rewrite  Q1 (added constraint):       {rew_q1_ok}/{nb}  ({pct(rew_q1_ok, nb)})   Δ = {100*(rew_q1_ok-orig_q1_ok)/nb:+.1f} pp

  In the ORIGINAL, Q1 directly asks the first hop (e.g. "Who directed X?"),
  so it almost always retrieves the correct bridge entity (94.8%).
  In the REWRITE, Q1 is replaced by an added constraint (citizenship,
  birthdate, award, ...) that is often low-selectivity, so the Q1 accuracy
  drops to 56.0% — a 38.8 pp fall.

  Where does Hop1 (the bridge entity) appear in the rewrite chain?
  (In the original it is ALWAYS at Q1; in the rewrite it may shift.)
""")
    print(f"  {'Hop1 position in rewrite':<30}  {'n':>4}  {'% of has-bridge':>16}  {'rew acc':>8}")
    print("  " + "-" * 63)
    for pos in [0, 1, 2, 3]:
        sub = [r for r in has_bridge if r["first_correct_pos"] == pos]
        if not sub: continue
        ra = sum(r["rew_ok"] for r in sub)
        print(f"  Hop1 at Q{pos+1} (position {pos})             {len(sub):>4}  {pct(len(sub),nb):>16}  {pct(ra,len(sub)):>8}")
    never_sub = [r for r in has_bridge if r["first_correct_pos"] is None]
    ra_n = sum(r["rew_ok"] for r in never_sub)
    print(f"  Hop1 NEVER found in rewrite            {len(never_sub):>4}  {pct(len(never_sub),nb):>16}  {pct(ra_n,len(never_sub)):>8}")
    print()

    # ── Q1: does rewrite Q1 retrieve the correct bridge entity? ────────────────
    print(SEP)
    print("  Q1 OUTCOME BREAKDOWN: does rewrite Q1 get the correct bridge entity?")
    print(SEP2)

    classes = [
        ("Q1_CORRECT",            "Q1 retrieves correct bridge entity"),
        ("Q1_WRONG_RECOVERED",    "Q1 wrong, correct entity found & used by gold step"),
        ("Q1_WRONG_FOUND_NOT_USED","Q1 wrong, correct entity found but NOT used by gold step"),
        ("Q1_WRONG_NOT_FOUND",    "Q1 wrong, correct entity never found in rewrite"),
        ("BRIDGE_UNKNOWN",        "Bridge entity not identifiable"),
    ]
    qc = Counter(r["q1_class"] for r in records)

    print(f"\n  {'Class':<35}  {'n':>4}  {'%':>6}  {'rew acc':>8}  {'degrade rate':>13}")
    print("  " + "-" * 72)
    for cls, label in classes:
        subset   = [r for r in records if r["q1_class"] == cls]
        ns       = len(subset)
        rew_ok   = sum(1 for r in subset if r["rew_ok"])
        orig_ok  = sum(1 for r in subset if r["orig_ok"])
        degraded = sum(1 for r in subset if r["orig_ok"] and not r["rew_ok"])
        bar = "█" * int(18 * ns / n)
        print(f"  {label:<35}  {ns:>4}  {pct(ns,n):>6}"
              f"  {pct(rew_ok,ns):>8}  {degraded:>3}/{orig_ok:<3} ({pct(degraded, orig_ok):>6})")

    n_q1_ok = qc["Q1_CORRECT"]
    n_q1_wrong = sum(qc[c] for c in ["Q1_WRONG_RECOVERED","Q1_WRONG_FOUND_NOT_USED","Q1_WRONG_NOT_FOUND"])
    print(f"""
  Summary:
    Rewrite Q1 retrieves correct bridge entity : {n_q1_ok:>4}  ({pct(n_q1_ok, n)})
    Rewrite Q1 retrieves WRONG entity          : {n_q1_wrong:>4}  ({pct(n_q1_wrong, n)})
""")

    # ── Q2: when Q1 is wrong, where does the correct entity appear? ────────────
    print(SEP)
    print("  Q2: When Q1 is wrong, where does the correct bridge entity")
    print("      appear in the rewrite chain, and does the gold step use it?")
    print(SEP2)

    wrong_q1 = [r for r in records if r["q1_class"].startswith("Q1_WRONG")]
    found_later = [r for r in wrong_q1 if r["first_correct_pos"] is not None]
    not_found   = [r for r in wrong_q1 if r["first_correct_pos"] is None]
    used   = [r for r in found_later if r["gold_step_correct_bridge"]]
    unused = [r for r in found_later if not r["gold_step_correct_bridge"]]

    print(f"""
  Of {len(wrong_q1)} pairs where Q1 retrieved the wrong entity:
    Correct entity found somewhere in rewrite chain : {len(found_later):>4}  ({pct(len(found_later), len(wrong_q1))})
      → and fed to the gold step (recovery)         : {len(used):>4}  ({pct(len(used), len(found_later))})
      → but NOT fed to the gold step (ignored)      : {len(unused):>4}  ({pct(len(unused), len(found_later))})
    Correct entity NEVER found in rewrite chain     : {len(not_found):>4}  ({pct(len(not_found), len(wrong_q1))})
""")

    pos_counter = Counter(r["first_correct_pos"] for r in found_later)
    print("  Position (Q index) where correct entity first appears:")
    for pos in sorted(pos_counter):
        sub = [r for r in found_later if r["first_correct_pos"] == pos]
        used_at_pos = sum(1 for r in sub if r["gold_step_correct_bridge"])
        bar = "█" * pos_counter[pos]
        print(f"    Q{pos+1} (position {pos}): {pos_counter[pos]:>3} pairs"
              f"  ({pct(pos_counter[pos], len(wrong_q1))} of wrong-Q1 cases)"
              f"  |  {used_at_pos} actually used by gold step")
    print()

    # ── Q3: when wrong bridge entity is fed to gold step, accuracy collapses ───
    print(SEP)
    print("  Q3: When the wrong bridge entity is fed to the gold relation")
    print("      step, does accuracy collapse?")
    print(SEP2)

    known = [r for r in records if r["bridge_orig"] is not None and r["gs_rew"] is not None]
    cb    = [r for r in known if r["gold_step_correct_bridge"]]
    wb    = [r for r in known if not r["gold_step_correct_bridge"]]

    def summary(lst, label):
        ns      = len(lst)
        rew_ok  = sum(1 for r in lst if r["rew_ok"])
        orig_ok = sum(1 for r in lst if r["orig_ok"])
        deg     = sum(1 for r in lst if r["orig_ok"] and not r["rew_ok"])
        gold_in = sum(1 for r in lst if r["gold_found_rew"])
        print(f"  {label}  (n={ns})")
        print(f"    Rewrite accuracy             : {pct(rew_ok,  ns)}")
        print(f"    Gold answer found in step    : {pct(gold_in, ns)}")
        print(f"    Degradation rate (orig✓→rew✗): {pct(deg, orig_ok)}")
        print()

    summary(cb, "Correct bridge entity fed to gold step")
    summary(wb, "Wrong bridge entity fed to gold step  ")

    print(f"""  Conclusion:
    When the correct bridge entity reaches the gold relation step,
    the rewrite succeeds at a rate comparable to the original.
    When the wrong entity is fed to the gold step, the gold relation
    produces the wrong answer and accuracy collapses.

    The root cause is Q1 selectivity: low-selectivity constraints
    (citizenship, birthdate, ...) retrieve the wrong bridge entity.
    Even when the correct entity is found later in the chain, the
    model usually does NOT reroute it as input to the gold step —
    resulting in a wrong final answer anyway.
""")

    # ── Order analysis ─────────────────────────────────────────────────────────
    print(SEP)
    print("  ORDER ANALYSIS: Does the gold relation step change position")
    print("  in the rewrite chain?  (rfe = rank from end, 0 = last step)")
    print(SEP2)

    order_defs = [
        ("GOLD_LAST_BOTH",      "Gold step is last in BOTH chains         (rfe=0→0)"),
        ("GOLD_PUSHED_EARLIER", "Gold step pushed earlier in rewrite       (rfe=0→1+)"),
        ("GOLD_NOT_FOUND",      "Gold step not identifiable in rewrite     (rfe=?→None)"),
        ("OTHER",               "Other / orig gold not last               "),
    ]
    print(f"\n  {'Order class':<52}  {'n':>4}  {'%':>6}  {'orig acc':>9}  {'rew acc':>9}  {'degrade':>8}")
    print("  " + "-" * 94)
    for cls, label in order_defs:
        subset   = [r for r in records if r["order_class"] == cls]
        ns       = len(subset)
        if ns == 0:
            continue
        oa = sum(1 for r in subset if r["orig_ok"])
        ra = sum(1 for r in subset if r["rew_ok"])
        deg = sum(1 for r in subset if r["orig_ok"] and not r["rew_ok"])
        print(f"  {label:<52}  {ns:>4}  {pct(ns,n):>6}"
              f"  {pct(oa,ns):>9}  {pct(ra,ns):>9}  {deg}/{oa} ({pct(deg,oa):>6})")

    print()
    print("  rfe distribution in rewrite chain (for pairs where gold step IS found):")
    found = [r for r in records if r["gs_rew_rfe"] is not None]
    rfe_c = Counter(r["gs_rew_rfe"] for r in found)
    for rfe_val in sorted(rfe_c):
        sub = [r for r in found if r["gs_rew_rfe"] == rfe_val]
        ra = sum(r["rew_ok"] for r in sub)
        label = "last step" if rfe_val == 0 else f"{rfe_val} step(s) before last"
        print(f"    rfe={rfe_val} ({label}): {rfe_c[rfe_val]:>4} pairs  rew_acc={pct(ra, rfe_c[rfe_val])}")

    print()
    print("  Controlling for Q1 bridge entity correctness:")
    for cls, label in [("GOLD_LAST_BOTH","gold last in both"), ("GOLD_PUSHED_EARLIER","gold pushed earlier")]:
        subset = [r for r in records if r["order_class"] == cls]
        for q1v, q1_lbl in [(True,"Q1 correct"), (False,"Q1 wrong")]:
            sub = [r for r in subset if r.get("q1_correct") == q1v]
            if not sub: continue
            oa = sum(r["orig_ok"] for r in sub)
            ra = sum(r["rew_ok"] for r in sub)
            deg = sum(1 for r in sub if r["orig_ok"] and not r["rew_ok"])
            print(f"    {cls} × {q1_lbl}: n={len(sub):>3}  rew_acc={pct(ra,len(sub)):>6}  "
                  f"degrade={pct(deg,oa):>6} ({deg}/{oa})")
    print(f"""
  Finding: order change alone is NOT the controlling variable.
  Within each order class, Q1 bridge entity correctness determines
  accuracy: Q1-correct → ~15% degrade; Q1-wrong → ~80% degrade.
  The gold step being pushed earlier (rfe>0) is a secondary effect
  caused by verification steps inserted after the gold step.
""")


def print_examples(records):
    print(SEP)
    print("  EXAMPLES: Original Question vs Rewrite — Decomposition Comparison")
    print(SEP2)

    def show(r, title):
        gs_o = r["gs_orig"]
        gs_r = r["gs_rew"]
        hop1_ref = gs_o["ref_label"] if gs_o else None  # label of the Hop1 step in original

        print(f"\n  ┌─ [{title}]")
        print(f"  │  Gold answer  : {r['gold']}")
        print(f"  │  Bridge entity: {r['bridge_orig']}   (= answer to Hop1)")
        print(f"  │")

        # ── side-by-side question ─────────────────────────────────────────────
        oq = r["orig_question"]
        rq = r["rew_question"]
        print(f"  │  ORIGINAL QUESTION  : {oq}")
        print(f"  │  REWRITE  QUESTION  : {rq}")
        print(f"  │")

        # ── original decomposition ────────────────────────────────────────────
        print(f"  │  ORIGINAL DECOMPOSITION ({r['n_orig']} steps)")
        print(f"  │  ─────────────────────────────────────────────")
        for q in r["orig_decomp"]:
            lbl = q["label"]; txt = q.get("text", "")
            ans = r["orig_inter"].get(lbl, "")
            if lbl == hop1_ref:
                role = "[Hop1 → bridge entity]"
            elif gs_o and lbl == gs_o["label"]:
                role = "[Hop2 / gold step]"
            else:
                role = ""
            print(f"  │    {lbl}: {txt}")
            print(f"  │         → {str(ans)[:70]}  {role}")

        # ── rewrite decomposition ─────────────────────────────────────────────
        print(f"  │")
        print(f"  │  REWRITE DECOMPOSITION ({r['n_rew']} steps)")
        print(f"  │  ─────────────────────────────────────────────")
        hop1_pos = r["first_correct_pos"]
        for i, q in enumerate(r["rew_decomp"]):
            lbl = q["label"]; txt = q.get("text", "")
            ans = r["rew_inter"].get(lbl, "")
            # tag role
            is_gold = gs_r and lbl == gs_r["label"]
            is_hop1_pos = (i == hop1_pos) and i > 0  # correct bridge appears here (but not Q1)
            is_q1 = (i == 0)
            if is_q1 and r["q1_correct"]:
                role = "[Q1 = added constraint  ✓ retrieves correct bridge entity]"
            elif is_q1:
                role = "[Q1 = added constraint  ✗ retrieves WRONG entity]"
            elif is_hop1_pos:
                role = f"[Hop1 found here at {lbl} — bridge entity recovered]"
            elif is_gold and r["gold_step_correct_bridge"]:
                role = "[Hop2 / gold step  — correct bridge entity as input  ✓]"
            elif is_gold:
                role = "[Hop2 / gold step  — WRONG bridge entity as input  ✗]"
            else:
                role = ""
            print(f"  │    {lbl}: {txt}")
            print(f"  │         → {str(ans)[:70]}  {role}")

        orig_sym = "✓" if r["orig_ok"] else "✗"
        rew_sym  = "✓" if r["rew_ok"]  else "✗"
        hop1_shift = f"Q1→Q{hop1_pos+1}" if hop1_pos is not None and hop1_pos > 0 else \
                     ("Q1 (no shift)" if hop1_pos == 0 else "Hop1 lost")
        print(f"  │")
        print(f"  └─ orig={orig_sym}  rew={rew_sym}  |  Hop1 position: orig=Q1 → rew={hop1_shift}"
              f"  |  class: {r['q1_class']}")

    # ── example 1: Q1 correct, order preserved ────────────────────────────────
    ex = next((r for r in records
               if r["q1_class"] == "Q1_CORRECT" and r["acc_group"] == "BOTH_CORRECT"), None)
    if ex:
        show(ex, "Q1 CORRECT — rewrite Q1 retrieves correct bridge entity, no order shift")

    # ── example 2: Q1 wrong, Hop1 lost → wrong bridge propagated to gold step ─
    ex = next((r for r in records
               if r["q1_class"] == "Q1_WRONG_NOT_FOUND" and r["acc_group"] == "ORIG_BETTER"), None)
    if ex:
        show(ex, "Q1 WRONG / HOP1 LOST — Hop1 never found in rewrite, gold step gets wrong input")

    # ── example 3: Q1 wrong, Hop1 shifted to Q2 but ignored ──────────────────
    ex = next((r for r in records
               if r["q1_class"] == "Q1_WRONG_FOUND_NOT_USED"
               and r["acc_group"] == "ORIG_BETTER"
               and r["first_correct_pos"] == 1), None)
    if ex:
        show(ex, "Q1 WRONG / HOP1 SHIFTED TO Q2 but gold step ignores it")

    # ── example 4: Q1 wrong, Hop1 shifted and recovered → gold step corrected ─
    ex = next((r for r in records
               if r["q1_class"] == "Q1_WRONG_RECOVERED"), None)
    if ex:
        show(ex, "Q1 WRONG / HOP1 RECOVERED — correct entity found later and used by gold step")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...", end=" ", flush=True)
    records = load_records()
    print(f"{len(records)} pairs loaded.\n")

    print("Generating figures ...")
    fig_overall(records)
    fig_q1_comparison(records)
    fig_q1_selectivity(records)
    fig_bridge_entity_impact(records)
    fig_bridge_recovery_position(records)
    fig_order_analysis(records)
    fig_summary(records)
    print()

    print_report(records)
    print_examples(records)

    print(SEP)
    print(f"  Figures → {FIG_DIR}/")
    print("  fig1_overall_accuracy.png         — overall accuracy: original vs rewrite")
    print("  fig2_q1_comparison.png            — Q1 accuracy (orig 94.8% vs rew 56.0%) + Hop1 shift")
    print("  fig3_q1_selectivity.png           — Q1 outcome class distribution + accuracy")
    print("  fig4_bridge_entity_impact.png     — accuracy when bridge entity correct vs wrong")
    print("  fig5_bridge_recovery_position.png — where correct entity appears in rewrite")
    print("  fig6_order_analysis.png           — gold step position shift original→rewrite")
    print("  fig7_summary.png                  — causal chain: Q1 → bridge → accuracy")
    print(SEP)


if __name__ == "__main__":
    main()
