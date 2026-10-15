import json
import random
from pathlib import Path

GEN_DIR = Path("generations")
OUT_DIR = Path("comparisons")
SEED = 42


def stack_them_up(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    byid = {}
    for row in rows:
        byid[row["id"]] = row
    return byid


def who_is_in_both(left, right):
    return sorted(set(left) & set(right))


def count_the_strays(left, right):
    return len(set(left) ^ set(right))


def toss_for_slots(coin, first_name, second_name):
    if coin.random() < 0.5:
        return first_name, second_name
    return second_name, first_name


def build_one_row(one_id, a_name, b_name, source_by_name):
    from_a = source_by_name[a_name][one_id]
    from_b = source_by_name[b_name][one_id]

    return {
        "id": one_id,
        "instruction": from_a["instruction"],
        "response_a": from_a["output"],
        "response_b": from_b["output"],
        "_hidden": {"a": a_name, "b": b_name},
    }


def tally_slot_a(pairs, first_name):
    seen = 0
    for row in pairs:
        if row["_hidden"]["a"] == first_name:
            seen = seen + 1
    return seen


def pair_them_off(first, first_name, second, second_name, where):
    left = stack_them_up(GEN_DIR / first)
    right = stack_them_up(GEN_DIR / second)

    source_by_name = {first_name: left, second_name: right}
    shared = who_is_in_both(left, right)
    missed = count_the_strays(left, right)

    coin = random.Random(SEED)
    plan = []

    for one_id in shared:
        a_name, b_name = toss_for_slots(coin, first_name, second_name)
        plan.append((one_id, a_name, b_name))

    pairs = []
    for one_id, a_name, b_name in plan:
        pairs.append(build_one_row(one_id, a_name, b_name, source_by_name))

    where.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for row in pairs:
        lines.append(json.dumps(row, ensure_ascii=False))
    where.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(where.name, "-", len(pairs), "pairs,", missed, "ids skipped")
    print("   ", first_name, "sits in slot A", tally_slot_a(pairs, first_name), "times")
    return pairs


if __name__ == "__main__":
    line_ups = [
        ("sft_alpaca_eval.json", "sft", "dpo_uf_alpaca_eval.json", "dpo_uf", "uf_dpo_vs_sft.jsonl"),
        ("sft_hh_eval.json", "sft", "dpo_hh_eval.json", "dpo_hh", "hh_dpo_vs_sft.jsonl"),
    ]

    for one_file, one_name, other_file, other_name, saved_as in line_ups:
        pair_them_off(one_file, one_name, other_file, other_name, OUT_DIR / saved_as)