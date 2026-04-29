import json

# Current (canonical) report
with open("GNN_Neural_Network/artifacts/vocabulary_report.json", "r", encoding="utf-8") as f:
    current = json.load(f)

# Previous raw baseline values (from progress context - before any canonicalization review)
raw_baseline = {
    "retained_edges": 36821,
    "retained_persons": 9797,
    "retained_hobbies": 1349,
    "canonical_singleton_ratio": 0.8511,
}

print("=== RETAINED COVERAGE COMPARISON ===")
print(f"{'':25s} {'Raw baseline':>14s}  {'Canonical now':>14s}  {'Delta':>10s}")
re = current["retained_edges"]
rp = current["retained_persons"]
rh = current["retained_hobbies"]
sr = current["canonical_singleton_ratio"]
print(f"  retained_edges:        {raw_baseline['retained_edges']:>14,}  {re:>14,}  {re - raw_baseline['retained_edges']:>+10,}")
print(f"  retained_persons:      {raw_baseline['retained_persons']:>14,}  {rp:>14,}  {rp - raw_baseline['retained_persons']:>+10,}")
print(f"  retained_hobbies:      {raw_baseline['retained_hobbies']:>14,}  {rh:>14,}  {rh - raw_baseline['retained_hobbies']:>+10,}")
print(f"  singleton_ratio:       {raw_baseline['canonical_singleton_ratio']:>14.4f}  {sr:>14.4f}  {sr - raw_baseline['canonical_singleton_ratio']:>+10.4f}")
print()
print("=== GATE CRITERIA ===")
print(f"  retained_edges increased:   {re >= raw_baseline['retained_edges']}")
print(f"  retained_persons increased: {rp >= raw_baseline['retained_persons']}")
print(f"  retained_hobbies decreased: {rh < raw_baseline['retained_hobbies']}")
print(f"  singleton_ratio decreased:  {sr < 0.8340}")
print()

# Check canonical_hobby_examples for suspicious large clusters
with open("GNN_Neural_Network/artifacts/canonical_hobby_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

print(f"=== CANONICAL HOBBY EXAMPLES: {len(examples)} hobbies ===")
print()

# Sort by example count descending to spot over-merges
sorted_hobbies = sorted(examples.items(), key=lambda x: -len(x[1]))

print("=== TOP 30 BY EXAMPLE COUNT (check for over-merging) ===")
for canonical, raw_list in sorted_hobbies[:30]:
    count = len(raw_list)
    print(f"  {canonical:35s} ({count:4d} raw variants)")
    for s in raw_list[:3]:
        print(f"      - {s}")

print()
print("=== ALL HOBBIES WITH 1 EXAMPLE ONLY (singletons that survived min_degree) ===")
one_example = [(c, r) for c, r in sorted_hobbies if len(r) == 1]
print(f"  Count: {len(one_example)}")
for c, r in one_example[:10]:
    print(f"  {c}: {r[0]}")
