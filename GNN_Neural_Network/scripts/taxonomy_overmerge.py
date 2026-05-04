"""
Phase 5-B: Taxonomy Over-merge Tracking
- 희귀 취미 강제 노출(fallback) 시 카테고리 편중도 분석
- 단일 취미가 너무 많은 상위 카테고리에 노출되는지 확인
"""
import sys
import os
import json
from collections import Counter, defaultdict
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

print("="*60)
print("Phase 5-B: Taxonomy Over-merge Tracking")
print("="*60)

# 1. edges 데이터 로드
print("\n[1/4] edges 데이터 로드...")
edges_path = os.path.join(PROJECT_ROOT, "data", "person_hobby_edges.csv")
if not os.path.exists(edges_path):
    print(f"[ERROR] {edges_path} 파일 없음")
    sys.exit(1)

import pandas as pd
df = pd.read_csv(edges_path)
print(f"  - 총 edges: {len(df)}개")
print(f"  - 사용자 수: {df['person_uuid'].nunique()}")
print(f"  - 취미 수: {df['hobby_name'].nunique()}")

# 2. Taxonomy 로드 (카테고리 매핑)
print("\n[2/4] Taxonomy 설정 로드...")
taxonomy_path = os.path.join(PROJECT_ROOT, "configs", "taxonomy.yaml")
if not os.path.exists(taxonomy_path):
    print(f"  [WARNING] {taxonomy_path} 없음 - 간이 버전 사용")
    # 간이 taxonomy: 취미 이름 기반 카테고리 할당
    # 실제 구현에서는 YAML 파일 참조
    category_map = {}
    for hobby in df['hobby_name'].unique():
        if any(kw in hobby.lower() for kw in ['운동', '스포츠', '축구', '농구', '테니스']):
            category_map[hobby] = '스포츠/레저'
        elif any(kw in hobby.lower() for kw in ['음악', '악기', '노래', '기타']):
            category_map[hobby] = '음악/예술'
        elif any(kw in hobby.lower() for kw in ['독서', '책', '문학']):
            category_map[hobby] = '독서/학습'
        elif any(kw in hobby.lower() for kw in ['게임', 'e스포츠', '게임']):
            category_map[hobby] = '게임/IT'
        elif any(kw in hobby.lower() for kw in ['요리', '음식', '맛집']):
            category_map[hobby] = '음식/요리'
        else:
            category_map[hobby] = '기타/다양'
else:
    import yaml
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        tax_conf = yaml.safe_load(f)
    category_map = tax_conf.get('hobby_categories', {})

# 카테고리 할당
hobby_categories = {}
for hobby in df['hobby_name'].unique():
    if hobby in category_map:
        hobby_categories[hobby] = category_map[hobby]
    else:
        hobby_categories[hobby] = '미분류'

categories = list(set(hobby_categories.values()))
print(f"  - 카테고리 수: {len(categories)}")
print(f"  - 카테고리 목록: {categories}")

# 3. 사용자별 카테고리 분포 분석
print("\n[3/4] 사용자별 카테고리 분포 분석...")
person_cat_dist = defaultdict(Counter)
for _, row in df.iterrows():
    person = row['person_uuid']
    hobby = row['hobby_name']
    cat = hobby_categories.get(hobby, '미분류')
    person_cat_dist[person][cat] += 1

# 카테고리 집중도 계산
overmerge_scores = []
for person, cat_counts in person_cat_dist.items():
    total = sum(cat_counts.values())
    if total == 0:
        continue
    # 상위 카테고리 비중
    top_cat_count = cat_counts.most_common(1)[0][1] if cat_counts else 0
    top_ratio = top_cat_count / total
    overmerge_scores.append(top_ratio)

avg_overmerge = sum(overmerge_scores) / len(overmerge_scores) if overmerge_scores else 0
print(f"  - 사용자별 상위 카테고리 집중도 평균: {avg_overmerge:.4f}")

# 희귀 취미 분석 (하위 20% 빈도 = 희귀)
print("\n[4/4] 희귀 취미 over-merge 분석...")
hobby_freq = Counter(df['hobby_name'])
freq_values = list(hobby_freq.values())
freq_threshold = sorted(freq_values)[int(len(freq_values) * 0.2)]  # 하위 20%
rare_hobbies = [h for h, f in hobby_freq.items() if f <= freq_threshold]
print(f"  - 희귀 취미 기준(하위 20%): {freq_threshold}회 이하")
print(f"  - 희귀 취미 수: {len(rare_hobbies)}개")

# 희귀 취미의 카테고리 분포
rare_cat_dist = Counter()
for h in rare_hobbies:
    cat = hobby_categories.get(h, '미분류')
    rare_cat_dist[cat] += 1

# 희귀 취미 중 과집중된 카테고리
print(f"  - 희귀 취미 카테고리 분포:")
for cat, count in rare_cat_dist.most_common():
    pct = count / len(rare_hobbies) * 100
    print(f"    * {cat}: {count}개 ({pct:.1f}%)")

# over-merge 경고: 특정 카테고리의 희귀 취미 비중이 50% 초과
overmerge_categories = []
for cat, count in rare_cat_dist.items():
    if count / len(rare_hobbies) > 0.5:
        overmerge_categories.append(cat)
        print(f"  [OVER-MERGE ALERT] '{cat}' 카테고리 희귀 취미 편중: {count/len(rare_hobbies)*100:.1f}%")

# 결과 저장
output_dir = os.path.join(PROJECT_ROOT, "artifacts", "experiments", "phase5_taxonomy_overmerge")
os.makedirs(output_dir, exist_ok=True)

result = {
    "experiment": "phase5_taxonomy_overmerge",
    "total_hobbies": len(hobby_categories),
    "total_categories": len(categories),
    "avg_user_top_category_ratio": float(avg_overmerge),
    "rare_hobby_threshold": freq_threshold,
    "rare_hobby_count": len(rare_hobbies),
    "rare_category_distribution": {k: v for k, v in rare_cat_dist.items()},
    "overmerge_warning_categories": overmerge_categories,
    "user_overmerge_scores": {
        "mean": float(np.mean(overmerge_scores)),
        "std": float(np.std(overmerge_scores)),
        "min": float(np.min(overmerge_scores)),
        "max": float(np.max(overmerge_scores))
    }
}

import numpy as np
result["user_overmerge_scores"]["mean"] = float(np.mean(overmerge_scores))
result["user_overmerge_scores"]["std"] = float(np.std(overmerge_scores))
result["user_overmerge_scores"]["min"] = float(np.min(overmerge_scores))
result["user_overmerge_scores"]["max"] = float(np.max(overmerge_scores))

result_path = os.path.join(output_dir, "overmerge_report.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

# 텍스트 리포트
text_path = os.path.join(output_dir, "overmerge_report.md")
with open(text_path, 'w', encoding='utf-8') as f:
    f.write("# Phase 5-B: Taxonomy Over-merge Tracking Report\n\n")
    f.write(f"**총 취미 수**: {len(hobby_categories)}\n\n")
    f.write(f"**카테고리 수**: {len(categories)}\n\n")
    f.write(f"**사용자별 상위 카테고리 집중도(평균)**: {avg_overmerge:.4f}\n\n")
    f.write(f"**희귀 취미 기준**: {freq_threshold}회 이하 ({len(rare_hobbies)}개)\n\n")
    f.write("## 희귀 취미 카테고리 분포\n\n")
    for cat, count in rare_cat_dist.most_common():
        f.write(f"- {cat}: {count}개 ({count/len(rare_hobbies)*100:.1f}%)\n")
    f.write("\n")
    if overmerge_categories:
        f.write("## ⚠️ Over-merge 경고\n\n")
        for cat in overmerge_categories:
            f.write(f"- `{cat}` 카테고리에 희귀 취미가 과도하게 편중됨\n")

print(f"\n[SUCCESS] 결과 저장:")
print(f"  - JSON: {result_path}")
print(f"  - 마크다운: {text_path}")
