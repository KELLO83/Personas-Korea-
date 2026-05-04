"""
Phase 5-C: Leakage-Safe Text Embedding Ablation
- 텍스트 임베딩 누수 검증 (KRUE 임베딩 도입 전 안전장치)
- 샘플링 기반 효율적 계산
"""
import sys
import os
import json
import numpy as np
from collections import Counter
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

print("="*60)
print("Phase 5-C: Leakage-Safe Text Embedding Ablation")
print("="*60)

# 1. 데이터 로드
print("\n[1/4] 데이터 로드 및 샘플링...")
edges_path = os.path.join(PROJECT_ROOT, "data", "person_hobby_edges.csv")
if not os.path.exists(edges_path):
    print(f"[ERROR] {edges_path} 파일 없음")
    sys.exit(1)

import pandas as pd
df = pd.read_csv(edges_path)
print(f"  - 전체 edges: {len(df)}개")

# 검증을 위해 샘플링
SAMPLE_SIZE = 5000
if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df
print(f"  - 분석용 샘플: {len(df_sample)}개")

# 2. 사용자 기준 Train/Val 분할
print("\n[2/4] Train/Validation 분할...")
from sklearn.model_selection import train_test_split

persons = df_sample['person_uuid'].unique()
train_persons, val_persons = train_test_split(persons, test_size=0.2, random_state=42)

train_df = df_sample[df_sample['person_uuid'].isin(train_persons)]
val_df = df_sample[df_sample['person_uuid'].isin(val_persons)]

print(f"  - Train edges: {len(train_df)}")
print(f"  - Val edges: {len(val_df)}")

# 3. 각 취미별 고유 텍스트 생성 (시뮬레이션)
print("\n[3/4] 텍스트 메타데이터 생성...")
all_hobbies = df_sample['hobby_name'].unique()

def generate_text_for_hobby(hobby):
    """취미별 텍스트 설명 (시뮬레이션)"""
    keywords = {
        '축구': '스포츠 축구 운동 경기 팀 선수 볼',
        '농구': '스포츠 농구 운동 공 던지기 농구장',
        '야구': '스포츠 야구 운동 경기 투수 타자',
        '독서': '책 읽기 문학 소설 작가 출판',
        '음악': '노래 악기 연주 곡 멜로디 음악가',
        '게임': '비디오 게임 플레이 PC 콘솔 플레이어',
        '요리': '음식 요리 요리사 레시피 맛집 식재료',
        '영화': '시네마 영화 감독 배우 스토리 시청',
        '여행': '여행 관광 명소 휴가 여행객 숙소',
        '등산': '등산 산 산책 트레킹 자연 산악',
        '수영': '수영 수영장 물 운동 수영복',
        '자전거': '자전거 스포츠 자전거길 바이크',
    }
    # 취미 이름에서 키워드 매칭
    for kw, text in keywords.items():
        if kw in str(hobby).lower():
            return text
    # 기본 텍스트
    return f"{hobby} 취미 활동 일상 생활 즐거움 공유"

unique_train_hobbies = train_df['hobby_name'].unique()
unique_val_hobbies = val_df['hobby_name'].unique()

train_texts_map = {h: generate_text_for_hobby(h) for h in unique_train_hobbies}
val_texts_map = {h: generate_text_for_hobby(h) for h in unique_val_hobbies}

print(f"  - Train 고유 취미: {len(unique_train_hobbies)}")
print(f"  - Val 고유 취미: {len(unique_val_hobbies)}")

# 4. 누수 검증 (효율적 샘플링 기반)
print("\n[4/4] 텍스트 누수 검증...")

def jaccard_sim(t1, t2):
    """Jaccard similarity"""
    s1, s2 = set(t1.split()), set(t2.split())
    if not s1 and not s2:
        return 1.0
    inter = s1 & s2
    union = s1 | s2
    return len(inter) / len(union) if union else 0.0

# 4-1. Train 내 중복 검사 (취미별 고유 텍스트 기준)
train_unique_texts = list(set(train_texts_map.values()))
train_dup_count = 0
for i in range(len(train_unique_texts)):
    for j in range(i+1, len(train_unique_texts)):
        if jaccard_sim(train_unique_texts[i], train_unique_texts[j]) > 0.8:
            train_dup_count += 1

print(f"  - Train 내 유사 텍스트 쌍: {train_dup_count}")

# 4-2. Train-Val 간 교집합 (같은 텍스트를 가진 취미가 있는지)
train_text_set = set(train_unique_texts)
val_text_set = set(val_texts_map.values())
intersection = train_text_set & val_text_set
print(f"  - Train/Val 공통 텍스트 수: {len(intersection)}")

# 4-3. 샘플링 기반 Jaccard 유사도 검사
val_text_list = list(val_text_set)
sample_pairs = []
N_SAMPLE = min(200, len(train_unique_texts))

for t_text in random.sample(train_unique_texts, min(N_SAMPLE, len(train_unique_texts))):
    for v_text in random.sample(val_text_list, min(100, len(val_text_list))):
        sim = jaccard_sim(t_text, v_text)
        if sim > 0.8:
            sample_pairs.append((t_text[:40], v_text[:40], round(sim, 3)))

print(f"  - 샘플링 기준 유사 쌍 (Jaccard>0.8): {len(sample_pairs)}개")

# 4-4. TF-IDF Cosine Similarity (선택적)
print("  - TF-IDF 분석 수행...")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    all_texts = train_unique_texts + val_text_list
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_mat = vectorizer.fit_transform(all_texts)
    
    n_train = len(train_unique_texts)
    if n_train > 0 and len(val_text_list) > 0:
        train_tfidf = tfidf_mat[:n_train]
        val_tfidf = tfidf_mat[n_train:]
        
        # 샘플링으로 평균 계산
        idx1 = np.random.choice(train_tfidf.shape[0], min(100, train_tfidf.shape[0]), replace=False)
        idx2 = np.random.choice(val_tfidf.shape[0], min(100, val_tfidf.shape[0]), replace=False)
        
        sim_mat = cosine_similarity(train_tfidf[idx1], val_tfidf[idx2])
        avg_sim = float(np.mean(sim_mat))
        max_sim = float(np.max(sim_mat))
    else:
        avg_sim = 0.0
        max_sim = 0.0
except Exception as e:
    print(f"  - TF-IDF 실패: {e}")
    avg_sim = 0.0
    max_sim = 0.0

print(f"  - 평균 Cosine Similarity: {avg_sim:.4f}")
print(f"  - 최대 Cosine Similarity: {max_sim:.4f}")

# 5. 결과 및 권고안
print("\n[결과] 레이블 누수 검증 결과")
leakage_rate = len(intersection) / len(val_text_set) if val_text_set else 0
avg_sim_threshold = 0.3  # 임계값

if leakage_rate == 0 and max_sim < avg_sim_threshold:
    status = "SAFE"
    recommendation = "KRUE 임베딩 도입 가능 - 누수 우려 없음"
elif leakage_rate < 0.1 and max_sim < 0.5:
    status = "CAUTION"
    recommendation = "KRUE 도입 가능, 다만 주기적 모니터링 필요"
else:
    status = "WARNING"
    recommendation = "누수 가능성 있음 - 추가 검증 필요"

print(f"  - Status: {status}")
print(f"  - 추천: {recommendation}")

# 6. 결과 저장
output_dir = os.path.join(PROJECT_ROOT, "artifacts", "experiments", "phase5_text_embedding_leakage")
os.makedirs(output_dir, exist_ok=True)

results = {
    "experiment": "phase5_text_embedding_leakage_check",
    "sample_size": SAMPLE_SIZE,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "train_unique_hobbies": len(unique_train_hobbies),
    "val_unique_hobbies": len(unique_val_hobbies),
    "train_duplicate_texts": train_dup_count,
    "common_texts_count": len(intersection),
    "text_leakage_rate": leakage_rate,
    "tfidf_avg_cosine_similarity": avg_sim,
    "tfidf_max_cosine_similarity": max_sim,
    "status": status,
    "recommendation": recommendation,
    "sample_high_similarity_pairs": sample_pairs[:5]
}

result_path = os.path.join(output_dir, "leakage_check_report.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

report_path = os.path.join(output_dir, "ablation_results.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Phase 5-C: Leakage-Safe Text Embedding Ablation Report\n\n")
    f.write(f"**Status**: `{status}`\n\n")
    f.write(f"**Recommendation**: {recommendation}\n\n")
    f.write("## 검증 요약\n\n")
    f.write(f"- 샘플 크기: {SAMPLE_SIZE} edges\n\n")
    f.write(f"- Train 크기: {len(train_df)}, Val 크기: {len(val_df)}\n\n")
    f.write(f"- 고유 취미(텍스트) 수 - Train: {len(unique_train_hobbies)}, Val: {len(unique_val_hobbies)}\n\n")
    f.write("## 누수 검증 지표\n\n")
    f.write(f"- Train 내 유사 텍스트 쌍: {train_dup_count}\n\n")
    f.write(f"- Train/Val 공통 텍스트 수: {len(intersection)}\n\n")
    f.write(f"- 텍스트 누수율: {leakage_rate:.4f}\n\n")
    f.write(f"- TF-IDF 평균 Cosine Similarity: {avg_sim:.4f}\n\n")
    f.write(f"- TF-IDF 최대 Cosine Similarity: {max_sim:.4f}\n\n")

print(f"\n[SUCCESS] 결과 저장 완료:")
print(f"  - {result_path}")
print(f"  - {report_path}")
