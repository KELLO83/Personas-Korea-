# Phase 5-C: Leakage-Safe Text Embedding Ablation Report

**Status**: `WARNING`

**Recommendation**: 누수 가능성 있음 - 추가 검증 필요

## 검증 요약

- 샘플 크기: 5000 edges

- Train 크기: 3986, Val 크기: 1014

- 고유 취미(텍스트) 수 - Train: 3983, Val: 1013

## 누수 검증 지표

- Train 내 유사 텍스트 쌍: 45

- Train/Val 공통 텍스트 수: 13

- 텍스트 누수율: 0.0152

- TF-IDF 평균 Cosine Similarity: 0.2668

- TF-IDF 최대 Cosine Similarity: 1.0000

