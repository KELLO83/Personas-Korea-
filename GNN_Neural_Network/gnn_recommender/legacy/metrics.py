"""Legacy Metrics Module (Import Compatibility)
legacy/train.py가 요구하는 메트릭 함수들을 메인 모듈로부터 연결하는 브릿지 모듈.
"""

# 메인 metrics.py로부터 모든 함수들 가져오기
from ..metrics import (
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    intra_list_diversity_at_k,
    # summarize_ranking_metrics가 메인에 있다면 임포트, 없으면 스텁 생성
)

try:
    from ..metrics import summarize_ranking_metrics
except ImportError:
    def summarize_ranking_metrics(*args, **kwargs):
        """Legacy train.py 호환용 더미 함수"""
        print("[metrics] summarize_ranking_metrics stub executed.")
        return {}

__all__ = [
    "recall_at_k", "hit_rate_at_k", "ndcg_at_k",
    "intra_list_diversity_at_k", "summarize_ranking_metrics"
]
