import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# 프로젝트 전반에서 공유할 캐싱 디렉토리 설정
EMBEDDING_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "GNN_Neural_Network", "artifacts", "embeddings_cache")


class KUREEncoder:
    """
    KURE-v1 모델을 싱글톤(또는 전역)으로 관리하여, 
    텍스트(페르소나, 취미) 간의 의미론적 유사도를 효율적으로 계산하는 인코더.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if KUREEncoder._model is None:
            logger.info("[KURE] Loading model (this may take a while on first run)...")
            KUREEncoder._model = SentenceTransformer("nlpai-lab/KURE-v1")
            logger.info("[KURE] Model loaded successfully.")

    @property
    def model(self):
        return KUREEncoder._model

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩 벡터 배열로 변환"""
        return self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    def compute_similarity(self, source_texts: List[str], target_items: List[str]) -> np.ndarray:
        """
        Source(페르소나)와 Target(취미) 간의 코사인 유사도 행렬 계산.
        캐싱을 통해 중복 계산을 방지합니다.
        """
        source_emb = self.encode_texts(source_texts)
        
        # Target 임베딩 캐싱 로직
        cache_path = os.path.join(EMBEDDING_CACHE_DIR, "kure_hobby_cache.npy")
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
        
        target_emb = []
        if os.path.exists(cache_path):
            logger.debug(f"[KURE] Loading cached hobby embeddings from {cache_path}")
            cached_data = np.load(cache_path, allow_pickle=True).item()
            # 캐시에서 찾고, 없는 것만 새로 인코딩
            for item in target_items:
                if item in cached_data:
                    target_emb.append(cached_data[item])
                else:
                    new_emb = self.model.encode([item], convert_to_numpy=True)[0]
                    cached_data[item] = new_emb
                    target_emb.append(new_emb)
            # 업데이트된 캐시 저장
            np.save(cache_path, cached_data)
        else:
            logger.debug(f"[KURE] Creating new cache at {cache_path}")
            cached_data = {}
            for item in target_items:
                emb = self.model.encode([item], convert_to_numpy=True)[0]
                cached_data[item] = emb
                target_emb.append(emb)
            np.save(cache_path, cached_data)

        target_emb = np.array(target_emb)
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(source_emb, target_emb)
        return similarity_matrix


# 전역에서 쉽게 접근하기 위한 싱글톤 인스턴스 생성 함수
def get_kure_encoder() -> KUREEncoder:
    return KUREEncoder()