import pandas as pd
import os
import sys

# 모듈 임포트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_recommender.data import build_domain_tagged_persona_text, PersonContext
from src.embeddings.kure_encoder import get_kure_encoder
from collections import defaultdict

def load_persona_context(csv_path: str) -> dict[str, PersonContext]:
    """
    CSV에서 Persona 데이터를 로드하여 PersonContext 객체 딕셔너리로 반환합니다.
    """
    df = pd.read_csv(csv_path)
    contexts = {}
    for _, row in df.iterrows():
        ctx = PersonContext(
            person_uuid=row['uuid'],
            age=str(row.get('age', '')),
            age_group=row.get('age_group', ''),
            sex=row.get('sex', ''),
            occupation=row.get('occupation', ''),
            district=row.get('district', ''),
            province=row.get('province', ''),
            family_type=row.get('family_type', ''),
            housing_type=row.get('housing_type', ''),
            education_level=row.get('education_level', ''),
            persona_text=row.get('persona', ''),
            professional_text=row.get('professional_persona', ''),
            sports_text=row.get('sports_persona', ''),
            arts_text=row.get('arts_persona', ''),
            travel_text=row.get('travel_persona', ''),
            culinary_text=row.get('culinary_persona', ''),
            family_text=row.get('family_persona', ''),
            skills_text=row.get('skills_and_expertise', ''),
            career_goals=row.get('career_goals_and_ambitions', ''),
            embedding_text=row.get('embedding_text', '')
        )
        contexts[ctx.person_uuid] = ctx
    return contexts

def build_ranker_dataset_with_kure(contexts: dict[str, PersonContext], output_path: str):
    """
    KURE 임베딩 Feature가 추가된 최종 Ranker 학습용 데이터셋을 생성합니다.
    """
    print("[INFO] KURE 인코더 로드 중...")
    encoder = get_kure_encoder()
    
    print("[INFO] 데이터프레임 병합 및 전처리 중...")
    rows = []
    hobby_list = ["독서", "영화 감상", "산책", "골프", "요리", "낚시", "노래방"]  # 예시 후보군
    
    for uuid, ctx in contexts.items():
        # 1. 도메인 태깅 텍스트 생성
        tagged_text = build_domain_tagged_persona_text(ctx)
        
        # 2. KURE 유사도 계산 (임시 후보군에 대해)
        similarity_scores = encoder.compute_similarity([tagged_text], hobby_list)[0]
        
        # 3. 기본 정형 데이터 + 텍스트 유사도 Feature 병합
        row_data = {
            'person_uuid': uuid,
            'age_group': ctx.age_group,
            'sex': ctx.sex,
            'occupation': ctx.occupation,
            'district': ctx.district,
            # ... 기타 정형 데이터 (실제 구현 시에는 원본 엣지 데이터 조인 필요)
        }
        
        # KURE 유사도 Feature 추가 (컬럼명: kure_sim_{hobby})
        for hobby, score in zip(hobby_list, similarity_scores):
            safe_hobby = hobby.replace(' ', '_')
            row_data[f'kure_sim_{safe_hobby}'] = round(float(score), 4)
        
        rows.append(row_data)
    
    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] KURE 피처가 추가된 데이터셋 저장 완료: {output_path}")
    print(f"[INFO] 생성된 칼럼: {final_df.columns.tolist()}")
    print(f"[INFO] 데이터 형태: {final_df.shape}")

if __name__ == "__main__":
    # 경로 설정 (예시)
    persona_csv = "../../data/raw/nemotron-personas-korea/train_sample.csv" # 10K 서브셋 가정
    output_csv = "artifacts/experiments/phase5_c_text_embedding/ranker_dataset_with_kure.csv"
    
    # 데이터셋 로드
    if not os.path.exists(persona_csv):
        print(f"[ERROR] 샘플 데이터를 찾을 수 없습니다: {persona_csv}")
        print("[INFO] 실제 파이프라인에서는 Neo4j 또는 전처리된 엣지 데이터를 사용합니다.")
        # 데모용 더미 데이터 생성 로직이 필요할 수 있으나, 현재 구조상 스킵하고 파이프라인 구조만 보여줌
    else:
        contexts = load_persona_context(persona_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        build_ranker_dataset_with_kure(contexts, output_csv)