import argparse
import logging
import polars as pl
from typing import Optional
import torch
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.data.sampling import normalize_age_group_tokens, sample_age_groups
from src.data.parallel_preprocessor import chunk_dataframe, execute_parallel
from src.graph.loader import GraphLoader


logger = logging.getLogger(__name__)


DEFAULT_TARGET_AGE_GROUPS = "10,20,30"
DEFAULT_TARGET_PERSONS = 10_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset loading to the specified number of rows before age filtering. Omit to load full dataset.",
    )
    parser.add_argument(
        "--age-groups",
        default=DEFAULT_TARGET_AGE_GROUPS,
        help="Comma-separated age groups to keep. Defaults to '10,20,30'.",
    )
    parser.add_argument(
        "--target-persons",
        type=int,
        default=DEFAULT_TARGET_PERSONS,
        help="Maximum total personas after age filtering. Defaults to 10000.",
    )
    parser.add_argument(
        "--age-sample-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic age-group sampling.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Neo4j graph nodes and relationships before loading.",
    )
    # 🔥 병렬 처리 워커 수 (기본: 사용 가능한 CPU 수)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers for preprocessing. Default: CPU count.",
    )
    return parser.parse_args()


def _get_existing_uuids(loader: GraphLoader) -> set[str]:
    """Neo4j에 이미 존재하는 모든 Person UUID를 조회하여 반환"""
    query = "MATCH (p:Person) RETURN p.uuid AS uuid"
    with loader.driver.session(database=loader.database) as session:
        result = session.run(query)
        return {record["uuid"] for record in result if record["uuid"]}


def _filter_target_age_groups(
    df: pl.DataFrame,
    age_groups: list[str],
    target_persons: int | None,
    random_seed: int,
) -> pl.DataFrame:
    if not age_groups:
        logger.info("age-groups is empty. 건너뜁니다.")
        return df

    if target_persons is None:
        logger.info("Filtering age groups=%s without row limit", age_groups)
        return df.filter(pl.col("age_group").is_in(age_groups))

    if target_persons <= 0:
        logger.warning("target-persons=%d is not valid. Skipping sampling.", target_persons)
        return df.filter(pl.lit(False))

    sampled = sample_age_groups(df, age_groups=age_groups, max_rows=target_persons, random_seed=random_seed)
    logger.info("Age filtering complete: requested=%d rows from groups=%s. Kept=%d rows", target_persons, age_groups, sampled.height)
    return sampled


def _load_to_neo4j(df: pl.DataFrame, reset: bool, batch_size: int) -> None:
    """데이터프레임을 Neo4j에 적재합니다. reset=True인 경우 적재 직전에 기존 데이터를 삭제합니다."""
    loader = GraphLoader()
    try:
        loader.create_schema()
        
        if reset:
            logger.info("🗑️  기존 그래프 데이터를 삭제합니다 (적재 직전 초기화)...")
            deleted = loader.clear_graph()
            logger.info(f"삭제 완료: {deleted:,}개의 노드가 제거되었습니다.")
        
        logger.info("Neo4j 신규 데이터 적재를 시작합니다 (Batch Size: %d)...", batch_size)
        loaded_count = loader.load_personas(df, batch_size=batch_size)
        logger.info(f"✅ Neo4j 적재 완료! 총 {loaded_count:,}개의 페르소나가 저장되었습니다.")
    finally:
        loader.close()


import re

def _batch_create_hobby_relationships(loader, reset: bool, batch_size: int = 5000) -> None:
    """
    Python에서 hobbies_text를 파싱하여 Cypher UNWIND 배치 쿼리로 관계 생성
    - APoC 미설치 환경 대응 (Python side 분리)
    - 대량의 관계 생성을 단일 트랜잭션으로 최적화
    """
    # Step 1: reset=True인 경우 기존 데이터 초기화
    if reset:
        logger.info("🗑️  기존 취미(Hobby) 노드와 LIKES 관계를 초기화합니다...")
        with loader.driver.session(database=loader.database) as session:
            session.run("MATCH ()-[r:LIKES]->() DELETE r")
            session.run("MATCH (h:Hobby) DETACH DELETE h")
        logger.info("기존 취미 데이터 초기화 완료")

    # Step 2: 대상 Person 데이터 추출 (기존 관계 무관하게 모두 재생성)
    extract_query = """
    MATCH (p:Person)
    WHERE p.hobbies_and_interests IS NOT NULL AND trim(p.hobbies_and_interests) <> ""
    RETURN p.uuid AS uuid, p.hobbies_and_interests AS hobbies
    """
    
    total_processed = 0
    with loader.driver.session(database=loader.database) as session:
        result = session.run(extract_query)
        
        batch_rows = []
        for record in result:
            uuid = record["uuid"]
            hobbies_text = record["hobbies"]
            
            # Python 정규식으로 다중 구분자([;,.]) 처리 및 토큰화
            # ex) "독서, 게임; 배드민턴. 낚시" -> ['독서', '게임', '배드민턴', '낚시']
            hobbies = [
                h.strip() 
                for h in re.split(r'[;,.,]\s*', hobbies_text) 
                if h.strip() and len(h.strip()) > 1
            ]
            
            for hobby_name in hobbies:
                batch_rows.append({"uuid": uuid, "hobby_name": hobby_name})
            
            # 배치 사이즈 도달 시 UNWIND 쿼리 실행
            if len(batch_rows) >= batch_size:
                _execute_hobby_batch(session, batch_rows)
                total_processed += len(batch_rows)
                batch_rows = []
        
        # 남은 데이터 처리
        if batch_rows:
            _execute_hobby_batch(session, batch_rows)
            total_processed += len(batch_rows)
    
    logger.info(f"✅ 총 {total_processed}개의 Person-Hobby 관계 생성 완료")

def _execute_hobby_batch(session, batch_rows):
    """UNWIND을 사용한 배치 관계 생성 쿼리 실행"""
    batch_query = """
    UNWIND $batch AS row
    MATCH (p:Person {uuid: row.uuid})
    MERGE (h:Hobby {name: row.hobby_name})
    MERGE (p)-[:LIKES]->(h)
    """
    session.run(batch_query, batch=batch_rows)

def _create_hobby_relationships(reset: bool) -> None:
    """[유지보수용] Person-Hobby 관계 생성 메인 진입점 (배치 로직 호출)"""
    from src.graph.loader import GraphLoader
    loader = GraphLoader()
    try:
        _batch_create_hobby_relationships(loader, reset=reset, batch_size=5000)
    finally:
        loader.close()


def _export_gnn_person_hobby_edges() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, "GNN_Neural_Network", "scripts", "export_person_hobby_edges.py")
    logger.info("GNN Person-Hobby edge/context CSV export를 시작합니다...")
    subprocess.run([sys.executable, script_path], check=True, cwd=project_root)


def _preprocess_single_chunk(chunk_df: pl.DataFrame, fast_mode: bool) -> pl.DataFrame:
    """
    멀티 프로세싱을 위해 데이터 청크를 pickle화하여 전달하기 편하도록 래핑된 함수.
    """
    return preprocess(chunk_df, fast_mode=fast_mode)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()

    # --- GPU 설정 확인 (병렬 프로세스 간 충돌 방지) ---
    # 멀티 프로세스에서 여러 CUDA 컨텍스트가 동시에 생성되는 것을 방지하기 위해
    # 가용 메모리와 프로세스 수를 제한합니다.
    if torch.cuda.is_available():
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[GPU] 사용 가능한 VRAM: {total_gpu_mem:.1f} GB")
        # VRAM이 적을 경우 워커 수 강제 제한
        if total_gpu_mem < 4 and args.workers > 2:
            logger.warning(f"[GPU] VRAM 부족으로 병렬 워커 수를 2로 제한합니다.")
            args.workers = 2

    # 1. 기존 데이터의 UUID 조회 (--reset=False인 경우에만)
    existing_uuids: set[str] = set()
    if not args.reset:
        loader = GraphLoader()
        logger.info("기존 Neo4j 데이터베이스에서 UUID 목록을 조회합니다...")
        existing_uuids = _get_existing_uuids(loader)
        logger.info(f"이미 존재하는 페르소나 수: {len(existing_uuids):,}명")
        loader.close()

    sample_size = args.sample_size if args.sample_size is not None else None
    
    # 2. LazyFrame으로 변경하여 데이터 로딩 최적화 (빠른 필터링을 위해 scan 사용)
    #    -> loader.load_dataset는 Eager이므로, 여기서 Lazy 전환을 위한 별도 로직 필요
    #    -> 현재는 loader를 그대로 두되, chunk processing을 통해 메모리 문제를 해결함.
    raw_df = load_dataset(sample_size=sample_size)
    
    # 3. 빠른 전처리 (Fast Preprocess)
    logger.info("모수(Population)에 대한 고속 전처리를 시작합니다...")
    df = preprocess(raw_df, fast_mode=True)

    age_groups = normalize_age_group_tokens(args.age_groups)
    if not age_groups:
        age_groups = normalize_age_group_tokens(DEFAULT_TARGET_AGE_GROUPS)

    target_persons = args.target_persons

    logger.info("연령대 필터링 및 샘플링을 시작합니다... (대상: %s, 목표: %s명)", age_groups, target_persons)
    df = _filter_target_age_groups(df, age_groups=age_groups, target_persons=target_persons, random_seed=args.age_sample_seed)
    logger.info("샘플링 완료. 최종 데이터 크기: %d행", df.height)

    # 4. 중복 데이터 필터링 (기존 UUID와 겹치는 행 제거)
    if existing_uuids:
        before_rows = df.height
        df = df.filter(~pl.col("uuid").is_in(list(existing_uuids)))
        after_rows = df.height
        duplicate_count = before_rows - after_rows

        if duplicate_count > 0:
            logger.info(f"✨ 중복 제거 완료: {duplicate_count:,}행 건너뜀")
            logger.info(f"   남은 신규 데이터: {after_rows:,}행")

        if after_rows == 0:
            logger.info("🎉 신규 추가할 데이터가 없습니다. 안전하게 종료합니다.")
            return

    # ===== 🚀 [핵심 변경] 병렬 전처리 및 GPU 배치 파이프라인 =====
    
    # 5. 데이터프레임을 청크로 분할 (전처리 병렬화를 위해)
    #    GPU 병렬 처리를 고려하여, 배치 사이즈와 비슷한 크기(예: 2048)로 조각냄
    chunk_size_for_processing = 2048  # GPU 메모리 오버플로우 방지를 위한 적절한 청크 크기
    chunks = chunk_dataframe(df, chunk_size=chunk_size_for_processing)
    logger.info(f"💡 {len(chunks)}개의 청크로 분할하여 병렬 처리 시작 (워커: {args.workers})")

    # 6. 멀티 프로세싱으로 전처리 및 임베딩 생성 병렬 실행
    #    참고: 각 프로세스는 KURE 모델을 독립적으로 로드하므로 메모리가 꽤 소모됩니다.
    from functools import partial
    preprocess_func = partial(_preprocess_single_chunk, fast_mode=False)
    
    processed_chunks = execute_parallel(
        func=preprocess_func,
        data_chunks=chunks,
        max_workers=args.workers,
        description="Embedding 생성 및 전처리 (병렬)"
    )

    # 7. 모든 청크 병합
    final_df = pl.concat(processed_chunks)
    logger.info(f"🧩 청크 병합 완료. 최종 전처리 데이터 수: {final_df.height:,}행")

    # 8. Neo4j 최종 적재
    _load_to_neo4j(df=final_df, reset=args.reset, batch_size=args.batch_size)
    
    # 9. Person-Hobby 관계(Edge) 생성 (그래프 구조 복구)
    _create_hobby_relationships(reset=args.reset)

    # 10. GNN 학습용 Person-Hobby edge/context CSV export
    _export_gnn_person_hobby_edges()


if __name__ == "__main__":
    main()
